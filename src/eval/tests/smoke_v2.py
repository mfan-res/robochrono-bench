#!/usr/bin/env python3
# coding: utf-8
"""端到端冒烟：④ 出的题能不能被 ⑥ 完整处理。**不调模型。**

    python3 src/eval/tests/smoke_v2.py

走的是真实的那条路 —— 载入 `data/vqa/eval/` → 切 Unit → 组请求 parts →
喂一个假答案 → 打分 → 汇总。除了网络调用之外全覆盖。

**为什么需要这一条。** `pack.py` 自己的出厂检查只验字段形状；
「评测端能不能真的处理」得由评测端自己回答。
两边各写一套判据的话，迟早会对不上 —— 这个项目已经栽过三次
（词表两套、校验判据两套、元表读取两套）。

假答案的用法：
- 选择题喂正确字母 → 准确率必须是 1.0。**不是随便喂个字母看它不崩**，
  而是让「打分确实在打分」这件事本身可验证。
- time 喂真值区间 → 同理。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import paths as _P  # noqa: E402
from robochrono.tasks import choice, time_eqa  # noqa: E402
from robochrono.tasks.base import CallContext, load_items  # noqa: E402

PACKED = _P.EVAL.parents[1] / "data" / "vqa" / "eval"


def fake_answer(task: str, unit) -> str:
    """假装模型答对了 —— 这样准确率必须是 1.0，打分链路才算被验证过。"""
    if task == "time":
        # 多题一次调用，评测端认的是「按题目 id 索引」的 JSON。
        # 用序号（1. 2. 3.）它解析不了 —— 那是我第一版写错的地方。
        return json.dumps({it["id"]: it["answer"] for it in unit.items}, ensure_ascii=False)
    return unit.items[0]["answer"]


def main() -> int:
    if not PACKED.exists():
        print(f"❌ 没有 {PACKED} —— 先跑 src/vqa/pack.py --write")
        return 1

    print(f"{'族/题型':<28}{'题':>5}{'调用':>6}{'媒体':>6}{'准确率':>8}  说明")
    print("-" * 74)
    bad = 0
    for path in sorted(PACKED.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        family, task = path.parent.name, path.stem
        items = load_items(path)[:6]              # 每格取几道就够，验的是链路不是规模
        impl = time_eqa.build() if task == "time" else choice.build(task)

        units = impl.units(items)
        rows, media = [], 0
        try:
            for u in units:
                parts = impl.parts(u)
                media += sum(1 for p in parts if p["type"] in ("image", "video"))
                rows += impl.rows(u, fake_answer(task, u), CallContext())
        except Exception as exc:                   # noqa: BLE001
            print(f"{family + '/' + task:<28}{len(items):>5}{'—':>6}{'—':>6}{'✗':>8}  {type(exc).__name__}: {exc}")
            bad += 1
            continue

        # **time 的结果行里没有 `correct`** —— 它报的是 tIoU 这类连续指标，
        # 「多少算答对」这个阈值整个评测端都没有定义（披露清单第 10 条）。
        # 所以这里按题型换判据：选择题看 correct，time 看 tIoU 是否为 1。
        if task == "time":
            got = [r for r in rows if isinstance(r.get("tIoU"), (int, float))]
            acc = sum(1 for r in got if abs(float(r["tIoU"]) - 1.0) < 1e-6) / len(got) if got else 0.0
            note = "✓ tIoU=1" if acc == 1.0 else f"⚠ 喂了真值区间，tIoU=1 的只有 {acc:.0%}"
        else:
            got = [r for r in rows if r.get("correct") is not None]
            acc = sum(bool(r["correct"]) for r in got) / len(got) if got else 0.0
            note = "✓" if acc == 1.0 else f"⚠ 喂了正确答案却只得 {acc:.0%}"
        if acc != 1.0:
            bad += 1
        print(f"{family + '/' + task:<28}{len(items):>5}{len(units):>6}{media:>6}{acc:>8.0%}  {note}")

    print("-" * 74)
    print("全部通过 —— ④ 的题能被 ⑥ 完整处理" if not bad else f"❌ {bad} 格有问题")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
