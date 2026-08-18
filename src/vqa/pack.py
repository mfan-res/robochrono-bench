#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第七步：把 `items.jsonl` 投影成评测端能吃的形状。

    python3 src/vqa/pack.py            # 投影并校验，不写盘
    python3 src/vqa/pack.py --write    # 写 data/vqa/eval/

为什么需要一个投影层
--------------------
④ 的产物走新契约（`prompt` / `truth` / `provenance`），
而 ⑥ 评测读的仍是 v1 的字段（`question` / `options` / `answer` / `input.clip_path`）。

**干净的契约留在 `items.jsonl`，适配消费者是这一步的职责。**
这样评测端一行都不用改，它那六套回归也就仍然有效 ——
改评测端等于同时动了「被测的东西」和「测它的尺子」。

顺手掐掉 v1 的一条泄漏
----------------------
v1 的 time 题在 `input` 里带着 `start` / `end`，而那**恰好等于答案**。
本步不发这两个字段。已确认评测端不读它们 ——
`time_eqa.py` 里对 `start`/`end` 的读取全部是在解析**模型输出**，不是读题目。

媒体路径写成相对路径
--------------------
相对于输出文件自身所在目录。**不写绝对路径** ——
BC-08 曾把绝对路径就地写进九个 QA 文件，结果掩盖了「走 normalized 与走原始 QA
发出去的内容其实不同」这个差别，事后才发现。

输出布局照搬 v1
---------------
`data/vqa/eval/<族>/<题型>.json`，与 v1 的 `datasets/QA/…` 同构，
评测端现有的配置与矩阵不用改。
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "data" / "vqa" / "items.jsonl"
OUT = ROOT / "data" / "vqa" / "eval"


def render(stem: str, options: list[dict[str, Any]]) -> str:
    """题干 + 选项，拼成 v1 的 `question` 形状。

    v1 把渲染后的整段存进 `question` 字段，评测端直接发这一整段。
    我们的 `items.jsonl` 里题干与选项是分开的（单一事实来源），
    在这里拼 —— 拼的规则只存在这一处。
    """
    if not options:
        return stem
    lines = "\n".join(f"{o['id']}. {o['text']}" for o in options)
    return f"{stem}\nOptions:\n{lines}"


def project(item: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    prov = item["provenance"]
    media = item["prompt"]["media"]
    rel = [os.path.relpath(ROOT / m["path"], out_dir) for m in media]
    key = "video_path" if item["task"] == "time" else "clip_path"

    row: dict[str, Any] = {
        "id": item["id"],
        "source_id": prov.get("segment_id"),
        "video_id": f"{item['family']}/{prov['episode']}",   # time 按它分组，须全局唯一
        "type": item["task"],
        "question": render(item["prompt"]["stem"], item["prompt"]["options"]),
        "answer": item["truth"]["answer"],
        "answer_text": item["truth"].get("answer_text"),
        # ⚠ input 里【只放媒体路径】。v1 在这里还放了 start/end，而那等于答案
        "input": {key: rel[0], **({f"{key}s": rel} if len(rel) > 1 else {})},
        "provenance": prov,
    }
    row["Q"] = row["question"]          # v1 的别名，冻结脚本读的是这个
    row["A"] = row["answer"]

    if item["prompt"]["options"]:
        row["options"] = [{"id": o["id"], "text": o["text"], "is_none_option": False}
                          for o in item["prompt"]["options"]]
        row["correct_option"] = next(o for o in row["options"]
                                     if o["id"] == item["truth"]["answer"])
    if item["task"] == "time":
        row["answer_seconds"] = item["truth"]["extra"]["seconds"]
    return row


def main() -> int:
    if not ITEMS.exists():
        print(f"❌ 没有 {ITEMS.relative_to(ROOT)} —— 先跑 src/vqa/compose.py --write")
        return 1
    items = [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines()]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for i in items:
        groups[(i["family"], i["task"])].append(i)

    print(f"投影 {len(items)} 道题 → {len(groups)} 个文件")
    written: list[tuple[str, str, int, Path]] = []
    problems: list[str] = []

    for (family, task), rows in sorted(groups.items()):
        out_dir = OUT / family
        packed = [project(i, out_dir) for i in rows]

        # ── 出厂检查 ──
        for p, orig in zip(packed, rows):
            for k in ("start", "end", "answer_seconds", "seconds"):
                if k in p["input"]:
                    problems.append(f"{p['id']}: input 里出现了 {k}（可反推答案）")
            for m, r in zip(orig["prompt"]["media"], [p["input"][k] for k in p["input"]][:1]):
                if not (out_dir / r).resolve().exists():
                    problems.append(f"{p['id']}: 媒体解析不到 {r}")
            if p["question"].count("Options:") > 1:
                problems.append(f"{p['id']}: 题干里重复出现 Options:")
        if task != "time":
            for p in packed:
                if p["answer"] not in {o["id"] for o in p["options"]}:
                    problems.append(f"{p['id']}: answer 不在 options 里")

        written.append((family, task, len(packed), out_dir / f"{task}.json"))
        if "--write" in sys.argv:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{task}.json").write_text(
                json.dumps({"items": packed}, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8")

    by_fam: dict[str, int] = defaultdict(int)
    for f, _, n, _ in written:
        by_fam[f] += n
    print(f"  {dict(by_fam)}")
    print(f"  题型 {dict(Counter(t for _, t, _, _ in written))}")

    if problems:
        print(f"\n❌ {len(problems)} 项不合格：")
        for x in problems[:6]:
            print(f"   {x}")
        return 1
    print("\n出厂检查：input 无真值字段 ✓　媒体可解析 ✓　答案在选项内 ✓")

    if "--write" in sys.argv:
        manifest = {"items": len(items), "files": [str(p.relative_to(ROOT)) for _, _, _, p in written],
                    "fingerprint": json.loads((ROOT / "build" / "plan.json")
                                              .read_text(encoding="utf-8")).get("fingerprint")}
        (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
        size = sum(p.stat().st_size for _, _, _, p in written)
        print(f"已写入 {OUT.relative_to(ROOT)}/（{size / 1e6:.1f} MB，含 manifest）")
    else:
        print("加 --write 写入 data/vqa/eval/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
