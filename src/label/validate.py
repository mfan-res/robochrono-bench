#!/usr/bin/env python3
# coding: utf-8
"""标注校验器 —— **遍历全族 + 出报表**。判据本身在 ``checks.py``。

    python3 src/label/validate.py            # 全部族
    python3 src/label/validate.py wash tea   # 只看某几族

一套判据，两处使用
------------------
标注工具保存时跑（``serve.py`` 的 ``review()``），以及对存量离线跑（本文件）。
**两边 import 同一个 ``checks.check_document``** —— 这不是约定，是结构。

> 此前不是这样：`serve.py` 自己抄了一份只有三类的检查，而四处文档都写着
> 「共用同一份判据」。那正是 v1 时代 `check_labels.py` 与标注工具各写各的
> 那个坑的重演（tea2 显示「21/21 齐全」而实际只有 20 集可用）。

八类检查的由来、以及「重叠为什么必须走帧号」，见 ``checks.py`` 的说明。

只报不修
--------
修法要逐条讨论 —— 有的能在出题阶段绕开（P-01 裁到第 0 集），
有的要还原数据（P-03），有的只能改能力定义（P-05）。代价差很多。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LABEL = ROOT / "data" / "label"
RAW = ROOT / "data" / "raw"

from checks import SEVERITY, check_document  # noqa: E402

# `episode_bounds` 只有一份实现，在 `src/vqa/index.py`。
# **这里曾经自己抄了一份**，于是那份里的两个 bug（只读第一个 parquet、
# 靠列顺序猜主视角）在 index.py 修好之后仍然留在这里 ——
# 正是「同一个东西存两份，不一致时不报错」的现场（D-42）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vqa"))
from index import episode_bounds  # noqa: E402


@dataclass
class Finding:
    kind: str
    family: str
    episode: str
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, kind: str, family: str, episode: str, detail: str) -> None:
        self.findings.append(Finding(kind, family, episode, detail))


def check_family(family: str, report: Report) -> None:
    """遍历一个族的所有标注文件。**判据不在这里** —— 见 checks.check_document。"""
    base = LABEL / family
    # 缺 subtasks.json 说明这个目录不是一个成形的族（例如刚删了一半）。
    # **报告，不崩溃** —— 崩溃会让其余六族的检查结果一起拿不到。
    if not (base / "subtasks.json").exists():
        report.add("引用", family, "-", "缺 subtasks.json，跳过该目录")
        return
    defined = json.loads((base / "subtasks.json").read_text(encoding="utf-8"))["subtasks"]
    subtasks = {s["id"] for s in defined}
    texts = {s["id"]: s["text"] for s in defined}
    bounds = episode_bounds(family)
    meta_path = RAW / family / "meta.json"
    fps = (json.loads(meta_path.read_text(encoding="utf-8")).get("fps")
           if meta_path.exists() else None)

    counts: Counter = Counter()
    for path in sorted((base / "segments").glob("*_segments.json")):
        episode = path.stem.replace("_segments", "")
        document = json.loads(path.read_text(encoding="utf-8"))
        found, sub = check_document(document, subtasks=subtasks, texts=texts,
                                    fps=fps, bounds=bounds.get(episode))
        counts.update(sub)
        for f in found:
            report.add(f.kind, family, episode, f.detail)
    report.stats[family] = dict(counts)


def main() -> int:
    only = sys.argv[1:]
    families = sorted(p.name for p in LABEL.iterdir()
                      if p.is_dir() and (not only or p.name in only))
    report = Report()
    for family in families:
        check_family(family, report)

    print(f"{'族':<13}{'段数':>6}{'污染文件':>9}{'帧重叠':>8}{'派生不符':>9}"
          f"{'打包视频':>9}{'漏标集':>8}{'歧义':>6}{'可疑':>6}")
    print("-" * 74)
    for family in families:
        s = report.stats[family]
        print(f"{family:<13}{s.get('segments',0):>6}{s.get('polluted_files',0):>9}"
              f"{s.get('frame_overlap',0):>8}{s.get('derived_mismatch',0):>9}"
              f"{s.get('packed_files',0):>9}{s.get('episodes_unlabeled',0):>8}"
              f"{s.get('ambiguous_files',0):>6}"
              f"{s.get('zero_length',0)+s.get('same_start',0):>6}")
    print("-" * 74)

    by_kind = Counter(f.kind for f in report.findings)
    if not report.findings:
        print("八类检查全部通过")
        return 0
    print(f"\n共 {len(report.findings)} 条发现：{dict(by_kind)}\n")
    for kind in ("污染", "引用", "派生", "重叠", "覆盖", "序列", "歧义", "可疑"):
        items = [f for f in report.findings if f.kind == kind]
        if not items:
            continue
        print(f"{SEVERITY[kind]} {kind}（{len(items)} 条）")
        seen: set[str] = set()
        for f in items:
            key = f"{f.family}|{f.detail[:40]}"
            if key in seen:
                continue
            seen.add(key)
            print(f"    {f.family}/{f.episode}  {f.detail}")
            if len(seen) >= 4:
                rest = len(items) - 4
                if rest > 0:
                    print(f"    …… 另有 {rest} 条同类")
                break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
