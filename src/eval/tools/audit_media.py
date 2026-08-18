#!/usr/bin/env python3
# coding: utf-8
"""审计各任务族的媒体路径能否解析，并报告按文件名解析的效果。

    python tools/audit_media.py                     # 全部族
    python tools/audit_media.py --family tea2       # 单个族
    python tools/audit_media.py --list-unresolved   # 列出解析不了的文件名

只读，不改任何文件。解析在内存里完成，用于判断某个族是否可以投入评测。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from robochrono import tasks  # noqa: E402
from robochrono.mediaindex import ResolveStats, index_for_qa, resolve_items  # noqa: E402
from robochrono.tasks.base import load_items  # noqa: E402

# 用 tasks.qa_path 而不是自己拼路径 —— 一半的族 QA 多嵌了一层子目录，
# 写死布局会让它们整族显示「无 QA 文件」。
RUNS = ["understanding", "left_right", "image_in_video", "time",
        "planning", "planning_2", "step_order"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, default=EVAL_ROOT / "datasets")
    parser.add_argument("--family", default=None)
    parser.add_argument("--list-unresolved", action="store_true")
    args = parser.parse_args()

    qa_root = args.datasets_root / "QA"
    families = ([args.family] if args.family else
                sorted({p.name for group in ("planning", "understanding")
                        for p in (qa_root / group).iterdir() if p.is_dir()}))

    print(f"{'族':<13} {'run':<15} {'题数':>5} {'路径':>6} {'原本就对':>8} {'解析回来':>8} "
          f"{'重名':>5} {'找不到':>6}")
    print("-" * 78)
    grand = ResolveStats()
    unresolved_all: Counter[str] = Counter()

    for family in families:
        family_total = ResolveStats()
        printed = False
        for run in RUNS:
            try:
                qa_path = tasks.qa_path(args.datasets_root, family, run)
            except ValueError as exc:
                print(f"{family:<13} {run:<15} ⚠️ {exc}".split(chr(10))[0])
                continue
            if not qa_path.exists():
                continue
            printed = True
            items = load_items(qa_path)
            stats = ResolveStats()
            resolve_items(items, index_for_qa(qa_path), stats)
            for attribute in ("total", "already_ok", "resolved", "provenance_skipped"):
                setattr(family_total, attribute,
                        getattr(family_total, attribute) + getattr(stats, attribute))
                setattr(grand, attribute,
                        getattr(grand, attribute) + getattr(stats, attribute))
            family_total.ambiguous += stats.ambiguous
            family_total.unresolved += stats.unresolved
            grand.ambiguous += stats.ambiguous
            grand.unresolved += stats.unresolved
            unresolved_all.update(stats.unresolved)
            flag = "" if not (stats.ambiguous or stats.unresolved) else "  ⚠️"
            print(f"{family:<13} {run:<15} {len(items):>5} {stats.total:>6} "
                  f"{stats.already_ok:>8} {stats.resolved:>8} "
                  f"{len(stats.ambiguous):>5} {len(stats.unresolved):>6}{flag}")
        if not printed:
            print(f"{family:<13} {'（无 QA 文件）':<15}")
        elif family_total.total:
            print(f"{'':13} {'└ 小计':<15} {'':>5} {family_total.total:>6} "
                  f"{family_total.already_ok:>8} {family_total.resolved:>8} "
                  f"{len(family_total.ambiguous):>5} {len(family_total.unresolved):>6}   "
                  f"可用率 {family_total.usable / family_total.total:.1%}")

    print("-" * 78)
    print(f"合计：{grand.summary()}")
    if grand.total:
        print(f"总可用率 {grand.usable / grand.total:.1%}")

    if args.list_unresolved and unresolved_all:
        print("\n找不到的文件名（前 30）：")
        for name, count in unresolved_all.most_common(30):
            print(f"   {count:>4}×  {name}")
    if grand.ambiguous:
        print(f"\n重名未解析（前 10，共 {len(grand.ambiguous)}）：")
        for entry in list(dict.fromkeys(grand.ambiguous))[:10]:
            print(f"   {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
