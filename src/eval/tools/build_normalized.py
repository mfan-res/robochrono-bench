#!/usr/bin/env python3
# coding: utf-8
"""构建规范化数据集。

    python tools/build_normalized.py --verify         # 只查产物是否过期，不构建
    python tools/build_normalized.py --check          # 预演构建，不写盘
    python tools/build_normalized.py                  # 全部族
    python tools/build_normalized.py --family wash    # 单个族

产物在 ``datasets/normalized/``（gitignore，可完全重建）。
原始 ``datasets/QA/`` 只读，不会被改动。

``--verify`` 适合放进 CI 或跑评测前 —— 评测本身也会查（加载时自动校验），
但提前查能在花几小时跑矩阵之前就发现「忘了重建」。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from robochrono.normalize import build, check_freshness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, default=EVAL_ROOT / "datasets")
    parser.add_argument("--out", type=Path, default=None,
                        help="默认 <datasets-root>/normalized")
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--check", action="store_true", help="预演构建，不写盘")
    parser.add_argument("--verify", action="store_true",
                        help="只检查现有产物是否过期，不构建；过期时退出码非零")
    parser.add_argument("--source-sha", default="",
                        help="远端仓库 commit，写进 manifest 供过期检测")
    args = parser.parse_args()

    out_root = args.out or (args.datasets_root / "normalized")

    if args.verify:
        state = check_freshness(args.datasets_root, out_root)
        if state.ok:
            print(f"产物是最新的（{out_root}）")
            return 0
        print(f"产物已过期（{len(state.reasons)} 项）：")
        for reason in state.reasons[:20]:
            print(f"  - {reason}")
        if len(state.reasons) > 20:
            print(f"  …… 另有 {len(state.reasons) - 20} 项")
        print("\n重建：python tools/build_normalized.py")
        return 1

    manifest = build(args.datasets_root, out_root, families=args.family,
                     dry_run=args.check, source_sha=args.source_sha)

    print(f"{'族':<22} {'布局':<10} {'schema':<9} {'任务':>5} {'题数':>7} "
          f"{'媒体':>7} {'未解析':>7} {'重名':>6}")
    print("-" * 82)
    totals = dict(items=0, media=0, unresolved=0, ambiguous=0)
    for family, entry in sorted(manifest["families"].items()):
        runs = entry.get("runs", {})
        if not runs:
            print(f"{family:<22} {entry.get('layout','?'):<10} {'—':<9} {0:>5}"
                  f"{'   （无可用 QA）':>28}")
            continue
        sums = {k: sum(r[k] for r in runs.values())
                for k in ("items", "media", "unresolved", "ambiguous")}
        for key in totals:
            totals[key] += sums[key]
        flag = "  ⚠️" if entry.get("missing_runs") else ""
        print(f"{family:<22} {entry.get('layout','?'):<10} "
              f"{entry.get('schema_version','n/a'):<9} {len(runs):>5} {sums['items']:>7} "
              f"{sums['media']:>7} {sums['unresolved']:>7} {sums['ambiguous']:>6}{flag}")
    print("-" * 82)
    print(f"{'合计':<22} {'':<10} {'':<9} {'':>5} {totals['items']:>7} "
          f"{totals['media']:>7} {totals['unresolved']:>7} {totals['ambiguous']:>6}")
    if totals["media"]:
        usable = totals["media"] - totals["unresolved"] - totals["ambiguous"]
        print(f"媒体可用率 {usable / totals['media']:.1%}")

    issues = manifest["issues"]
    if issues:
        print(f"\n问题清单（{len(issues)} 条）：")
        for issue in issues[:12]:
            detail = issue.get("detail") or issue.get("examples") or ""
            print(f"  [{issue['kind']}] {issue.get('family','')}/{issue.get('run','')}"
                  f"  {issue.get('count','')}  {str(detail)[:56]}")
        if len(issues) > 12:
            print(f"  … 另有 {len(issues) - 12} 条，见 manifest.json")

    if args.check:
        print("\n这是预演（--check），未写盘。")
    else:
        print(f"\n已写入 {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
