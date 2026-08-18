#!/usr/bin/env python3
# coding: utf-8
"""把结果 JSONL 按 id 去重压实。

结果文件是追加日志：一题失败会写 error 行，续跑成功后再追加一行成功的，
同一个 id 于是出现多次。读取侧（``ResultStore.final_rows``）已经会取舍，
所以**这个工具不是正确性必需的** —— 它只是让文件变小、变干净，方便人直接看。

取舍规则与 ``final_rows`` 完全一致，两边不能分叉：
  有成功行取最后一条成功行，全失败取最后一条失败行。

    python tools/compact_results.py results/full            # 只报告
    python tools/compact_results.py results/full --write    # 实际改写
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robochrono.store import ResultStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true", help="实际改写文件，默认只报告")
    args = parser.parse_args()

    files = sorted(args.root.rglob("*.jsonl"))
    if not files:
        print(f"{args.root} 下没有 .jsonl")
        return 1

    total_before = total_after = touched = 0
    print(f"{'文件':<52} {'原始':>7} {'去重后':>7} {'error':>6}")
    print("-" * 76)

    for path in files:
        store = ResultStore(path, meta={})
        raw = sum(1 for _ in store.rows())
        final = store.final_rows()
        errors = sum(1 for r in final if r.get("error"))
        total_before += raw
        total_after += len(final)

        if len(final) != raw:
            touched += 1
            label = str(path.relative_to(args.root))
            print(f"{label:<52} {raw:>7} {len(final):>7} {errors:>6}")
            if args.write:
                # 先写临时文件再原子改名，中途挂掉不会毁掉原始结果
                tmp = path.with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as handle:
                    for row in final:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                tmp.replace(path)

    print("-" * 76)
    print(f"{len(files)} 个文件，其中 {touched} 个有重复；总行数 {total_before} → {total_after}")
    if touched and not args.write:
        print("这是预演。加 --write 实际改写。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
