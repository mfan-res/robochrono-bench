#!/usr/bin/env python3
# coding: utf-8
"""并发等价性回归：``concurrency=N`` 的结果必须与 ``concurrency=1`` 完全一致。

API 侧串行跑不完（一个任务族 2,450 次调用约 13 小时，真实矩阵 245,000 次
要几周），所以 engine 里加了线程池。但并发只有在**不改变结果**时才可用，
这个测试就是那条判据。

用 replay provider：输出由录制表决定，是确定性的，所以任何差异都只可能
来自并发本身（共享状态被踩、结果串行化出错、落盘交错）。

同时验两件事：
  内容   逐行比对，包括打分字段与 model_output
  完整   并发路径不能丢行或重复行

不需要 GPU 或 API key。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _P  # noqa: E402
sys.path.insert(0, str(_P.EVAL))

from robochrono import engine  # noqa: E402
from robochrono.store import ResultStore  # noqa: E402
from robochrono.tasks import choice, time_eqa, trajectory  # noqa: E402
from robochrono.tasks.base import load_items  # noqa: E402

BASELINE = _P.BASELINE
QA = _P.QA

# timing 是墙钟耗时，并发下必然不同 —— 这是预期的，不算差异。
IGNORE_ROW = {"timing"}
IGNORE_SUMMARY = {"elapsed_seconds"}

TASKS = [
    ("understanding", QA / "understanding/stack_cubes/understanding_vqa.json"),
    ("left_right", QA / "understanding/stack_cubes/left_right_vqa.json"),
    ("image_in_video", QA / "understanding/stack_cubes/image_in_video_vqa.json"),
    ("planning", QA / "planning/stack_cubes/planning_vqa.json"),
    ("planning_2", QA / "planning/stack_cubes/planning_2_vqa.json"),
    ("step_order", QA / "planning/stack_cubes/step_order_vqa.json"),
    ("time", QA / "understanding/stack_cubes/time_vqa.json"),
    ("trajectory_2D", QA / "planning/stack_cubes/trajectory_qa_2d.json"),
    ("trajectory_3D", QA / "planning/stack_cubes/trajectory_qa_3d.json"),
]

CONCURRENCY = 8


def make_task(name: str):
    if name == "time":
        return time_eqa.build()
    if name.startswith("trajectory"):
        return trajectory.build(name)
    return choice.build(name)


def strip(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in IGNORE_ROW}


def compare(serial: list[dict], concurrent: list[dict]) -> list[str]:
    """按 id 比对。并发下完成顺序不确定，所以比集合而非序列。"""
    diffs: list[str] = []
    a = {str(r["id"]): strip(r) for r in serial}
    b = {str(r["id"]): strip(r) for r in concurrent}

    if len(a) != len(serial):
        diffs.append(f"串行结果里 id 重复：{len(serial)} 行 / {len(a)} 个 id")
    if len(b) != len(concurrent):
        diffs.append(f"并发结果里 id 重复：{len(concurrent)} 行 / {len(b)} 个 id")
    for missing in sorted(set(a) - set(b)):
        diffs.append(f"并发丢了 id={missing}")
    for extra in sorted(set(b) - set(a)):
        diffs.append(f"并发多了 id={extra}")

    for key in sorted(set(a) & set(b)):
        if a[key] != b[key]:
            fields = [f for f in set(a[key]) | set(b[key])
                      if a[key].get(f) != b[key].get(f)]
            diffs.append(f"id={key} 字段不同：{','.join(sorted(fields))}")
    return diffs


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="robochrono-conc-"))
    print(f"并发度 {CONCURRENCY}，provider=replay\n")
    print(f"{'task':<16} {'units':>6} {'rows':>6} {'行差异':>7} {'汇总差异':>9}  status")
    print("-" * 66)
    failures = 0

    try:
        for name, qa_path in TASKS:
            baseline_path = BASELINE / f"{name}.json"
            if not baseline_path.exists():
                print(f"{name:<16} {'-':>6} {'-':>6} {'-':>7} {'-':>9}  基线缺失")
                failures += 1
                continue

            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            old_rows = {str(r["id"]): r for r in baseline["results"]}
            items = [i for i in load_items(qa_path) if str(i.get("id")) in old_rows]

            task = make_task(name)
            units = task.units(items)
            table = {
                unit.key: old_rows[str(unit.items[0]["id"])].get("model_output")
                for unit in units
            }

            def run(concurrency: int) -> tuple[list[dict], dict]:
                # 每次用全新的 runtime 与 store —— 不让两次跑共享任何状态
                runtime: dict[str, Any] = {"type": "replay", "replay_table": dict(table)}
                store = ResultStore(work / f"{name}-c{concurrency}.jsonl",
                                    meta={"task": name, "provider": "replay"})
                summary = engine.run(task, items, runtime, store, overwrite=True,
                                     concurrency=concurrency)
                return list(store.rows()), summary

            serial_rows, serial_summary = run(1)
            conc_rows, conc_summary = run(CONCURRENCY)

            row_diffs = compare(serial_rows, conc_rows)
            sum_diffs = [
                f"{k}: {serial_summary[k]!r} != {conc_summary.get(k)!r}"
                for k in serial_summary
                if k not in IGNORE_SUMMARY and serial_summary[k] != conc_summary.get(k)
            ]

            ok = not row_diffs and not sum_diffs
            failures += 0 if ok else 1
            status = "OK" if ok else (row_diffs + sum_diffs)[0][:26]
            print(f"{name:<16} {len(units):>6} {len(serial_rows):>6} "
                  f"{len(row_diffs):>7} {len(sum_diffs):>9}  {status}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("-" * 66)
    print("并发与串行结果一致" if failures == 0 else f"{failures} 个任务出现差异")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
