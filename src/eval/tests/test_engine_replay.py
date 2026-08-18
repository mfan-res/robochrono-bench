#!/usr/bin/env python3
# coding: utf-8
"""端到端回归：用 replay provider 把整条执行链路跑一遍。

比 test_replay_regression.py 更强 —— 那个只验打分函数，这个把
engine 循环、JSONL 存储、断点恢复、导出一起验了，跑的是真实代码路径。

判据仍然是：所有 BC 开关关闭时，汇总指标与冻结脚本逐字节一致。
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

IGNORE = {"elapsed_seconds", "parse_failure_rate", "accuracy_answered", "parse_recovered"}

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


def make_task(name: str):
    if name == "time":
        return time_eqa.build()
    if name.startswith("trajectory"):
        return trajectory.build(name)
    return choice.build(name)


def diff_summary(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    diffs = []
    for key, old_value in old.items():
        if key in IGNORE or key not in new:
            continue
        new_value = new[key]
        if isinstance(old_value, float) and isinstance(new_value, (int, float)):
            if abs(old_value - float(new_value)) > 1e-9:
                diffs.append(f"{key}: {old_value} != {new_value}")
        elif old_value != new_value:
            diffs.append(f"{key}: {old_value!r} != {new_value!r}")
    return diffs


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="robochrono-replay-"))
    print(f"{'task':<16} {'rows':>6} {'汇总差异':>9} {'断点':>6}  status")
    print("-" * 60)
    failures = 0

    try:
        for name, qa_path in TASKS:
            baseline_path = BASELINE / f"{name}.json"
            if not baseline_path.exists():
                print(f"{name:<16} {'-':>6} {'-':>9} {'-':>6}  基线缺失")
                failures += 1
                continue

            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            old_rows = {str(r["id"]): r for r in baseline["results"]}
            items = [i for i in load_items(qa_path) if str(i.get("id")) in old_rows]

            task = make_task(name)
            # replay 表按 unit key 建：多数任务是 item id，time 是 video_id
            table = {
                unit.key: old_rows[str(unit.items[0]["id"])].get("model_output")
                for unit in task.units(items)
            }
            runtime: dict[str, Any] = {"type": "replay", "replay_table": table}

            store = ResultStore(work / f"{name}.jsonl", meta={"task": name, "provider": "replay"})
            summary = engine.run(task, items, runtime, store, overwrite=True)

            # 再跑一次，验证断点恢复：不应新增任何行
            before = len(list(store.rows()))
            engine.run(task, items, runtime, store)
            resumed_ok = len(list(store.rows())) == before

            diffs = diff_summary(baseline.get("summary", {}), summary)
            ok = not diffs and resumed_ok
            failures += 0 if ok else 1
            status = "OK" if ok else (diffs[0] if diffs else "断点恢复未生效")
            print(f"{name:<16} {before:>6} {len(diffs):>9} {'✓' if resumed_ok else '✗':>6}  {status[:44]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("-" * 60)
    print("端到端回归通过" if failures == 0 else f"{failures} 个任务未通过")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
