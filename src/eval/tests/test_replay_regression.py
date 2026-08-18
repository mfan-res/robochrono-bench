#!/usr/bin/env python3
# coding: utf-8
"""Replay 回归：用基线里录下的真实模型输出重放，比对新旧实现的打分与汇总。

阶段 1 的门禁。在所有 BC 开关关闭的前提下，新实现对同一份模型输出必须给出
与冻结脚本**完全相同**的每题打分和汇总指标，否则重构就改变了评分。

不需要 GPU，不需要 API —— 模型输出来自 eval/results/baseline/。
基线由 eval/tools/smoke_all.sh 用冻结脚本生成。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _P  # noqa: E402
sys.path.insert(0, str(_P.EVAL))

from robochrono.tasks import choice, time_eqa, trajectory  # noqa: E402
from robochrono.mediaindex import index_for_qa, resolve_items  # noqa: E402
from robochrono.tasks.base import CallContext, load_items  # noqa: E402
# 为什么这里要解析媒体路径
# ------------------------
# 原始 QA 里的路径有三种风格（相对 / 生成机绝对 / Windows 反斜杠），
# 三种在本机都不能直接用。生产路径上由 `mediaindex` 在构建期解析，
# 任务代码拿到的已经是本机绝对路径。测试若喂未解析的 item，
# 比的就不是生产里跑的那条路。
#
# 这一步以前不需要，是因为 BC-08 曾把绝对路径就地写进九个 QA 文件，
# 掩盖了差别。那次改写已还原（原始数据只读），所以差别显出来了。


BASELINE = _P.BASELINE
QA = _P.QA

# 汇总里必然不同的字段（耗时）与阶段 1 新增的字段，不参与比对
SUMMARY_IGNORE = {
    "elapsed_seconds",
    "parse_failure_rate",
    "accuracy_answered",
    "parse_recovered",
}

TASKS: list[tuple[str, Path, Any]] = [
    ("understanding", QA / "understanding/stack_cubes/understanding_vqa.json", None),
    ("left_right", QA / "understanding/stack_cubes/left_right_vqa.json", None),
    ("image_in_video", QA / "understanding/stack_cubes/image_in_video_vqa.json", None),
    ("planning", QA / "planning/stack_cubes/planning_vqa.json", None),
    ("planning_2", QA / "planning/stack_cubes/planning_2_vqa.json", None),
    ("step_order", QA / "planning/stack_cubes/step_order_vqa.json", None),
    ("time", QA / "understanding/stack_cubes/time_vqa.json", None),
    ("trajectory_2D", QA / "planning/stack_cubes/trajectory_qa_2d.json", None),
    ("trajectory_3D", QA / "planning/stack_cubes/trajectory_qa_3d.json", None),
]


def make_task(name: str):
    if name == "time":
        return time_eqa.build()
    if name.startswith("trajectory"):
        return trajectory.build(name)
    return choice.build(name)


def compare_rows(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """比对旧结果行与新结果行的共有字段。新行按 BC-04 精简过，只比对交集。"""
    skip = {"prompt", "model_output", "model_prediction", "timing", "model_answer"}
    # 媒体路径按**文件名**比，不按绝对路径。
    # 基线是在旧仓库录的，绝对路径必然不同；但直接忽略会掩盖「切错了片段」这类真错误，
    # 所以退一步比文件名 —— 文件名里带着族/集/帧号，切错了照样能发现。
    PATH_KEYS = {"video_path", "clip_path", "image_path", "initial_image", "image",
                 "video_paths", "clip_paths", "image_paths"}

    def names(v: Any) -> Any:
        if isinstance(v, str):
            return Path(v).name
        if isinstance(v, list):
            return [names(x) for x in v]
        if isinstance(v, dict):
            return {k: names(x) for k, x in v.items()}
        return v

    diffs = []
    for key, old_value in old.items():
        if key in skip or key not in new:
            continue
        new_value = new[key]
        if key in PATH_KEYS:
            if names(old_value) != names(new_value):
                diffs.append(f"{key}（按文件名比）: {names(old_value)} != {names(new_value)}")
            continue
        if isinstance(old_value, float) and isinstance(new_value, (int, float)):
            if abs(old_value - float(new_value)) > 1e-9:
                diffs.append(f"{key}: {old_value!r} != {new_value!r}")
        elif old_value != new_value:
            diffs.append(f"{key}: {old_value!r} != {new_value!r}")
    return diffs


def main() -> int:
    print(f"{'task':<16} {'items':>6} {'行差异':>7} {'汇总差异':>9}  status")
    print("-" * 62)
    failures = 0

    for name, qa_path, _ in TASKS:
        baseline_path = BASELINE / f"{name}.json"
        if not baseline_path.exists() or not qa_path.exists():
            print(f"{name:<16} {'-':>6} {'-':>7} {'-':>9}  基线或 QA 缺失")
            failures += 1
            continue

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        old_rows = {str(r["id"]): r for r in baseline["results"]}
        recorded = {rid: r.get("model_output") for rid, r in old_rows.items()}

        items = [i for i in load_items(qa_path) if str(i.get("id")) in old_rows]
        resolve_items(items, index_for_qa(qa_path))
        task = make_task(name)
        ctx = CallContext()

        new_rows: list[dict[str, Any]] = []
        for unit in task.units(items):
            text = recorded.get(str(unit.items[0]["id"]))
            if text is None:
                new_rows.extend(task.error_rows(unit, str(old_rows[unit.key].get("error") or "no output")))
                continue
            try:
                new_rows.extend(task.rows(unit, text, ctx))
            except Exception as exc:  # noqa: BLE001
                new_rows.extend(task.error_rows(unit, f"{type(exc).__name__}: {exc}"))

        row_diffs = 0
        first_diff = ""
        for row in new_rows:
            old = old_rows.get(row["id"])
            if old is None:
                continue
            diffs = compare_rows(old, row)
            if diffs:
                row_diffs += 1
                if not first_diff:
                    first_diff = f"{row['id']}  " + " ｜ ".join(diffs[:2])

        new_summary = task.summarize(new_rows, 0.0)
        old_summary = baseline.get("summary", {})
        summary_diffs = []
        for key, old_value in old_summary.items():
            if key in SUMMARY_IGNORE or key not in new_summary:
                continue
            new_value = new_summary[key]
            if isinstance(old_value, float) and isinstance(new_value, (int, float)):
                if abs(old_value - float(new_value)) > 1e-9:
                    summary_diffs.append(f"{key}: {old_value} != {new_value}")
            elif old_value != new_value:
                summary_diffs.append(f"{key}: {old_value!r} != {new_value!r}")

        ok = row_diffs == 0 and not summary_diffs
        failures += 0 if ok else 1
        status = "OK" if ok else (first_diff or summary_diffs[0])[:60]
        print(f"{name:<16} {len(new_rows):>6} {row_diffs:>7} {len(summary_diffs):>9}  {status}")
        if summary_diffs and row_diffs == 0:
            for d in summary_diffs[:3]:
                print(f"    汇总: {d}")

    print("-" * 62)
    print("回归通过" if failures == 0 else f"{failures} 个任务未通过")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
