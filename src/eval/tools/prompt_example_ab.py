#!/usr/bin/env python3
# coding: utf-8
"""A/B：schema 里的示例点写法，对轨迹预测的影响。

起因：实测发现 3D 上 **58% 的预测是逐字抄回 prompt 里的示例点**
``[0.12, -0.03, 0.45]``（SenseNova 92%、Qwen3-VL-8B 74%）。
那个 schema 看起来不像格式说明，像一个填好的答案。

本脚本用同一模型、同一批题、同样的生成参数，只改 schema 的示例写法：

    legacy       [[0.12, -0.03, 0.45]]                     冻结版
    placeholder  [[<x1>, <y1>, <z1>], [<x2>, ...], ...]    占位符 + 多点

对比三件事：照抄率、解析出的点数、分数。

    python tools/prompt_example_ab.py --model SenseNova-SI-1.1-InternVL3-2B --dim 3D -n 30
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from robochrono.tasks import trajectory as T  # noqa: E402
from robochrono.tasks.base import CallContext, load_items  # noqa: E402
from robochrono.vlm_api import call_vlm, runtime_config  # noqa: E402

PROVIDERS = {
    "RynnBrain-2B": "local_rynnbrain_2b",
    "Qwen3-VL-8B-Instruct": "local_qwen",
    "SenseNova-SI-1.1-InternVL3-2B": "local_sensenova_si_1_1_internvl3_2b",
    "Cosmos3-Edge-2B": "local_cosmos3_edge_2b",
}
QA = {"2D": "trajectory_qa_2d.json", "3D": "trajectory_qa_3d.json"}
EXAMPLE = {"2D": [123.4, 256.7], "3D": [0.12, -0.03, 0.45]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="SenseNova-SI-1.1-InternVL3-2B")
    parser.add_argument("--dim", choices=["2D", "3D"], default="3D")
    parser.add_argument("-n", "--items", type=int, default=30)
    parser.add_argument("--styles", nargs="+", default=["legacy", "placeholder"])
    parser.add_argument("--pose", nargs="+", type=int, default=[0],
                        help="0=不给初始位姿 1=给；可同时给两者做 A/B")
    args = parser.parse_args()

    runtime = runtime_config(config_path=EVAL_ROOT / "configs/providers.json",
                             provider_name=PROVIDERS[args.model], default_model="")
    items = load_items(EVAL_ROOT / "datasets/QA/planning/stack_cubes" / QA[args.dim])[: args.items]
    example = EXAMPLE[args.dim]

    print(f"模型 {args.model}   {args.dim}   {len(items)} 题\n")
    print(f"{'示例写法':<14} {'照抄示例':>8} {'点数中位':>9} {'空/1点':>8} {'平均分':>8} {'跑赢盲基线?':>11}")
    print("-" * 68)
    blind = 3.19 if args.dim == "2D" else 0.75

    combos = [(st, bool(po)) for st in args.styles for po in args.pose]
    for style, pose in combos:
        task = T.TrajectoryTask(f"trajectory_{args.dim}", example_style=style,
                                include_initial_pose=pose)
        echo = degenerate = 0
        counts: list[int] = []
        scores: list[float] = []
        for item in items:
            unit = type("U", (), {"key": str(item["id"]), "items": [item]})()
            meta: dict[str, Any] = {}
            try:
                _, text = call_vlm(runtime, task.parts(unit), meta)
            except Exception as exc:  # noqa: BLE001
                print(f"    {item['id']}: {type(exc).__name__}: {exc}")
                continue
            row = task.rows(unit, text, CallContext(frames_used=meta.get("frames_used", {}),
                                                    usage=meta.get("usage", {}),
                                                    media_transforms=[]))[0]
            gripper = (row.get("scored_grippers") or [None])[0]
            points = (row.get("predicted_trajectory") or {}).get(gripper) or []
            counts.append(len(points))
            if points and all(p == example for p in points):
                echo += 1
            if len(points) <= 1:
                degenerate += 1
            scores.append(float(row.get("score") or 0.0))

        n = max(1, len(scores))
        mean = statistics.mean(scores) if scores else 0.0
        label = f"{style}{'+pose' if pose else ''}"
        print(f"{label:<14} {echo:>4}/{n:<3} {statistics.median(counts) if counts else 0:>9.0f} "
              f"{degenerate:>4}/{n:<3} {mean:>8.2f} {'✓' if mean > blind else '✗':>11}")

    print("-" * 68)
    print(f"盲基线（固定点重复十次）= {blind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
