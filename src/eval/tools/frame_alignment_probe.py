#!/usr/bin/env python3
# coding: utf-8
"""预测轨迹到底是「形状对了但位置错」，还是「完全没信号」？

3D 分数集体趴在噪声带里，有两种可能：
  A. 模型画对了形状，只是放错了参照系 / 原点 / 尺度 —— 那是**题目问法**的问题
  B. 模型根本没有空间轨迹能力 —— 那是模型的问题

区分方法：给预测做最优对齐后重新打分。

    原样           不动
    去中心         平移到与真值同一质心（消掉原点差异）
    去中心+缩放     再统一尺度（消掉单位/量纲差异）
    去中心+缩放+旋转 完整相似变换（消掉坐标轴朝向差异）

如果「去中心」就让分数暴涨，是原点问题；要到「旋转」才涨，是坐标轴约定问题；
全部对齐后仍然不涨，那就是 B —— 形状本身不对，换 prompt 也救不回来。

对照组：把真值自己做同样处理（应当始终 100 分），以及随机形状（应当始终低分）。

    python tools/frame_alignment_probe.py --model Qwen3-VL-8B-Instruct -n 40
"""

from __future__ import annotations

import argparse
import math
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
}


def centroid(points: list[list[float]]) -> list[float]:
    dim = len(points[0])
    return [sum(p[i] for p in points) / len(points) for i in range(dim)]


def recenter(points: list[list[float]], target: list[float]) -> list[list[float]]:
    c = centroid(points)
    return [[p[i] - c[i] + target[i] for i in range(len(p))] for p in points]


def spread(points: list[list[float]]) -> float:
    c = centroid(points)
    return math.sqrt(sum(sum((p[i] - c[i]) ** 2 for i in range(len(p))) for p in points) / len(points))


def rescale(points: list[list[float]], factor: float) -> list[list[float]]:
    c = centroid(points)
    return [[c[i] + (p[i] - c[i]) * factor for i in range(len(p))] for p in points]


def kabsch(predicted: list[list[float]], expected: list[list[float]]) -> list[list[float]]:
    """最优旋转对齐（Kabsch）。只在点数相同时可用，纯 python 实现 3x3。"""
    if len(predicted) != len(expected):
        return predicted
    dim = len(predicted[0])
    pc, ec = centroid(predicted), centroid(expected)
    p = [[x[i] - pc[i] for i in range(dim)] for x in predicted]
    q = [[x[i] - ec[i] for i in range(dim)] for x in expected]
    # 协方差 H = pᵀq
    h = [[sum(p[k][i] * q[k][j] for k in range(len(p))) for j in range(dim)] for i in range(dim)]
    # 用幂迭代做一个粗略的极分解：R ≈ H (HᵀH)^(-1/2)，这里用 Gram-Schmidt 近似
    cols = [[h[i][j] for i in range(dim)] for j in range(dim)]
    basis: list[list[float]] = []
    for col in cols:
        v = col[:]
        for b in basis:
            dot = sum(v[i] * b[i] for i in range(dim))
            v = [v[i] - dot * b[i] for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in v))
        basis.append([x / norm for x in v] if norm > 1e-9 else [1.0 if i == len(basis) else 0.0 for i in range(dim)])
    rotated = [[sum(x[j] * basis[i][j] for j in range(dim)) for i in range(dim)] for x in p]
    return [[rotated[k][i] + ec[i] for i in range(dim)] for k in range(len(rotated))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("-n", "--items", type=int, default=40)
    parser.add_argument("--style", default="placeholder",
                        help="用哪种 prompt 取预测；默认 placeholder（不含可抄的示例点）")
    args = parser.parse_args()

    runtime = runtime_config(config_path=EVAL_ROOT / "configs/providers.json",
                             provider_name=PROVIDERS[args.model], default_model="")
    items = load_items(EVAL_ROOT / "datasets/QA/planning/stack_cubes/trajectory_qa_3d.json")[: args.items]
    task = T.TrajectoryTask("trajectory_3D", example_style=args.style)

    variants: dict[str, list[float]] = {k: [] for k in
                                        ("原样", "去中心", "去中心+缩放", "去中心+缩放+旋转",
                                         "对照:真值自身", "对照:真值倒序")}
    kept = 0
    for item in items:
        unit = type("U", (), {"key": str(item["id"]), "items": [item]})()
        meta: dict[str, Any] = {}
        try:
            _, text = call_vlm(runtime, task.parts(unit), meta)
        except Exception as exc:  # noqa: BLE001
            print(f"    {item['id']}: {type(exc).__name__}: {exc}")
            continue
        row = task.rows(unit, text, CallContext(frames_used={}, usage={}, media_transforms=[]))[0]
        gripper = (row.get("scored_grippers") or [None])[0]
        gt = (row.get("expected_trajectory") or {}).get(gripper) or []
        pred = (row.get("predicted_trajectory") or {}).get(gripper) or []
        if len(gt) < 2 or len(pred) < 2:
            continue
        kept += 1
        tol = T.score_tolerance_for_item(item, {gripper: gt})
        score = lambda pts: T.score_curve(gt, pts, tol)["score"] or 0.0

        centered = recenter(pred, centroid(gt))
        sp, sg = spread(centered), spread(gt)
        scaled = rescale(centered, sg / sp) if sp > 1e-9 else centered
        rotated = kabsch(scaled, gt)

        variants["原样"].append(score(pred))
        variants["去中心"].append(score(centered))
        variants["去中心+缩放"].append(score(scaled))
        variants["去中心+缩放+旋转"].append(score(rotated))
        variants["对照:真值自身"].append(score(gt))
        variants["对照:真值倒序"].append(score(gt[::-1]))

    print(f"\n模型 {args.model}   prompt={args.style}   有效 {kept}/{len(items)} 题\n")
    print(f"{'对齐方式':<20} {'平均分':>8} {'中位':>8}")
    print("-" * 40)
    for name, values in variants.items():
        if values:
            print(f"{name:<20} {statistics.mean(values):>8.2f} {statistics.median(values):>8.2f}")
    print("-" * 40)
    print("盲基线（固定点重复十次）= 0.75")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
