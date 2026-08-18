#!/usr/bin/env python3
# coding: utf-8
"""按提案口径评测轨迹，并与冻结版口径对照。

提案含 prompt 与评分两组改动，本脚本把它们一起跑出来：

prompt（example_style="proposal"）
  - schema 示例换成本题真实初始位姿，重复 N 次（合法 JSON，2B 模型不掉格式；
    逐题变化，"照抄"不再是错误行为；抄成 N 个相同点会被位移分判 0）
  - 告知动作时长与"按时间均分"（现状完全缺失）
  - 点数从 approximately 改为 exactly

评分
  - 排除第 0 点：它是给定的，不该计分
  - 位置分 + 位移分两个数：位移分让"有序"第一次被真正测到
    （真值倒序：位置 67.41 分，位移 3.85 分）
  - 容差统一为轨迹相对，2D 不再用图像对角线
"""
from __future__ import annotations
import argparse, math, statistics as st, sys
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))
from robochrono.tasks import trajectory as T                      # noqa: E402
from robochrono.tasks.base import CallContext, load_items         # noqa: E402
from robochrono.vlm_api import call_vlm, runtime_config           # noqa: E402

PROVIDERS = {"RynnBrain-2B": "local_rynnbrain_2b",
             "Qwen3-VL-8B-Instruct": "local_qwen",
             "SenseNova-SI-1.1-InternVL3-2B": "local_sensenova_si_1_1_internvl3_2b"}
QA = {"2D": "trajectory_qa_2d.json", "3D": "trajectory_qa_3d.json"}


def unified_tolerance(gt: list[list[float]], dim: int) -> float:
    """两个维度统一用轨迹相对容差。现状 2D 用图像对角线 5%（恒定），与 3D 不可比。"""
    extent = T.point_cloud_extent(gt) or 0.0
    floor = 10.0 if dim == 2 else 0.02
    return max(floor, 0.1 * extent)


def deltas(points: list[list[float]]) -> list[list[float]]:
    return [[points[i + 1][d] - points[i][d] for d in range(len(points[0]))]
            for i in range(len(points) - 1)]


def dual_score(gt: list[list[float]], pred: list[list[float]], dim: int) -> tuple[float, float]:
    """返回 (位置分, 位移分)。均已排除第 0 点。"""
    if len(gt) < 3 or len(pred) < 2:
        return 0.0, 0.0
    tol = unified_tolerance(gt, dim)
    pos = T.score_curve(gt[1:], pred[1:], tol)["score"] or 0.0
    step = st.median([math.dist(gt[k], gt[k + 1]) for k in range(len(gt) - 1)]) or 1e-6
    gd, pd = deltas(gt), deltas(pred)
    dsp = (T.score_curve(gd, pd, max(step * 0.5, 1e-6))["score"] or 0.0) if pd else 0.0
    return pos, dsp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    ap.add_argument("--dim", choices=["2D", "3D"], default="3D")
    ap.add_argument("-n", "--items", type=int, default=40)
    ap.add_argument("--styles", nargs="+", default=["legacy", "proposal"])
    args = ap.parse_args()

    rt = runtime_config(config_path=EVAL_ROOT / "configs/providers.json",
                        provider_name=PROVIDERS[args.model], default_model="")
    items = load_items(EVAL_ROOT / "datasets/QA/planning/stack_cubes" / QA[args.dim])[: args.items]
    d = 2 if args.dim == "2D" else 3

    print(f"模型 {args.model}   {args.dim}   {len(items)} 题\n")
    print(f"{'prompt':<10} {'点数中位':>8} {'退化':>6} {'旧口径分':>9} {'位置分':>8} {'位移分':>8}")
    print("-" * 58)

    for style in args.styles:
        task = T.TrajectoryTask(f"trajectory_{args.dim}", example_style=style)
        old, pos_s, dsp_s, counts, degen = [], [], [], [], 0
        for item in items:
            unit = type("U", (), {"key": str(item["id"]), "items": [item]})()
            try:
                _, text = call_vlm(rt, task.parts(unit), {})
            except Exception as exc:                              # noqa: BLE001
                print(f"    {item['id']}: {type(exc).__name__}: {exc}"); continue
            row = task.rows(unit, text, CallContext(frames_used={}, usage={}, media_transforms=[]))[0]
            g = (row.get("scored_grippers") or [None])[0]
            gt = (row.get("expected_trajectory") or {}).get(g) or []
            pred = (row.get("predicted_trajectory") or {}).get(g) or []
            counts.append(len(pred))
            if len(pred) <= 1:
                degen += 1
            old.append(float(row.get("score") or 0.0))
            p, q = dual_score(gt, pred, d)
            pos_s.append(p); dsp_s.append(q)
        n = max(1, len(old))
        print(f"{style:<10} {st.median(counts) if counts else 0:>8.0f} {degen:>3}/{n:<2} "
              f"{st.mean(old):>9.2f} {st.mean(pos_s):>8.2f} {st.mean(dsp_s):>8.2f}")

    # 同口径下的参照线
    print("-" * 58)
    base = {"锚点原地重复": [], "首尾直线": [], "真值倒序": []}
    for item in items:
        exp = T.expected_trajectory(item); g = T.active_gripper_for_item(item, exp)
        sc = T.grippers_to_score(g)
        if len(sc) != 1: continue
        gt = exp[sc[0]]
        if len(gt) < 3: continue
        lerp = lambda a, b, t: [a[k] + (b[k] - a[k]) * t for k in range(d)]
        base["锚点原地重复"].append(dual_score(gt, [gt[0][:] for _ in gt], d))
        base["首尾直线"].append(dual_score(gt, [lerp(gt[0], gt[-1], k / (len(gt) - 1))
                                             for k in range(len(gt))], d))
        base["真值倒序"].append(dual_score(gt, gt[::-1], d))
    for k, v in base.items():
        print(f"{'基线:' + k:<10} {'':>8} {'':>6} {'':>9} "
              f"{st.mean([x[0] for x in v]):>8.2f} {st.mean([x[1] for x in v]):>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
