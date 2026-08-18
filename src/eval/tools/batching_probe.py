#!/usr/bin/env python3
# coding: utf-8
"""time 任务的打包大小对「答全率」的影响。

背景：``time`` 是九个任务里唯一把多题合并成一次调用的 —— 每个视频 6 道题一次发出，
要求模型返回带 id 的 6 条答案。实测 RynnBrain-2B 漏答 48%，而 Qwen3-VL-8B 与
SenseNova 是 0%。需要判断这是个别模型的毛病，还是「一次 6 题」这个协议本身的问题。

做法：同一批视频、同一套题，只改**每次调用打包几道题**（1 / 2 / 3 / 6），
数模型返回了多少个被要求的 id。抽帧与生成参数全程不变，打包大小是唯一变量。

    python tools/batching_probe.py --model RynnBrain-2B --videos 10
    python tools/batching_probe.py --model RynnBrain-2B --videos 10 --max-new-tokens 1024

``--max-new-tokens`` 用来排除「输出预算不够」这个解释（默认 256）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from robochrono.tasks import time_eqa  # noqa: E402
from robochrono.tasks.base import load_items  # noqa: E402
from robochrono.vlm_api import call_vlm, runtime_config  # noqa: E402

QA = EVAL_ROOT / "datasets/QA/understanding/stack_cubes/time_vqa.json"

# plan.json 里的模型名 → providers.json 里的 provider 名
PROVIDERS = {
    "RynnBrain-2B": "local_rynnbrain_2b",
    "Qwen3-VL-8B-Instruct": "local_qwen",
    "SenseNova-SI-1.1-InternVL3-2B": "local_sensenova_si_1_1_internvl3_2b",
}


def chunk(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RynnBrain-2B")
    parser.add_argument("--videos", type=int, default=10)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 6])
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    provider = PROVIDERS.get(args.model)
    if provider is None:
        print(f"未知模型 {args.model}；已知：{sorted(PROVIDERS)}")
        return 1

    runtime = runtime_config(config_path=EVAL_ROOT / "configs/providers.json",
                             provider_name=provider, default_model="")
    runtime["frames"] = {"mode": "fps", "value": args.fps,
                         "video_sample_fps": args.fps, "num_segments": 1}
    runtime["align_fps_to_segments"] = True
    if args.max_new_tokens:
        runtime["max_new_tokens"] = args.max_new_tokens

    task = time_eqa.build()
    units = task.units(load_items(QA))[:args.videos]
    print(f"模型 {args.model}   视频 {len(units)} 个   fps={args.fps:g}   "
          f"max_new_tokens={runtime['max_new_tokens']}\n")
    print(f"{'打包大小':>8} {'调用数':>6} {'要求 id':>8} {'返回 id':>8} {'答全率':>7}  按题位漏答率")
    print("-" * 78)

    summary: dict[int, float] = {}
    for size in args.sizes:
        asked = got = calls = 0
        missing_by_pos: dict[int, list[int]] = {}
        for unit in units:
            for position_offset, group in enumerate(chunk(unit.items, size)):
                ids = [str(i["id"]) for i in group]
                parts = [
                    *(p for p in task.parts(type(unit)(key=unit.key, items=group))
                      if p.get("type") == "video"),
                    {"type": "text", "text": time_eqa.build_prompt(group)},
                ]
                meta: dict[str, Any] = {}
                try:
                    _, text = call_vlm(runtime, parts, meta)
                except Exception as exc:  # noqa: BLE001
                    print(f"    调用失败: {type(exc).__name__}: {exc}")
                    continue
                calls += 1
                parsed = time_eqa.parse_multi_interval_text(text, ids)
                asked += len(ids)
                for local_index, qid in enumerate(ids):
                    absolute = position_offset * size + local_index + 1
                    hit = parsed.get(qid) is not None
                    got += int(hit)
                    missing_by_pos.setdefault(absolute, []).append(0 if hit else 1)

        rate = got / asked if asked else 0.0
        summary[size] = rate
        detail = " ".join(
            f"{p}:{sum(v)/len(v):.0%}" for p, v in sorted(missing_by_pos.items())
        )
        print(f"{size:>8} {calls:>6} {asked:>8} {got:>8} {rate:>6.0%}  {detail}")

    print("-" * 78)
    if len(summary) > 1:
        best, worst = max(summary, key=summary.get), min(summary, key=summary.get)
        print(f"打包 {best} 题答全率 {summary[best]:.0%}，打包 {worst} 题 {summary[worst]:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
