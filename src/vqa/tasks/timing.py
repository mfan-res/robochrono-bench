#!/usr/bin/env python3
# coding: utf-8
"""time —— 一集一组，用全长视频，答案是时间区间而非选项。

出题规则住在这里，编排住在 `plan.py`。每个 emitter 只认 `Ctx`：
往 `ctx.items` 追加、往 `ctx.skipped` 记原因、经 `ctx.need` 申报素材，
**不读别的题型的任何状态** —— 这是能按题型分文件的前提。
"""
from __future__ import annotations

from ._base import (
    Ctx, RECIPE_VERSION, STEMS, VIEW,
)

def emit_time(ctx: Ctx) -> None:
    """time —— 一集一组，用全长视频。

    只往 ``ctx.items`` 追加、往 ``ctx.skipped`` 记原因、经 ``ctx.need`` 申报素材；
    **不读别的题型的任何状态** —— 这是按题型分文件的前提。
    """
    family, episode, segments = ctx.family, ctx.episode, ctx.segments
    texts, need, items, skipped = ctx.texts, ctx.need, ctx.items, ctx.skipped
    time_repeats = ctx.time_repeats

    # ---- time：一集一组，用全长视频 ----
    if not episode["full_video_usable"]:
        skipped["time:该集有未标注的 episode"] += 1
        return
    group = f"{family}/{episode['episode']}@time"
    asked = 0
    for segment in segments:
        if time_repeats == "skip" and "ambiguous_repeat" in segment["reasons"]:
            skipped["time:同集内动作重复（P-05）"] += 1
            continue
        items.append({
            "id": f"{family}/{episode['episode']}/{segment['id']}@time",
            "family": family, "task": "time", "group": group,
            "stem": STEMS["time"].format(action=texts[segment["subtask"]].rstrip(".")),
            "answer_subtask": segment["subtask"], "answer_text": None,
            "answer_seconds": [segment["start"], segment["end"]],
            "distractors": [],                       # time 不是选择题
            "media": [need("video", family, episode["episode"])],
            "provenance": {
                "episode": episode["episode"], "segment_id": segment["id"],
                "subtask": segment["subtask"], "next_subtask": None,
                "recipe": {
                    "version": RECIPE_VERSION,
                    "clip": {"mode": "full_video", "view": VIEW,
                             "start_frame": 0, "end_frame": episode["frames"] - 1,
                             "seconds": episode["duration"]},
                    "distractors": {}, "synthetic": False,
                },
            },
        })
        asked += 1
    if asked == 0:
        skipped["time:整组无可用动作"] += 1
