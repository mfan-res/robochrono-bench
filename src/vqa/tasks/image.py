#!/usr/bin/env python3
# coding: utf-8
"""图选项两题：left_right（2×2 腕部视角）与 image_in_video（片段 + 四帧）。

出题规则住在这里，编排住在 `plan.py`。每个 emitter 只认 `Ctx`：
往 `ctx.items` 追加、往 `ctx.skipped` 记原因、经 `ctx.need` 申报素材，
**不读别的题型的任何状态** —— 这是能按题型分文件的前提。
"""
from __future__ import annotations
from typing import Any

from ._base import (
    ANCHOR, Ctx, IMAGE_DISTRACTORS, IV_MIN_SEGMENT_GAP, MIN_CLIP_SECONDS,
    MUTUAL_RATIO, RECIPE_VERSION, STEMS, VIEW, clip_for, frame_key,
    other_blocks, phase_frame,
)

def emit_left_right(ctx: Ctx) -> None:
    """left_right —— 2×2：{本时刻, 别的动作块} × {左腕, 右腕}。

    只往 ``ctx.items`` 追加、往 ``ctx.skipped`` 记原因、经 ``ctx.need`` 申报素材；
    **不读别的题型的任何状态** —— 这是按题型分文件的前提。
    """
    family, episode, segments = ctx.family, ctx.episode, ctx.segments
    looks, need, items, skipped = ctx.looks, ctx.need, ctx.items, ctx.skipped

    # ---- left_right：2×2，四个选项全部同集 ----
    # 选项 = {本时刻, 别的动作块} × {左腕, 右腕}
    # 每个视角各出现两次、每个时刻各出现两次 —— **任何单一线索都不指向答案**，
    # 必须同时定「哪个相机」和「哪个时刻」。认出侧别只能到 50%（已披露）。
    for i, segment in enumerate(segments):
        mid = phase_frame(segment, ANCHOR)
        tid_base = f"{family}/{episode['episode']}/{segment['id']}"
        views = ["wrist_left", "wrist_right"]
        anchors = {v: frame_key(family, episode["episode"], v, mid)
                   for v in views}
        if not all(looks.has(k) for k in anchors.values()):
            skipped["left_right:缺手腕视角"] += 2
            continue

        # 别的动作块：一个块同时供左右两张干扰图，
        # 两张都要与【两个正确项】都够远（左右两道题共用这个块）。
        chosen = None
        for block, phase, frame in other_blocks(f"{tid_base}@left_right",
                                                segments, i):
            keys = {v: frame_key(family, episode["episode"], v, frame)
                    for v in views}
            if not all(looks.has(k) for k in keys.values()):
                continue
            if all(looks.far_enough(a, b, family, views)
                   for a in anchors.values() for b in keys.values()):
                chosen = (block, phase, frame, keys)
                break
        if chosen is None:
            skipped["left_right:同集找不到够远的别的动作块"] += 2
            continue
        block, phase, frame, other_keys = chosen

        for side in views:
            flip = "wrist_right" if side == "wrist_left" else "wrist_left"
            tid = f"{tid_base}@left_right_{side.split('_')[1]}"
            # 对侧手腕同一时刻。实测它比典型的不同动作对更可分
            # （比值 1.13–1.78），所以【不再】因为它被丢题 ——
            # 第一版按脏基准算的下限错丢了 98 道 wash。
            if not looks.far_enough(anchors[side], anchors[flip], family, views):
                skipped["left_right:对侧同刻与正确项画面差不足"] += 1
                continue
            correct = need("frame", family, episode["episode"], mid, mid,
                           view=side)
            opts = [correct,
                    need("frame", family, episode["episode"], mid, mid,
                         view=flip),
                    need("frame", family, episode["episode"], frame, frame,
                         view="wrist_left"),
                    need("frame", family, episode["episode"], frame, frame,
                         view="wrist_right")]
            items.append({
                "id": tid, "family": family, "task": "left_right", "group": tid,
                "stem": STEMS["left_right"].format(side=side.split("_")[1]),
                "answer_subtask": segment["subtask"],
                "answer_text": f"{side.split('_')[1]} gripper camera view",
                "distractors": [], "image_options": opts,
                "correct_option": correct,
                "media": [need("frame", family, episode["episode"], mid, mid,
                               view="main")],
                "provenance": {
                    "episode": episode["episode"], "segment_id": segment["id"],
                    "subtask": segment["subtask"], "next_subtask": None,
                    "recipe": {"version": RECIPE_VERSION,
                               "clip": {"mode": "frame", "view": "main",
                                        "start_frame": mid, "end_frame": mid,
                                        "seconds": 0.0},
                               "side": side.split("_")[1],
                               # `distractors` 按契约只放【来源计数】——
                               # 元数据放同级键，别挤进计数表（schema 会拦）
                               "distractors": {"opposite_wrist_same_moment": 1,
                                               "other_block_frames": 2},
                               "layout": "2x2",
                               "same_episode": True,
                               "other_block": block["id"],
                               "other_subtask": block["subtask"],
                               "other_phase": phase,
                               "synthetic": False}},
            })

def emit_image_in_video(ctx: Ctx) -> None:
    """image_in_video —— 片段 + 四个同相机、不同动作块的帧。

    只往 ``ctx.items`` 追加、往 ``ctx.skipped`` 记原因、经 ``ctx.need`` 申报素材；
    **不读别的题型的任何状态** —— 这是按题型分文件的前提。
    """
    family, episode, segments = ctx.family, ctx.episode, ctx.segments
    fps, looks, need = ctx.fps, ctx.looks, ctx.need
    items, skipped, window = ctx.items, ctx.skipped, ctx.window

    # ---- image_in_video：片段 + 四个【同相机、不同动作块】的帧 ----
    # 四个选项同为 main 视角，分别来自四个不同的动作块（全部同集）。
    # 没有视角线索、没有时刻线索 —— **盲测天花板就是 25%**，
    # 唯一解法是认出片段里发生的是哪个动作块。
    for i, segment in enumerate(segments):
        clip = clip_for(segment, window, fps)
        if (clip[1] - clip[0] + 1) / fps < MIN_CLIP_SECONDS:
            skipped["image_in_video:段过短"] += 1
            continue
        mid = phase_frame(segment, ANCHOR)
        if not clip[0] <= mid <= clip[1]:
            # 正确项必须真的在片段里，否则这道题没有正确答案
            skipped["image_in_video:锚点不在片段内"] += 1
            continue
        tid = f"{family}/{episode['episode']}/{segment['id']}@image_in_video"
        a_key = frame_key(family, episode["episode"], "main", mid)
        if not looks.has(a_key):
            skipped["image_in_video:缺锚点帧"] += 1
            continue

        # **挑最接近等边的那一组四点**，不是「按顺序取第一个够远的」。
        # 顺序贪心会挑出扁的集合：干扰项都离答案远、彼此却挤在一起，
        # 于是**答案成了离群点**（实测排第一占 53%，白送 28 个百分点）。
        # 这里每次加入到已选各点最小距离最大的那个候选 ——
        # 四个点互相撑开，答案在构造上不再特殊。
        cands: list[tuple[dict[str, Any], float, int, str]] = []
        seen_keys: set[str] = set()
        for block, phase, frame in other_blocks(tid, segments, i):
            key = frame_key(family, episode["episode"], "main", frame)
            if key in seen_keys or not looks.has(key):
                continue
            seen_keys.add(key)
            cands.append((block, phase, frame, key))

        floor = looks.floor(family, ["main"]) * MUTUAL_RATIO
        picked: list[tuple[dict[str, Any], float, int]] = []
        picked_keys: list[str] = [a_key]
        used_blocks: set[str] = set()
        for _slot in range(IMAGE_DISTRACTORS):
            best = None
            for block, phase, frame, key in cands:
                if key in picked_keys:
                    continue
                gap = min(looks.distance(key, k) for k in picked_keys)
                if gap < floor:
                    continue
                # 平手时按 key 定序 —— 出题必须确定
                if best is None or (gap, key) > (best[0], best[3][3]):
                    best = (gap, block, phase, (block, phase, frame, key))
            if best is None:
                break
            _gap, block, phase, entry = best
            picked.append((entry[0], entry[1], entry[2]))
            picked_keys.append(entry[3])
            used_blocks.add(block["id"])

        if len(picked) < IMAGE_DISTRACTORS:
            skipped["image_in_video:同集凑不齐够远的别的动作块"] += 1
            continue
        # **三条干扰项不能全来自同一个动作块。** 那样它们会聚成一簇，
        # 正确项就是唯一的异类 —— 不看视频也能挑出来。
        # 两个块不存在这个问题：簇是 {X,X}、{Y}、{Z}，答案与 Y 一样是单点。
        if len(used_blocks) < 2:
            skipped["image_in_video:三条干扰项同属一个动作块（答案会成异类）"] += 1
            continue

        correct = need("frame", family, episode["episode"], mid, mid, view="main")
        opts = [correct] + [need("frame", family, episode["episode"],
                                 f, f, view="main") for _b, _p, f in picked]
        items.append({
            "id": tid, "family": family, "task": "image_in_video", "group": tid,
            "stem": STEMS["image_in_video"],
            "answer_subtask": segment["subtask"],
            "answer_text": "the option image that appeared in the clip",
            "distractors": [], "image_options": opts, "correct_option": correct,
            "media": [need("clip", family, episode["episode"], *clip)],
            "provenance": {
                "episode": episode["episode"], "segment_id": segment["id"],
                "subtask": segment["subtask"], "next_subtask": None,
                "recipe": {"version": RECIPE_VERSION,
                           "clip": {"mode": window, "view": VIEW,
                                    "start_frame": clip[0], "end_frame": clip[1],
                                    "seconds": round((clip[1] - clip[0] + 1) / fps, 3)},
                           "min_segment_gap": IV_MIN_SEGMENT_GAP,
                           "distractors": {
                               "other_block_frames": len(picked)},
                           "layout": "same_camera_other_blocks",
                           "same_episode": True,
                           "blocks": [b["subtask"] for b, _p, _f in picked],
                           "phases": [p for _b, p, _f in picked],
                           "distinct_blocks": len(used_blocks),
                           "synthetic": False}},
        })
