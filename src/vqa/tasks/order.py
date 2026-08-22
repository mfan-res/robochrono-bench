#!/usr/bin/env python3
# coding: utf-8
"""step_order —— 同集三个动作块各一帧，打乱后问先后。

出题规则住在这里，编排住在 `plan.py`。每个 emitter 只认 `Ctx`：
往 `ctx.items` 追加、往 `ctx.skipped` 记原因、经 `ctx.need` 申报素材，
**不读别的题型的任何状态** —— 这是能按题型分文件的前提。
"""
from __future__ import annotations
import hashlib
import itertools

from ._base import (
    ANCHOR, Ctx, DISTRACTORS_PER_QUESTION, MUTUAL_RATIO, PHASES,
    RECIPE_VERSION, STEMS, STEP_ORDER_FRAMES, VIEW, frame_key,
    order_text, other_blocks, phase_frame,
)

def emit_step_order(ctx: Ctx) -> None:
    """step_order —— 同集三个动作块各一帧，打乱后问先后。

    只往 ``ctx.items`` 追加、往 ``ctx.skipped`` 记原因、经 ``ctx.need`` 申报素材；
    **不读别的题型的任何状态** —— 这是按题型分文件的前提。
    """
    family, episode, segments = ctx.family, ctx.episode, ctx.segments
    looks, need, items, skipped = ctx.looks, ctx.need, ctx.items, ctx.skipped

    # ---- step_order：同集三个动作块各一帧，打乱后问时间顺序 ----
    # **不拼宫格。** v1 把若干结果图拼成一张再发 —— 那正是 BC-16 那个坑
    # （当年要用 jpegtran 无损拆回去）。这里发三张独立的图各自带标号，
    # 评测端本来就支持多图，拼图只会引入编码损失和标号渲染问题。
    #
    # 三帧的选取照搬 image_in_video：同集、动作各不同、两两够远
    # （最大化最小距离）。**按帧集合去重** —— 只有 3 个动作块的族，
    # 每集所有段会挑出同一组三帧，不去重就是同一道题出三遍。
    seen_sets: set[frozenset[str]] = set()
    for i, segment in enumerate(segments):
        tid = f"{family}/{episode['episode']}/{segment['id']}@step_order"
        anchor_frame = phase_frame(segment, ANCHOR)
        a_key = frame_key(family, episode["episode"], "main", anchor_frame)
        if not looks.has(a_key):
            skipped["step_order:缺锚点帧"] += 1
            continue
        floor = looks.floor(family, ["main"]) * MUTUAL_RATIO

        cands: list[tuple[str, int, str]] = []
        for block_seg in segments:
            if block_seg["subtask"] == segment["subtask"]:
                continue
            for phase in (ANCHOR, PHASES[0], PHASES[-1]):
                fr = phase_frame(block_seg, phase)
                k = frame_key(family, episode["episode"], "main", fr)
                if looks.has(k):
                    cands.append((block_seg["subtask"], fr, k))

        chosen = [(segment["subtask"], anchor_frame, a_key)]
        for _slot in range(STEP_ORDER_FRAMES - 1):
            best = None
            subs = {c[0] for c in chosen}
            keys_so_far = {c[2] for c in chosen}
            for sub, fr, k in cands:
                if sub in subs or k in keys_so_far:
                    continue
                gap = min(looks.distance(k, c[2]) for c in chosen)
                if gap < floor:
                    continue
                if best is None or (gap, k) > (best[0], best[1][2]):
                    best = (gap, (sub, fr, k))
            if best is None:
                break
            chosen.append(best[1])
        if len(chosen) < STEP_ORDER_FRAMES:
            skipped["step_order:同集凑不齐够远的三个动作块"] += 1
            continue

        sig = frozenset(c[2] for c in chosen)
        if sig in seen_sets:
            skipped["step_order:同一组三帧已出过题"] += 1
            continue
        seen_sets.add(sig)

        # 呈现顺序按哈希取 6 种排列之一 —— **不能按时间顺序摆**，
        # 否则「Image 1 → Image 2 → Image 3」永远是答案。
        by_time = sorted(chosen, key=lambda c: c[1])
        perms = list(itertools.permutations(range(STEP_ORDER_FRAMES)))
        k = int(hashlib.md5(tid.encode()).hexdigest(), 16)
        shown = [by_time[j] for j in perms[k % len(perms)]]

        # 正确选项 = 把呈现标号排成时间顺序的那个序列
        label_of = {c[2]: n + 1 for n, c in enumerate(shown)}
        answer_text = order_text([label_of[c[2]] for c in by_time])
        wrong = [order_text(list(seq)) for seq in
                 itertools.permutations(range(1, STEP_ORDER_FRAMES + 1))
                 if order_text(list(seq)) != answer_text]
        picks = [wrong[(k + n * 7919) % len(wrong)] for n in range(len(wrong))]
        distractors: list[str] = []
        for text in picks:
            if text not in distractors:
                distractors.append(text)
            if len(distractors) >= DISTRACTORS_PER_QUESTION:
                break

        items.append({
            "id": tid, "family": family, "task": "step_order", "group": tid,
            "stem": STEMS["step_order"],
            "answer_subtask": segment["subtask"],
            "answer_text": answer_text,
            "distractors": distractors,
            "media": [need("frame", family, episode["episode"], c[1], c[1],
                           view="main") for c in shown],
            "provenance": {
                "episode": episode["episode"], "segment_id": segment["id"],
                "subtask": segment["subtask"], "next_subtask": None,
                "recipe": {"version": RECIPE_VERSION,
                           "clip": {"mode": "frame", "view": VIEW,
                                    "start_frame": shown[0][1],
                                    "end_frame": shown[-1][1],
                                    "seconds": 0.0},
                           "distractors": {"permutations":
                                           len(distractors)},
                           "layout": "labeled_frames",
                           "same_episode": True,
                           "frames": STEP_ORDER_FRAMES,
                           "blocks": [c[0] for c in shown],
                           "shown_order": [c[1] for c in shown],
                           "synthetic": False}},
        })
