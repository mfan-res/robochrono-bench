#!/usr/bin/env python3
# coding: utf-8
"""understanding / planning / planning_2 —— 片段 + 四个文字选项。

出题规则住在这里，编排住在 `plan.py`。每个 emitter 只认 `Ctx`：
往 `ctx.items` 追加、往 `ctx.skipped` 记原因、经 `ctx.need` 申报素材，
**不读别的题型的任何状态** —— 这是能按题型分文件的前提。
"""
from __future__ import annotations

from ._base import (
    Ctx, DISTRACTORS_PER_QUESTION, MIN_CLIP_SECONDS, NONE_TEXT, RECIPE_VERSION,
    STEMS, TASK_NAMES, VIEW, assert_no_leak, build_options, clip_for,
)

def emit_text(ctx: Ctx) -> None:
    """understanding / planning / planning_2 —— 片段 + 四个文字选项。

    只往 ``ctx.items`` 追加、往 ``ctx.skipped`` 记原因、经 ``ctx.need`` 申报素材；
    **不读别的题型的任何状态** —— 这是按题型分文件的前提。
    """
    family, episode, segments = ctx.family, ctx.episode, ctx.segments
    fps, texts, compat, borrowable = ctx.fps, ctx.texts, ctx.compat, ctx.borrowable
    need, items, skipped = ctx.need, ctx.items, ctx.skipped
    window, cap, none_option = ctx.window, ctx.cap, ctx.none_option
    per_episode = 0
    # 近邻候选按「同集内出现过」优先 —— 同集的动作在画面上更接近，
    # 比族里另一个八竿子打不着的动作更难排除

    for i, segment in enumerate(segments):
        if cap is not None and per_episode >= cap:
            skipped["cap"] += 1
            continue
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        # 下一段必须在同一 episode 内 —— 跨 episode 的「接下来」不成立
        if nxt is not None and nxt["episode_index"] != segment["episode_index"]:
            nxt = None

        base = f"{family}/{episode['episode']}/{segment['id']}"
        clip = clip_for(segment, window, fps)

        if (clip[1] - clip[0] + 1) / fps < MIN_CLIP_SECONDS:
            skipped[f"片段题:段短于 {MIN_CLIP_SECONDS}s（疑似标注误按）"] += 3
            continue

        for task in ("understanding", "planning", "planning_2"):
            if task != "understanding" and nxt is None:
                skipped[f"{task}:段尾无下一段"] += 1
                continue
            answer_id = segment["subtask"] if task == "understanding" else nxt["subtask"]
            answer_text = texts[answer_id]

            # 只从「与答案共现过」的动作里挑 —— 见 cooccurrence()
            # ⚠ **必须 sorted**：这是一个 set，而 Python 的字符串哈希
            # 每个进程都不同，直接遍历会让选项**每次运行都不一样**。
            # 实测三次构建三个不同的指纹 —— 而「同样输入必得同样一批题」
            # 是这条流水线写在文档里的承诺。已加构建自检（见 main）。
            usable = [texts[i] for i in sorted(compat.get(answer_id, set()))]
            options, source = build_options(
                f"{base}@{task}", answer_text, usable, borrowable)
            if len(options) < DISTRACTORS_PER_QUESTION:
                skipped[f"{task}:干扰项不足"] += 1
                continue

            assert_no_leak(clip, segment, task)
            items.append({
                "id": f"{base}@{task}",
                "family": family, "task": task, "group": f"{base}@{task}",
                "stem": STEMS[task].format(task_name=TASK_NAMES[family]),
                "answer_subtask": answer_id, "answer_text": answer_text,
                "distractors": options,
                # [6] 照此组装选项。放在 plan 而非 compose，是因为它
                # 决定题量与基线，属于「出哪些题」而不是「怎么排版」
                "none_option": NONE_TEXT if none_option != "off" else None,
                "media": [need("clip", family, episode["episode"], *clip)],
                "provenance": {
                    "episode": episode["episode"],
                    "segment_id": segment["id"],
                    "subtask": segment["subtask"],
                    "next_subtask": nxt["subtask"] if nxt else None,
                    "recipe": {
                        "version": RECIPE_VERSION,
                        "clip": {"mode": window, "view": VIEW,
                                 "start_frame": clip[0], "end_frame": clip[1],
                                 "seconds": round((clip[1] - clip[0] + 1) / fps, 3)},
                        "distractors": source,
                        "synthetic": False,
                    },
                },
            })
            per_episode += 1
