#!/usr/bin/env python3
# coding: utf-8
"""④ 出题的不变量。**重构 plan.py 之前先跑它，之后再跑一遍，指纹必须一字不差。**

    python3 src/vqa/tests/test_plan_invariants.py

为什么这一套是重构的前置
------------------------
`plan.py` 有 900 多行，`build()` 一个函数就 448 行，七个题型的规则混在一个
循环里。要按题型拆开，就必须先有一个**锁死全部 10,178 道题内容**的判据 ——
否则「拆完还是原来那批题」只是希望，不是事实。

指纹就是那个判据，但**它此前是漏的**：只算 `[id, answer_text, *distractors]`，
那是文字选项题的形状。图选项题的 `answer_text` 是常量、`distractors` 是空的，
`time` 也一样，于是三个题型共 **3,904 道（占 38%）完全不在覆盖范围内**。
实测把 1,264 道 image_in_video 的选项顺序全反转、把 2,640 道 left_right 的
正确答案全换掉，**指纹都不变**。

这三个都是后加的题型 —— 加题型时没人回来看那个函数，而它不报错，
只是悄悄少管一块。`plan.fingerprint()` 已改成按题型取真正的内容，
并且**新题型若两个字段都取不到会显式报错**。

这里守两件事
------------
1. **指纹锁**：当前产物的指纹必须等于 `EXPECTED`。改题目设计时**改这个常量**，
   并在提交信息里说明改了什么 —— 那正是「题目变了」这件事该被看见的地方。
2. **核心函数的行为**：`clip_for` / `assert_no_leak` / `cooccurrence` /
   `build_options`。它们是选题与选项构造的地基，此前一个都没测。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "vqa"))

import plan as P  # noqa: E402
# 判据与选项构造在 `tasks/_base.py`；`plan.py` 只做编排。
from tasks import _base as B  # noqa: E402

BUILD = ROOT / "build"

# 当前产物的内容指纹。**改题目设计时改这里**，并说明改了什么。
EXPECTED = "cd029ee53291"


def check_fingerprint() -> list[str]:
    path = BUILD / "plan.json"
    if not path.exists():
        return []
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    got = P.fingerprint(items)
    if got != EXPECTED:
        return [f"指纹 {got} ≠ 预期 {EXPECTED}（{len(items)} 道题）—— "
                "题目内容变了。若是有意的，改 EXPECTED 并在提交信息里说明。"]
    return []


def check_fingerprint_sensitivity() -> list[str]:
    """指纹必须对**每个题型**的内容变化敏感。

    只验「当前值等于预期」是不够的 —— 一个恒返回常量的指纹也能全过。
    这里逐题型改一处，确认它真的会变。
    """
    path = BUILD / "plan.json"
    if not path.exists():
        return []
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    base = P.fingerprint(items)
    bad: list[str] = []

    def mutate(label: str, fn) -> None:
        copy = json.loads(json.dumps(items))
        changed = fn(copy)
        if not changed:
            return
        if P.fingerprint(copy) == base:
            bad.append(f"改了「{label}」（{changed} 处）指纹却不变")

    def flip_images(rows) -> int:
        n = 0
        for it in rows:
            if it["task"] == "image_in_video" and it.get("image_options"):
                it["image_options"] = list(reversed(it["image_options"]))
                n += 1
        return n

    def swap_answer(rows) -> int:
        n = 0
        for it in rows:
            if it["task"] == "left_right" and it.get("correct_option"):
                it["correct_option"] = "换一张图"
                n += 1
        return n

    def swap_distractor(rows) -> int:
        n = 0
        for it in rows:
            if it["task"] == "understanding" and it.get("distractors"):
                it["distractors"][0] = "换一条干扰项"
                n += 1
        return n

    def swap_media(rows) -> int:
        n = 0
        for it in rows:
            if it["task"] == "time" and it.get("media"):
                it["media"] = ["换一段视频"]
                n += 1
        return n

    mutate("图选项题的选项顺序", flip_images)
    mutate("图选项题的正确答案", swap_answer)
    mutate("文字题的干扰项", swap_distractor)
    mutate("time 的媒体", swap_media)
    return bad


def check_clip_never_leaks() -> list[str]:
    """`assert_no_leak`：planning 的片段不得越过本段段尾。

    **这是断言不是配置项** —— 片段越过段尾就等于把答案（下一个动作）
    直接放进画面里。这里正反两个方向都验。
    """
    bad: list[str] = []
    segment = {"id": "file-000@f000100", "subtask": "pick_bowl",
               "start_frame": 100, "end_frame": 210}
    try:
        B.assert_no_leak((100, 210), segment, "planning")
    except AssertionError:
        bad.append("片段正好到段尾，不该报")
    try:
        B.assert_no_leak((100, 211), segment, "planning")
        bad.append("片段越过段尾 1 帧，**没有报** —— 答案会漏进画面")
    except AssertionError:
        pass
    try:
        B.assert_no_leak((100, 999), segment, "understanding")
    except AssertionError:
        bad.append("understanding 不该受这条约束（它本来就看到当前动作）")
    return bad


def check_min_clip() -> list[str]:
    """最短片段：`pen_inbox/file-037@f000272` 那个 1 帧的段不该出题。"""
    bad: list[str] = []
    if B.MIN_CLIP_SECONDS <= 0:
        bad.append("MIN_CLIP_SECONDS 被关掉了")
    # 0.4 秒这条线要卡在「1 帧的孤例」与「次短的 17 帧段」之间
    if not (1 / 25 < B.MIN_CLIP_SECONDS < 17 / 30):
        bad.append(f"MIN_CLIP_SECONDS={B.MIN_CLIP_SECONDS} 落在了 1 帧与次短段（0.57s）之外 —— "
                   "要么拦不住那个孤例，要么开始误伤真实的短段")
    return bad


def check_build_options() -> list[str]:
    """`build_options`：四选一、答案不在干扰项里、同输入同输出。"""
    bad: list[str] = []
    actions = ["Pick the brush.", "Pick the bowl.", "Put the bowl.", "Wipe the bowl."]
    borrow = ["Move the gift.", "Pick the pen."]
    chosen, source = B.build_options("x@understanding", actions[0], actions, borrow)
    if len(chosen) != P.DISTRACTORS_PER_QUESTION:
        bad.append(f"干扰项 {len(chosen)} 条，应为 {P.DISTRACTORS_PER_QUESTION}")
    if actions[0] in chosen:
        bad.append("答案出现在干扰项里")
    if len(set(chosen)) != len(chosen):
        bad.append("干扰项内部有重复")
    again, _ = B.build_options("x@understanding", actions[0], actions, borrow)
    if chosen != again:
        bad.append("同样输入两次得到不同结果 —— 出题必须确定")
    other, _ = B.build_options("y@understanding", actions[0], actions, borrow)
    if chosen == other and len(actions) > P.DISTRACTORS_PER_QUESTION + 1:
        bad.append("不同题目拿到完全相同的干扰项 —— 轮转没生效")
    # 小族：动作不够时必须借。**borrow 不能与本族动作重叠** ——
    # 生产里 `borrowable` 是 `set(别族) - set(本族)` 算出来的，天然不重叠。
    small = ["Pick the gift.", "Move the gift.", "Put the gift."]
    outside = ["Pick the pen.", "Wipe the bowl."]
    picked, src = B.build_options("z@understanding", small[0], small, outside)
    if src.get("borrowed", 0) < 1:
        bad.append("三动作族没有借用，凑不满四选一")
    if len(set(picked)) != len(picked):
        bad.append(f"三动作族的干扰项有重复：{picked}")

    # **重叠时会不会出重复** —— 生产上不会发生（borrowable 已减去本族），
    # 但它是个上了膛的坑：谁把 borrowable 的构造改成不减，
    # 四选一会静默变成三选一，而随机基线仍按 25% 报。
    overlap = B.build_options("w@understanding", small[0], small, ["Move the gift."])[0]
    if len(set(overlap)) != len(overlap):
        bad.append(f"borrowable 与本族动作重叠时产生了重复选项：{overlap} —— "
                   "四选一会静默变成三选一。生产上 borrowable 已减去本族所以不发生，"
                   "但 build_options 自己不设防")
    return bad


def main() -> int:
    if not (BUILD / "plan.json").exists():
        print(f"跳过：没有 {BUILD / 'plan.json'}。先跑 python3 src/vqa/plan.py --write")
        return 0

    checks = [
        ("指纹锁（全部 10,178 道题的内容）", check_fingerprint),
        ("指纹对每个题型都敏感", check_fingerprint_sensitivity),
        ("planning 片段不越段尾（assert_no_leak）", check_clip_never_leaks),
        ("最短片段闸门", check_min_clip),
        ("build_options 的四条不变量", check_build_options),
    ]
    failures = 0
    width = max(len(name) for name, _ in checks)
    for name, fn in checks:
        bad = fn()
        print(f"{'✓' if not bad else '✗'} {name:<{width}}")
        for line in bad:
            print(f"    {line}")
        failures += bool(bad)

    print()
    if failures:
        print(f"❌ {failures} 项不成立。**重构 plan.py 之前必须全绿** —— "
              "否则「拆完还是原来那批题」只是希望，不是事实。")
        return 1
    print(f"④ 出题的不变量成立。指纹 {EXPECTED}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
