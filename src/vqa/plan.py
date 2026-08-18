#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第四步：决定出哪些题，以及每道题要什么素材。

    python3 src/vqa/plan.py                  # 打印题量与素材统计
    python3 src/vqa/plan.py --write          # 写 build/plan.json
    python3 src/vqa/plan.py --time-repeats keep --cap 5    # 换参数

**这一步不生产任何东西**，只出两张表：

    items    每道题的内容（问谁、答案是什么、用哪几条干扰项、要哪些素材）
    media    去重后的素材需求清单（[5] 照着切）

拆开的理由（D-05）
------------------
把「要什么素材」先枚举出来，重复的就能在生产前合并。
实测旧的 `planning_clips` 8.9 GB 与 `understanding_clips` 是同一批片段的
重复编码 —— 同分辨率同帧数同时长，像素差只有 0.1–0.3 的编码噪声。
理解题与规划题用的**本来就是同一个片段**，只是问法不同。

一条必须守住的规则：**片段不得越过本段段尾**
--------------------------------------------
planning 问「接下来会发生什么」，答案是**下一段**的动作。
片段只要越过本段段尾，露出的就是答案本身。

这不是假想。stack_cubes 被回写进标注层的出题窗口（P-03），四段全部越界 2 秒：

```
                  人工标注           回写的窗口
pick_red_cube     0.00 → 25.70      10.00 → 27.70     ← 段尾之后 +2.00s
place_red_cube   25.70 → 33.05      31.05 → 35.05     ← +2.00s
```

`window_for_segment` 的 `after_window` 参数在 planning 上就是泄漏。
所以这里把它写成**断言**（`assert_no_leak`），而不是一个可配置项 ——
可配置意味着有人会配错，而配错了不报错。

> 起点后移是另一回事，**那是合理的**：段的开头往往还是上一个动作的余波
> （P-06），不后移的话片段里看不到目标动作。起点怎么取由 `--window` 决定，
> 终点则一律不越界。

三个待决项在这里生效
--------------------
都做成参数，人拍板后改默认值即可，不必改代码：

``--time-repeats``  P-05：wash 每集洗两个盘子，「pick the plate 在第几秒」有两个答案。
                    ``skip``（默认）只对不重复的动作出时间题；``keep`` 全出。
``--window``        A1：``raw`` 用标注段原始起止（默认，时长自然但本身泄答案）；
                    ``tail`` 只取段尾若干秒（时长恒定，但看不到动作全过程）。
``--cap``           A3：每族每集最多出几道，``None``（默认）不封顶。
``--none-option``   A7：见下。

A7 ·「都不对」这个选项在旧题里从来不是答案
------------------------------------------
全量统计 `eval/datasets/QA`：**10,787 道题有「All other options are wrong.」，
它是正确答案的有 0 道。**

于是它是白送的一次排除 —— 名义六选一，**实际五选一**，
随机基线不是 16.7% 而是 20%。三种处置：

``off``         （默认）不放这个选项，五选一。基线 20%，与实际难度一致。
``fixed``       放，但永不为答案。复刻 v1，仅供 A4 对照用。
``answerable``  放，并在一部分题里**抽掉正确答案**让它成为真答案。
                这会真正考「能不能说出都不对」，但改变了题型语义，属 ① 能力定义。

默认选 ``off`` 是因为它**不改变任何一道题的实际难度**，只是让写下来的基线
等于真实基线。``answerable`` 需要人拍板。
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BUILD = ROOT / "build"

PLAN_VERSION = "1"
RECIPE_VERSION = "v2.0"

# 出题用的视角。三视角都有，但 main 是唯一被标注参照的那个 ——
# 其余视角靠时间对齐，段边界在它们上面未经核验。
VIEW = "main"

TASKS = ("understanding", "planning", "planning_2", "time",
         "left_right", "image_in_video")

STEMS = {
    "left_right": "Given the image captured by the head camera, which option shows the "
                  "{side} gripper camera's view at this moment?",
    "image_in_video": "Given this video clip of an action segment, "
                      "which option image appeared in the clip?",
    "understanding": "Based on the egocentric video up to now, choose the ONE option "
                     "that best matches what is happening RIGHT NOW?",
    "planning": "Based on the current visual state, what should happen next?",
    "planning_2": "The overall task is {task_name}. Based on the current visual state, "
                  "what should happen next?",
    "time": 'When did the action "{action}" happen?',
}

# 族的自然语言任务名，planning_2 的题干要用
TASK_NAMES = {
    "airpods": "putting the earphones into the airpods case",
    "gift_inhand": "handing over a gift",
    "pen_inbox": "putting a pen into a box",
    "stack_cubes": "stacking cubes",
    "tea": "making tea",
    "tea2": "brewing tea with a teapot",
    "wash": "washing dishes",
}

# 三条干扰项 = 四选一。**不是任选的数字**，是盲测出来的：
#   三选一  零借用    合计干净，但分族散（wash +2.6σ / gift_inhand −3.1σ）
#   四选一  借最少    合计 −0.2σ，借的族 −0.3σ，没借的族 −0.1σ  ← 唯一三处全贴基线
#   五选一  借 1–2    借的族 +2.9σ
#   六选一  借 1–3    借的族 +6.3σ
# 选项数每多一个，小族就得多借一条别族动作，而借来的动作看过视频就能排除。
DISTRACTORS_PER_QUESTION = 3

# 图选项题型也用三条干扰项 = 四选一，与文字题型一致 ——
# 于是【全部选择题】的随机基线都是 25%，报告里不用按题型换算。
# v1 用六选一，其中一个是永不为答案的「都不对」（A7 已定不放）。
IMAGE_DISTRACTORS = 3

# image_in_video 的「同集别动作」干扰项至少隔这么多段。
# 人工复核（T2-A）看到 tea 的相邻段画面几乎一样 —— 固定机位、小物体操作，
# 帧间差别只有机械臂位置。隔开之后才是「难」而不是「无解」。
IV_MIN_SEGMENT_GAP = 2
NONE_TEXT = "All other options are wrong."   # v1 的原文，保持一致便于对照

# 片段题的最短时长。**不是任选的阈值** —— `pen_inbox/file-037@f000272`
# 是个 1 帧（0.05 秒）的段（标注时连按了两次 K），照样出了三道题、
# 切出了一个 13 KB 的单帧「视频」。次短的段是 17 帧（0.57 秒），
# 所以 0.4 秒这条线只拦掉那一个孤例，不误伤。
# **写成通用闸门而不是针对那一条打补丁** —— 重标之后还会有下一个。
MIN_CLIP_SECONDS = 0.4


def assert_no_leak(clip: tuple[int, int], segment: dict[str, Any], task: str) -> None:
    """片段不得越过本段段尾。**断言而非配置项** —— 见模块 docstring。"""
    if task.startswith("planning") and clip[1] > segment["end_frame"]:
        raise AssertionError(
            f"{task} 片段 {clip} 越过段尾 {segment['end_frame']}"
            f"（{segment['id']} / {segment['subtask']}）—— 露出的正是答案")


def clip_for(segment: dict[str, Any], mode: str, fps: float) -> tuple[int, int]:
    """片段的起止帧。终点恒为段尾，只有起点随 mode 变。"""
    if mode == "raw":
        return segment["start_frame"], segment["end_frame"]
    if mode == "tail":                       # A1 的备选：时长恒定，消掉时长线索
        span = int(round(3.0 * fps))
        return max(segment["start_frame"], segment["end_frame"] - span), segment["end_frame"]
    raise ValueError(f"未知的 window 模式：{mode}")


def cooccurrence(episodes: list[dict[str, Any]]) -> dict[str, set[str]]:
    """哪些动作曾在同一集里同时出现过。

    **干扰项只能从「与答案共现过」的动作里挑。** 词表里可能有互斥的两组：
    wash 的 38 集用 left/right 盘子，file-000 / 001 机位不同用 far/near，
    两组永不同时出现。不加这条限制时实测：720 道含相对位置的题里
    **487 道（68%）混进了本集根本不存在的那条轴**，
    有效选项从 4 个掉到平均 3.0，等效基线 25% → 34%，最差的 264 道只剩 2 个选项。

    「本集不存在的动作」看过视频就能排除，所以它不是干扰项，是白送的一次排除。

    判据做成共现关系而不是写死 left/right ↔ far/near ——
    **以后任何一次词表拆分都会造出同样的问题**，写死只能挡住这一次。
    """
    graph: dict[str, set[str]] = {}
    for ep in episodes:
        here = {s["subtask"] for s in ep["segments"]}
        for a in here:
            graph.setdefault(a, set()).update(here)
    return graph


def pick_frames(item_id: str, pool: list[dict[str, Any]], episode: str, segment_id: str,
                count: int, same_ep_index: int | None = None,
                min_gap: int = 0, from_same_ep: int = 0) -> list[tuple[dict[str, Any], str]]:
    """从帧池里挑干扰帧，返回 [(帧记录, 左右侧)]。

    **所有干扰帧都来自同一个池子** —— 与正确帧走同一条抽帧路径、同一套参数。
    若两者用不同参数抽，图像统计本身就成了线索（这是图选项题型里
    对应「编出来的文字选项零样本可辨」的那个坑）。

    `min_gap` > 0 时，同一集内的帧必须与本段隔开这么多段 ——
    相邻段在固定机位下画面几乎一样，那样的干扰项是无解不是难（T2-A）。

    左右侧按哈希混着给：若干扰帧全取同侧，那唯一的对侧图会变成「异类」
    而被白排除，等于少一个选项。
    """
    def usable(r: dict[str, Any]) -> bool:
        if r["segment_id"] == segment_id:
            return False
        if r["episode"] == episode and same_ep_index is not None:
            return abs(r["seg_index"] - same_ep_index) >= min_gap
        return True

    k = int(hashlib.md5(item_id.encode()).hexdigest(), 16)
    picked, seen = [], set()

    def take(cands: list[dict[str, Any]], n: int) -> None:
        if not cands:
            return
        for step in range(len(cands)):
            r = cands[(k + step * 7919) % len(cands)]   # 7919 与池长多互质，散得开
            if r["segment_id"] in seen:
                continue
            seen.add(r["segment_id"])
            picked.append((r, "left" if (k + len(picked)) % 2 == 0 else "right"))
            if len([1 for _ in picked]) >= n:
                break

    # **同集的干扰项要【定额分配】，不能靠碰巧。**
    # 一个族的池子有两百来条，同集合格的只有三四条 —— 均匀抽的话
    # 实测 4,170 个干扰项里只有 54 个来自本集，而那正是最难的一类（T2-A）。
    if from_same_ep:
        same = [r for r in pool if r["episode"] == episode and usable(r)]
        take(same, from_same_ep)
    take([r for r in pool if usable(r) and r["episode"] != episode or
          (not from_same_ep and usable(r))], count)
    return picked[:count]


def build_options(item_id: str, answer: str, actions: list[str],
                  borrowable: list[str]) -> tuple[list[str], dict[str, int]]:
    """选出 3 条干扰项，凑成统一的四选一。**这是选项构造的唯一实现** ——
    `blind.py` 直接导入它，两边不可能分叉。

    取用顺序：本族其它真实动作 → 别族的真实动作。两者都按 `md5(题目id)` 轮转，
    并优先取词数与答案接近的（防「最长的那个是对的」）。

    为什么干扰项**必须是在别处当过答案的真实标签**
    ------------------------------------------
    盲测（不给视频、只发题干和选项给纯文本模型）把三种造法都测了一遍：

    ```
    LLM 自由生成                    +11.9%（6.1σ）  40% 的干扰项引入了场景外物体
    本族物体 × 语料动词 重组          + 6.6%（5.7σ）  «Wipe the stirrer.» 这种一眼假
    全部用真实标签                    + 1.2%（p=.29） 与随机不可分
    ```

    按「选项里有没有编出来的」二分，2,394 次作答：

    ```
    全是真实标签    n=598   20.9%   p=0.58   ← 干净
    含 ≥1 个编的   n=1796  26.4%   p<1e-4   ← 泄 6–8 点，和编了几个无关
    ```

    **一个可识别的假选项就把四选一变成三选一。** 限定词汇表、过语义闸门、
    过合理性闸门都不够 —— 只有「在别处当过答案的文字」才不带破绽。

    小族怎么凑满
    ------------
    gift_inhand 与 pen_inbox 只有 3 个动作，填不满 3 条干扰项，**各借 1 条**。
    其余五族都够，一条不借。
    借来的仍是真实标签（在那个族里当过答案），所以「是不是真标签」不再有区分度。

    **借的动作必须提到本场景没有的物体** —— 否则可能碰巧也是真的
    （tea 与 tea2 都有 `Pour the tea.`）。代价是这类选项看过视频就能排除，
    于是小族的**等效**选项数仍等于它的动作数。那是标注词表的属性，
    改选项设计改不掉；组间难度不要求一致（见 D-38），所以不处理，只记录。
    """
    def rotate(seq: list[str], salt: str) -> list[str]:
        if not seq:
            return seq
        k = int(hashlib.md5(f"{item_id}|{salt}".encode()).hexdigest(), 16) % len(seq)
        return seq[k:] + seq[:k]

    n = len(normalize(answer).split())
    inside = rotate([a for a in actions if a != answer], "in")
    outside = sorted(rotate([a for a in borrowable if a != answer], "out"),
                     key=lambda t: abs(len(normalize(t).split()) - n))

    chosen = (inside + outside)[:DISTRACTORS_PER_QUESTION]
    within = set(actions)
    return chosen, {"in_family": sum(c in within for c in chosen),
                    "borrowed": sum(c not in within for c in chosen)}


def build(index: dict[str, Any], vocab: dict[str, Any],
          window: str, time_repeats: str, cap: int | None,
          none_option: str) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    media: dict[str, dict[str, Any]] = {}       # 内容寻址去重
    skipped: Counter = Counter()

    def need(kind: str, family: str, episode: str,
             start: int | None = None, end: int | None = None,
             view: str = VIEW) -> str:
        """登记一条素材需求，返回它的 key。同样的需求只登记一次。

        `kind` 为 ``frame`` 时抽单帧（图选项题型用），``clip`` 切片，``video`` 整段。
        `view` 默认 main —— 只有 `left_right` 会用到手腕视角。
        """
        key = (f"{family}/{episode}/{view}/{kind}" +
               (f"/{start:06d}-{end:06d}" if start is not None else ""))
        media.setdefault(key, {
            "key": key, "kind": kind, "family": family, "episode": episode,
            "view": view, "start_frame": start, "end_frame": end,
            "source": f"data/source/{family}/{episode}/{view}.mp4",
            "used_by": [],
        })
        return key

    # 图选项题型的帧池：每段一条记录（三个视角同一时刻）
    pool: dict[str, list[dict[str, Any]]] = {}
    for fam in sorted(index):
        rows = []
        for ep in index[fam]["episodes"]:
            for i, seg in enumerate(ep["segments"]):
                rows.append({"family": fam, "episode": ep["episode"], "seg_index": i,
                             "segment_id": seg["id"], "subtask": seg["subtask"],
                             "frame": (seg["start_frame"] + seg["end_frame"]) // 2})
        pool[fam] = rows

    for family in sorted(index):
        entry = index[family]
        fps = entry["fps"]
        texts = {s["id"]: s["text"] for s in vocab[family]["subtasks"]}
        actions = list(texts.values())
        compat = cooccurrence(entry["episodes"])
        # 可借的：别族的真实动作，且其宾语不在本场景里（否则可能碰巧也是真的）
        here = {s["object"] for s in vocab[family]["subtasks"] if s["object"]}
        borrowable = sorted({s["text"] for f, v in vocab.items() if f != family
                             for s in v["subtasks"]
                             if s["object"] and s["object"] not in here}
                            - set(actions))

        for episode in entry["episodes"]:
            segments = episode["segments"]
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

            # ---- left_right：主视角一帧当题面，手腕视角当选项 ----
            for i, segment in enumerate(segments):
                mid = (segment["start_frame"] + segment["end_frame"]) // 2
                for side in ("left", "right"):
                    tid = f"{family}/{episode['episode']}/{segment['id']}@left_right_{side}"
                    correct = need("frame", family, episode["episode"], mid, mid,
                                   view=f"wrist_{side}")
                    # 最难的干扰项：**对侧手腕的同一时刻**。答对必须真做左右判断。
                    flip = "right" if side == "left" else "left"
                    other = need("frame", family, episode["episode"], mid, mid,
                                 view=f"wrist_{flip}")
                    picks = pick_frames(tid, pool[family], episode["episode"],
                                        segment["id"], IMAGE_DISTRACTORS - 1)
                    opts = [correct, other] + [
                        need("frame", family, r["episode"], r["frame"], r["frame"],
                             view=f"wrist_{sd}") for r, sd in picks]
                    if len(opts) < IMAGE_DISTRACTORS + 1:
                        skipped["left_right:干扰帧不足"] += 1
                        continue
                    items.append({
                        "id": tid, "family": family, "task": "left_right", "group": tid,
                        "stem": STEMS["left_right"].format(side=side),
                        "answer_subtask": segment["subtask"],
                        "answer_text": f"{side} gripper camera view",
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
                                       "side": side,
                                       "distractors": {"symmetric": 1,
                                                       "other_frames": len(picks)},
                                       "synthetic": False}},
                    })

            # ---- image_in_video：片段（复用 understanding 那一批）+ 图选项 ----
            for i, segment in enumerate(segments):
                clip = clip_for(segment, window, fps)
                if (clip[1] - clip[0] + 1) / fps < MIN_CLIP_SECONDS:
                    skipped["image_in_video:段过短"] += 1
                    continue
                mid = (segment["start_frame"] + segment["end_frame"]) // 2
                tid = f"{family}/{episode['episode']}/{segment['id']}@image_in_video"
                correct = need("frame", family, episode["episode"], mid, mid, view="main")
                # 1 条来自本集（隔 ≥2 段，最难的那类）+ 2 条来自别集
                picks = pick_frames(tid, pool[family], episode["episode"], segment["id"],
                                    IMAGE_DISTRACTORS, same_ep_index=i,
                                    min_gap=IV_MIN_SEGMENT_GAP, from_same_ep=1)
                opts = [correct] + [need("frame", family, r["episode"], r["frame"],
                                         r["frame"], view="main") for r, _ in picks]
                if len(opts) < IMAGE_DISTRACTORS + 1:
                    skipped["image_in_video:干扰帧不足"] += 1
                    continue
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
                                   "distractors": {"other_frames": len(picks)},
                                   "synthetic": False}},
                })

            # ---- time：一集一组，用全长视频 ----
            if not episode["full_video_usable"]:
                skipped["time:该集有未标注的 episode"] += 1
                continue
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

    for item in items:
        # **`image_options` 也要计入引用。** 图选项题型把选项图放在
        # `image_options` 而不是 `media` 里（compose 需要区分「题面」与「选项」），
        # 漏统计的后果不只是账算错 —— 若哪天按「没被引用」清理素材，
        # 4,170 张里有 1,390 张会被误删。
        for key in list(item["media"]) + list(item.get("image_options", [])):
            media[key]["used_by"].append(item["id"])

    choices = DISTRACTORS_PER_QUESTION + 1 + (none_option != "off")
    return {
        "plan_version": PLAN_VERSION,
        "recipe_version": RECIPE_VERSION,
        "options": {"window": window, "time_repeats": time_repeats, "cap": cap,
                    "view": VIEW, "distractors_per_question": DISTRACTORS_PER_QUESTION,
                    "none_option": none_option,
                    "choices_per_question": choices,
                    "random_baseline": round(1 / choices, 4)},
        "items": items,
        "media": list(media.values()),
        "skipped": dict(skipped),
    }


def report(plan: dict[str, Any], index: dict[str, Any]) -> None:
    by_family: dict[str, Counter] = defaultdict(Counter)
    for item in plan["items"]:
        by_family[item["family"]][item["task"]] += 1

    print(f"{'族':<13}" + "".join(f"{t:>13}" for t in TASKS) + f"{'合计':>8}")
    print("-" * 78)
    total: Counter = Counter()
    for family in sorted(by_family):
        counts = by_family[family]
        total.update(counts)
        print(f"{family:<13}" + "".join(f"{counts[t]:>13}" for t in TASKS)
              + f"{sum(counts.values()):>8}")
    print("-" * 78)
    print(f"{'合计':<13}" + "".join(f"{total[t]:>13}" for t in TASKS)
          + f"{sum(total.values()):>8}")

    from collections import Counter as _C
    kinds = _C(m["kind"] for m in plan["media"])
    clips = [m for m in plan["media"] if m["kind"] == "clip"]
    frames = [m for m in plan["media"] if m["kind"] == "frame"]
    reuse = sum(len(m["used_by"]) for m in clips)
    freuse = sum(len(m["used_by"]) for m in frames)
    print(f"\n素材需求：{dict(kinds)}")
    print(f"  切片被 {reuse} 道题引用 —— 去重省下 {reuse - len(clips)} 次编码"
          f"（understanding / planning / planning_2 / image_in_video 共用同一片段）")
    if frames:
        byview = _C(m["view"] for m in frames)
        print(f"  帧被 {freuse} 次引用，去重后 {len(frames)} 张 —— 省下 {freuse - len(frames)} 次抽帧")
        print(f"    按视角 {dict(byview)}")
        print(f"    **所有选项图同一条抽帧路径、同一套参数** —— 正确图与干扰图之间"
              f"没有图像统计上的差别可辨")

    if plan["skipped"]:
        print("\n没出成的：")
        for why, n in sorted(plan["skipped"].items(), key=lambda kv: -kv[1]):
            print(f"  {n:>6}  {why}")


def main() -> int:
    def arg(name: str, default: str | None = None) -> str | None:
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    window = arg("--window", "raw")
    time_repeats = arg("--time-repeats", "skip")
    none_option = arg("--none-option", "off")
    cap_raw = arg("--cap")
    cap = int(cap_raw) if cap_raw else None

    index = json.loads((BUILD / "index.json").read_text(encoding="utf-8"))["families"]
    vocab = json.loads((BUILD / "vocab.json").read_text(encoding="utf-8"))["families"]
    # `data/llm_cache/` 三代（v1-vendor / v2 / v3）全部退场，只作留档 ——
    # 干扰项改为一律取自真实标签，不再需要任何生成物（D-37 / D-38）。
    plan = build(index, vocab, window, time_repeats, cap, none_option)

    # 确定性自检：同样输入必须得到同样一批题。
    # **构建两遍比对**，因为这类 bug（遍历 set、用 dict 顺序、掺进时间戳）
    # 不会报错，只会让每次出的题悄悄不同 —— 而下游的盲测结论就此失效。
    again = build(index, vocab, window, time_repeats, cap, none_option)
    fp = [hashlib.md5(json.dumps([[i["id"], i["answer_text"], *i["distractors"]]
                                  for i in p["items"]], ensure_ascii=False,
                                 sort_keys=True).encode()).hexdigest()[:12]
          for p in (plan, again)]
    if fp[0] != fp[1]:
        print(f"❌ 构建不确定：两次指纹 {fp[0]} ≠ {fp[1]}")
        print("   同样输入得到了不同的题。常见原因：遍历 set / dict、掺进时间或随机数。")
        return 1
    plan["fingerprint"] = fp[0]
    opt = plan["options"]
    print(f"指纹 {plan['fingerprint']}（两次构建一致）")
    print(f"配方 {RECIPE_VERSION}   window={window}  time-repeats={time_repeats}  "
          f"cap={cap}  none-option={none_option}")
    print(f"选择题 {opt['choices_per_question']} 选一，随机基线 "
          f"{opt['random_baseline']:.1%}"
          + ("" if none_option != "fixed" else
             "  ⚠ fixed 下「都不对」永不为答案，真实基线其实是 "
             f"{1 / (opt['choices_per_question'] - 1):.1%}") + "\n")
    report(plan, index)

    if "--write" in sys.argv:
        BUILD.mkdir(exist_ok=True)
        out = BUILD / "plan.json"
        out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写入 {out.relative_to(ROOT)}（{out.stat().st_size / 1e6:.1f} MB）")
    else:
        print("\n加 --write 写入 build/plan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
