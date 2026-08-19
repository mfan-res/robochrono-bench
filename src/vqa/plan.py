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
import itertools
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
         "left_right", "image_in_video", "step_order")

STEMS = {
    "left_right": "Given the image captured by the head camera, which option shows the "
                  "{side} gripper camera's view at this moment?",
    "image_in_video": "Given this video clip of an action segment, "
                      "which option image appeared in the clip?",
    "step_order": "The images above show three moments from the same episode, "
                  "presented in random order. Which option lists them in the "
                  "correct chronological order?",
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

# step_order 用几张图。**三张不是任选的**：
#   三张 → 6 种排列，取 4 个当选项，随机基线 25%，与其余六道口径一致
#   四张 → 24 种排列，选项数与基线都对不齐；且 gift_inhand / pen_inbox
#          每集只有 3 个动作块，取四张这两族直接出不了题
STEP_ORDER_FRAMES = 3

# 图选项题型也用三条干扰项 = 四选一，与文字题型一致 ——
# 于是【全部选择题】的随机基线都是 25%，报告里不用按题型换算。
# v1 用六选一，其中一个是永不为答案的「都不对」（A7 已定不放）。
IMAGE_DISTRACTORS = 3

# 「别的动作块」优先隔这么多段；隔 1 段的必须额外过画面差下限（人定）。
# 人工复核（T2-A）看到 tea 的相邻段画面几乎一样 —— 固定机位、小物体操作，
# 帧间差别只有机械臂位置。隔开之后才是「难」而不是「无解」。
IV_MIN_SEGMENT_GAP = 2
NONE_TEXT = "All other options are wrong."   # v1 的原文，保持一致便于对照

# 段内相位。与 `frames.py` 的 PHASES / ANCHOR 必须一致 ——
# 两处分叉的话 plan 会去要一张池子里没有的帧。
PHASES = (0.1, 0.3, 0.5, 0.7, 0.9)
ANCHOR = 0.5

# 四个选项【两两】都要过同一条下限 —— 六条边一视同仁。
#
# 曾经把「干扰项彼此」放宽到一半，理由是「那只为了不出现双胞胎，门槛该低些」。
# **那个理由是错的，而且错得可测**：规则一旦不对称，答案就成了离群点 ——
# 实测答案在「离其它三个最远」上排第一的比例是 **53%**（应为 25%，
# gift_inhand 79% / pen_inbox 75%），「挑最不像的那个」白送 28 个百分点。
# 盲测先抓到的是它的反面：2B 模型偏爱居中的那张，于是【低于】随机 8.3 点（−3.6σ）。
#
# 六条边同一条门槛，答案在构造上就不再特殊。
#
# 系数 0.85 不是随手取的：p25 是给【三条边】标定的，六条边都要过时
# 同一个数字严格得多 —— 实测只剩 813 道，airpods/gift_inhand/pen_inbox
# 分别只有 31/5/6，三个族等于没有。0.85 下六族齐全、1,264 道，
# 而离群检查仍然均匀：**对称性来自约束的形状，不来自它的高低。**
MUTUAL_RATIO = 0.85

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


class Looks:
    """帧的「看起来像不像」。由 `frames.py` 预先算好，这里只查表。

    **为什么图选项需要这个** —— 结构判据（来自别集 / 隔了两段）预设了
    「不同集看起来不一样」。这批数据里前提不成立：同一套脚本、固定机位，
    **同一个动作在任何一集里都是同一张图**（18 组里 15 组，同动作·别集的
    画面距离显著小于不同动作）。人工复核判「无解」的 7 道，
    **结构上全部合规**（D-56 / D-57）。

    下限只在【不同动作帧对】上标定 —— 混进同动作近似对的话基准是脏的，
    一边漏拦一边误伤（第一版据此错丢了 98 道 `left_right`）。
    """

    def __init__(self, payload: dict[str, Any] | None, desc: Any = None) -> None:
        self.on = bool(payload)
        if not self.on:
            self.floors, self.index, self.desc = {}, {}, None
            return
        self.floors = payload["floors"]
        self.index = {k: i for i, k in enumerate(payload["order"])}
        self.desc = desc.astype("float32")
        self.meta = payload["frames"]

    def has(self, key: str) -> bool:
        return not self.on or key in self.index

    def distance(self, a_key: str, b_key: str) -> float:
        ia, ib = self.index.get(a_key), self.index.get(b_key)
        if ia is None or ib is None:
            return float("inf")            # 池子里没有就不拦，交给别的判据
        d = self.desc[ia] - self.desc[ib]
        return float((d @ d / d.size) ** 0.5)

    def floor(self, family: str, views: list[str]) -> float:
        """取涉及视角里**最松**的下限 —— left_right 的选项跨左右两个腕视角，
        按最严的那个会过度丢题。"""
        if not self.on:
            return 0.0
        return min((self.floors[f"{family}/{v}"] for v in views
                    if f"{family}/{v}" in self.floors), default=0.0)

    def far_enough(self, a_key: str, b_key: str, family: str, views: list[str]) -> bool:
        return self.distance(a_key, b_key) >= self.floor(family, views)


def frame_key(family: str, episode: str, view: str, frame: int) -> str:
    return f"{family}/{episode}/{view}/frame/{frame:06d}-{frame:06d}"


def order_text(labels: list[int]) -> str:
    """把标号序列写成选项文字。**只此一处** —— 正确答案与干扰项若各写各的，
    一个空格的差异就会让「答案在选项内」的出厂检查漏判。"""
    return " -> ".join(f"Image {n}" for n in labels)


def phase_frame(segment: dict[str, Any], phase: float) -> int:
    """与 `frames.py.phase_frames` 同一个算式，不可分叉。"""
    return segment["start_frame"] + int(round(
        (segment["end_frame"] - segment["start_frame"]) * phase))


def other_blocks(item_id: str, segments: list[dict[str, Any]], i: int,
                 ) -> list[tuple[dict[str, Any], float, int]]:
    """同集里【别的动作块】的候选帧，按取用优先级排好。

    人定的三条约束在这里落地：
      **只从同集取** —— 参数就是本集的 segments，跨集根本进不来
      **动作必须不同** —— `subtask != 本段 subtask`
      **相位可变** —— 每个块贡献 5 个相位，让相邻块也能给出够远的帧

    优先级：先隔 ≥2 段的块（画面天然更远），再隔 1 段的；
    同一个块内先给锚点相位，再给两端相位（离得最远的先来）。
    块之间按 `md5(题目id)` 轮转，避免永远取同一个块。
    """
    here = segments[i]
    far = [(j, t) for j, t in enumerate(segments)
           if t["subtask"] != here["subtask"] and abs(j - i) >= IV_MIN_SEGMENT_GAP]
    near = [(j, t) for j, t in enumerate(segments)
            if t["subtask"] != here["subtask"] and abs(j - i) == 1]
    k = int(hashlib.md5(item_id.encode()).hexdigest(), 16)
    out: list[tuple[dict[str, Any], float, int]] = []
    for tier in (far, near):
        if not tier:
            continue
        for step in range(len(tier)):
            _j, block = tier[(k + step * 7919) % len(tier)]
            for phase in (ANCHOR, PHASES[0], PHASES[-1], PHASES[1], PHASES[-2]):
                out.append((block, phase, phase_frame(block, phase)))
    return out


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
          none_option: str, looks: "Looks") -> dict[str, Any]:
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
    frames_path = BUILD / "frames.json"
    if not frames_path.exists():
        print("❌ 缺 build/frames.json —— 图选项题型要靠它判「干扰项在画面上分不分得开」。\n"
              "   先跑 python3 src/vqa/frames.py --write")
        return 1
    looks_payload = json.loads(frames_path.read_text(encoding="utf-8"))
    import numpy
    looks_desc = numpy.load(BUILD / "frames_desc.npy")
    vocab = json.loads((BUILD / "vocab.json").read_text(encoding="utf-8"))["families"]
    # `data/llm_cache/` 三代（v1-vendor / v2 / v3）全部退场，只作留档 ——
    # 干扰项改为一律取自真实标签，不再需要任何生成物（D-37 / D-38）。
    looks = Looks(looks_payload, looks_desc)
    plan = build(index, vocab, window, time_repeats, cap, none_option, looks)

    # 确定性自检：同样输入必须得到同样一批题。
    # **构建两遍比对**，因为这类 bug（遍历 set、用 dict 顺序、掺进时间戳）
    # 不会报错，只会让每次出的题悄悄不同 —— 而下游的盲测结论就此失效。
    again = build(index, vocab, window, time_repeats, cap, none_option, looks)
    fp = [hashlib.md5(json.dumps([[i["id"], i["answer_text"], *i["distractors"]]
                                  for i in p["items"]], ensure_ascii=False,
                                 sort_keys=True).encode()).hexdigest()[:12]
          for p in (plan, again)]
    if fp[0] != fp[1]:
        print(f"❌ 构建不确定：两次指纹 {fp[0]} ≠ {fp[1]}")
        print("   同样输入得到了不同的题。常见原因：遍历 set / dict、掺进时间或随机数。")
        return 1
    # 把【实际用的门槛】写进产物 —— 出厂检查照着这个验，
    # 而不是自己另算一遍。检查与规则分叉会在合规的题上报警（踩过）。
    plan["option_floor"] = {
        "percentile_source": "build/frames_floors.json",
        "scale": {"image_in_video": MUTUAL_RATIO, "left_right": 1.0},
        "applies_to": {"image_in_video": "四个选项两两（六条边）",
                       "left_right": "正确项与每条干扰项"},
    }
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
