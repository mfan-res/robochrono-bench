#!/usr/bin/env python3
# coding: utf-8
"""④ 出题的共用地基 —— 常量、判据、选项构造。**不含任何题型分支。**

`plan.py` 曾经是 900 多行、`build()` 一个函数 448 行，七个题型的规则挤在同一个
循环里。加一个题型就往里插一段 —— `frames.py` 那次就是这么插的，连步骤编号
都插乱了（两个文件都自称「第四步」）。

拆分的判据是**「加一个题型要碰几处已有代码」**：
共用的东西放这里，题型各自的规则放 `tasks/<类>.py`，
`plan.py` 只剩编排（读产物 → 遍历段 → 调题型 → 去重素材 → 自检 → 报表）。

**这个文件里的每个常量都有实测依据**，改之前先读它上面那段注释 ——
它们不是可调参数，是量出来的：`MUTUAL_RATIO` 是六条边同门槛下的产量拐点、
`MIN_CLIP_SECONDS` 卡在 1 帧孤例与 0.57 秒次短段之间、
`IV_MIN_SEGMENT_GAP` 来自人工复核「相邻段画面几乎一样」。

回归：`src/vqa/tests/test_plan_invariants.py`。**改这里之前先跑它** ——
它锁着全部 10,178 道题的内容指纹。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

# `normalize` 只有一份实现，在 vocab.py —— 「同一个动作有 11 种表示」那件事的
# 直接后果就是这类归一化必须只存一处（D-25）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vocab import normalize  # noqa: E402

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

    # **自己去重，不指望调用方。** 生产上 `borrowable` 是 `set(别族) - set(本族)`
    # 算出来的，天然不与 inside 重叠，所以这一步当前是恒等的（指纹不变）。
    # 但它是个上了膛的坑：谁把 borrowable 的构造改成不减本族，
    # 四选一就会**静默变成三选一**，而随机基线仍按 25% 报、盲测结论跟着失效。
    # 选项数是这套题的地基，不该依赖上游某处的减法还在。
    pool: list[str] = []
    for text in inside + outside:
        if text not in pool:
            pool.append(text)
    chosen = pool[:DISTRACTORS_PER_QUESTION]
    within = set(actions)
    return chosen, {"in_family": sum(c in within for c in chosen),
                    "borrowed": sum(c not in within for c in chosen)}


