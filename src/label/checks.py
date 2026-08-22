#!/usr/bin/env python3
# coding: utf-8
"""标注的判据 —— **只有这一份实现**。

这个模块只做一件事：给它一份标注文档 + 上下文，它告诉你有哪些问题。
**不读目录、不写报表、不管界面。** 那些是消费者的事：

    validate.py        遍历全族 → 调这里 → 出离线报表
    serve.py review()  保存前拿当前文档 → 调这里 → 变成浏览器提示

为什么必须只有一份
------------------
D-25 那次的洞察是「同一个动作在系统里被存了 11 份，每一份都是一次分叉的机会」，
解法是**文字只存在 subtasks.json 一处，其余存 ID**。那条在**数据层**落实了。

**但判据层没落实**：`serve.py` 曾自己抄了一份只有三类的检查，
而 `serve.py` / `validate.py` / README / AGENTS.md **四处都写着「共用同一份判据」**。
AGENTS.md 甚至标着「风险：高 —— 判据分叉过一次，代价很大」——
指的是 v1 时代 `check_labels.py` 与标注工具各写各的，导致 tea2 显示
「21/21 齐全」而实际只有 20 集可用。**同样的事又发生了一次，文档还在宣称它没发生。**

这个模块就是把那句话变成真的。

八类检查，每条对应一个踩过的坑
------------------------------
=====  ============================================  ==========================
检查    针对                                          怎么发现的
=====  ============================================  ==========================
污染    出题产物被回写进 label 层                      P-03：stack_cubes 带 metadata
覆盖    整个 episode 没被标注                          P-01：tea2 只标了第一集
歧义    同一 episode 内 subtask 重复                   P-01 / P-05：Time EQA 真值不唯一
重叠    段与段真重叠（**走帧号，不走秒**）              P-02b / P-04
派生    start/end 与 start_frame/end_frame 不自洽      上游 end=(f+1)/fps 的隐含语义
引用    subtask 引用了 subtasks.json 里没有的 ID       ID 化之后的新风险
序列    动作序列讲不通                                 手里没拿着的东西不能放/用
可疑    零长度段、同起点                               由人看画面决定是不是误标
=====  ============================================  ==========================

为什么重叠必须走帧号
--------------------
``end = (end_frame + 1) / fps``（闭区间转半开）使相邻段的**秒区间**必然重叠一帧。
全量 631 处。按秒判断「落在恰好一个段内」会在这 631 个点上判到两个段；
**帧号层是干净的**。这个坑很容易再踩，所以判据写死在这里。

回归
----
``src/label/tests/test_checks.py`` —— 夹具是 ``segments.before_*`` 与
``segments.polluted``，即每条检查当初要抓的那次真实事故。
**改这个文件之前先跑它。**
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vqa"))
from vocab import parse  # noqa: E402

# 只有这些键是标注该有的。多出来的一律视为污染 —— 尤其 metadata，
# 那是出题器 window_for_segment 写的（P-03）。
#
# ⚠ 这份名单与 `src/common/schemas/segments.json` 的 `additionalProperties: false`
# 是同一件事。**schema 是权威**；这里保留一份是为了在没装 jsonschema 时仍能查污染。
# 两者由 `check_schema()` 交叉核对，不一致会自己报出来。
ALLOWED_SEGMENT_KEYS = {"id", "subtask", "start_frame", "end_frame",
                        "start", "end", "start_time", "end_time", "episode_index"}

# ✗ = 必须修；⚠ = 待人判断。「序列」是 ✗ —— 动作讲不通只有两种可能：
# 标错了物体，或者漏标了一段，两种都要改数据。
SEVERITY = {"污染": "✗", "覆盖": "⚠", "歧义": "⚠", "重叠": "✗",
            "派生": "✗", "引用": "✗", "序列": "✗", "可疑": "⚠", "结构": "✗"}

BLOCKING = {k for k, v in SEVERITY.items() if v == "✗"}

# 动词分类，用于「序列」检查
TAKE = {"pick", "pick up", "take", "grasp"}
DROP = {"put", "place", "put down"}
PREPS = {"with", "in", "on", "into", "onto", "to", "from", "at"}

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "common" / "schemas" / "segments.json"


@dataclass
class Finding:
    kind: str
    detail: str

    @property
    def severity(self) -> str:
        return SEVERITY.get(self.kind, "⚠")

    @property
    def blocking(self) -> bool:
        return self.kind in BLOCKING


def check_schema(document: dict[str, Any]) -> list[Finding]:
    """按 `schemas/segments.json` 校验整份文档。

    **这个契约此前没有任何代码加载**（4.4）—— 它定义了 `id` 的 pattern、
    `subtask` 的 pattern、`source` 的必填字段，而校验器只手抄了其中
    「多余字段」那一条。README 却写着「每个层间边界都有 schema + 校验器」。

    没装 jsonschema 就跳过并说明，不静默通过 —— 抄 `vqa/compose.py` 的现成做法。
    """
    try:
        import jsonschema
    except ImportError:
        return [Finding("结构", "跳过 schema 校验（没装 jsonschema）")]
    if not SCHEMA_PATH.exists():
        return [Finding("结构", f"找不到契约 {SCHEMA_PATH}")]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    out: list[Finding] = []
    for error in sorted(validator.iter_errors(document), key=lambda e: list(e.path))[:8]:
        where = "/".join(str(x) for x in error.path) or "(根)"
        out.append(Finding("结构", f"{where}: {error.message[:120]}"))
    return out


def check_document(
    document: dict[str, Any],
    *,
    subtasks: set[str],
    texts: dict[str, str],
    fps: float | None,
    bounds: list[list[float]] | None = None,
    schema: bool = True,
) -> tuple[list[Finding], Counter]:
    """核一份标注文档。返回 (发现, 计数)。

    ``document``  一份 ``*_segments.json`` 的完整内容（含 ``source`` 与 ``segments``）
    ``subtasks``  该族定义过的 subtask id
    ``texts``     id → 文字，「序列」检查要用它解析动词与宾语
    ``fps``       该族的 fps，来自 ``data/raw/<族>/meta.json``；缺就跳过「派生」
    ``bounds``    该视频里各 episode 的时间区间；``None`` 或只有一集时跳过「覆盖」
    ``schema``    是否跑 `schemas/segments.json`（在线保存时可关，省一次解析）
    """
    findings: list[Finding] = []
    counts: Counter = Counter()

    def add(kind: str, detail: str) -> None:
        findings.append(Finding(kind, detail))

    if schema:
        findings.extend(check_schema(document))

    segments = document.get("segments") or []
    counts["segments"] += len(segments)
    fps_by_family = fps
    eps = bounds

    # ① 污染 —— 出题产物回写
    extra = {k for seg in segments for k in seg} - ALLOWED_SEGMENT_KEYS
    if extra:
        counts["polluted_files"] += 1
        add("污染", f"多出字段 {sorted(extra)}")

    # ② 引用 —— subtask 必须存在于定义里
    #
    # **缺这个键也走这条**，不要崩。本文件开头对「缺 subtasks.json」写过
    # 「报告，不崩溃 —— 崩溃会让其余六族的检查结果一起拿不到」，
    # 那条原则同样适用于段里缺 subtask：v1 形状的数据（存 narration 而非
    # subtask id）喂进来时，此前会 KeyError 掉整个族。
    # 而「拿 v1 数据当回归夹具」正是我们想做的事（tests/test_checks.py）。
    for seg in segments:
        if seg.get("subtask") not in subtasks:
            add("引用", f"未定义的 subtask {seg.get('subtask')!r}")

    # ③ 派生 —— start/end 必须与帧号自洽
    if fps_by_family:
        for seg in segments:
            want_start = round(seg["start_frame"] / fps_by_family, 3)
            want_end = round((seg["end_frame"] + 1) / fps_by_family, 3)
            if abs(seg["start"] - want_start) > 0.002 or abs(seg["end"] - want_end) > 0.002:
                counts["derived_mismatch"] += 1
                add("派生",
                           f"{seg['id']}: 文件 {seg['start']}–{seg['end']}，"
                           f"按 fps={fps_by_family} 应为 {want_start}–{want_end}")

    # ④ 重叠 —— 走帧号。共享边界（前 end_frame == 后 start_frame）不算重叠
    ordered = sorted(segments, key=lambda s: (s["start_frame"], s["end_frame"]))
    for a, b in zip(ordered, ordered[1:]):
        if b["start_frame"] < a["end_frame"]:
            counts["frame_overlap"] += 1
            add("重叠",
                       f"{a['id']} 帧 {a['start_frame']}–{a['end_frame']} 与 "
                       f"{b['id']} 帧 {b['start_frame']}–{b['end_frame']} 重叠")

    # ④b 可疑段：零长度、同起点。不判错 —— 由人看画面决定是不是误标
    for seg in segments:
        if seg["end_frame"] <= seg["start_frame"]:
            counts["zero_length"] += 1
            add("可疑",
                       f"{seg['id']} 长度为 {seg['end_frame'] - seg['start_frame'] + 1} 帧"
                       f"（{seg.get('subtask')}）—— 疑似误按")
    starts = Counter(s["start_frame"] for s in segments)
    for frame, n in starts.items():
        if n > 1:
            counts["same_start"] += 1
            add("可疑",
                       f"{n} 段从同一帧 {frame} 开始 —— id 需加后缀区分")

    # ⑤ 覆盖 —— 多 episode 的视频里，有没有整集没被标注
    if eps and len(eps) > 1:
        counts["packed_files"] += 1
        spans = [(s["start"], s["end"]) for s in segments]
        missed = [i for i, (lo, hi) in enumerate(eps)
                  if not any(lo - 0.5 <= a and b <= hi + 0.5 for a, b in spans)]
        if missed:
            counts["episodes_unlabeled"] += len(missed)
            add("覆盖",
                       f"视频含 {len(eps)} 个 episode，第 {missed} 个完全没有标注")

    # ⑤b 序列 —— 手里没拿着的东西不能放/用，拿着的东西不能再拿一次。
    #
    # **这条抓的是前六项都抓不到的一类错**：不重叠、不越界、不污染、
    # 词表内、覆盖完整，但动作序列讲不通。实测抓出两处真错误：
    #   wash/file-009  「pick_plate → pick_rag → wipe_bowl_with_brush」
    #                  抽帧看：机械臂拿着盘子用抹布擦。两段标错了物体
    #   wash/file-030  两次 pick_plate 之间没有 put_plate，中间 5.4 秒空隙
    #                  抽帧看：正在把盘子放进沥水架 —— 漏标了一段
    #
    # ⚠ **只对本族词表里「有 pick 动作」的物体配对。** tea 的 tea leaves
    # 只有 put 没有 pick，不加这个守卫会 39/39 集全报，
    # 而那只是词表没定义拿的动作 —— 第一版就是这么误报了 39 条。
    takeable = {parse(texts[i])["object"] for i in texts
                if parse(texts[i])["verb"] in TAKE}
    held: set[str] = set()
    for seg in sorted(segments, key=lambda s: s["start_frame"]):
        got = parse(texts.get(seg.get("subtask"), ""))
        verb, obj = got["verb"], got["object"]
        if verb in TAKE:
            if obj in held:
                add("序列",
                           f"{seg['start']:.2f}s 又拿了一次「{obj}」，上一个还没放下")
            held.add(obj)
        elif verb in DROP:
            if obj in takeable and obj not in held:
                add("序列",
                           f"{seg['start']:.2f}s 放下「{obj}」，但没拿起过")
            held.discard(obj)
        else:
            need = [obj, *(w for w in got["modifier"].split() if w not in PREPS)]
            miss = [o for o in need if o in takeable and o not in held]
            if miss:
                add("序列",
                           f"{seg['start']:.2f}s「{verb} {obj}」但手里没有 {miss}")

    # ⑥ 歧义 —— 同一 episode 内同一 subtask 出现多次，时间类任务真值不唯一
    per_episode: dict[int, Counter] = defaultdict(Counter)
    for seg in segments:
        idx = 0
        if eps:
            idx = next((i for i, (lo, hi) in enumerate(eps)
                        if lo - 0.5 <= seg["start"] <= hi + 0.5), 0)
        per_episode[idx][seg.get("subtask")] += 1
    for idx, tally in per_episode.items():
        dup = {k: v for k, v in tally.items() if v > 1}
        if dup:
            counts["ambiguous_files"] += 1
            add("歧义",
                       f"episode {idx} 内 {dup} —— 按 subtask 问时刻的题真值不唯一")
            break

    return findings, counts
