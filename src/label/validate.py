#!/usr/bin/env python3
# coding: utf-8
"""标注校验器 —— 六条检查，每条对应一个已经踩过的坑。

一套代码，两处使用
------------------
标注工具保存时跑，以及对存量 320 份离线跑。**必须是同一份代码** ——
之前 `check_labels.py` 和标注工具各写各的判据，结果两边口径不一致，
tea2 显示「21/21 齐全」而实际只有 20 集可用。

六条检查的由来
--------------
每一条都不是想出来的，是踩出来的：

=====  ============================================  ==========================
检查    针对                                          怎么发现的
=====  ============================================  ==========================
污染    出题产物被回写进 label 层                      P-03：stack_cubes 带 metadata
覆盖    整个 episode 没被标注                          P-01：tea2 只标了第一集
歧义    同一 episode 内 subtask 重复                   P-01 / P-05：Time EQA 真值不唯一
重叠    段与段真重叠（**走帧号，不走秒**）              P-02b / P-04
派生    start/end 与 start_frame/end_frame 不自洽      上游 end=(f+1)/fps 的隐含语义
引用    subtask 引用了 subtasks.json 里没有的 ID       ID 化之后的新风险
=====  ============================================  ==========================

为什么重叠必须走帧号
--------------------
``end = (end_frame + 1) / fps``（闭区间转半开）使相邻段的**秒区间**必然重叠一帧。
全量 631 处。按秒判断「落在恰好一个段内」会在这 631 个点上判到两个段；
**帧号层是干净的**。这个坑很容易再踩，所以判据写死在这里。

只报不修
--------
修法要逐条讨论 —— 有的能在出题阶段绕开（P-01 裁到第 0 集），
有的要还原数据（P-03），有的只能改能力定义（P-05）。代价差很多。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LABEL = ROOT / "data" / "label"
RAW = ROOT / "data" / "raw"

# 只有这些键是标注该有的。多出来的一律视为污染 —— 尤其 metadata，
# 那是出题器 window_for_segment 写的（P-03）。
ALLOWED_SEGMENT_KEYS = {"id", "subtask", "start_frame", "end_frame",
                        "start", "end", "start_time", "end_time", "episode_index"}

# ✗ = 必须修；⚠ = 待人判断。「序列」是 ✗ —— 动作讲不通只有两种可能：
# 标错了物体，或者漏标了一段，两种都要改数据。
SEVERITY = {"污染": "✗", "覆盖": "⚠", "歧义": "⚠", "重叠": "✗",
            "派生": "✗", "引用": "✗", "序列": "✗", "可疑": "⚠"}


@dataclass
class Finding:
    kind: str
    family: str
    episode: str
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, kind: str, family: str, episode: str, detail: str) -> None:
        self.findings.append(Finding(kind, family, episode, detail))


# `episode_bounds` 只有一份实现，在 `src/vqa/index.py`。
# **这里曾经自己抄了一份**，于是那份里的两个 bug（只读第一个 parquet、
# 靠列顺序猜主视角）在 index.py 修好之后仍然留在这里 ——
# 正是「同一个东西存两份，不一致时不报错」的现场（D-42）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vqa"))
from index import episode_bounds  # noqa: E402
from vocab import parse  # noqa: E402

# 动词分类，用于「序列」检查
TAKE = {"pick", "pick up", "take", "grasp"}
DROP = {"put", "place", "put down"}
PREPS = {"with", "in", "on", "into", "onto", "to", "from", "at"}


def check_family(family: str, report: Report) -> None:
    base = LABEL / family
    # 缺 subtasks.json 说明这个目录不是一个成形的族（例如刚删了一半）。
    # **报告，不崩溃** —— 崩溃会让其余六族的检查结果一起拿不到。
    if not (base / "subtasks.json").exists():
        report.add("引用", family, "-", "缺 subtasks.json，跳过该目录")
        return
    defined = json.loads((base / "subtasks.json").read_text(encoding="utf-8"))["subtasks"]
    subtasks = {s["id"] for s in defined}
    texts = {s["id"]: s["text"] for s in defined}
    bounds = episode_bounds(family)
    fps_by_family = json.loads((RAW / family / "meta.json").read_text(encoding="utf-8")).get("fps")

    counts = Counter()
    for path in sorted((base / "segments").glob("*_segments.json")):
        episode = path.stem.replace("_segments", "")
        segments = json.loads(path.read_text(encoding="utf-8"))["segments"]
        counts["segments"] += len(segments)

        # ① 污染 —— 出题产物回写
        extra = {k for seg in segments for k in seg} - ALLOWED_SEGMENT_KEYS
        if extra:
            counts["polluted_files"] += 1
            report.add("污染", family, episode, f"多出字段 {sorted(extra)}")

        # ② 引用 —— subtask 必须存在于定义里
        for seg in segments:
            if seg.get("subtask") not in subtasks:
                report.add("引用", family, episode, f"未定义的 subtask {seg.get('subtask')!r}")

        # ③ 派生 —— start/end 必须与帧号自洽
        if fps_by_family:
            for seg in segments:
                want_start = round(seg["start_frame"] / fps_by_family, 3)
                want_end = round((seg["end_frame"] + 1) / fps_by_family, 3)
                if abs(seg["start"] - want_start) > 0.002 or abs(seg["end"] - want_end) > 0.002:
                    counts["derived_mismatch"] += 1
                    report.add("派生", family, episode,
                               f"{seg['id']}: 文件 {seg['start']}–{seg['end']}，"
                               f"按 fps={fps_by_family} 应为 {want_start}–{want_end}")

        # ④ 重叠 —— 走帧号。共享边界（前 end_frame == 后 start_frame）不算重叠
        ordered = sorted(segments, key=lambda s: (s["start_frame"], s["end_frame"]))
        for a, b in zip(ordered, ordered[1:]):
            if b["start_frame"] < a["end_frame"]:
                counts["frame_overlap"] += 1
                report.add("重叠", family, episode,
                           f"{a['id']} 帧 {a['start_frame']}–{a['end_frame']} 与 "
                           f"{b['id']} 帧 {b['start_frame']}–{b['end_frame']} 重叠")

        # ④b 可疑段：零长度、同起点。不判错 —— 由人看画面决定是不是误标
        for seg in segments:
            if seg["end_frame"] <= seg["start_frame"]:
                counts["zero_length"] += 1
                report.add("可疑", family, episode,
                           f"{seg['id']} 长度为 {seg['end_frame'] - seg['start_frame'] + 1} 帧"
                           f"（{seg['subtask']}）—— 疑似误按")
        starts = Counter(s["start_frame"] for s in segments)
        for frame, n in starts.items():
            if n > 1:
                counts["same_start"] += 1
                report.add("可疑", family, episode,
                           f"{n} 段从同一帧 {frame} 开始 —— id 需加后缀区分")

        # ⑤ 覆盖 —— 多 episode 的视频里，有没有整集没被标注
        eps = bounds.get(episode)
        if eps and len(eps) > 1:
            counts["packed_files"] += 1
            spans = [(s["start"], s["end"]) for s in segments]
            missed = [i for i, (lo, hi) in enumerate(eps)
                      if not any(lo - 0.5 <= a and b <= hi + 0.5 for a, b in spans)]
            if missed:
                counts["episodes_unlabeled"] += len(missed)
                report.add("覆盖", family, episode,
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
            got = parse(texts.get(seg["subtask"], ""))
            verb, obj = got["verb"], got["object"]
            if verb in TAKE:
                if obj in held:
                    report.add("序列", family, episode,
                               f"{seg['start']:.2f}s 又拿了一次「{obj}」，上一个还没放下")
                held.add(obj)
            elif verb in DROP:
                if obj in takeable and obj not in held:
                    report.add("序列", family, episode,
                               f"{seg['start']:.2f}s 放下「{obj}」，但没拿起过")
                held.discard(obj)
            else:
                need = [obj, *(w for w in got["modifier"].split() if w not in PREPS)]
                miss = [o for o in need if o in takeable and o not in held]
                if miss:
                    report.add("序列", family, episode,
                               f"{seg['start']:.2f}s「{verb} {obj}」但手里没有 {miss}")

        # ⑥ 歧义 —— 同一 episode 内同一 subtask 出现多次，时间类任务真值不唯一
        per_episode: dict[int, Counter] = defaultdict(Counter)
        for seg in segments:
            idx = 0
            if eps:
                idx = next((i for i, (lo, hi) in enumerate(eps)
                            if lo - 0.5 <= seg["start"] <= hi + 0.5), 0)
            per_episode[idx][seg["subtask"]] += 1
        for idx, tally in per_episode.items():
            dup = {k: v for k, v in tally.items() if v > 1}
            if dup:
                counts["ambiguous_files"] += 1
                report.add("歧义", family, episode,
                           f"episode {idx} 内 {dup} —— 按 subtask 问时刻的题真值不唯一")
                break

    report.stats[family] = dict(counts)


def main() -> int:
    only = sys.argv[1:]
    families = sorted(p.name for p in LABEL.iterdir()
                      if p.is_dir() and (not only or p.name in only))
    report = Report()
    for family in families:
        check_family(family, report)

    print(f"{'族':<13}{'段数':>6}{'污染文件':>9}{'帧重叠':>8}{'派生不符':>9}"
          f"{'打包视频':>9}{'漏标集':>8}{'歧义':>6}{'可疑':>6}")
    print("-" * 74)
    for family in families:
        s = report.stats[family]
        print(f"{family:<13}{s.get('segments',0):>6}{s.get('polluted_files',0):>9}"
              f"{s.get('frame_overlap',0):>8}{s.get('derived_mismatch',0):>9}"
              f"{s.get('packed_files',0):>9}{s.get('episodes_unlabeled',0):>8}"
              f"{s.get('ambiguous_files',0):>6}"
              f"{s.get('zero_length',0)+s.get('same_start',0):>6}")
    print("-" * 74)

    by_kind = Counter(f.kind for f in report.findings)
    if not report.findings:
        print("六条检查全部通过")
        return 0
    print(f"\n共 {len(report.findings)} 条发现：{dict(by_kind)}\n")
    for kind in ("污染", "引用", "派生", "重叠", "覆盖", "序列", "歧义", "可疑"):
        items = [f for f in report.findings if f.kind == kind]
        if not items:
            continue
        print(f"{SEVERITY[kind]} {kind}（{len(items)} 条）")
        seen: set[str] = set()
        for f in items:
            key = f"{f.family}|{f.detail[:40]}"
            if key in seen:
                continue
            seen.add(key)
            print(f"    {f.family}/{f.episode}  {f.detail}")
            if len(seen) >= 4:
                rest = len(items) - 4
                if rest > 0:
                    print(f"    …… 另有 {rest} 条同类")
                break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
