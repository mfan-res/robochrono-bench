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

SEVERITY = {"污染": "✗", "覆盖": "⚠", "歧义": "⚠", "重叠": "✗", "派生": "✗", "引用": "✗"}


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


def episode_bounds(family: str) -> dict[str, list[tuple[float, float]]]:
    """从 LeRobot 的 meta/episodes 读各视频里 episode 的时间区间。

    ⚠ parquet 里有多个 ``*_file_index`` 列（``data/…`` 与 ``videos/<view>/…``）。
    **必须显式取 ``videos/`` 开头的那一列** —— 取错会把「状态打包成一个 parquet」
    误读成「一个视频装 40 轮」（D-19 记过这个坑）。

    ⚠ 元表本身不完整：tea/wash 只记了 10 集而实际有 39/40。
    所以查不到的视频返回**空**，调用方必须把「查不到」与「只有一集」区别对待。
    """
    tables = list((RAW / family / "meta" / "episodes").rglob("*.parquet"))
    if not tables:
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    frame = pq.read_table(tables[0]).to_pandas()
    cols = {suffix: next((c for c in frame.columns
                          if c.startswith("videos/") and c.endswith(suffix)), None)
            for suffix in ("file_index", "from_timestamp", "to_timestamp")}
    if not all(cols.values()):
        return {}
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for _, row in frame.iterrows():
        key = f"file-{int(row[cols['file_index']]):03d}"
        out[key].append((float(row[cols["from_timestamp"]]), float(row[cols["to_timestamp"]])))
    return {k: sorted(v) for k, v in out.items()}


def check_family(family: str, report: Report) -> None:
    base = LABEL / family
    subtasks = {s["id"] for s in
                json.loads((base / "subtasks.json").read_text(encoding="utf-8"))["subtasks"]}
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
          f"{'打包视频':>9}{'漏标集':>8}{'歧义文件':>9}")
    print("-" * 74)
    for family in families:
        s = report.stats[family]
        print(f"{family:<13}{s.get('segments',0):>6}{s.get('polluted_files',0):>9}"
              f"{s.get('frame_overlap',0):>8}{s.get('derived_mismatch',0):>9}"
              f"{s.get('packed_files',0):>9}{s.get('episodes_unlabeled',0):>8}"
              f"{s.get('ambiguous_files',0):>9}")
    print("-" * 74)

    by_kind = Counter(f.kind for f in report.findings)
    if not report.findings:
        print("六条检查全部通过")
        return 0
    print(f"\n共 {len(report.findings)} 条发现：{dict(by_kind)}\n")
    for kind in ("污染", "引用", "派生", "重叠", "覆盖", "歧义"):
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
