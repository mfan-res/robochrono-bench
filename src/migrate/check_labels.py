#!/usr/bin/env python3
# coding: utf-8
"""③ 标注核验 —— 全量检查人工标注，把 P-01 那类问题一次性挖干净。

为什么要这一段
--------------
P-01（tea2 的 Time EQA 真值系统性歧义）是 subagent 顺手挖出来的，
不是我们系统性查出来的。**其余六族有没有类似问题，此前不知道。**

七个任务的真值**全部**源自这 320 份标注。标注错一处，题就错一片，
而且错得不报错 —— 这正是七段框架里「⑤ 验题」缺失的代价，
但很多问题在 ③ 就能查出来，不必等到出完题。

查六项
------
1. **对齐**：标注帧号不越界，`start_frame/start` 隐含的 fps 与视频实际一致
2. **覆盖率**：标注跨度 / 视频全长。P-01 里 tea2 只有 37%，其余 84–92%
3. **只覆盖第一集？**：与 `meta/episodes` 的边界比对 —— P-01 的直接成因
4. **重复动作**：同一视频内 narration 重复。重复本身不是错
   （wash 反复搓洗是真实的），但**跨轮重复**会让时间类真值有歧义
5. **用词受控**：narration 的取值集合有多干净 —— D-04 要拿它派生词表
6. **可疑标注**：时长异常、边界倒置、空 narration、疑似笔误

**不修，只报。** 修法要逐条讨论 —— 有的能在出题阶段绕开（P-01），
有的要回去重标（笔误），代价差很多。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SOURCE = ROOT / "data" / "source"
LABEL = ROOT / "data" / "label"


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def frame_count(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True).stdout
    try:
        return int(out.strip())
    except ValueError:
        return 0


def episode_bounds(family: str) -> dict[str, list[tuple[float, float]]]:
    """从 meta/episodes 读出每个视频文件里各 episode 的时间区间。

    ⚠ parquet 里有多个 ``*_file_index`` 列（``data/…`` 与 ``videos/<view>/…``）。
    必须显式取 ``videos/`` 开头的那一列 —— 取错会把「状态打包成 1 个 parquet」
    误读成「一个视频装 40 轮」。
    ⚠ 这张表本身不完整（tea/wash 只记 10 集，实际 39/40），
    只能用来判断「谁打包了」，不能枚举全部 episode。
    """
    tables = list((RAW / family / "meta" / "episodes").rglob("*.parquet"))
    if not tables:
        return {}
    import pyarrow.parquet as pq

    frame = pq.read_table(tables[0]).to_pandas()
    file_col = next((c for c in frame.columns
                     if c.startswith("videos/") and c.endswith("file_index")), None)
    from_col = next((c for c in frame.columns
                     if c.startswith("videos/") and c.endswith("from_timestamp")), None)
    to_col = next((c for c in frame.columns
                   if c.startswith("videos/") and c.endswith("to_timestamp")), None)
    if not (file_col and from_col and to_col):
        return {}
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for _, row in frame.iterrows():
        out[f"file-{int(row[file_col]):03d}"].append((float(row[from_col]), float(row[to_col])))
    return {k: sorted(v) for k, v in out.items()}


def check_family(family: str) -> dict[str, Any]:
    meta = json.loads((RAW / family / "meta.json").read_text(encoding="utf-8"))
    physical = meta["views"]["main"].replace("observation.images.", "")
    bounds = episode_bounds(family)

    rows: list[dict[str, Any]] = []
    for seg_path in sorted((LABEL / family / "segments").glob("*_segments.json")):
        episode = seg_path.stem.replace("_segments", "")
        video = next((RAW / family / "videos" / f"observation.images.{physical}")
                     .rglob(f"{episode}.mp4"), None)
        if video is None:
            rows.append({"episode": episode, "error": "无对应视频"})
            continue
        segments = json.loads(seg_path.read_text(encoding="utf-8"))["segments"]
        if not segments:
            rows.append({"episode": episode, "error": "标注为空"})
            continue

        total, frames = duration(video), frame_count(video)
        starts = [float(s["start"]) for s in segments]
        ends = [float(s["end"]) for s in segments]
        span = max(ends) - min(starts)

        # ① 对齐：隐含 fps 与实际是否一致；帧号是否越界
        implied = [s["start_frame"] / s["start"] for s in segments
                   if s.get("start_frame") and float(s.get("start", 0)) > 0.5]
        real_fps = frames / total if total else 0
        fps_off = (max(abs(f - real_fps) for f in implied) if implied else 0.0)
        over = sum(1 for s in segments if (s.get("end_frame") or 0) > frames + 2)

        # ③ 只覆盖第一集？
        eps = bounds.get(episode, [])
        first_only = None
        if len(eps) > 1:
            first_only = max(ends) <= eps[0][1] + 1.0

        # ④ 重复动作
        narrations = [str(s.get("narration") or "").strip().rstrip(".").lower() for s in segments]
        dup = len(narrations) - len(set(narrations))

        rows.append({
            "episode": episode, "segments": len(segments),
            "duration": total, "span": span,
            "coverage": span / total if total else 0,
            "fps_off": fps_off, "frame_overflow": over,
            "episodes_in_file": len(eps), "first_episode_only": first_only,
            "dup_narrations": dup, "narrations": narrations,
            "gap_max": max((b - a for a, b in zip(sorted(ends)[:-1], sorted(starts)[1:])), default=0.0),
            "shortest": min(e - s for s, e in zip(starts, ends)),
            "inverted": sum(1 for s, e in zip(starts, ends) if e <= s),
        })
    return {"family": family, "rows": rows}


def main() -> int:
    only = sys.argv[1:] or None
    families = json.loads((ROOT / "data" / "families.json").read_text(encoding="utf-8"))["families"]
    targets = [f for f, v in families.items()
               if v.get("status") != "excluded" and (not only or f in only)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(check_family, targets))

    print("① 对齐 / ② 覆盖率 / ③ 是否只覆盖第一集")
    print(f"{'族':<13}{'集':>4}{'段/集':>7}{'覆盖率':>8}{'最低':>7}"
          f"{'fps偏差':>8}{'帧越界':>7}{'多轮打包':>9}{'只覆盖首轮':>11}")
    print("-" * 82)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for rep in reports:
        rows = [r for r in rep["rows"] if "error" not in r]
        all_rows[rep["family"]] = rows
        if not rows:
            continue
        cov = [r["coverage"] for r in rows]
        packed = [r for r in rows if r["episodes_in_file"] > 1]
        first = [r for r in packed if r["first_episode_only"]]
        print(f"{rep['family']:<13}{len(rows):>4}"
              f"{sum(r['segments'] for r in rows) / len(rows):>7.1f}"
              f"{sum(cov) / len(cov):>8.0%}{min(cov):>7.0%}"
              f"{max(r['fps_off'] for r in rows):>8.2f}"
              f"{sum(r['frame_overflow'] for r in rows):>7}"
              f"{len(packed):>9}{(str(len(first)) if packed else '—'):>11}")

    print("\n④ 重复动作 / ⑤ 用词 / ⑥ 可疑项")
    print(f"{'族':<13}{'词表':>5}{'有重复的集':>11}{'最短段':>8}{'最大空隙':>9}{'倒置':>6}  可疑")
    print("-" * 82)
    for family, rows in all_rows.items():
        if not rows:
            continue
        vocab = Counter(n for r in rows for n in r["narrations"])
        dup_eps = sum(1 for r in rows if r["dup_narrations"] > 0)
        odd: list[str] = []
        rare = [w for w, c in vocab.items() if c == 1]
        if rare:
            odd.append(f"仅出现 1 次的说法 {len(rare)} 种")
        if min(r["shortest"] for r in rows) < 0.5:
            odd.append("有 <0.5s 的段")
        print(f"{family:<13}{len(vocab):>5}{dup_eps:>11}"
              f"{min(r['shortest'] for r in rows):>7.2f}s"
              f"{max(r['gap_max'] for r in rows):>8.1f}s"
              f"{sum(r['inverted'] for r in rows):>6}  {'; '.join(odd) or '—'}")

    print("\n⑤ 各族 narration 词表（D-04 将据此派生候选动作集）")
    for family, rows in all_rows.items():
        vocab = Counter(n for r in rows for n in r["narrations"])
        rare = sorted(w for w, c in vocab.items() if c == 1)
        print(f"\n  [{family}] {len(vocab)} 种说法")
        for word, count in vocab.most_common(6):
            print(f"      {count:>4}× {word}")
        if rare:
            print(f"      仅 1 次（可疑笔误）：{rare[:6]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
