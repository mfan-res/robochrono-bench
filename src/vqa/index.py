#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第一步：把「有什么可用来出题」扫成一张表。

    python3 src/vqa/index.py            # 打印概览
    python3 src/vqa/index.py --write    # 同时写 build/index.json

为什么单独一步
--------------
后面每一步（选题、切片、组装）都需要知道：哪些集可用、每集多长多少帧、
fps 与分辨率、视角有哪些、episode 边界在哪、每段落在哪个 episode、
哪些段能出题哪些不能。这些信息散在 `data/source/*/meta.json`、
`data/label/*/segments/*.json`、`data/raw/*/meta/episodes/*.parquet` 三处，
每一步各扫一遍既慢又容易口径不一。**扫一次，后面都读它。**

「能不能出题」在这里就判掉
--------------------------
不是所有标注都能出题。限制分三个层级，**记在哪一级取决于它影响谁**：

段级 ``reasons``（字段 ``usable``）
    ``ambiguous_repeat`` —— 同一 episode 内重复的动作。wash 每集洗两个盘子，
    问「pick the plate 在第几秒」有两个都对的答案（P-05，处置待定）。

集级 ``full_video_usable``
    tea2 的视频装 2–3 集但只标了第一集。**这不能记在段上** ——
    落在未标注 episode 里的段根本不存在，标在段上一条也匹配不到（实测 0 条）。
    真正受影响的是**用全长视频的任务**（time）：后两集里有动作但没有真值，
    模型答对了反而被判错（P-01）。

帧区间 ``unlabeled_frame_gaps``
    段与段之间的空隙。stack_cubes 每集结尾有 8.6 秒无标注区间，
    从那里抽帧问「现在在做什么」，六个选项里没有正确答案。
    它既不否定某个段也不否定某一集，**只否定某些抽帧位置**，所以单独一张表。

**只标记，不丢弃。** 下游自己决定要不要用，这里不悄悄少给一批。

两个覆盖率不要混用
------------------
``span_coverage``     首段起点到末段终点 / 总帧数 —— 标注是否覆盖了主体时段，查漏标
``labeled_fraction``  各段长度之和 / 总帧数     —— 有多少位置可供抽帧

airpods 前者 80% 而后者 20%，因为它动作短、间隔长（每段约 1.8 秒）——
**那不是漏标，是任务本身如此。** 只看后者会误判成数据缺失。
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BUILD = ROOT / "build"

INDEX_VERSION = "1"

# 物理视角名 → 逻辑名的映射**只有一份**，在 `migrate/fetch_raw.py`。
# 这里导入而不是另写一份 —— 三个采集平台三套命名（top / left_eye / cam_color
# 都是 main），各写一份迟早对不上，而对不上时不报错，只是元表「查不到」。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "migrate"))
from fetch_raw import VIEW_MAP  # noqa: E402


def probe(path: Path) -> tuple[int, float]:
    """(帧数, 时长)。用容器元数据，不解码 —— `-count_frames` 会跑几十分钟。"""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-show_entries", "format=duration",
         "-of", "json", str(path)], capture_output=True, text=True).stdout
    try:
        data = json.loads(out)
        return int(data["streams"][0]["nb_frames"]), float(data["format"]["duration"])
    except Exception:  # noqa: BLE001
        return 0, 0.0


def episode_bounds(family: str) -> dict[str, list[list[float]]]:
    """各视频里 episode 的时间区间。

    ⚠ **`meta/episodes/` 下可能有多个 parquet，必须全读。**
    stack_cubes 有 24 个、tea/wash 各 4 个。此前只读 `tables[0]`，
    于是 tea/wash 只拿到 10 集、stack_cubes 只拿到 10 集 ——
    **曾被记成「上游元表不完整」，其实是这里的 bug**（D-42）。
    全读之后六个族都是每文件恰好一集，tea2 是唯一 1–3 集的。

    ⚠ parquet 里 ``*_file_index`` 列有好几组：``data/…`` 一组，
    **每个视角各一组**。取 ``data/…`` 会把「状态打包成一个 parquet」
    误读成「一个视频装 40 轮」；而各视角那几组**取值互不相同**
    （tea2 的 wrist_left 从第 2 集起就与 top 错开一个文件），
    所以按 `MAIN_STREAM` 显式指定，不能「取第一个 videos/ 开头的」。

    查不到仍返回空 —— 调用方必须把「查不到」与「只有一集」区别对待。
    """
    tables = list((DATA / "raw" / family / "meta" / "episodes").rglob("*.parquet"))
    if not tables:
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    import pandas
    frame = pandas.concat([pq.read_table(t).to_pandas() for t in sorted(tables)],
                          ignore_index=True)
    prefix = next((f"videos/observation.images.{n}/" for n, logical in VIEW_MAP.items()
                   if logical == "main"
                   and any(c.startswith(f"videos/observation.images.{n}/")
                           for c in frame.columns)), None)
    if prefix is None:
        return {}
    cols = {k: next((c for c in frame.columns
                     if c.startswith(prefix) and c.endswith(k)), None)
            for k in ("file_index", "from_timestamp", "to_timestamp")}
    if not all(cols.values()):
        return {}
    out: dict[str, list[list[float]]] = defaultdict(list)
    for _, row in frame.iterrows():
        out[f"file-{int(row[cols['file_index']]):03d}"].append(
            [float(row[cols["from_timestamp"]]), float(row[cols["to_timestamp"]])])
    return {k: sorted(v) for k, v in out.items()}


def episode_of(start: float, bounds: list[list[float]] | None) -> int:
    if not bounds:
        return 0
    return next((i for i, (lo, hi) in enumerate(bounds) if lo - 0.5 <= start <= hi + 0.5), 0)


def index_family(family: str, cap: int | None) -> dict[str, Any]:
    source_meta = json.loads((DATA / "source" / family / "meta.json").read_text(encoding="utf-8"))
    subtasks = json.loads((DATA / "label" / family / "subtasks.json").read_text(encoding="utf-8"))
    fps = float(source_meta["fps"])
    bounds_by_episode = episode_bounds(family)

    episodes: list[dict[str, Any]] = []
    for directory in sorted(p for p in (DATA / "source" / family).iterdir()
                            if p.is_dir() and p.name.startswith("file-")):
        name = directory.name
        if cap is not None and int(name.split("-")[1]) >= cap:
            continue                                  # families.json 的 episode_cap
        seg_path = DATA / "label" / family / "segments" / f"{name}_segments.json"
        if not seg_path.exists():
            continue
        frames, duration = probe(directory / "main.mp4")
        bounds = bounds_by_episode.get(name)
        segments = json.loads(seg_path.read_text(encoding="utf-8"))["segments"]

        # 同一 episode 内重复的 subtask —— 时间类题真值不唯一
        per_episode: dict[int, Counter] = defaultdict(Counter)
        for seg in segments:
            per_episode[episode_of(seg["start"], bounds)][seg["subtask"]] += 1
        repeated = {i: {k for k, n in c.items() if n > 1} for i, c in per_episode.items()}

        # 被标注覆盖的 episode
        labeled_eps = {episode_of(s["start"], bounds) for s in segments}
        missing_eps = ([i for i in range(len(bounds)) if i not in labeled_eps]
                       if bounds and len(bounds) > 1 else [])

        rows: list[dict[str, Any]] = []
        ordered = sorted(segments, key=lambda s: s["start_frame"])
        for seg in ordered:
            ep = episode_of(seg["start"], bounds)
            reasons = []
            if seg["subtask"] in repeated.get(ep, set()):
                reasons.append("ambiguous_repeat")
            rows.append({
                "id": seg["id"], "subtask": seg["subtask"], "episode_index": ep,
                "start_frame": seg["start_frame"], "end_frame": seg["end_frame"],
                "start": seg["start"], "end": seg["end"],
                "duration": round(seg["end"] - seg["start"], 3),
                "usable": not reasons, "reasons": reasons,
            })

        # 未标注区间：出题抽帧必须避开（六个选项里没有正确答案）
        gaps = []
        cursor = 0
        for seg in ordered:
            if seg["start_frame"] > cursor:
                gaps.append([cursor, seg["start_frame"] - 1])
            cursor = max(cursor, seg["end_frame"] + 1)
        if frames and cursor < frames:
            gaps.append([cursor, frames - 1])

        episodes.append({
            "episode": name, "frames": frames, "duration": round(duration, 3),
            "episode_bounds": bounds,
            "labeled_episodes": sorted(labeled_eps), "unlabeled_episodes": missing_eps,
            "unlabeled_frame_gaps": gaps,
            # 两个指标测的是不同的事，见模块 docstring —— 不要合并成一个「覆盖率」
            "span_coverage": round(
                (ordered[-1]["end_frame"] - ordered[0]["start_frame"] + 1) / frames, 3)
                if frames and ordered else 0.0,
            "labeled_fraction": round(
                sum(s["end_frame"] - s["start_frame"] + 1 for s in ordered) / frames, 3)
                if frames else 0.0,
            # 用全长视频的任务（time）能不能用这一集
            "full_video_usable": not missing_eps,
            "full_video_reason": (f"含 {len(bounds or [])} 个 episode，"
                                  f"第 {missing_eps} 个没有标注 —— 那些动作没有真值")
                                 if missing_eps else None,
            "segments": rows,
        })

    return {
        "family": family,
        "fps": fps,
        "resolution": source_meta["output"],
        "views": source_meta["views"],
        "timestamp_bar_cropped": source_meta.get("timestamp_bar_rows_cropped", 0),
        "subtasks": subtasks["subtasks"],
        "episodes": episodes,
    }


def main() -> int:
    write = "--write" in sys.argv
    registry = json.loads((DATA / "families.json").read_text(encoding="utf-8"))["families"]
    active = {f: v for f, v in registry.items()
              if v.get("status") not in ("excluded", "parked") and (DATA / "source" / f).is_dir()}

    index = {"index_version": INDEX_VERSION, "families": {}}
    print(f"{'族':<13}{'集':>4}{'段':>6}{'可用段':>7}{'跨度覆盖':>9}{'标注占比':>9}"
          f"{'全长可用':>9}  分辨率 · fps · subtask")
    print("-" * 88)
    totals = Counter()

    for family, info in sorted(active.items()):
        entry = index_family(family, info.get("episode_cap"))
        index["families"][family] = entry
        segs = [s for e in entry["episodes"] for s in e["segments"]]
        usable = sum(1 for s in segs if s["usable"])
        n = len(entry["episodes"]) or 1
        span = sum(e["span_coverage"] for e in entry["episodes"]) / n
        frac = sum(e["labeled_fraction"] for e in entry["episodes"]) / n
        full = sum(1 for e in entry["episodes"] if e["full_video_usable"])
        totals["episodes"] += len(entry["episodes"])
        totals["segments"] += len(segs)
        totals["usable"] += usable
        totals["full"] += full
        full_ratio = f"{full}/{len(entry['episodes'])}"
        res = f"{entry['resolution'][0]}×{entry['resolution'][1]}"
        print(f"{family:<13}{len(entry['episodes']):>4}{len(segs):>6}{usable:>7}"
              f"{span:>9.0%}{frac:>9.0%}{full_ratio:>9}  "
              f"{res} · {entry['fps']:g} · {len(entry['subtasks'])}")

    print("-" * 88)
    full_ratio = f"{totals['full']}/{totals['episodes']}"
    print(f"{'合计':<13}{totals['episodes']:>4}{totals['segments']:>6}{totals['usable']:>7}"
          f"{'':>9}{'':>9}{full_ratio:>9}")

    reasons = Counter(r for e in index["families"].values()
                      for ep in e["episodes"] for s in ep["segments"] for r in s["reasons"])
    if reasons:
        print(f"\n段不可用的原因：{dict(reasons)}")
        print("  ambiguous_repeat   同 episode 内动作重复，时间类题真值不唯一（P-05）")
    bad = [(f, e["episode"], e["full_video_reason"])
           for f, entry in index["families"].items()
           for e in entry["episodes"] if not e["full_video_usable"]]
    if bad:
        print(f"\n全长视频不可用（影响 time 任务）：{len(bad)} 集")
        for f, ep, why in bad[:2]:
            print(f"  {f}/{ep}  {why}")
        print(f"  …… 共 {len({f for f, _, _ in bad})} 个族")

    if write:
        BUILD.mkdir(exist_ok=True)
        out = BUILD / "index.json"
        out.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写入 {out.relative_to(ROOT)}（{out.stat().st_size / 1e6:.1f} MB）")
    else:
        print("\n加 --write 写入 build/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
