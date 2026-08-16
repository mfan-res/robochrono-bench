#!/usr/bin/env python3
# coding: utf-8
"""还原 stack_cubes 的人工标注（P-03）。

现状
----
`data/label/stack_cubes/` 存的**不是人工标注**，是出题器 `window_for_segment`
（`legacy_pickplace` 模式）的输出被回写了回来：

    人工      0.00–25.70 pick_red │ 25.70–33.05 place_red │ …   4 段，零缺口
    存储     10.00–27.70 pick_red │ 26.70–32.05 move_red（合成）│
             31.05–35.05 place_red │ …                           6 段，200 处重叠

三处偏差都能精确反解：``pick`` 起点 +10.0（`pick_before_window`）、
所有段终点 +2.0（`after_window`）、``place`` 起点 = `end − 2.0`；
``move`` 是凭空合成的（id 形如 `file-000-1_file-000-2_move`，且无 `original_*`）。

为什么必须还原
--------------
不还原就带着两个错误进 ④：

1. **偏移会叠加。** 新出题器读 label 时会把 `pick_start_offset` 的结果
   当成人工真值，再套一次窗口计算。
2. **`place` 越界 2 秒是错的标注。** `place_red` 的存储终点 35.05s
   落在人标的 `pick_yellow`（33.05 起）范围内 —— 那 2 秒的内容被标成了错的动作。

还原不等于丢掉 `move`
---------------------
`move` 是有意义的动作，团队决定保留 —— 但它应当在 ④ 出题阶段**显式合成**：
窗口规则写进 recipe（可讨论、可改），产物标 `synthetic: true`，不伪装成人工标注。
**还原是把「要不要 move、怎么定义 move」从既成事实变回一个可讨论的选择。**

做法
----
从 `metadata.original_start/original_end` 直接取值 —— 那就是人工标注本身，
不需要反解公式。**写到新目录**，不动现有文件（沿用 D-21 的先例）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LABEL = ROOT / "data" / "label" / "stack_cubes"
SOURCE = ROOT / "data" / "source" / "stack_cubes"
OUT = LABEL / "segments.restored"


def video_meta(episode: str) -> tuple[float, int]:
    path = SOURCE / episode / "main.mp4"
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-show_entries", "format=duration",
         "-of", "json", str(path)], capture_output=True, text=True).stdout
    data = json.loads(out)
    return float(data["format"]["duration"]), int(data["streams"][0]["nb_read_frames"])


def main() -> int:
    apply = "--apply" in sys.argv
    fps = json.loads((ROOT / "data" / "raw" / "stack_cubes" / "meta.json")
                     .read_text(encoding="utf-8"))["fps"]
    subtasks = json.loads((LABEL / "subtasks.json").read_text(encoding="utf-8"))["subtasks"]
    keep = {s["id"] for s in subtasks if not s["id"].startswith("move_")}

    print(f"fps={fps}  保留的 subtask：{sorted(keep)}")
    print(f"丢弃（出题器合成）：{sorted({s['id'] for s in subtasks} - keep)}\n")
    print(f"{'集':<10}{'原 6 段':>8}{'还原':>6}{'覆盖率':>9}{'空隙':>7}{'重叠':>7}")
    print("-" * 50)

    totals = {"in": 0, "out": 0, "gap": 0, "overlap": 0}
    for path in sorted((LABEL / "segments").glob("*_segments.json")):
        episode = path.stem.replace("_segments", "")
        doc = json.loads(path.read_text(encoding="utf-8"))
        human = [s for s in doc["segments"]
                 if (s.get("metadata") or {}).get("original_start") is not None]
        totals["in"] += len(doc["segments"])

        rows: list[dict[str, Any]] = []
        for index, seg in enumerate(sorted(human, key=lambda s: s["metadata"]["original_start"]), 1):
            start = float(seg["metadata"]["original_start"])
            end = float(seg["metadata"]["original_end"])
            # 帧号才是权威；秒是派生量。四舍五入回帧，再按 core 的公式重算秒。
            start_frame = int(round(start * fps))
            end_frame = int(round(end * fps)) - 1        # end 是半开区间 → 闭区间
            rows.append({
                "id": f"{episode}-{index}",
                "subtask": seg["subtask"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start": round(start_frame / fps, 3),
                "end": round((end_frame + 1) / fps, 3),
            })

        duration, frames = video_meta(episode)
        gaps = sum(1 for a, b in zip(rows, rows[1:]) if b["start_frame"] > a["end_frame"] + 1)
        overlaps = sum(1 for a, b in zip(rows, rows[1:]) if b["start_frame"] < a["end_frame"])
        coverage = (rows[-1]["end"] - rows[0]["start"]) / duration if duration else 0
        totals["out"] += len(rows)
        totals["gap"] += gaps
        totals["overlap"] += overlaps

        if apply:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / path.name).write_text(json.dumps({
                "source": {
                    "video": f"stack_cubes/{episode}/main.mp4",
                    "fps": fps, "total_frames": frames,
                    "tool_version": "restored-from-metadata/1",
                    "categories_sha256": "",
                    "episode_bounds": None,
                    "_restored": "由 metadata.original_start/original_end 还原；"
                                 "出题器合成的 move 段已丢弃（将在 ④ 显式重建）",
                },
                "segments": rows,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if path.name.endswith(("000_segments.json", "001_segments.json")):
            print(f"{episode:<10}{len(doc['segments']):>8}{len(rows):>6}"
                  f"{coverage:>9.0%}{gaps:>7}{overlaps:>7}")

    print("-" * 50)
    print(f"{'合计':<10}{totals['in']:>8}{totals['out']:>6}"
          f"{'':>9}{totals['gap']:>7}{totals['overlap']:>7}")
    if not apply:
        print("\n这是预演。加 --apply 才写盘（写到 segments.restored/，不动现有文件）。")
    else:
        print(f"\n已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
