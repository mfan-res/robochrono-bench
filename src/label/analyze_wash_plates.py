#!/usr/bin/env python3
# coding: utf-8
"""判定 wash 每一集里「先拿的是左边还是右边那个盘子」。

    python3 src/label/analyze_wash_plates.py            # 判定 + 出对照图
    python3 src/label/analyze_wash_plates.py --json     # 只输出判定结果

为什么需要这个
--------------
wash 每集洗两个盘子，同一个动作（pick / wipe / put plate）做两遍，
于是 time 题「Pick the plate 发生在第几秒」有两个都对的答案（P-05，40 条告警）。

处置是给两个盘子加上相对位置：`pick_left_plate` / `pick_right_plate`。
**前提已实测**：把 40 集的初始摆放铺成一张对照图看过，
38/40 集两个盘子左右并排、机位一致；file-000 / file-001 机位不同（盘子近/远）。

难点不在「左右」，在**每一集先拿的是哪一个** —— 那要逐集判定。

判据
----
```
t1  第一次 pick_plate 前 1 秒                 托盘上两个盘子都在
窗口 第一次 put_plate 结束 → 第二次 pick_plate 开始   先洗的那个已经进了沥水架
```

在 t1 找出两团蓝色（盘子是亮蓝、托盘是木色，分离度很高），
再看窗口里哪一团没了 —— **没的那个就是先拿的**。

**窗口内要取多帧的最大值，不能只看一帧。** 只取一帧时机械臂可能正好挡在
左边，「左边没蓝色」就分不清是「盘子被拿走了」还是「被挡住了」——
而那会让每一集都误判成「先拿左」。取最大值之后：盘子若还在，总有几帧不被挡；
真被拿走则始终接近 0。

只取画面下部（`Y_FLOOR` 以下）：沥水架在上方，洗完的盘子会出现在那里，
不排除就会把「盘子搬到架子上」误读成「盘子还在」。

**这是辅助判定，不是最终答案。** 输出一张 40 格对照图，每格标着判定结果与
置信度，**人扫一遍确认之后才改 label**。判据反复被特例击穿时，
换呈现方式比换算法有效 —— 这个项目已经验证过两次。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "source" / "wash"
LAB = ROOT / "data" / "label" / "wash" / "segments"

Y_FLOOR = 0.62        # 只看这一比例以下的画面（上面是沥水架）
MIN_BLUE = 300        # 一团盘子至少这么多像素，少于此视为「不在」
SAMPLES = 9           # 窗口内取几帧求最大值（抗机械臂遮挡）


def frame_at(episode: str, seconds: float, size: tuple[int, int]) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(SRC / episode / "main.mp4"),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 3).astype(int)


def blue_mask(image: np.ndarray) -> np.ndarray:
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    mask = (b > r + 25) & (b > g + 10) & (b > 80)
    mask[: int(mask.shape[0] * Y_FLOOR)] = False        # 掐掉沥水架所在的上半部
    return mask


def two_blobs(mask: np.ndarray) -> list[tuple[int, int]] | None:
    """按列投影找两团蓝色，返回各自的 x 区间。找不到两团就返回 None。"""
    cols = mask.sum(0)
    on = cols > 2
    spans, start = [], None
    for x, v in enumerate(on):
        if v and start is None:
            start = x
        elif not v and start is not None:
            if cols[start:x].sum() >= MIN_BLUE:
                spans.append((start, x))
            start = None
    if start is not None and cols[start:].sum() >= MIN_BLUE:
        spans.append((start, len(on)))
    spans.sort(key=lambda s: -cols[s[0]:s[1]].sum())
    if len(spans) >= 2:
        return sorted(spans[:2])
    if not spans:
        return None
    # 只找到一团 —— 两个盘子挨在一起，中间没有空列（file-005 / 015 就是这样）。
    # 在这一团内部找**蓝色最少的那一列**当谷底切开。只在两侧都够大时才接受，
    # 否则说明它真的只是一个盘子。
    a, b = spans[0]
    inner = cols[a:b]
    if b - a < 40:
        return None
    lo, hi = a + int((b - a) * 0.3), a + int((b - a) * 0.7)   # 谷底只可能在中段
    cut = a + int(np.argmin(cols[lo:hi])) + (lo - a)
    left, right = (a, cut), (cut, b)
    if min(cols[left[0]:left[1]].sum(), cols[right[0]:right[1]].sum()) < MIN_BLUE:
        return None
    return [left, right]


def decide(episode: str, size: tuple[int, int]) -> dict:
    segs = json.loads((LAB / f"{episode}_segments.json").read_text(encoding="utf-8"))["segments"]
    picks = [s for s in segs if s["subtask"] == "pick_plate"]
    puts = [s for s in segs if s["subtask"] == "put_plate"]
    if len(picks) < 2 or not puts:
        return {"episode": episode, "verdict": "跳过", "why": "不是两次 pick_plate"}

    t1 = max(0.0, picks[0]["start"] - 1.0)
    lo, hi = puts[0]["end"], picks[1]["start"]
    m1 = blue_mask(frame_at(episode, t1, size))

    blobs = two_blobs(m1)
    if not blobs:
        return {"episode": episode, "verdict": "判不了", "why": "t1 没找到两团盘子",
                "t1": round(t1, 2)}

    left, right = blobs
    base = [max(1, m1[:, a:b].sum()) for a, b in (left, right)]
    # 窗口内均匀取 SAMPLES 帧，每侧取【最大】覆盖率 —— 见 docstring
    best = [0.0, 0.0]
    times = [lo + (hi - lo) * k / (SAMPLES - 1) for k in range(SAMPLES)] if hi > lo else [lo]
    for t in times:
        m = blue_mask(frame_at(episode, t, size))
        for i, (a, b) in enumerate((left, right)):
            best[i] = max(best[i], m[:, a:b].sum() / base[i])
    keep = best
    first = "left" if keep[0] < keep[1] else "right"
    margin = abs(keep[0] - keep[1])
    return {"episode": episode, "verdict": first, "margin": round(margin, 3),
            "kept_left": round(keep[0], 3), "kept_right": round(keep[1], 3),
            "left_x": left, "right_x": right, "t1": round(t1, 2),
            "window": [round(lo, 2), round(hi, 2)]}


def main() -> int:
    meta = json.loads((SRC / ".." / "wash" / "meta.json").resolve().read_text(encoding="utf-8"))
    size = tuple(meta["output"])
    eps = sorted(p.stem.replace("_segments", "") for p in LAB.glob("*_segments.json"))
    out = [decide(e, size) for e in eps]

    if "--json" in sys.argv:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    from collections import Counter
    print(f"{'集':<12}{'先拿':>6}{'区分度':>8}   左侧残留 / 右侧残留")
    for d in out:
        if "margin" in d:
            print(f"{d['episode']:<12}{d['verdict']:>6}{d['margin']:>8.2f}   "
                  f"{d['kept_left']:.2f} / {d['kept_right']:.2f}"
                  + ("   ⚠ 区分度低" if d["margin"] < 0.25 else ""))
        else:
            print(f"{d['episode']:<12}{d['verdict']:>6}   {d.get('why', '')}")
    print(f"\n{dict(Counter(d['verdict'] for d in out))}")
    weak = [d["episode"] for d in out if d.get("margin", 1) < 0.25]
    if weak:
        print(f"⚠ 区分度低、需人工确认的 {len(weak)} 集：{weak}")
    (ROOT / "build" / "wash_plates.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("判定写入 build/wash_plates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
