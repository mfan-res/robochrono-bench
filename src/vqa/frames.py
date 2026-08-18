#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第四步（新增）：抽出候选帧池，并量出每个族「有多少视觉余量」。

    python3 src/vqa/frames.py             # 只统计，不写
    python3 src/vqa/frames.py --write     # 抽帧 + 写 build/frames.json

为什么要单独一步，而且必须排在 plan 之前
----------------------------------------
`left_right` / `image_in_video` 的干扰项是**图**。图的干扰项好不好，
不能靠「来自别的集」「隔了两段」这类**结构判据**来保证 ——
人工复核（T1-A / T2-A 的第二轮）24 道里判了 7 道「无解」，
而那 7 道在结构上全都合规。

结构判据预设了「不同集看起来不一样」。这个前提在两个族上不成立：

```
gift_inhand   30 集是同一套脚本、固定机位 —— 别集的帧和同集的一样难分
airpods       腕部相机全是特写，两两画面差中位 40.7（wash 是 54.1）
```

所以要**直接量画面差**，而量画面差就得先有帧。plan 决定用哪些帧、
assets 才去抽 —— 那个顺序下 plan 拿不到像素。于是把「抽候选池」提前：

    index → vocab → distract → **frames** → plan → assets → compose → pack

候选池 = 每段中点 × 每个视角，正好就是 assets 后来要抽的那批
（1,390 段 × 3 视角 = 4,170），所以**这一步不增加任何抽帧量**，
只是把它提前。文件名与 `assets.py` 完全一致，assets 跑到时全是「已存在」。

下限为什么按族取分位数，而不是取一个绝对值
------------------------------------------
airpods 的 40 和 wash 的 40 不是一回事。用全局阈值等于按族施加不同难度，
而人已经定过：**族间不需要可比，但同一族内要讲得通**。
取该族该视角自身分布的 p25，含义是「别挑这个族里最像的那四分之一」——
它不声称能复现人工判定（实测复现不了 L3），只保证**近似重复不会被选中**。

代价实测（重挑干扰项，而不是丢题）：left_right 96%、image_in_video 100%
仍可出题。丢的那 4% 全是 `left_right` 的「对侧手腕同一时刻」——
那条是题型核心，不可替换，过不了下限就只能不出这道题。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
OUT = ROOT / "data" / "vqa" / "assets"

FRAMES_VERSION = "1"

# 描述子边长。32×32 灰度足以判「是不是近似重复」，
# 又小到可以把 4,170×4,170 的距离矩阵直接算出来。
# **不是用来判语义相似的** —— 它只回答「这两张图看起来是不是几乎一样」。
DESC = 32

# 选项帧的 JPEG 质量。必须与 assets.py 的 JPEG_Q 一致 ——
# 正确图与干扰图若用不同参数抽，图像统计本身就成了线索。
JPEG_Q = 3

# 下限取该族该视角自身距离分布的这个分位数。
FLOOR_PERCENTILE = 25

WORKERS = 8


def frame_key(family: str, episode: str, view: str, frame: int) -> str:
    """与 `plan.need()` / `assets.dest_of()` 同一套命名，不可分叉。"""
    return f"{family}/{episode}/{view}/frame/{frame:06d}-{frame:06d}"


def extract(job: tuple[str, Path, float, int]) -> tuple[str, str]:
    key, src, fps, frame = job
    dst = OUT / f"{key}.jpg"
    if dst.exists() and dst.stat().st_size > 0 and dst.open("rb").read(2) == b"\xff\xd8":
        return key, "已存在"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.part.jpg")
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{frame / fps:.6f}",
                        "-i", str(src), "-frames:v", "1", "-q:v", str(JPEG_Q), str(tmp)],
                       check=True)
        # 校验魔数再落盘。D-54：`-c copy` 曾把单帧 MP4 写成 `.jpg`，
        # 产出全程不报错，直到评测端读图才炸。
        if tmp.stat().st_size == 0 or tmp.open("rb").read(2) != b"\xff\xd8":
            tmp.unlink(missing_ok=True)
            return key, "抽出来不是 JPEG"
        os.replace(tmp, dst)
        return key, "新建"
    except subprocess.CalledProcessError:
        tmp.unlink(missing_ok=True)
        return key, "ffmpeg 失败"


def describe(key: str) -> tuple[str, list[int]]:
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(OUT / f"{key}.jpg"), "-vf",
                        f"scale={DESC}:{DESC},format=gray", "-frames:v", "1",
                        "-f", "rawvideo", "-"], capture_output=True)
    return key, list(r.stdout[: DESC * DESC])


def floors(desc: dict[str, list[int]], keys_by_group: dict[str, list[str]]) -> dict[str, float]:
    """每个 (族, 视角) 的距离分位数。**全对算，不抽样** —— 抽样要引入随机数，
    而这份产物要进 plan，出题必须确定。最大的一组 517 帧 = 13 万对，秒级。"""
    out = {}
    for group, keys in sorted(keys_by_group.items()):
        if len(keys) < 2:
            continue
        m = numpy.array([desc[k] for k in sorted(keys)], dtype=numpy.float32)
        # RMS 距离矩阵的上三角
        sq = (m * m).sum(1)
        d2 = numpy.maximum(sq[:, None] + sq[None, :] - 2 * m @ m.T, 0) / m.shape[1]
        iu = numpy.triu_indices(len(keys), k=1)
        out[group] = float(numpy.percentile(numpy.sqrt(d2[iu]), FLOOR_PERCENTILE))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    index = json.loads((BUILD / "index.json").read_text(encoding="utf-8"))["families"]

    jobs: list[tuple[str, Path, float, int]] = []
    meta: dict[str, dict[str, Any]] = {}
    for family in sorted(index):
        entry = index[family]
        fps = entry["fps"]
        for episode in entry["episodes"]:
            for i, segment in enumerate(episode["segments"]):
                frame = (segment["start_frame"] + segment["end_frame"]) // 2
                for view in entry["views"]:
                    src = ROOT / f"data/source/{family}/{episode['episode']}/{view}.mp4"
                    if not src.exists():
                        continue
                    key = frame_key(family, episode["episode"], view, frame)
                    if key in meta:
                        continue
                    jobs.append((key, src, fps, frame))
                    meta[key] = {"family": family, "episode": episode["episode"],
                                 "view": view, "frame": frame, "seg_index": i,
                                 "segment_id": segment["id"]}
    print(f"候选帧池 {len(jobs)} 张（{len({(m['family'], m['episode'], m['frame']) for m in meta.values()})} 段 "
          f"× 各自的视角数）")
    if not args.write:
        print("（未写。加 --write 抽帧并写 build/frames.json）")
        return 0

    with ThreadPoolExecutor(WORKERS) as pool:
        states = dict(pool.map(extract, jobs))
    from collections import Counter
    tally = Counter(states.values())
    print(f"抽帧 {dict(tally)}")
    bad = [k for k, v in states.items() if v not in ("已存在", "新建")]
    if bad:
        print(f"❌ {len(bad)} 张没抽出来，前三：{bad[:3]}")
        return 1

    with ThreadPoolExecutor(WORKERS) as pool:
        desc = dict(pool.map(describe, sorted(meta)))
    short = [k for k, v in desc.items() if len(v) != DESC * DESC]
    if short:
        print(f"❌ {len(short)} 个描述子长度不对，前三：{short[:3]}")
        return 1

    by_group: dict[str, list[str]] = {}
    for key, m in meta.items():
        by_group.setdefault(f"{m['family']}/{m['view']}", []).append(key)
    fl = floors(desc, by_group)

    print(f"\n每个族每个视角的画面差 p{FLOOR_PERCENTILE}（下限）")
    print(f"{'族/视角':<28}{'帧数':>6}{'下限':>8}")
    for group in sorted(fl):
        print(f"{group:<28}{len(by_group[group]):>6}{fl[group]:8.1f}")

    head = {"version": FRAMES_VERSION, "desc_size": DESC,
            "floor_percentile": FLOOR_PERCENTILE, "jpeg_q": JPEG_Q, "floors": fl,
            "counts": {g: len(k) for g, k in sorted(by_group.items())}}
    payload = {**head, "frames": meta,
               "descriptors": {k: v for k, v in sorted(desc.items())}}
    (BUILD / "frames.json").write_text(json.dumps(payload), encoding="utf-8")
    # **判据进 git，像素不进。** 下限是要被 review 的决定（每族一个数字）；
    # 描述子有 19 MB 且完全可再生，进仓库只会淹没历史。
    (BUILD / "frames_floors.json").write_text(json.dumps(head, indent=1,
                                                         ensure_ascii=False),
                                              encoding="utf-8")
    print(f"\n写入 build/frames.json（{(BUILD / 'frames.json').stat().st_size / 1e6:.1f} MB，不进 git）")
    print(f"     build/frames_floors.json（判据，进 git）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
