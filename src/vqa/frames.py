#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第 3 步：抽出候选帧池，并量出「什么叫画面上分得开」。

> **编号说明**：这一步是后加的（图选项题型需要帧池），插进来时沿用了
> `plan.py` 的「第四步」，于是一段时间里**两个文件都自称第四步**。
> 现已重编号为 1–7。原本的「第三步」是 `distract.py` / `pool.py`
> 两代 LLM 干扰项生成器，自 D-38 起干扰项改为一律取自真实标签，它们已退场
> （留档见 `data/llm_cache/README.md`），编号由本步接替。

    python3 src/vqa/frames.py             # 只统计，不抽
    python3 src/vqa/frames.py --write     # 抽帧 + 写 build/frames.{json,npy}

这一步存在的理由
----------------
`left_right` / `image_in_video` 的干扰项是**图**。图的干扰项好不好，
不能靠「来自别的集」「隔了两段」这类**结构判据**保证 ——
人工复核 24 道判了 7 道「无解」，而那 7 道**在结构上全部合规**。

根因是采集方式，不是挑选的运气：同一套脚本、固定机位，
**同一个动作在任何一集里都是同一张图**。实测 18 个（族×视角）里 15 个，
「同动作·别集」的画面距离显著小于「不同动作」：

```
airpods/main       同动作·别集 26.6   不同动作 52.4   0.51  ← 近一倍
pen_inbox/main     27.5              47.3            0.58
wash/main          41.3              55.4            0.75
stack_cubes/main   38.8              41.6            0.93  ← 唯一接近的
```

于是干扰项改成三条约束一起（人定）：
  **只从同一集取** —— 跨集的差异是假的，同集之内不同动作块的差异才是真的
  **动作必须不同** —— 语义判据，来自标注，可解释
  **画面还要够远** —— 兜底，语义判据漏掉的个例（wash 有一对只差 3.4）

为什么每段抽五个相位而不是只抽中点
----------------------------------
只抽中点等于**专门取每个动作最典型的那一帧**，撞车是系统性的。
段内不同相位的画面是真的不一样，这让相邻动作块也能贡献够远的干扰项 ——
实测 stack_cubes 的产量从 26/200 回到 173/200，就是靠这个。

下限为什么只在「不同动作对」上标定
----------------------------------
第一版在**全部帧对**上取 p25，而那个分布里混了大量同动作近似对 ——
基准是脏的：一边被压低（漏掉该拦的），一边又误伤（`left_right` 的
「对侧手腕同一时刻」被错误地丢了 98 道，而实测它比典型的不同动作对
更可分，比值 1.13–1.78）。现在只在不同动作对上算，基准干净。

下限按族按视角取，不取绝对值 —— airpods 的 40 和 wash 的 40 不是一回事。
人已定过：族间不需要可比，同一族内讲得通即可。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
OUT = ROOT / "data" / "vqa" / "assets"

FRAMES_VERSION = "2"

# 段内相位。**不含 0 和 1** —— 段边界是分段的交接点，
# 那里的画面往往还是上一个（或下一个）动作的余波（P-06）。
PHASES = (0.1, 0.3, 0.5, 0.7, 0.9)

# 0.5 这个相位是「本时刻」的标准帧：正确项、left_right 的题面都用它。
ANCHOR = 0.5

# 描述子边长。32×32 灰度只回答「这两张图看起来是不是几乎一样」，
# **不是用来判语义相似的**。
DESC = 32

# 选项帧的 JPEG 质量。必须与 assets.py 的 JPEG_Q 一致 ——
# 正确图与干扰图若用不同参数抽，图像统计本身就成了线索。
JPEG_Q = 3

FLOOR_PERCENTILE = 25
WORKERS = 8


def frame_key(family: str, episode: str, view: str, frame: int) -> str:
    """与 `plan.need()` / `assets.dest_of()` 同一套命名，不可分叉。"""
    return f"{family}/{episode}/{view}/frame/{frame:06d}-{frame:06d}"


def phase_frames(start: int, end: int) -> dict[float, int]:
    """段内各相位的帧号。段太短时不同相位会落到同一帧，去重后返回 ——
    17 帧的段（族里最短）在 0.1/0.3 上就会重合。"""
    out: dict[float, int] = {}
    seen: set[int] = set()
    for p in PHASES:
        f = start + int(round((end - start) * p))
        if f in seen and p != ANCHOR:
            continue
        seen.add(f)
        out[p] = f
    out[ANCHOR] = start + int(round((end - start) * ANCHOR))   # 锚点必须在
    return out


def produce(job: tuple[str, Path, float, int]) -> tuple[str, str, list[int]]:
    """抽一帧并顺手算描述子 —— 分两趟要多跑两万次 ffmpeg。"""
    key, src, fps, frame = job
    dst = OUT / f"{key}.jpg"
    state = "已存在"
    if not (dst.exists() and dst.stat().st_size > 0
            and dst.open("rb").read(2) == b"\xff\xd8"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.part.jpg")
        try:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{frame / fps:.6f}",
                            "-i", str(src), "-frames:v", "1", "-q:v", str(JPEG_Q),
                            str(tmp)], check=True)
            # 校验魔数再落盘。D-54：`-c copy` 曾把单帧 MP4 写成 `.jpg`，
            # 产出全程不报错，直到评测端读图才炸。
            if tmp.stat().st_size == 0 or tmp.open("rb").read(2) != b"\xff\xd8":
                tmp.unlink(missing_ok=True)
                return key, "抽出来不是 JPEG", []
            os.replace(tmp, dst)
            state = "新建"
        except subprocess.CalledProcessError:
            tmp.unlink(missing_ok=True)
            return key, "ffmpeg 失败", []
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(dst), "-vf",
                        f"scale={DESC}:{DESC},format=gray", "-frames:v", "1",
                        "-f", "rawvideo", "-"], capture_output=True)
    return key, state, list(r.stdout[: DESC * DESC])


def floors(desc: numpy.ndarray, order: list[str],
           meta: dict[str, dict[str, Any]]) -> dict[str, float]:
    """每个 (族, 视角) 的下限：**只在「不同动作」的帧对上**取 p25。

    全对精确算，不抽样 —— 抽样要引入随机数，而这份产物要进 plan，
    出题必须确定。
    """
    index = {k: i for i, k in enumerate(order)}
    by_group: dict[str, list[str]] = defaultdict(list)
    for key, m in meta.items():
        by_group[f"{m['family']}/{m['view']}"].append(key)

    out: dict[str, float] = {}
    for group, keys in sorted(by_group.items()):
        keys = sorted(keys)
        if len(keys) < 2:
            continue
        m = desc[[index[k] for k in keys]].astype(numpy.float32)
        sq = (m * m).sum(1)
        d = numpy.sqrt(numpy.maximum(sq[:, None] + sq[None, :] - 2 * m @ m.T, 0)
                       / m.shape[1])
        subs = numpy.array([meta[k]["subtask"] for k in keys])
        iu = numpy.triu_indices(len(keys), k=1)
        differ = (subs[:, None] != subs[None, :])[iu]
        if differ.sum() < 10:
            continue
        out[group] = float(numpy.percentile(d[iu][differ], FLOOR_PERCENTILE))
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
                phases = phase_frames(segment["start_frame"], segment["end_frame"])
                for view in entry["views"]:
                    src = ROOT / f"data/source/{family}/{episode['episode']}/{view}.mp4"
                    if not src.exists():
                        continue
                    for phase, frame in sorted(phases.items()):
                        key = frame_key(family, episode["episode"], view, frame)
                        if key in meta:
                            continue
                        jobs.append((key, src, fps, frame))
                        meta[key] = {"family": family, "episode": episode["episode"],
                                     "view": view, "frame": frame, "phase": phase,
                                     "seg_index": i, "segment_id": segment["id"],
                                     "subtask": segment["subtask"]}

    segs = len({(m["family"], m["episode"], m["segment_id"]) for m in meta.values()})
    print(f"候选帧池 {len(jobs)} 张 = {segs} 段 × 视角 × {len(PHASES)} 相位（短段相位重合后去重）")
    if not args.write:
        print("（未写。加 --write 抽帧并写 build/frames.json + frames_desc.npy）")
        return 0

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(produce, jobs))
    tally = Counter(state for _k, state, _d in results)
    print(f"抽帧 {dict(tally)}")
    broken = [k for k, state, _d in results if state not in ("已存在", "新建")]
    short = [k for k, _s, d in results if len(d) != DESC * DESC]
    if broken or short:
        print(f"❌ 抽失败 {len(broken)}，描述子长度不对 {len(short)}，"
              f"前三：{(broken + short)[:3]}")
        return 1

    order = sorted(meta)
    lookup = {k: d for k, _s, d in results}
    desc = numpy.array([lookup[k] for k in order], dtype=numpy.uint8)
    fl = floors(desc, order, meta)

    print(f"\n下限 = 该族该视角【不同动作帧对】距离的 p{FLOOR_PERCENTILE}")
    print(f"{'族/视角':<28}{'帧数':>7}{'下限':>8}")
    counts = Counter(f"{m['family']}/{m['view']}" for m in meta.values())
    for group in sorted(fl):
        print(f"{group:<28}{counts[group]:>7}{fl[group]:8.1f}")

    head = {"version": FRAMES_VERSION, "desc_size": DESC, "phases": list(PHASES),
            "anchor": ANCHOR, "floor_percentile": FLOOR_PERCENTILE, "jpeg_q": JPEG_Q,
            "floors": fl, "counts": dict(sorted(counts.items()))}
    (BUILD / "frames.json").write_text(
        json.dumps({**head, "order": order, "frames": meta}), encoding="utf-8")
    numpy.save(BUILD / "frames_desc.npy", desc)
    # **判据进 git，像素不进。** 下限是要被 review 的决定（每族一个数字）。
    (BUILD / "frames_floors.json").write_text(
        json.dumps(head, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 build/frames.json + frames_desc.npy"
          f"（{desc.nbytes / 1e6:.0f} MB 描述子，不进 git）")
    print(f"     build/frames_floors.json（判据，进 git）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
