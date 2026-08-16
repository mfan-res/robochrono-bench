#!/usr/bin/env python3
# coding: utf-8
"""把 ``data/raw`` 规范化成 ``data/source`` —— 下游只面对一种格式。

做什么
------
一次编码，同时完成四件事：

    裁掉烧录叠加   A 组 crop=<w>:<h-44>:0:44      D-16 全量核验：叠加下沿 ≤43
    对齐像素预算   等比缩到 ≈307,200 px            D-17：不对齐宽高比，只对齐信息量
    统一编码       h264 / yuv420p / crf 18
    全帧内         -g 1                            让后续切片能 -c copy 无损

**不做**：不统一 fps（D-14：抽帧与原生 fps 无关，重采样只会丢帧或造帧）；
不拼接多视角（拼哪几个、什么顺序是出题的选择，不是采集数据的属性）；
不改状态数据（原样复制）。

为什么规范化要落地成一层
------------------------
``data/raw`` 在四个维度上异构（视角命名、分辨率、fps、叠加）。不落这一层，
**每个下游都得知道这些差异** —— 出题、验题、盲基线、将来重标。
落了这一层，异构只在这里处理一次。

为什么用全帧内
--------------
视频的裁剪缩放没有无损办法，每转一次多一代有损压缩。全帧内让**之后所有切片
都能 ``-c copy``**，于是全流程只有这一代编码 —— 和「不落规范化层、切片时一趟做完」
的编码代数相同，但下游干净得多。

实测（wash 单集）：普通 GOP 9.5 MB 但切片必须重编码；每 0.4 秒关键帧 15.1 MB
且切片边界误差 0.080s；全帧内 40.1 MB，切片边界误差 **0.000s**。

``main_right`` 不转
-------------------
只有 gripper 三族有，七个任务都没用。**不转，但在 manifest 里标出来** ——
raw 还在，将来要用时单独补即可。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SOURCE = ROOT / "data" / "source"

BAR_ROWS = 44                 # D-16：全量核验的叠加下沿 +1
TARGET_PIXELS = 640 * 480     # D-17：B 组原生像素数，作为统一预算
CRF = 18
SKIP_VIEWS = {"main_right"}   # 七个任务都没用，不转但记录
WORKERS = 6


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-show_entries", "format=duration,size", "-of", "json", str(path)],
        capture_output=True, text=True).stdout
    return json.loads(out)


def plan_transform(native: tuple[int, int], burned: bool) -> tuple[str, tuple[int, int]]:
    """给定原生尺寸，算出 ffmpeg 的 -vf 与输出尺寸。"""
    width, height = native
    usable_h = height - BAR_ROWS if burned else height
    factor = math.sqrt(TARGET_PIXELS / (width * usable_h))
    out_w = max(2, round(width * factor) // 2 * 2)
    out_h = max(2, round(usable_h * factor) // 2 * 2)
    steps = []
    if burned:
        steps.append(f"crop={width}:{usable_h}:0:{BAR_ROWS}")
    if (out_w, out_h) != (width, usable_h):
        steps.append(f"scale={out_w}:{out_h}")
    return (",".join(steps) or "null"), (out_w, out_h)


def convert(src: Path, dst: Path, vf: str) -> tuple[bool, int]:
    """返回 (是否新转, 输出字节数)。已存在则跳过 —— 支持断点续跑。"""
    if dst.exists() and dst.stat().st_size > 0:
        return False, dst.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 临时名必须保留 .mp4 —— ffmpeg 从扩展名推断容器，".part" 会让它报错退出
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.part.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if vf != "null":
        cmd += ["-vf", vf]
    # -g 1 = 全帧内；后续切片才能 -c copy 无损且边界精确
    cmd += ["-c:v", "libx264", "-crf", str(CRF), "-preset", "veryfast",
            "-g", "1", "-pix_fmt", "yuv420p", "-an", str(tmp)]
    subprocess.run(cmd, check=True)
    os.replace(tmp, dst)
    return True, dst.stat().st_size


def normalize_family(canon: str, meta: dict) -> dict:
    native = tuple(meta["native"])
    burned = bool(meta["timestamp_burned_in"])
    vf, out_size = plan_transform(native, burned)

    jobs: list[tuple[Path, Path]] = []
    skipped_views: list[str] = []
    for logical, physical in meta["views"].items():
        if logical in SKIP_VIEWS:
            skipped_views.append(logical)
            continue
        phys = physical.replace("observation.images.", "")
        for src in sorted((RAW / canon / "videos" / f"observation.images.{phys}").rglob("*.mp4")):
            jobs.append((src, SOURCE / canon / src.stem / f"{logical}.mp4"))

    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda ab: convert(ab[0], ab[1], vf), jobs))
    total = sum(size for _, size in results)
    fresh = sum(1 for new, _ in results if new)

    # 状态数据原样复制，不改打包方式（轨迹任务已搁置，没有消费者）
    states = 0
    for pqf in sorted((RAW / canon / "data").rglob("*.parquet")):
        dst = SOURCE / canon / "_states" / pqf.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(pqf, dst)
        states += 1

    raw_bytes = sum(f.stat().st_size
                    for logical, physical in meta["views"].items() if logical not in SKIP_VIEWS
                    for f in (RAW / canon / "videos" /
                              f"observation.images.{physical.replace('observation.images.', '')}").rglob("*.mp4"))

    entry = {
        "native": list(native), "output": list(out_size),
        "vf": vf, "crf": CRF, "gop": 1,
        "timestamp_bar_rows_cropped": BAR_ROWS if burned else 0,
        "fps": meta.get("fps"),                       # 原样保留，不重采样
        "views": [v for v in meta["views"] if v not in SKIP_VIEWS],
        "views_skipped": skipped_views,
        "views_skipped_reason": "七个任务都没用；raw 仍在，需要时单独补转",
        "videos": len(jobs), "states_files": states,
        "raw_bytes": raw_bytes, "source_bytes": total,
        "ratio": round(total / raw_bytes, 3) if raw_bytes else None,
        "seconds": round(time.time() - started, 1),
        "states_aligned": meta.get("states_aligned", True),
    }
    (SOURCE / canon / "meta.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  {canon:<13}{str(native):>11} → {str(out_size):<11}"
          f"{len(jobs):>4} 视频  {raw_bytes/1e9:>6.2f} → {total/1e9:.2f} GB"
          f"  ×{entry['ratio']}  {entry['seconds']:.0f}s  新转 {fresh}", flush=True)
    return entry


def main() -> int:
    only = sys.argv[1:] or None
    families = json.loads((ROOT / "data" / "families.json").read_text(encoding="utf-8"))["families"]
    active = [f for f, v in families.items() if v.get("status") != "excluded"]
    targets = [f for f in active if not only or f in only]

    print(f"裁叠加 {BAR_ROWS} 行（仅 A 组）· 等比缩到 ≈{TARGET_PIXELS:,} px · crf {CRF} · 全帧内\n")
    print(f"  {'族':<13}{'原生':>11}   {'输出':<11}{'视频':>4}  {'raw → source':^18}"
          f"  {'比':^6}{'耗时':>6}")
    print("-" * 96)
    total_raw = total_out = 0
    for fam in targets:
        meta = json.loads((RAW / fam / "meta.json").read_text(encoding="utf-8"))
        entry = normalize_family(fam, meta)
        total_raw += entry["raw_bytes"]
        total_out += entry["source_bytes"]
    print("-" * 96)
    print(f"  合计  {total_raw/1e9:.2f} → {total_out/1e9:.2f} GB"
          f"  ×{total_out/total_raw:.2f}" if total_raw else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
