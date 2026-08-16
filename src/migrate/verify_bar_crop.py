#!/usr/bin/env python3
# coding: utf-8
"""全量核验：裁掉顶部 N 行之后，烧录叠加层有没有残留。

为什么要全量
------------
A 组三族的画面顶部有**两种**采集时烧录的叠加：

    main         整行黑条  ``2026-07-14 11:20:05.176439 | epoch_ns=…``（多相机同步）
    wrist_left   约 1/3 宽 ``mono_ms=4093705.028``（单调时钟）
    main_right / wrist_right   无

`mono_ms` 的数字位数会变（4093705 / 33622 / 1092506），宽度随之变；高度理应由字号
固定，但那是推断。抽样看过 12 个视角没问题，**全量才能排除例外**。

判据
----
前面试过四种判据，三种被「画面顶部本身很暗」骗过（深色天花板、阴影），
第四种（同一视频内取 3 帧比恒定性）同样失败 ——
**同一段视频里深色背景本来就基本不变**，于是被当成合成叠加。

有效的判据是**跨集恒定性**：

    每一集各取 1 帧 → 全族汇总 → 逐像素 std < 4 且 mean < 15 → 合成叠加背景

集与集之间场景、光照、人物全都不同，唯一逐像素不变的只能是采集软件画上去的东西。

**叠加属于「族 + 视角」这个组合，不属于单个视频** —— 它由采集软件用固定字号
画在固定位置，只有数字位数会让宽度变化。所以「全量」的正确含义是
*每一集都参与构建这张恒定性图*，而不是逐个视频各判一次。

输出
----
逐 (族, 视角) 报告叠加层下沿的分布，以及**裁 N 行后是否仍有残留**。
任何一个视频报出残留，就说明 N 不够。
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CROP_ROWS = 44          # 待验证的裁剪行数
SCAN_ROWS = 100         # 只看顶部这么多行
WORKERS = 8


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def strip(path: Path, when: float) -> np.ndarray | None:
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{when:.2f}", "-i", str(path),
                          "-frames:v", "1", "-vf", f"crop=iw:{SCAN_ROWS}:0:0",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    try:
        return np.asarray(Image.open(io.BytesIO(out)).convert("L")).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


def one_frame(path: Path) -> np.ndarray | None:
    total = duration(path)
    return strip(path, total * 0.45) if total > 0 else None


def overlay_geometry(frames: list[np.ndarray]) -> tuple[int, np.ndarray]:
    """跨集汇总，返回 (叠加层下沿行数, 每行叠加像素占比)。"""
    stack = np.stack(frames)
    constant = (stack.std(0) < 4) & (stack.mean(0) < 15)
    share = constant.mean(axis=1)
    rows = [i for i in range(SCAN_ROWS) if share[i] > 0.10]
    return ((max(rows) + 1) if rows else 0), share


def main() -> int:
    families = json.loads((ROOT / "data" / "families.json").read_text(encoding="utf-8"))["families"]
    active = [f for f, v in families.items() if v.get("status") != "excluded"]

    print(f"判据：每集各取 1 帧汇总，跨集逐像素恒定且暗 → 合成叠加。裁剪 = {CROP_ROWS} 行\n")
    print(f"{'族/视角':<30}{'集数':>5}{'叠加行段':>12}{'下沿':>6}{'占宽':>7}  结论")
    print("-" * 84)
    worst = 0
    failures: list[str] = []

    for fam in sorted(active):
        meta = json.loads((RAW / fam / "meta.json").read_text(encoding="utf-8"))
        for logical, physical in meta["views"].items():
            phys = physical.replace("observation.images.", "")
            videos = sorted((RAW / fam / "videos" / f"observation.images.{phys}").rglob("*.mp4"))
            if not videos:
                continue
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                frames = [f for f in pool.map(one_frame, videos) if f is not None]
            if len(frames) < 3:
                print(f"  {fam + '/' + logical:<28}{len(frames):>5}  样本不足")
                continue
            bottom, share = overlay_geometry(frames)
            worst = max(worst, bottom)
            seg = f"{min(i for i in range(SCAN_ROWS) if share[i] > 0.10)}–{bottom - 1}" if bottom else "无"
            wide = f"{share[:bottom].max():.0%}" if bottom else "—"
            ok = "✓ 44 覆盖" if bottom <= CROP_ROWS else f"✗ 需要 {bottom}"
            if bottom > CROP_ROWS:
                failures.append(f"{fam}/{logical}: 下沿 {bottom}")
            print(f"  {fam + '/' + logical:<28}{len(frames):>5}{seg:>12}{bottom:>6}{wide:>7}  {ok}")

    print("-" * 84)
    print(f"叠加层下沿最大值 = {worst}（裁剪行数 {CROP_ROWS}）")
    if failures:
        print("\n有残留：")
        for f in failures:
            print("  -", f)
        return 1
    print("裁 44 行后全部干净，无一例外")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
