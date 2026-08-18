#!/usr/bin/env python3
# coding: utf-8
"""把「一个文件装多集」的视频切成一集一文件。**当前不要运行 —— 见文末。**

⚠ **本脚本的核心假设是错的**：它假设三个视角共享同一套文件结构。
实测 tea2 不成立 —— LeRobot 按【视角各自】算字节预算打包，
手腕视角压缩率不同、装下的集数也不同，从第 2 集起 wrist_left 就与 main
错开一个文件（D-42）。正确做法是按各视角自己的 `file_index` + 时间戳分别切。

**没有改写，因为同事正在重切并重标 tea2**，本族已 `status: parked`。
保留此文件是为了把「测出来的东西」留在代码旁边：切法本身（`-c copy`
逐帧比对无损）是验证过的，错的只是「视角共享文件结构」这个前提。

    python3 src/migrate/split_packed_episodes.py --family tea2 --dry-run
    python3 src/migrate/split_packed_episodes.py --family tea2 --limit 1   # 先切一个
    python3 src/migrate/split_packed_episodes.py --family tea2

为什么会有多集打包
------------------
LeRobot v3 的导出参数 `video_files_size_in_mb`：往一个视频文件里追加 episode，
写到这么大就换下一个。六个族设成 0.001–1 MB（比任何一集都小，等于不打包），
**tea2 设成 200 MB**，于是每个文件装了 2–3 集。

不是任务长度、不是 fps，就是这一个配置值（D-41）。

为什么在 `data/source` 层切，而不是出题时裁
------------------------------------------
D-20 原本定的是「不改数据，出题时传 start/end 裁到第 0 集」。改主意的理由：

- `data/source` 是**可再生层**，切它不碰 `raw` / `label` 这两个不可再生的
- 切完之后 tea2 与其余六族结构一致，`full_video_usable` 这个特例、
  以及出题时的裁剪分支都能删掉 —— **少一个特例，胜过多一处正确处理**
- 每集 309s → 121s，与 tea 98s / wash 71s 对齐，
  「视频时长差 12 倍」这个跨族混淆变量同时缓解

命名：第 0 集沿用原名
--------------------
`file-000` 仍是第 0 集，后续集叫 `file-000-e1` / `-e2`。

**不能给第 0 集改名** —— 段 id 由「视频名 + 起始帧」派生（D-28），
改名会让该族全部标注段的 id 失效。而第 0 集恰好从 0.0s 开始（20/20 实测），
所以帧号一个都不用动。

切法：`-c copy`，逐帧比对验证过
------------------------------
`data/source` 是全帧内的（`-g 1`，D-18），每一帧都是关键帧，
所以按任意帧切都无损且精确。实测：切出来的段与原文件对应区间
**逐帧解码 md5 完全一致**（第 0 集与中间集各验一次）。

耗时 0.2 秒/文件，总字节数不变（213 MB → 72+72+69）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vqa"))
from index import episode_bounds  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "source"


def frames_of(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def cut(src: Path, dst: Path, first: int, count: int, fps: float) -> int:
    """切 [first, first+count) 帧到 dst，返回实际帧数。`-c copy`，不重编码。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if first:
        cmd += ["-ss", f"{first / fps:.6f}"]           # 全帧内，seek 到哪就是哪
    cmd += ["-i", str(src), "-c", "copy", "-frames:v", str(count), str(dst)]
    subprocess.run(cmd, check=True)
    return frames_of(dst)


def main() -> int:
    def arg(name: str, default: str | None = None) -> str | None:
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    family = arg("--family", "tea2")
    dry = "--dry-run" in sys.argv
    limit = int(arg("--limit", "0") or 0)

    meta = json.loads((SOURCE / family / "meta.json").read_text(encoding="utf-8"))
    fps = float(meta["fps"])
    views = meta["views"]
    bounds = episode_bounds(family)

    targets = [d for d in sorted((SOURCE / family).iterdir())
               if d.is_dir() and d.name.startswith("file-")
               and len(bounds.get(d.name) or []) > 1
               and not (SOURCE / family / f"{d.name}-e1").exists()]   # 已切过就跳过
    if limit:
        targets = targets[:limit]

    print(f"{family}：{len(targets)} 个文件待切（fps {fps:g}，视角 {views}）")
    if not targets:
        print("  没有待切的 —— 要么已经切过，要么本来就是一集一文件")
        return 0

    ok = bad = 0
    for d in targets:
        spans = [(round(a * fps), round(b * fps)) for a, b in bounds[d.name]]
        total = frames_of(d / f"{views[0]}.mp4")
        if spans[-1][1] != total:
            print(f"  ✗ {d.name} 元表帧数 {spans[-1][1]} ≠ 实际 {total}，跳过")
            bad += 1
            continue
        plan = [(d.name if i == 0 else f"{d.name}-e{i}", fa, fb - fa)
                for i, (fa, fb) in enumerate(spans)]
        print(f"  {d.name}  {total} 帧 → " +
              " + ".join(f"{n}({c})" for n, _, c in plan), flush=True)
        if dry:
            continue

        # 先全部切到暂存，核对帧数无误再落位 —— 中途失败不会留下半切的目录
        stage = SOURCE / family / f".split-{d.name}"
        if stage.exists():
            shutil.rmtree(stage)
        good = True
        for name, first, count in plan:
            for view in views:
                got = cut(d / f"{view}.mp4", stage / name / f"{view}.mp4", first, count, fps)
                if got != count:
                    print(f"    ✗ {name}/{view} 切出 {got} 帧，应为 {count}")
                    good = False
        if not good:
            shutil.rmtree(stage)
            bad += 1
            continue
        for name, _, _ in plan:
            dst = SOURCE / family / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(stage / name), str(dst))
        shutil.rmtree(stage)
        ok += 1

    print(f"\n完成 {ok}，失败 {bad}")
    if ok and not dry:
        print("下一步：更新 data/label 的 source 块（备份 + corrections.json + 回归），"
              "再重建 index / vocab / plan")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
