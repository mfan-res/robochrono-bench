#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第五步：按 `build/plan.json` 的素材清单，把片段切出来。

    python3 src/vqa/assets.py --dry-run     # 只看要做什么
    python3 src/vqa/assets.py --limit 5     # 先做 5 个
    python3 src/vqa/assets.py               # 全量（可断点续跑）

**这是 ④ 里唯一真正写盘的一步。** 前四步只读和算，改个参数重跑几十秒；
这一步动 ffmpeg，所以所有判断都在 [4] 做完了，这里只执行。

两类素材，两种做法
------------------
``clip``   段的片段（understanding / planning / planning_2 共用）
           `-c copy` 无损切。**不重编码** —— `data/source` 是全帧内的（`-g 1`，D-18），
           每一帧都是关键帧，按任意帧切都精确。实测逐帧解码 md5 与原片对应区间一致。

``video``  整段视频（time 用）
           **符号链接，不复制。** 内容与 `data/source` 完全相同，复制 5.6 GB 没有意义。
           先试硬链接（对读取方与真文件无异），本机 `/mnt/public` 是网络共享、
           `os.link` 返回 EPERM，于是回落到符号链接；再不行才复制。
           **用了哪种要记进清单** —— 符号链接在把 assets 目录单独拷走时会断，
           不记下来就会变成「文件莫名其妙不见了」。

去重已经在 [4] 做完
-------------------
清单里 1,391 段切片被 3,675 道题引用 —— 理解题与规划题用的**本来就是同一个片段**，
只是问法不同。实测旧的 `planning_clips` 8.9 GB 就是 `understanding_clips`
的重复编码（同分辨率同帧数同时长，像素差 0.1–0.3 的编码噪声）。

每一个产物都要核帧数
--------------------
切完立刻 `ffprobe` 核对帧数，不符就**报错并删掉**，不留下一个「看起来存在」的坏文件。
断点续跑时也按帧数判断是否已完成 —— 只看文件存在会把上次中断的半截当成完成品。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
OUT = ROOT / "data" / "vqa" / "assets"
WORKERS = 8


def frames_of(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def dest_of(item: dict[str, Any]) -> Path:
    return OUT / f"{item['key']}.mp4"


def produce(item: dict[str, Any], fps: float) -> dict[str, Any]:
    src = ROOT / item["source"]
    dst = dest_of(item)
    want = (item["end_frame"] - item["start_frame"] + 1
            if item["start_frame"] is not None else frames_of(src))

    if (dst.exists() or dst.is_symlink()) and frames_of(dst) == want:
        # ⚠ `stat()` 会跟随符号链接去统计目标大小 —— 那会把 5.6 GB 的
        # `data/source` 算进「输出」里。实占空间要用 `lstat()`。
        return {**item, "state": "已存在", "frames": want,
                "bytes": dst.lstat().st_size,
                "how": "symlink" if dst.is_symlink() else "clip"}

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.part.mp4")   # 扩展名必须留 .mp4
    how = "copy"
    try:
        if item["kind"] == "video":
            try:
                os.link(src, tmp)                                # 同盘零成本，最理想
                how = "hardlink"
            except OSError:
                try:
                    os.symlink(src.resolve(), tmp)               # 网络盘上只能这样
                    how = "symlink"
                except OSError:
                    subprocess.run(["cp", str(src), str(tmp)], check=True)
        else:
            cmd = ["ffmpeg", "-v", "error", "-y"]
            if item["start_frame"]:
                cmd += ["-ss", f"{item['start_frame'] / fps:.6f}"]
            cmd += ["-i", str(src), "-c", "copy", "-frames:v", str(want), str(tmp)]
            subprocess.run(cmd, check=True)
            how = "clip"
        got = frames_of(tmp)
        if got != want:
            tmp.unlink(missing_ok=True)
            return {**item, "state": "帧数不符", "frames": got, "want": want}
        os.replace(tmp, dst)
        return {**item, "state": "新建", "how": how, "frames": got,
                "bytes": dst.lstat().st_size}
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        return {**item, "state": "ffmpeg 失败", "error": str(exc)[:120]}


def main() -> int:
    def arg(name: str, default: str = "0") -> str:
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    plan = json.loads((BUILD / "plan.json").read_text(encoding="utf-8"))
    index = json.loads((BUILD / "index.json").read_text(encoding="utf-8"))["families"]
    media = plan["media"]
    limit = int(arg("--limit"))
    if limit:
        media = media[:limit]

    clips = sum(1 for m in media if m["kind"] == "clip")
    print(f"素材 {len(media)} 项：{clips} 段切片 + {len(media) - clips} 个全长视频")
    if "--dry-run" in sys.argv:
        for m in media[:5]:
            print(f"  {m['key']}  ←  {m['source']}"
                  + (f"  帧 {m['start_frame']}–{m['end_frame']}" if m["start_frame"] is not None else "  整段"))
        print(f"  …… 输出到 {OUT.relative_to(ROOT)}/")
        return 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        done = list(pool.map(lambda m: produce(m, index[m["family"]]["fps"]), media))

    tally = Counter(d["state"] for d in done)
    size = sum(d.get("bytes", 0) for d in done)
    ways = Counter(d.get("how") for d in done if d.get("how"))
    print(f"\n{dict(tally)}")
    print(f"实占 {size / 1e9:.2f} GB   产出方式 {dict(ways)}"
          f"（符号链接指向 data/source，本身不占空间）")

    bad = [d for d in done if d["state"] not in ("新建", "已存在")]
    if bad:
        print(f"\n❌ {len(bad)} 项失败 —— **不静默跳过**：")
        for d in bad[:5]:
            print(f"   {d['key']}  {d['state']}  {d.get('error') or f'得到 {d.get(chr(102)+chr(114)+chr(97)+chr(109)+chr(101)+chr(115))} 应为 {d.get(chr(119)+chr(97)+chr(110)+chr(116))}'}")
        return 1

    # 清单里没有了但盘上还在的 —— 上一版计划的残留。**删掉并报告**，
    # 留着会让「assets 目录里有什么」与「题目引用什么」悄悄分叉。
    want = {f"{m['key']}.mp4" for m in plan["media"]}
    orphan = [p for p in OUT.rglob("*.mp4") if str(p.relative_to(OUT)) not in want]
    for p in orphan:
        p.unlink()
    if orphan:
        print(f"清掉 {len(orphan)} 个孤儿素材（上一版计划的残留）")

    (BUILD / "assets.json").write_text(json.dumps(
        {"assets": [{k: d[k] for k in ("key", "kind", "frames", "bytes", "state")
                     if k in d} for d in done]},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"清单写入 build/assets.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
