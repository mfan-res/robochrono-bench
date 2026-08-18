#!/usr/bin/env python3
# coding: utf-8
"""⑤ 验题：图选项题型的盲基线题面。

    python3 src/vqa/blind_image.py --write
    cd src/eval && python -m robochrono --datasets-root ../../build/blind_v2 \\
        run --provider <provider> --families wash --runs left_right image_in_video

盲测件 = 正式题**挖掉 `input`**，选项原样。
只给四张选项图（或四张候选帧），不给主视角 / 不给片段 ——
高出 25% 的部分就是不看题面也能拿到的分。

为什么不另写一套 harness
------------------------
一开始想单独写个脚本调模型。**那样测的是那个脚本，不是评测链路。**
改成让评测端在缺主视角 / 缺片段时「不放这部分」而不是拿 `None` 去做图片
（`choice.py` 的 `media_head_and_options` / `media_clip_and_options`，
这条防御本身也该有），盲测于是走**完全相同的代码路径**。

`build/blind_v2/` 不进 git —— 它由 `data/vqa/eval/` 机械导出，
这个脚本才是可复现的那一份。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "vqa" / "eval"
DST = ROOT / "build" / "blind_v2"

# 只有选项本身就是媒体的题型才需要这种盲测：文字选项题走 `blind.py`
# （纯文本模型 + `--by-text`），两者问的是同一个问题、手段不同。
TASKS = ("left_right", "image_in_video")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    plan: list[tuple[Path, dict, int]] = []
    for path in sorted(SRC.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = doc["items"] if isinstance(doc, dict) else doc
        if not rows or rows[0]["type"] not in TASKS:
            continue
        for row in rows:
            row["input"] = {}
        plan.append((path, doc, len(rows)))

    total = sum(n for _, _, n in plan)
    print(f"盲测件 {total} 题，{len(plan)} 个文件")
    for path, _, n in plan:
        print(f"  {path.relative_to(SRC)}  {n}")
    if not args.write:
        print("\n（未写。加 --write 生成 build/blind_v2/）")
        return 0

    if DST.exists():
        shutil.rmtree(DST)
    for path, doc, _ in plan:
        out = DST / path.relative_to(SRC)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    # 媒体路径是相对 QA 文件的（`../../assets/...`），所以盲测件目录旁边
    # 也要有一个 assets 才解析得到。软链，不占空间。
    link = DST.parent / "assets"
    if not link.exists():
        link.symlink_to(Path("..") / "data" / "vqa" / "assets")
    print(f"\n已写入 {DST.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
