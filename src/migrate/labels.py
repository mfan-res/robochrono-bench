#!/usr/bin/env python3
# coding: utf-8
"""把标注搬进 ``data/label/<规范族名>/``。

搬两样：

``segments/``
    旧 ``datasets/json/<别名>/*_segments.json`` 原样复制。**一字不改** ——
    它是人类标注，是七个任务真值的唯一来源。

``categories.txt``
    候选动作标签集。**原始 txt 文件我们没有**（`category_label_path` 指向
    生成机上的路径），但它的内容被生成器完整写进了每个 QA 文件的
    ``option_design.category_labels``，所以能还原。还原出来的顺序按原数组，
    不排序 —— 顺序可能有含义，不知道就别动。

顺带做两件校验（这两条以后应当固化成 ``check_labels``）：

1. 标注集数 vs QA 引用的 video_id 数。允许标注更多 ——
   生成时可能因缺视角视频跳过某集（tea2 就跳了 file-020），有记录。
2. 标签条数 vs ``num_category_labels``。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

OLD = Path("/mnt/public/users/wbcd/workspace/michael/benchmark/eval")
NEW = Path("/mnt/public/users/wbcd/workspace/michael/bench/data")
sys.path.insert(0, str(OLD))


def main() -> int:
    from robochrono import tasks                       # noqa: E402
    from robochrono.tasks.base import load_items       # noqa: E402

    families = json.loads((NEW / "families.json").read_text(encoding="utf-8"))["families"]
    datasets = OLD / "datasets"

    print(f"{'族':<13}{'标注':>6}{'QA视频':>8}{'标签':>6}  说明")
    print("-" * 62)
    issues: list[str] = []
    total = 0

    for canon, info in families.items():
        src = datasets / "json" / info["label_dir"]
        dst = NEW / "label" / canon
        (dst / "segments").mkdir(parents=True, exist_ok=True)

        segs = sorted(src.glob("*_segments.json"))
        for p in segs:
            shutil.copy2(p, dst / "segments" / p.name)
        total += len(segs)

        qa_json = json.loads(tasks.qa_path(datasets, info["qa"], "understanding")
                             .read_text(encoding="utf-8"))
        design = qa_json.get("option_design") or {}
        labels = design.get("category_labels") or []
        # 顺序原样，不排序 —— 不知道有没有含义就别动
        (dst / "categories.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

        vids = {str(i.get("video_id"))
                for i in load_items(tasks.qa_path(datasets, info["qa"], "understanding"))}

        note = ""
        if len(segs) < len(vids):
            note = "✗ 标注少于 QA —— 不该发生"
            issues.append(f"{canon}: 标注 {len(segs)} < QA 视频 {len(vids)}")
        elif len(segs) > len(vids):
            skipped = qa_json.get("num_missing_media_skipped")
            note = f"标注多 {len(segs)-len(vids)} 集（生成时跳过，记录 num_missing_media_skipped={skipped}）"
        if labels and design.get("num_category_labels") not in (None, len(labels)):
            issues.append(f"{canon}: 标签数 {len(labels)} ≠ 声明 {design.get('num_category_labels')}")
            note += "  ✗ 标签数与声明不符"

        print(f"{canon:<13}{len(segs):>6}{len(vids):>8}{len(labels):>6}  {note}")

    print("-" * 62)
    print(f"{'合计':<13}{total:>6}")
    if issues:
        print("\n问题：")
        for i in issues:
            print("  -", i)
        return 1
    print("\n标注与类别标签已就位，校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
