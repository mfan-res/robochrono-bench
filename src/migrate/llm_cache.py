#!/usr/bin/env python3
# coding: utf-8
"""把 LLM 干扰项缓存搬进 ``data/llm_cache/<族>/<pipeline>.json``。

为什么这东西必须单独保存
------------------------
出题过程本身是**完全确定性**的（选项打乱用 ``md5(item_id|text)`` 排序，
无 ``random``，seed 固定）。**唯一的非确定性来源就是 LLM 生成的干扰项。**
换句话说：丢了这份缓存，重跑生成器会得到另一套题。

它和原始录像、人类标注一样属于「不可再生」，所以放在 ``data/vqa/`` 外面 ——
后者的定义是「删了能重建」。

四个族的缓存不存在，要反建
--------------------------
实测只有 gemini-3.5-flash 那批（gift_inhand / pen_inbox / tea / wash）
留下了 ``llm_distractors.json``；glm-5.2 那批
（airpods / express / stack_cubes / tea2）**没有**。

但干扰项本身还在 —— 它们就躺在 QA 的选项里，带着
``distractor_type == "generated_wrong_label"`` 标记。按正确答案分组抽出来，
就能还原出同样结构的缓存。

**反建出来的和原始缓存不等价，必须标清楚：**

原始缓存每个标签有 ``llm_distractors_per_label``（=6）条候选，出题时只抽 2 条用。
反建只能拿到**被用过的那些**。所以它够用来复现现有题目，
**不够用来重新抽样** —— 换个 seed 重新出题，候选池就少了。

两条流水线各有一份缓存
----------------------
``time_understanding`` 与 ``planning`` 是两个生成器，各自调 LLM、各自缓存。
实测两份内容不同，所以**不合并** —— 没验证等价的东西不合并，这是原则。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

OLD = Path("/mnt/public/users/wbcd/workspace/michael/benchmark/eval")
NEW = Path("/mnt/public/users/wbcd/workspace/michael/bench/data")
sys.path.insert(0, str(OLD))

# 哪些评测任务出自哪条生成流水线
PIPELINES = {"time_understanding": ["understanding"], "planning": ["planning", "planning_2"]}


def reconstruct(datasets: Path, qa_name: str, runs: list[str]) -> dict[str, Any]:
    """从 QA 的选项里反建缓存。"""
    from robochrono import tasks
    from robochrono.tasks.base import load_items

    pool: dict[str, set[str]] = defaultdict(set)
    used = 0
    for run in runs:
        try:
            path = tasks.qa_path(datasets, qa_name, run)
        except ValueError:
            continue
        if not path.exists():
            continue
        for item in load_items(path):
            label = item.get("answer_text")
            for option in item.get("options") or []:
                if option.get("distractor_type") == "generated_wrong_label" and option.get("text"):
                    pool[str(label)].add(str(option["text"]))
                    used += 1
    return {
        "task_category_distractors": {k: sorted(v) for k, v in sorted(pool.items())},
        "_reconstructed": True,
        "_reconstructed_from": f"QA 选项中 distractor_type=generated_wrong_label（{used} 次引用）",
        "_caveat": "只含被实际用过的干扰项；原始缓存每标签 6 条候选，这里少于 6 条。"
                   "够复现现有题目，不够重新抽样。",
    }


def main() -> int:
    families = json.loads((NEW / "families.json").read_text(encoding="utf-8"))["families"]
    datasets = OLD / "datasets"

    # 先把盘上现成的缓存按 (族, 流水线) 索引起来
    found: dict[tuple[str, str], Path] = {}
    for path in datasets.rglob("llm_distractors.json"):
        parts = path.parts
        pipeline = path.parent.name                      # time_understanding / planning
        family = parts[parts.index("QA") + 2]            # QA/<group>/<family>/...
        found[(family, pipeline)] = path

    print(f"{'族':<13}{'流水线':<20}{'来源':<10}{'标签':>5}{'干扰项':>7}  说明")
    print("-" * 74)
    reconstructed = 0

    for canon, info in families.items():
        out_dir = NEW / "llm_cache" / canon
        out_dir.mkdir(parents=True, exist_ok=True)
        for pipeline, runs in PIPELINES.items():
            src = found.get((info["qa"], pipeline))
            if src is not None:
                data = json.loads(src.read_text(encoding="utf-8"))
                origin, note = "原始", ""
            else:
                data = reconstruct(datasets, info["qa"], runs)
                origin, note = "**反建**", "原始缓存缺失，从 QA 选项还原"
                reconstructed += 1
            table = data.get("task_category_distractors") or {}
            (out_dir / f"{pipeline}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{canon:<13}{pipeline:<20}{origin:<10}{len(table):>5}"
                  f"{sum(len(v) for v in table.values()):>7}  {note}")

    print("-" * 74)
    print(f"16 份缓存已写入 data/llm_cache/，其中 {reconstructed} 份是反建的（文件内标了 _reconstructed）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
