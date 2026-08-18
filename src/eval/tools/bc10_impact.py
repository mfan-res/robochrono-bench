#!/usr/bin/env python3
# coding: utf-8
"""统计 BC-10 的影响面。

BC-10：left_right 与 image_in_video 的选项是图片，text 字段为 null。
冻结脚本用 str(option.get("text", "")) 取文本，None 变成字符串 "None"，
归一化后是 "none"，于是任何含 "none" 的回答都会命中第一个选项。

本工具离线重放已有结果，不调模型、不占卡。报告三个层次：

  susceptible   有多少条走到了「选项文本匹配」这一级 —— 只有这些才可能被影响。
                前两级（整串是字母、正则抓到孤立大写字母）不受影响。
  changed       打开修复后，预测选项发生变化的条数。
  correctness   其中对错发生翻转的条数，及方向。

用法：
    bc10_impact.py <结果文件...>          # 支持 .json（冻结格式）与 .jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))

from robochrono import parsing  # noqa: E402
from robochrono.tasks.base import load_items  # noqa: E402

QA = REPO / "eval/datasets/QA"
AFFECTED_RUNS = {"left_right", "image_in_video"}
QA_FILE = {
    "left_right": QA / "understanding/stack_cubes/left_right_vqa.json",
    "image_in_video": QA / "understanding/stack_cubes/image_in_video_vqa.json",
}


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    yield from data.get("results", [])


def resolution_level(text: str, options: dict[str, str]) -> str:
    """判断 extract_choice 会走到哪一级。只有第三级会受 BC-10 影响。"""
    import re

    valid = set(options)
    normalized = str(text).strip().upper()
    if normalized in valid:
        return "exact_letter"
    match = re.search(r"\b([A-Z])\b", normalized)
    if match and match.group(1) in valid:
        return "isolated_letter"
    return "text_match"


def choice_text_of(payload: Any) -> str:
    """取出真正送进 extract_choice 的那段文本。"""
    if isinstance(payload, dict):
        for key in ("choice", "answer", "option", "letter"):
            if payload.get(key):
                return str(payload[key])
        return ""
    return str(payload or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantify the BC-10 scoring bug.")
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    print(f"{'file':<34} {'run':<15} {'rows':>5} {'文本匹配':>8} {'含none':>7} "
          f"{'改变':>5} {'翻转':>10}")
    print("-" * 92)

    grand = {"rows": 0, "text_match": 0, "has_none": 0, "changed": 0, "to_correct": 0, "to_wrong": 0}

    for path in args.results:
        if not path.exists():
            print(f"{path.name:<34} 文件不存在")
            continue

        rows = list(read_rows(path))
        run = None
        for candidate in AFFECTED_RUNS:
            if candidate in path.stem or candidate in str(path):
                run = candidate
                break
        if run is None:
            print(f"{path.name:<34} {'—':<15} {len(rows):>5}   （非受影响任务，跳过）")
            continue

        items = {str(i["id"]): i for i in load_items(QA_FILE[run])}
        stats = {"text_match": 0, "has_none": 0, "changed": 0, "to_correct": 0, "to_wrong": 0}

        for row in rows:
            item = items.get(str(row.get("id")))
            output = row.get("model_output")
            if item is None or not output:
                continue

            buggy = parsing.options_from_item(item, null_text_fix=False)
            fixed = parsing.options_from_item(item, null_text_fix=True)

            old = parsing.parse_choice_answer(output, buggy)
            new = parsing.parse_choice_answer(output, fixed)

            if resolution_level(choice_text_of(old.get("parsed")) or output, buggy) == "text_match":
                stats["text_match"] += 1
            if "none" in str(output).lower():
                stats["has_none"] += 1

            if old["choice"] != new["choice"]:
                stats["changed"] += 1
                gold = str(item.get("answer") or item.get("A") or "").upper()
                was, now = old["choice"] == gold, new["choice"] == gold
                if now and not was:
                    stats["to_correct"] += 1
                elif was and not now:
                    stats["to_wrong"] += 1

        flip = f"+{stats['to_correct']}/-{stats['to_wrong']}"
        print(f"{path.name:<34} {run:<15} {len(rows):>5} {stats['text_match']:>8} "
              f"{stats['has_none']:>7} {stats['changed']:>5} {flip:>10}")

        grand["rows"] += len(rows)
        for key in stats:
            grand[key] += stats[key]

    print("-" * 92)
    print(f"{'合计':<34} {'':<15} {grand['rows']:>5} {grand['text_match']:>8} "
          f"{grand['has_none']:>7} {grand['changed']:>5} "
          f"{'+' + str(grand['to_correct']) + '/-' + str(grand['to_wrong']):>10}")
    print()
    print("文本匹配 = 走到第三级匹配的条数，只有这些可能受影响")
    print("含none   = 模型输出里出现 none 的条数")
    print("改变     = 打开修复后预测选项变化的条数")
    print("翻转     = 其中判定对错发生变化的条数（+转对 / -转错）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
