#!/usr/bin/env python3
# coding: utf-8
"""BC-08：把 QA JSON 里的媒体路径统一规范化成本机绝对路径。

数据集里的九个 VQA JSON 有三种互不兼容的路径风格：

  1. 相对路径（planning / planning_2 / step_order / trajectory_2d / trajectory_3d）
     —— 相对各自 JSON 所在目录，只有 cwd 恰好在那里时才能解析。
  2. 绝对路径（time / understanding / left_right）
     —— 指向生成时那台机器的 /home/llm/yyywv/...，本机不存在。
  3. 相对路径 + Windows 反斜杠（image_in_video）
     —— Linux 下 Path() 不拆反斜杠，整串会被当成一个文件名。

本工具遍历整棵 JSON，对每个「看起来像媒体路径」的字符串尝试解析到本机真实
文件；解析得到的绝对路径才会写回。解析不到的一律原样保留，并在报告里分类计数。

溯源字段（source_video_paths 等指向原始 LeRobot 数据集的路径）本机没有对应
文件，会被归入 provenance 类原样保留 —— 评测不需要它们。

默认 dry-run，只报告不落盘。加 --apply 才真正写入，原文件备份为 <name>.orig。
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

MEDIA_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".jpg", ".jpeg", ".png")

# 生成时那台机器的输出根 -> 本数据集里对应的族目录名
LEGACY_PREFIXES = {
    "/home/llm/yyywv/test_vlm/workflow_outputs/time_understanding": "understanding",
    "/home/llm/yyywv/test_vlm/workflow_outputs/planning": "planning",
    "D:/yyyywv/研三/test_vlm/workflow_outputs/time_understanding": "understanding",
    "D:/yyyywv/研三/test_vlm/workflow_outputs/planning": "planning",
}


def looks_like_media(text: str) -> bool:
    return text.lower().endswith(MEDIA_SUFFIXES)


def resolve(raw: str, family_dir: Path) -> tuple[str, str]:
    """返回 (新路径, 分类)。分类为 rewritten / already_ok / provenance / unresolved。"""
    text = raw.replace("\\", "/")

    candidate: Path | None = None
    if text.startswith("/") or (len(text) > 2 and text[1] == ":"):
        for prefix, _group in LEGACY_PREFIXES.items():
            if text.startswith(prefix + "/"):
                candidate = family_dir / text[len(prefix) + 1 :]
                break
        else:
            # 未知的绝对路径：本机就存在就接受，否则视为溯源字段
            existing = Path(text)
            if existing.exists():
                return str(existing.resolve()), "already_ok"
            return raw, "provenance"
    else:
        candidate = family_dir / text

    if candidate is None:
        return raw, "unresolved"
    if candidate.exists():
        resolved = str(candidate.resolve())
        return resolved, "already_ok" if resolved == raw else "rewritten"
    return raw, "unresolved"


def walk(node: Any, family_dir: Path, stats: Counter, samples: dict[str, list[str]]) -> Any:
    if isinstance(node, dict):
        return {k: walk(v, family_dir, stats, samples) for k, v in node.items()}
    if isinstance(node, list):
        return [walk(v, family_dir, stats, samples) for v in node]
    if isinstance(node, str) and looks_like_media(node):
        new, kind = resolve(node, family_dir)
        stats[kind] += 1
        if len(samples.setdefault(kind, [])) < 3:
            samples[kind].append(node if kind != "rewritten" else f"{node}  ->  {new}")
        return new
    return node


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize media paths in RoboChrono QA JSON files.")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets",
        help="Directory containing QA/ and json/.",
    )
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry-run).")
    args = parser.parse_args()

    qa_root = args.datasets_root / "QA"
    json_files = sorted(qa_root.glob("*/*/*.json"))
    if not json_files:
        raise SystemExit(f"No QA JSON found under {qa_root}")

    grand = Counter()
    print(f"{'file':<28} {'rewritten':>10} {'already_ok':>11} {'provenance':>11} {'unresolved':>11}")
    print("-" * 74)

    for path in json_files:
        family_dir = path.parent
        data = json.loads(path.read_text(encoding="utf-8"))
        stats: Counter = Counter()
        samples: dict[str, list[str]] = {}
        new_data = walk(data, family_dir, stats, samples)
        grand.update(stats)

        print(
            f"{path.name:<28} {stats['rewritten']:>10} {stats['already_ok']:>11} "
            f"{stats['provenance']:>11} {stats['unresolved']:>11}"
        )
        for kind in ("unresolved",):
            for sample in samples.get(kind, []):
                print(f"    [{kind}] {sample}")

        if args.apply and stats["rewritten"]:
            backup = path.with_suffix(path.suffix + ".orig")
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 74)
    print(
        f"{'TOTAL':<28} {grand['rewritten']:>10} {grand['already_ok']:>11} "
        f"{grand['provenance']:>11} {grand['unresolved']:>11}"
    )
    if not args.apply:
        print("\ndry-run，未写入任何文件。确认无误后加 --apply。")
    else:
        print("\n已写入，原文件备份为 *.orig。")
    return 1 if grand["unresolved"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
