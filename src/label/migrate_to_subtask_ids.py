#!/usr/bin/env python3
# coding: utf-8
"""把标注从「存渲染后的字符串」改成「存 subtask ID」。

为什么
------
同一个动作在系统里被存了 **11 份**：categories.txt、narration、main_verbs、
objects、answer_text、correct_option.text、options[].text、question 内嵌、
answer_action/answer_objects、source_time_eqa、LLM 缓存的键。

每一份都是一次渲染，每一次渲染都是一次分叉的机会，而没有任何机制保证它们收敛。
我们已知的五个事故全是其中两份漂移：

    h        LLM 缓存按 categories 建键、按 answer_text 查表 → 六族全部落空
    B-01     categories → narration 渲染时插错 the
    P-03     出题产物被回写进 narration 所在的层
    D-21 后遗症   我们改了 narration，没改 categories，两边永久不一致
    answer_text ≠ correct_option.text   同一个东西在图选项任务里分叉

**根因是系统里没有「动作」这个实体。** 每层存的都是字符串，不是引用。

改法
----
给每个 subtask 一个**永不改变**的 ID，标注段只存 ID，文字只在展示时渲染。
于是 11 份变 1 份，漂移在结构上不可能发生。

ID 一旦分配就不再变 —— 这是它的全部价值。将来修措辞（像 D-21 那样）
只改 `subtasks.json` 里一行 `text`，所有引用它的标注自动跟着变。

具体做三件事
------------
1. 生成 ``data/label/<族>/subtasks.json``（从现存 narration 派生，一对一）
2. 段里 ``narration`` / ``main_verbs`` / ``objects`` → 一个 ``subtask`` 字段
3. ``categories.txt`` 标为 deprecated（不删 —— v1 的题目建立在它上面，
   删了就无法解释旧结果）

**stack_cubes 的 ``metadata`` 保留不动** —— 它是 P-03 的证据，也是将来还原
4 段原件的依据。schema 会继续把它标为违规，那是**正确行为**，直到还原完成。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL = ROOT / "data" / "label"

_ARTICLES = {"a", "an", "the"}


def make_id(text: str) -> str:
    """ID 去掉冠词 —— 它是标识符，不是句子。

    不加序号：序号会与列表顺序耦合，而 ID 必须稳定。
    """
    words = [w for w in re.split(r"[^A-Za-z0-9]+", text.strip().rstrip(".").lower()) if w]
    kept = [w for w in words if w not in _ARTICLES]
    return "_".join(kept or words)


def main() -> int:
    apply = "--apply" in sys.argv
    print(f"{'族':<13}{'subtask':>8}{'段数':>7}{'ID 唯一':>8}  说明")
    print("-" * 66)
    plans: dict[str, tuple[list[dict[str, str]], dict[str, str]]] = {}

    for family in sorted(p.name for p in LABEL.iterdir() if p.is_dir()):
        counts: Counter[str] = Counter()
        for path in sorted((LABEL / family / "segments").glob("*_segments.json")):
            for seg in json.loads(path.read_text(encoding="utf-8"))["segments"]:
                counts[seg["narration"].strip()] += 1
        subtasks = [{"id": make_id(t), "text": t} for t in counts]
        by_text = {s["text"]: s["id"] for s in subtasks}
        ids = [s["id"] for s in subtasks]
        unique = len(set(ids)) == len(ids)
        plans[family] = (subtasks, by_text)
        note = "" if unique else "✗ ID 冲突，需人工消歧"
        print(f"{family:<13}{len(subtasks):>8}{sum(counts.values()):>7}"
              f"{'✓' if unique else '✗':>8}  {note}")
        if not unique:
            return 1

    if not apply:
        print("\n这是预演。加 --apply 才写盘。")
        print("\n示例（tea2）：")
        for s in plans["tea2"][0]:
            print(f"  {s['id']:<24} {s['text']}")
        return 0

    print()
    for family, (subtasks, by_text) in plans.items():
        base = LABEL / family
        backup = base / "segments.before_subtask_id"
        if not backup.exists():
            shutil.copytree(base / "segments", backup)

        (base / "subtasks.json").write_text(json.dumps({
            "family": family, "version": 1,
            "_note": "subtask 定义。ID 永不改变；修措辞只改 text，所有引用自动跟随。"
                     "标注段用 subtask 字段引用这里的 id，不再存渲染后的字符串。",
            "subtasks": subtasks,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        converted = kept_metadata = 0
        for path in sorted((base / "segments").glob("*_segments.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            for seg in doc["segments"]:
                seg["subtask"] = by_text[seg["narration"].strip()]
                for dead in ("narration", "main_verbs", "objects"):
                    seg.pop(dead, None)
                if "metadata" in seg:
                    kept_metadata += 1          # P-03 的证据，保留待还原
                converted += 1
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

        cats = base / "categories.txt"
        if cats.exists() and not (base / "categories.deprecated.txt").exists():
            cats.rename(base / "categories.deprecated.txt")

        note = f"，保留 {kept_metadata} 段的 metadata（P-03 证据）" if kept_metadata else ""
        print(f"  {family:<13}{len(subtasks)} 个 subtask，{converted} 段已转{note}")

    print("\n原件备份在各族的 segments.before_subtask_id/；"
          "categories.txt 改名为 categories.deprecated.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
