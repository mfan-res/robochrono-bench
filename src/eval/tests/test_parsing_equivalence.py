#!/usr/bin/env python3
# coding: utf-8
"""验证 eval/robochrono/parsing.py 与 test/ 下冻结脚本的解析行为逐条一致。

这是阶段 1 的第一道门禁。BC-02 关闭时，新实现对任意输入都必须给出与旧脚本
相同的 choice；否则重构就改变了评分。

对六个选择题任务各取真实 item，配上一组覆盖常见与边缘情况的模型输出，
逐条比对。不需要 GPU，不需要 API。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _P  # noqa: E402
sys.path.insert(0, str(_P.EVAL / "upstream"))   # 冻结的参考实现
sys.path.insert(0, str(_P.EVAL))

from robochrono import parsing  # noqa: E402

QA = _P.QA

# (任务名, QA 文件, 冻结脚本模块名, 是否用 choices 取选项)
TASKS = [
    ("understanding", QA / "understanding/stack_cubes/understanding_vqa.json", "understanding_glm_test", False),
    ("left_right", QA / "understanding/stack_cubes/left_right_vqa.json", "left_right_glm_test", False),
    ("image_in_video", QA / "understanding/stack_cubes/image_in_video_vqa.json", "image_in_video_glm_test", False),
    ("planning", QA / "planning/stack_cubes/planning_vqa.json", "planning_glm_test", False),
    ("planning_2", QA / "planning/stack_cubes/planning_2_vqa.json", "planning_2_glm_test", False),
    ("step_order", QA / "planning/stack_cubes/step_order_vqa.json", "step_order_glm_test", True),
]

# 覆盖：规范 JSON、带围栏、纯字母、小写、句子、选项原文、空、噪声、
# 思考块（BC-02 关闭时两边都应解析失败或都成功）
OUTPUTS = [
    '{{"choice": "{first}", "reason": "x"}}',
    '```json\n{{"choice": "{first}"}}\n```',
    '{first}',
    '{first_lower}',
    'The answer is {first}.',
    'I think the correct option is {first}) because of the gripper.',
    '{{"answer": "{first}"}}',
    '{{"option": "{first}"}}',
    '{{"letter": "{first}"}}',
    '{option_text}',
    '',
    'none of these',
    '   ',
    '{{"choice": null}}',
    '{{"choice": "Z"}}',
    'Options A and B both look plausible',
    '<think>Maybe {first}, maybe not.</think>{{"choice": "{first}"}}',
    '<think>reasoning only, no answer</think>',
    'Here is my answer:\n{{"choice": "{first}"}}\nDone.',
    '{{"choice": "{first}"}} extra trailing text',
]


def main() -> int:
    total = mismatch = 0
    print(f"{'task':<16} {'items':>6} {'cases':>7} {'mismatch':>9}  status")
    print("-" * 56)

    for name, qa_path, module_name, use_choices in TASKS:
        if not qa_path.exists():
            print(f"{name:<16} {'-':>6} {'-':>7} {'-':>9}  QA 文件缺失")
            continue

        legacy = __import__(module_name)
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        items = (data.get("items", data) if isinstance(data, dict) else data)[:40]

        n_case = n_bad = 0
        first_bad = ""
        for item in items:
            options = parsing.choices_from_item(item) if use_choices else parsing.options_from_item(item)
            if not options:
                continue
            first = sorted(options)[0]
            option_text = options[first] or "placeholder"

            for template in OUTPUTS:
                text = template.format(first=first, first_lower=first.lower(), option_text=option_text)
                old = legacy.parse_model_answer(text, item)["choice"]
                new = parsing.parse_choice_answer(
                    text, options, keep_hyphen=use_choices, strip_reasoning=False
                )["choice"]
                n_case += 1
                if old != new:
                    n_bad += 1
                    if not first_bad:
                        first_bad = f"{text[:40]!r} old={old} new={new}"

        total += n_case
        mismatch += n_bad
        status = "OK" if n_bad == 0 else first_bad
        print(f"{name:<16} {len(items):>6} {n_case:>7} {n_bad:>9}  {status}")

    print("-" * 56)
    print(f"共比对 {total} 条，不一致 {mismatch} 条")
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
