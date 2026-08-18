#!/usr/bin/env python3
# coding: utf-8
"""任务注册表。

九个 run 对应八个任务类型 —— trajectory 的 2D 与 3D 是两份独立输入、
两次独立运行、两组独立分数，所以按 run 建模。
"""

from __future__ import annotations

import json

from typing import Any

from . import choice, time_eqa, trajectory

# run 名 -> 该 run 在生成产物里对应的 QA 文件名
QA_FILENAME: dict[str, str] = {
    "time": "time_vqa.json",
    "understanding": "understanding_vqa.json",
    "left_right": "left_right_vqa.json",
    "image_in_video": "image_in_video_vqa.json",
    # BC-06：planning 与 planning_2 各自指向自己的输入。
    # 冻结版两个脚本都读 config 的 tasks.planning，而那一节当前指向
    # planning_2_vqa.json —— 跑 planning 不带 --input 会直接崩。
    "planning": "planning_vqa.json",
    "planning_2": "planning_2_vqa.json",
    "step_order": "step_order_vqa.json",
    "trajectory_2D": "trajectory_qa_2d.json",
    "trajectory_3D": "trajectory_qa_3d.json",
}

# run 名 -> 数据所在的组目录（生成流水线把产物分成两组）
QA_GROUP: dict[str, str] = {
    "time": "understanding",
    "understanding": "understanding",
    "left_right": "understanding",
    "image_in_video": "understanding",
    "planning": "planning",
    "planning_2": "planning",
    "step_order": "planning",
    "trajectory_2D": "planning",
    "trajectory_3D": "planning",
}

# run 名 -> 报表里的主指标
PRIMARY_METRIC: dict[str, str] = {
    "understanding": "accuracy",
    "left_right": "accuracy",
    "planning": "accuracy",
    "planning_2": "accuracy",
    "step_order": "accuracy",
    "image_in_video": "accuracy",
    "trajectory_2D": "mean_score",
    "trajectory_3D": "mean_score",
    "time": "mean_tIoU",
}

ALL_RUNS: tuple[str, ...] = tuple(QA_FILENAME)


def build(name: str, **flags: Any):
    """按 run 名构造任务实例。"""
    if name == "time":
        return time_eqa.build(**flags)
    if name.startswith("trajectory"):
        return trajectory.build(name, **flags)
    if name in choice.SPECS:
        return choice.build(name, **flags)
    raise ValueError(f"unknown run {name!r}; known runs: {list(ALL_RUNS)}")


class StaleNormalized(RuntimeError):
    """规范化产物相对当前代码或当前数据已过期。"""


_FRESHNESS_CHECKED: set[Any] = set()


def _require_fresh(root: Any) -> None:
    """每个进程对每个数据根只查一次，查过就记住。"""
    if root in _FRESHNESS_CHECKED:
        return
    from ..normalize import check_freshness

    state = check_freshness(root)
    if not state.ok:
        shown = "\n  ".join(state.reasons[:8])
        more = f"\n  …… 另有 {len(state.reasons) - 8} 条" if len(state.reasons) > 8 else ""
        raise StaleNormalized(
            f"规范化产物已过期，拒绝用它评测：\n  {shown}{more}\n"
            f"\n重建：python tools/build_normalized.py"
            f"\n若确实要绕过规范化走原始 QA，显式传 source=\"raw\"。")
    _FRESHNESS_CHECKED.add(root)


def load_run_items(datasets_root: Any, family: str, run: str,
                   *, source: str = "normalized") -> list[dict[str, Any]]:
    """取某个 (族, 任务) 的题目列表。**走哪条路必须显式声明。**

    ``source="normalized"``（默认）
        读 ``datasets/normalized/<family>/<run>.jsonl``。媒体路径已在构建期
        解析为绝对路径，不必每次运行重扫十几万个文件建索引。
        产物缺失或过期直接抛 :class:`StaleNormalized`，**不回退**。

    ``source="raw"``
        读原始 QA 并在内存里解析路径。replay 回归走这条 —— 它的基线就是
        规范化之前的行为，必须比对得上。

    为什么不再「优先规范化、缺了自动回退」
    ------------------------------------
    因为那让**走了哪条路取决于文件在不在**，而两条路给出的东西已经不同了。
    实测：把 ``stack_cubes/planning_2.jsonl`` 藏起来，评测照跑 300 题，
    只是每题的图从 3 张变回 1 张 —— BC-16 悄悄失效，没有任何提示。
    题数一样、不报错、分数变了，这是最难发现的一种错。

    触发场景都不是假想：升级构建器后忘了重建、只 ``--family`` 重建了一部分、
    别人 clone 仓库直接跑（``normalized/`` 是 gitignore 的）。
    所以宁可停下来报错，也不要「刚好还能跑」。

    两条路的等价性由 ``tests/test_normalized_equivalence.py`` 守着。
    """
    from pathlib import Path as _Path

    root = _Path(datasets_root)

    if source == "raw":
        from ..mediaindex import index_for_qa, resolve_items
        from .base import load_items

        path = qa_path(root, family, run)
        items = load_items(path)
        resolve_items(items, index_for_qa(path))
        return items

    if source != "normalized":
        raise ValueError(f"source 只能是 'normalized' 或 'raw'，收到 {source!r}")

    from ..normalize import canonical_family

    _require_fresh(root)
    jsonl = root / "normalized" / canonical_family(family) / f"{run}.jsonl"
    if not jsonl.exists():
        raise StaleNormalized(
            f"{family}/{run} 没有规范化产物（{jsonl}）。\n"
            f"重建：python tools/build_normalized.py --family {family}")
    items: list[dict[str, Any]] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line)["item"])
    return items


def qa_path(datasets_root: Any, family: str, run: str) -> Any:
    """定位某个 (族, 任务) 的 QA 文件。

    标准布局是 ``<root>/QA/<group>/<family>/<file>``，但数据集里有一半的族
    **多嵌了一层子目录**，而且那一层的名字不统一：

        QA/planning/stack_cubes/planning_vqa.json                  ← 标准
        QA/planning/gift_inhand/planning/planning_vqa.json         ← 多一层 planning/
        QA/understanding/tea/time_understanding/left_right_vqa.json
        QA/understanding/wash/time/time_vqa.json

    按固定布局拼路径会让这 4 个族**整族静默消失** —— matrix 只会记一句
    「QA 文件缺失」然后跳过。实测漏掉的是 6,937 题，占总量的一半以上。

    所以：先试标准位置，找不到就在族目录下递归找同名文件。
    **多处命中时报错而不是取第一个** —— 静默选错文件会让整族评错，
    而报错至少能被发现。
    """
    from pathlib import Path

    filename = QA_FILENAME[run]
    family_root = Path(datasets_root) / "QA" / QA_GROUP[run] / family
    direct = family_root / filename
    if direct.exists():
        return direct

    if not family_root.exists():
        return direct                      # 让调用方按「缺失」处理，路径用于报错信息

    hits = sorted(family_root.rglob(filename))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(
            f"{family}/{run} 的 QA 文件在 {len(hits)} 处同时存在，无法判定用哪个：\n  "
            + "\n  ".join(str(h) for h in hits)
        )
    return direct
