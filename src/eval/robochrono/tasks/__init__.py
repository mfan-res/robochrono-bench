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
# 每个任务用哪个指标当总分。
#
# ⚠ `time` 曾定为 `mean_tIoU`，**已改为 `tIoU@0.5`**。依据是在本批数据上
# 跑退化基线量出来的（不是照搬学界惯例）：
#
#   退化基线              mean_tIoU  @0.3  @0.5  @0.7  中心命中  有交集
#   整段视频都报              0.13    12%    0%    0%    15%    100%
#   报视频正中间 1 秒          0.02     1%    0%    0%    15%     17%
#   随便报同集另一个动作        0.00     0%    0%    0%     0%     17%
#   真值但整体平移 2 秒        0.47    78%   49%   19%    75%     92%
#   真值但起止各缩 20%        0.60   100%  100%    0%   100%    100%
#
# - `mean_tIoU` 有 **0.13 的地板** —— 「整段视频都报」就能拿到，不是 0
# - `pointing` 更糟，同一策略拿 100%
# - `@0.7` 太严：起止各缩 20% 这种合理回答得 0%。而段边界本就是分段的
#   交接点、不是动作的精确起始（P-06），要求 0.7 是在苛求标注没有的精度
# - `@0.5` 三个退化策略全 0%、合理回答 100%、平移 2 秒 49% —— 有区分度
#
# 换算成时间容差：段长 D 时纯平移 s 满足 (D−s)/(D+s) ≥ 0.5 即 s ≤ D/3。
# 中位段长 5.9 秒 → 约 ±2 秒，与标注本身的精度相称。
#
# 其余指标照报，只是不当总分。
PRIMARY_METRIC: dict[str, str] = {
    "understanding": "accuracy",
    "left_right": "accuracy",
    "planning": "accuracy",
    "planning_2": "accuracy",
    "step_order": "accuracy",
    "image_in_video": "accuracy",
    "trajectory_2D": "mean_score",
    "trajectory_3D": "mean_score",
    "time": "tIoU@0.5",
}

ALL_RUNS: tuple[str, ...] = tuple(QA_FILENAME)

# run 名 -> (指标, 下限, 这个下限是哪种策略拿到的)
#
# **「低于随机」与「低于最蠢的策略」都不是噪声，是信号。**
#
# 首轮全量里 time 拿了 `tIoU@0.5 = 0.0`，而上面那张表里「整段视频都报」
# 能拿 `mean_tIoU 0.13` —— 模型比最蠢的策略还差。
# 发现这件事靠的是人工诊断两小时，而 `errors=0`、熔断没触发、
# `matrix` 退出 0、报表照常出数，**没有任何一处会说出这句话**（D-63）。
# 那几个退化基线上一轮就量过了（就是上面 PRIMARY_METRIC 的注释），
# 只是从没进过代码。这里把它变成一个记号。
#
# 选择题的 0.25：④ 出题固定四选一（`src/vqa/plan.py` 的
# `DISTRACTORS_PER_QUESTION = 3`），名义随机基线 1/4。
# **用名义值而不是等效值** —— gift_inhand / pen_inbox 的 understanding
# 实际是三选一、等效基线 33%（docs/disclosures.md 第 2 条），
# 等效值只会让门槛更高，用名义值不会误报。
#
# trajectory 不设下限：它的退化基线没量过，而且已搁置（指标判定无效）。
RANDOM_BASELINE = 0.25

DEGENERATE_FLOOR: dict[str, tuple[str, float, str]] = {
    run: ("accuracy", RANDOM_BASELINE, "随机猜") for run in choice.SPECS
}
DEGENERATE_FLOOR["time"] = ("mean_tIoU", 0.13, "「整段视频都报」")


def floor_breach(run: str, summary: dict[str, Any]) -> str | None:
    """这个成绩是不是低到「不看视频也能拿到」。返回一句话，或 None 表示没问题。

    两条判据，都只在**确实答了题**的时候才生效 —— 全都没答上是另一类问题
    （解析失败 / 媒体缺失），由 `answered` 与 `parse_failure_rate` 各自反映，
    混进来会让这个记号失去意义。
    """
    if not summary.get("answered"):
        return None

    floor = DEGENERATE_FLOOR.get(run)
    if floor:
        metric, threshold, label = floor
        value = summary.get(metric)
        if isinstance(value, (int, float)) and value < threshold:
            return f"低于{label}（{metric} {value:.3g} < {threshold:g}）"

    # 主指标恰好为 0：三个退化策略在 tIoU@0.5 上也全是 0，
    # 所以「0 分」不代表比它们差，而是**与它们没有区分度**。同样要标出来。
    primary = PRIMARY_METRIC.get(run)
    if primary and summary.get(primary) == 0:
        return f"{primary} 为 0，与退化策略无区分度（答了 {summary['answered']} 题）"
    return None


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


def load_for_run(datasets_root: Any, family: str, run: str,
                 stats: Any = None) -> list[dict[str, Any]]:
    """**评测真正走的加载路径。preflight 与 run 共用这一个函数。**

    此前两边各写各的：`cli.run` 走这里的逻辑，`preflight.check_data` 走
    `load_run_items(source="normalized")`。于是自检验的不是运行走的那条路 ——
    v2 数据上自检 25 项全红，而 `run` 一路跑通（冒烟、四轮盲基线都过）。
    **自检与运行分叉，比没有自检更糟**：它在能跑的数据上报警，
    也可能在真出问题时沉默。

    媒体路径相对 QA 文件所在目录解析（`base=`）—— 这是 v2 的约定，
    换机器不用重新生成 QA。
    """
    from pathlib import Path as _Path

    from ..mediaindex import index_for_qa, resolve_items
    from .base import load_items

    path = qa_path(_Path(datasets_root), family, run)
    items = load_items(path)
    resolve_items(items, index_for_qa(path, datasets_root), stats, base=path.parent)
    return items


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
        # base=QA 文件所在目录 —— 让媒体可以写成相对它的路径（见 mediaindex）
        resolve_items(items, index_for_qa(path), base=path.parent)
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

    from pathlib import Path as _Path

    # ── 新布局优先：<root>/<族>/<题型>.json ──────────────────────────
    # v1 那层「组」（understanding / planning）是历史约定，不是结构 ——
    # time 归 understanding、step_order 归 planning，纯粹因为当初谁先做。
    # 新数据用扁平布局，一眼看得出谁是谁；v1 的数据继续走下面的兜底。
    flat = _Path(datasets_root) / family / f"{run}.json"
    if flat.exists():
        return flat

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
