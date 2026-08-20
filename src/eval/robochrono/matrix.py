#!/usr/bin/env python3
# coding: utf-8
"""矩阵展开：把 (模型 × 任务族 × 任务) 展成一张 run 列表。

冻结版没有这个概念 —— 每个组合都是手敲一条命令。15 模型 × 20 族 × 9 任务
意味着最多 2700 条命令，必须由程序展开。

三件事在这里完成：
  稀疏规则   不是每个任务都适用于每个族（比如视角识别只测双手任务）
  分片       多机分工，按稳定哈希切分，机器之间互不重叠也不遗漏
  排序       本地模型按 model-major 排，让同一个模型的所有 run 连在一起，
             权重只加载一次
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import tasks

# 只有这四个任务送视频，抽帧档位对其余五个（静态图）没有意义
VIDEO_RUNS = {"time", "understanding", "image_in_video", "planning"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model: str | None = None
    kind: str = "local"          # local | api
    note: str = ""

    @property
    def is_local(self) -> bool:
        return self.kind == "local"


@dataclass(frozen=True)
class RunSpec:
    model: ModelSpec
    family: str
    run: str
    # 抽帧档位。None 表示用 provider 的默认（API 模型改不了服务端，只能如此）。
    # 本地模型按团队定的协议跑 fps=1 与 fps=2 两档。
    frames_fps: float | None = None

    @property
    def variant(self) -> str:
        return "default" if self.frames_fps is None else f"fps{self.frames_fps:g}"

    @property
    def key(self) -> str:
        return f"{self.model.name}__{self.family}__{self.run}__{self.variant}"

    def qa_path(self, datasets_root: Path) -> Path:
        return tasks.qa_path(datasets_root, self.family, self.run)


@dataclass
class Plan:
    models: list[ModelSpec] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    # 本地模型的抽帧档位。空列表 = 沿用 provider 配置，不做多档。
    frame_variants: list[float] = field(default_factory=list)
    family_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_rules: list[dict[str, Any]] = field(default_factory=list)


def load_plan(path: Path) -> Plan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    models = [
        ModelSpec(
            name=str(m["name"]),
            provider=str(m["provider"]),
            model=m.get("model"),
            kind=str(m.get("kind", "local")),
            note=str(m.get("note", "")),
        )
        for m in raw.get("models", [])
    ]
    return Plan(
        models=models,
        families=[str(f) for f in raw.get("families", [])],
        runs=[str(r) for r in raw.get("runs", tasks.ALL_RUNS)],
        family_attrs=raw.get("family_attrs", {}),
        skip_rules=raw.get("skip_rules", []),
        frame_variants=[float(v) for v in raw.get("frame_variants", [])],
    )


def derive_family_attrs(datasets_root: Path, family: str) -> dict[str, Any]:
    """从数据本身推导族属性，而不是手工在 plan.json 里维护。

    ``two_handed``：left_right 的题里 ``target_side`` 同时出现 left 和 right。

    为什么要推：``testproject.md`` 写着「当前双手任务覆盖 pick cube」，
    据此 plan.json 只给 stack_cubes 标了 two_handed。但实测**八个族的
    left_right 全是左右各半的完整双手数据**（wash 有 1034 题）。
    照手工标注跑，会静默丢掉 7 个族的 left_right —— 共 3,002 题。
    文档与数据不一致时，以数据为准，并且让判据可复算。
    """
    from . import tasks
    from .tasks.base import load_items

    attrs: dict[str, Any] = {}
    try:
        path = tasks.qa_path(datasets_root, family, "left_right")
        if path.exists():
            sides = {str(i.get("target_side")) for i in load_items(path)}
            attrs["two_handed"] = {"left", "right"} <= sides
    except Exception:  # noqa: BLE001  推导失败就退回 plan.json 里的手工标注
        pass
    return attrs


def _skipped(plan: Plan, family: str, run: str,
             derived: dict[str, dict[str, Any]] | None = None) -> str | None:
    """返回跳过原因，None 表示不跳过。

    属性优先级：从数据推导出的 > plan.json 手工标注的。
    """
    attrs = {**plan.family_attrs.get(family, {}), **((derived or {}).get(family, {}))}
    for rule in plan.skip_rules:
        if rule.get("run") and rule["run"] != run:
            continue
        unless = rule.get("unless")
        if unless and not attrs.get(unless):
            return f"{run} requires family attribute `{unless}`"
        only_if = rule.get("only_if")
        if only_if and not attrs.get(only_if):
            return f"{run} requires family attribute `{only_if}`"
    return None


def shard_of(key: str, shards: int) -> int:
    """稳定哈希分片。同一个 key 在任何机器上都落到同一片。"""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % shards


def expand(
    plan: Plan,
    datasets_root: Path,
    *,
    shard: tuple[int, int] | None = None,
    only_kind: str | None = None,
    only_models: list[str] | None = None,
    only_runs: list[str] | None = None,
) -> tuple[list[RunSpec], list[tuple[str, str]]]:
    """展开矩阵。返回 (要跑的 run 列表, [(key, 跳过原因)])。

    ``shard`` 形如 (1, 4) 表示「四台机器里的第一台」。
    ``only_models`` 限定模型名 —— 按模型分派不同 python 环境时用到。
    ``only_runs``   限定题型 —— **只重跑某个题型时用**（D-62）。

    为什么 ``only_runs`` 是必要的而不是锦上添花
    ------------------------------------------
    没有它的时候，「只重跑 time」在 ``matrix`` 上无法表达。
    唯一看起来可行的写法是 ``--overwrite`` 起全矩阵再中途停手，
    而 ``--overwrite`` 会在**任何 unit 开跑之前**就清掉全部 42 个 run 的结果。
    实际后果见 ``ResultStore.displace`` 的说明。
    """
    selected: list[RunSpec] = []
    skipped: list[tuple[str, str]] = []
    derived = {f: derive_family_attrs(datasets_root, f) for f in plan.families}
    wanted = set(only_models) if only_models else None
    if wanted:
        unknown = wanted - {m.name for m in plan.models}
        if unknown:
            raise ValueError(f"plan 里没有这些模型：{sorted(unknown)}")
    wanted_runs = set(only_runs) if only_runs else None
    if wanted_runs:
        # **拼错要报错，不能静默跑空。** `--runs tiem` 若只是筛不出东西，
        # 表现是「矩阵为空」，看起来像数据没到位而不是打字错了。
        unknown_runs = wanted_runs - set(plan.runs)
        if unknown_runs:
            raise ValueError(
                f"plan 的 runs 里没有这些题型：{sorted(unknown_runs)}；"
                f"可选：{sorted(plan.runs)}")

    for model in plan.models:
        if only_kind and model.kind != only_kind:
            continue
        if wanted and model.name not in wanted:
            continue
        # 抽帧档位只对本地模型有意义 —— API 模型的抽帧在服务端，我们改不了。
        # 送静态图的任务也不受影响，只对视频任务展开多档。
        variants: list[float | None] = (
            [float(v) for v in plan.frame_variants]
            if (model.is_local and plan.frame_variants) else [None]
        )

        for family in plan.families:
            for run in plan.runs:
                if wanted_runs and run not in wanted_runs:
                    continue
                run_variants = variants if run in VIDEO_RUNS else [None]
                for fps in run_variants:
                    spec = RunSpec(model=model, family=family, run=run, frames_fps=fps)

                    reason = _skipped(plan, family, run, derived)
                    if reason:
                        skipped.append((spec.key, reason))
                        continue
                    if not spec.qa_path(datasets_root).exists():
                        skipped.append((spec.key, f"QA 文件缺失 {spec.qa_path(datasets_root)}"))
                        continue
                    if shard is not None:
                        index, total = shard
                        if shard_of(spec.key, total) != index - 1:
                            continue
                    selected.append(spec)

    # model-major：同一个模型的所有 run 连在一起，本地权重只加载一次。
    # 15 模型 × 9 任务原本要加载 135 次，排序后降到 15 次。
    selected.sort(key=lambda s: (s.model.name, s.family, s.run, s.variant))
    return selected, skipped
