#!/usr/bin/env python3
# coding: utf-8
"""矩阵执行：model-major 调度，本地模型用 GPU worker 池。

把 (模型 × 任务族 × 任务) 的矩阵按模型分组依次执行。对本地模型，
该模型下所有任务族、所有任务的 unit 汇成一个队列摊给各卡；
对 API 模型，瓶颈是网络往返而非算力，用线程池并发（``--api-concurrency``），
并发等价性由 tests/test_concurrency_equivalence.py 保证。

两种并发不叠加：本地模型即便走到单卡串行路径也保持 concurrency=1。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import engine, pool, tasks
from .matrix import ModelSpec, Plan, RunSpec
from .store import ResultStore
from .tasks.base import load_items
from .vlm_api import runtime_config


def _store_for(spec: RunSpec, results_root: Path, runtime_meta: dict[str, Any]) -> ResultStore:
    # 抽帧档位单独分目录，fps=1 与 fps=2 的结果不会互相覆盖
    out_dir = results_root / spec.model.name / spec.family / spec.variant
    return ResultStore(out_dir / f"{spec.run}.jsonl", meta=runtime_meta)


def _apply_frames(runtime: dict[str, Any], spec: RunSpec) -> dict[str, Any]:
    """把该 run 的抽帧档位写进 runtime。

    frames 只影响预处理、不影响权重，所以同一次模型加载可以服务多个档位，
    不需要为 fps=1/fps=2 各加载一遍。
    """
    if spec.frames_fps is None:
        return runtime
    runtime = dict(runtime)
    runtime["frames"] = {
        "mode": "fps",
        "value": spec.frames_fps,
        "video_sample_fps": spec.frames_fps,
        "num_segments": 1,
    }
    # 团队定的口径：按实际帧数对齐，num_segments 型换算成 round(时长 × fps)
    runtime["align_fps_to_segments"] = True
    return runtime


def _environment() -> dict[str, Any]:
    """记录产出这批结果的运行环境。

    这不是可有可无的元数据 —— 实测确认 **transformers 版本会改变模型输出**：
    4.57.6 与 5.15.0 对同一段视频拼出的 prompt 差 2 个 token
    （5.x 在整段视频外多包一层 `<|vision_start|>…<|vision_end|>`），
    画面完全相同但输出跟着变。详见 docs/environments.md。

    所以两批结果能不能放一起比，取决于它们是不是同一个版本产出的。
    不记下来，事后无从判断。
    """
    import platform
    import sys

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "executable": sys.executable,
    }
    for module in ("transformers", "torch", "qwen_vl_utils"):
        try:
            info[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001  API provider 不需要这些包
            info[module] = None
    return info


def _meta(spec: RunSpec, runtime: dict[str, Any], qa_path: Path, flags: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment": _environment(),
        "model_name": spec.model.name,
        "provider": runtime["provider"],
        "model": runtime["model"],
        "api_url": runtime["api_url"],
        "family": spec.family,
        "run": spec.run,
        "input": str(qa_path),
        "frames": runtime["frames"],
        "generation": {
            "temperature": runtime["temperature"],
            "thinking": runtime["thinking"],
            "max_new_tokens": runtime["max_new_tokens"],
        },
        "flags": dict(flags),
        "frames_variant": spec.variant,
    }


def run_matrix(
    plan: Plan,
    specs: list[RunSpec],
    *,
    config_path: Path,
    datasets_root: Path,
    results_root: Path,
    gpus: list[int],
    flags: dict[str, Any],
    limit_items: int | None = None,
    limit_groups: int | None = None,
    overwrite: bool = False,
    api_concurrency: int | None = None,
    api_rate_limit: float | None = None,
) -> int:
    by_model: dict[str, list[RunSpec]] = {}
    for spec in specs:
        by_model.setdefault(spec.model.name, []).append(spec)

    failures = 0
    for model_name, model_specs in by_model.items():
        model: ModelSpec = model_specs[0].model
        print(f"\n{'='*70}\n模型 {model_name}（{model.kind}，{len(model_specs)} 个 run）\n{'='*70}")

        use_pool = model.is_local and len(gpus) > 1
        if use_pool:
            failures += _run_local_pool(
                model, model_specs, config_path=config_path, datasets_root=datasets_root,
                results_root=results_root, gpus=gpus, flags=flags,
                limit_items=limit_items, limit_groups=limit_groups, overwrite=overwrite,
            )
        else:
            failures += _run_serial(
                model, model_specs, config_path=config_path, datasets_root=datasets_root,
                results_root=results_root, flags=flags,
                limit_items=limit_items, limit_groups=limit_groups, overwrite=overwrite,
                api_concurrency=api_concurrency, api_rate_limit=api_rate_limit,
            )
    return failures


def _prepare(spec: RunSpec, datasets_root: Path, config_path: Path, model: ModelSpec,
             flags: dict[str, Any], results_root: Path, overwrite: bool):
    qa_path = spec.qa_path(datasets_root)
    runtime = runtime_config(config_path=config_path, provider_name=model.provider,
                             default_model="", cli_model=model.model)
    runtime = _apply_frames(runtime, spec)
    store = _store_for(spec, results_root, _meta(spec, runtime, qa_path, flags))
    if overwrite and store.path.exists():
        store.path.unlink()
    store.open()
    # 优先读规范化产物；不存在时回退原始 QA 并在内存里解析媒体路径
    items = tasks.load_run_items(datasets_root, spec.family, spec.run)
    # 所有 flag 发给所有任务，各 build() 自己挑认识的
    return qa_path, runtime, store, items, tasks.build(spec.run, **flags)


def _run_serial(model, model_specs, *, config_path, datasets_root, results_root,
                flags, limit_items, limit_groups, overwrite,
                api_concurrency=None, api_rate_limit=None) -> int:
    failures = 0
    for spec in model_specs:
        print(f"\n--- {spec.family} × {spec.run} ---")
        try:
            _, runtime, store, items, task = _prepare(
                spec, datasets_root, config_path, model, flags, results_root, overwrite)
            # 本地模型即便走到串行路径（单卡）也不并发：瓶颈是 GPU 不是网络。
            if model.is_local:
                concurrency, rate_limit = 1, 0.0
            else:
                concurrency = api_concurrency if api_concurrency is not None else runtime.get("concurrency", 1)
                rate_limit = api_rate_limit if api_rate_limit is not None else runtime.get("rate_limit", 0.0)
            summary = engine.run(task, items, runtime, store,
                                 limit_items=limit_items, limit_groups=limit_groups,
                                 overwrite=False,
                                 concurrency=max(1, int(concurrency)),
                                 rate_limit=float(rate_limit))
            _write_summary(store, summary, spec)
        except Exception as exc:  # noqa: BLE001
            print(f"  RUN FAILED: {type(exc).__name__}: {exc}")
            failures += 1
    return failures


def _run_local_pool(model, model_specs, *, config_path, datasets_root, results_root,
                    gpus, flags, limit_items, limit_groups, overwrite) -> int:
    """该模型的所有 run 汇成一个队列，摊给各卡。权重每卡只加载一次。"""
    work: list[pool.WorkItem] = []
    stores: dict[str, ResultStore] = {}
    contexts: dict[str, tuple[RunSpec, Any]] = {}

    for spec in model_specs:
        try:
            _, _, store, items, task = _prepare(
                spec, datasets_root, config_path, model, flags, results_root, overwrite)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {spec.family}/{spec.run}: {type(exc).__name__}: {exc}")
            continue

        stores[spec.key] = store
        contexts[spec.key] = (spec, task)
        done = store.completed_ids()
        units = engine.limit_units(task.units(items), limit_items, limit_groups)
        for unit in units:
            if all(str(i.get("id")) in done for i in unit.items):
                continue
            work.append(pool.WorkItem(spec.key, spec.run, unit.key, unit.items,
                                      frames_fps=spec.frames_fps))

    if not work:
        print("  全部已完成，无需执行")
    else:
        print(f"  {len(work)} 个 unit 摊给 {len(gpus)} 张卡")

        def on_result(result: pool.WorkResult) -> None:
            store = stores.get(result.spec_key)
            if store is not None:
                store.append(result.rows)

        stats = pool.run_pool(
            work, gpus=gpus, config_path=config_path, provider=model.provider,
            model_override=model.model, task_flags=flags, on_result=on_result,
        )
        print(f"  完成 {stats['done']}，错误 {stats['errors']}")

    for key, (spec, task) in contexts.items():
        store = stores[key]
        summary = task.summarize(store.final_rows(), 0.0)
        _write_summary(store, summary, spec)
    return 0


def _write_summary(store: ResultStore, summary: dict[str, Any], spec: RunSpec) -> None:
    path = store.path.with_name(f"{spec.run}.summary.json")  # 与 jsonl 同目录，已按 variant 分开
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metric = tasks.PRIMARY_METRIC[spec.run]
    print(f"  {spec.family}/{spec.run}: {metric} = {summary.get(metric)}  "
          f"(answered {summary.get('answered')}/{summary.get('total')}, "
          f"errors {summary.get('errors')})")
