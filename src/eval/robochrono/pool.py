#!/usr/bin/env python3
# coding: utf-8
"""GPU worker 池。

冻结版是「一个脚本一个进程一次加载」，10 个本地模型 × 9 个任务要加载 90 次权重，
30B 级别每次几分钟。这里改成：

    主进程
      ├─ worker-0  CUDA_VISIBLE_DEVICES=0  ┐ 每个 worker 加载当前模型的一份副本，
      ├─ worker-1  CUDA_VISIBLE_DEVICES=1  │ 从共享队列领 unit，算完把结果行送回
      └─ worker-N  CUDA_VISIBLE_DEVICES=N  ┘

同一时刻所有卡跑**同一个模型**，把该模型的全部 (任务族 × 任务 × unit) 摊给各 worker；
这个模型做完再整体换下一个。既只加载一次权重，又吃满所有卡。

两个必须注意的实现细节：

1. ``CUDA_VISIBLE_DEVICES`` 必须在 **import torch 之前**设好，所以用 spawn 启动子进程，
   并在子进程入口第一件事就设环境变量。本包内所有 torch 都是函数内懒加载，符合这个前提。
2. 结果由**主进程单点写入**，worker 只回传结果行。避免多进程写同一个 JSONL 交错。

注意：多卡只解决「权重放不下」，不解决「单个大张量放不下」——
后者是 Time EQA OOM 的真正原因，详见 eval/docs/frame_sampling_investigation.md。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STOP = "__STOP__"


@dataclass
class WorkItem:
    spec_key: str
    run: str
    unit_key: str
    items: list[dict[str, Any]]
    # 抽帧档位。frames 只影响预处理，所以同一次模型加载可以服务多个档位。
    frames_fps: float | None = None


@dataclass
class WorkResult:
    spec_key: str
    unit_key: str
    rows: list[dict[str, Any]]
    seconds: float
    worker: int
    error: str = ""


def _worker(
    gpu_index: int,
    work_queue: Any,
    result_queue: Any,
    config_path: str,
    provider: str,
    model_override: str | None,
    task_flags: dict[str, Any],
) -> None:
    """子进程入口。设卡 → 加载模型 → 循环领活。"""
    # 必须在任何 torch 导入之前
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    from . import tasks
    from .tasks.base import CallContext
    from .vlm_api import call_vlm, runtime_config

    try:
        runtime = runtime_config(
            config_path=Path(config_path),
            provider_name=provider,
            default_model="",
            cli_model=model_override,
        )
        # 每个 worker 独占一张卡，模型放在这张卡上
        runtime["device_map"] = {"": 0}
    except Exception:
        result_queue.put(WorkResult("", "", [], 0.0, gpu_index, traceback.format_exc()))
        return

    task_cache: dict[str, Any] = {}

    while True:
        try:
            item = work_queue.get(timeout=5)
        except queue.Empty:
            continue
        if item == STOP:
            break

        started = time.perf_counter()
        task = task_cache.get(item.run)
        if task is None:
            task = task_cache[item.run] = tasks.build(item.run, **task_flags)

        from .tasks.base import Unit

        unit = Unit(key=item.unit_key, items=item.items)
        if item.frames_fps is not None:
            runtime["frames"] = {
                "mode": "fps", "value": item.frames_fps,
                "video_sample_fps": item.frames_fps, "num_segments": 1,
            }
            runtime["align_fps_to_segments"] = True
        meta: dict[str, Any] = {}
        try:
            _, text = call_vlm(runtime, task.parts(unit), meta)
            retry_hook = getattr(task, "retry_parts", None)
            attempt = 0
            while retry_hook is not None:
                retry = retry_hook(unit, text, attempt)
                if retry is None:
                    break
                attempt += 1
                _, text = call_vlm(runtime, retry, meta)
            ctx = CallContext(
                frames_used=meta.get("frames_used", {}),
                usage=meta.get("usage", {}),
                media_transforms=meta.get("media_transforms", []),
            )
            rows = task.rows(unit, text, ctx)
            error = ""
        except Exception as exc:  # noqa: BLE001
            rows = task.error_rows(unit, f"{type(exc).__name__}: {exc}")
            error = f"{type(exc).__name__}: {exc}"

        seconds = round(time.perf_counter() - started, 3)
        for row in rows:
            row["timing"] = {"seconds": seconds, "worker": gpu_index}
        result_queue.put(WorkResult(item.spec_key, item.unit_key, rows, seconds, gpu_index, error))


# 选择题类的 run。曾经用来门控 flag 分发，现在只剩 preflight/report 在用；
# flag 分发已改成「全发给所有任务，各自挑」，见 tasks.build。
_CHOICE_RUNS = {
    "understanding", "left_right", "image_in_video",
    "planning", "planning_2", "step_order",
}


def run_pool(
    work: list[WorkItem],
    *,
    gpus: list[int],
    config_path: Path,
    provider: str,
    model_override: str | None = None,
    task_flags: dict[str, Any] | None = None,
    on_result: Any = None,
) -> dict[str, int]:
    """把 work 摊给各 GPU worker，边收边交给 on_result 落盘。"""
    if not work:
        return {"done": 0, "errors": 0}

    ctx = mp.get_context("spawn")
    work_queue: Any = ctx.Queue()
    result_queue: Any = ctx.Queue()

    for item in work:
        work_queue.put(item)
    for _ in gpus:
        work_queue.put(STOP)

    procs = [
        ctx.Process(
            target=_worker,
            args=(gpu, work_queue, result_queue, str(config_path), provider,
                  model_override, task_flags or {}),
            daemon=True,
        )
        for gpu in gpus
    ]
    for proc in procs:
        proc.start()

    done = errors = 0
    started = time.perf_counter()
    while done < len(work):
        try:
            result: WorkResult = result_queue.get(timeout=600)
        except queue.Empty:
            alive = [p for p in procs if p.is_alive()]
            if not alive:
                print(f"  所有 worker 已退出，仍有 {len(work)-done} 个 unit 未完成", flush=True)
                break
            continue

        if not result.spec_key and result.error:
            print(f"  worker-{result.worker} 启动失败：\n{result.error}", flush=True)
            errors += 1
            break

        done += 1
        if result.error:
            errors += 1
        if on_result is not None:
            on_result(result)
        if done % 20 == 0 or done == len(work):
            rate = done / max(1e-6, time.perf_counter() - started)
            print(f"    [{done}/{len(work)}] {rate*60:.1f} unit/min  errors={errors}", flush=True)

    for proc in procs:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()

    return {"done": done, "errors": errors}


def visible_gpus(requested: int | None = None) -> list[int]:
    """可用 GPU 列表。requested 给出时取前 N 张。"""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    count = torch.cuda.device_count()
    return list(range(count if requested is None else min(requested, count)))
