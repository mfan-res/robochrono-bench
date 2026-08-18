#!/usr/bin/env python3
# coding: utf-8
"""Task 协议。

八个冻结脚本的差异收敛到四个钩子：取媒体、组 prompt、解析输出、算分。
其余约 1600 行是逐字重复的样板，由 engine 统一承担。

一个设计要点：``time`` 是按视频分组提问的 —— 一次模型调用回答该 episode 的
全部问题；其余任务一题一次调用。用 ``Unit`` 把两种形态统一起来：
一个 Unit 对应一次模型调用，产出一到多条结果行。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Unit:
    """一次模型调用的单位。

    key    断点续跑的唯一标识（item id，或 time 的 video_id）
    items  这次调用覆盖的题目。多数任务只有一道，time 是整个 episode 的若干道。
    """

    key: str
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CallContext:
    """执行时传给 Task 的运行期信息，供记录与自适应逻辑使用。"""

    frames_used: dict[str, int] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    # BC-11：为满足请求体预算而对媒体做的变换。必须落到结果里，否则无从审计。
    media_transforms: list[dict[str, Any]] = field(default_factory=list)


class Task(Protocol):
    """一个评测任务。实现里的 prompt / 打分 / 汇总均从冻结脚本逐字搬运。"""

    name: str

    def units(self, items: list[dict[str, Any]]) -> list[Unit]:
        """把题目列表切成「一次调用」的单位。"""
        ...

    def parts(self, unit: Unit) -> list[dict[str, Any]]:
        """组装发给模型的内容：text / image / video 三种 part。"""
        ...

    def rows(self, unit: Unit, text: str, ctx: CallContext) -> list[dict[str, Any]]:
        """解析模型输出并打分，每道题产出一行结果。"""
        ...

    def error_rows(self, unit: Unit, error: str) -> list[dict[str, Any]]:
        """调用失败时的占位结果行，语义与冻结脚本的 except 分支一致。"""
        ...

    def summarize(self, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        """汇总指标。逐字搬运自冻结脚本。"""
        ...


# --------------------------------------------------------------------------
# 共享样板
# --------------------------------------------------------------------------


def load_items(path: Path) -> list[dict[str, Any]]:
    """读 QA JSON。八个脚本此处逻辑一致。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Input must be a list or contain an `items` list: {path}")
    return [item for item in items if isinstance(item, dict)]


def one_item_per_unit(items: list[dict[str, Any]]) -> list[Unit]:
    """默认切分：一题一次调用。"""
    return [Unit(key=str(item.get("id")), items=[item]) for item in items]


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(path: Any) -> dict[str, Any]:
    return {"type": "image", "path": str(path)}


def video_part(path: Any) -> dict[str, Any]:
    return {"type": "video", "path": str(path)}


def base_row(item: dict[str, Any], prompt: str, text: str | None, ctx: CallContext | None) -> dict[str, Any]:
    """结果行的公共字段。

    与冻结脚本相比这里**不再复制整条原始 item**（BC-04）：原 item 靠 id 关联回
    QA 文件即可，复制它会让 trajectory 每条结果膨胀到 50 KB。导出兼容格式时再合并回去。

    新增 ``frames_used`` 与 ``usage``（BC-09 的管道改造）：冻结脚本把
    ``call_vlm`` 返回的原始响应直接丢弃，导致事后完全无法复盘模型看了多少帧。
    """
    row: dict[str, Any] = {
        "id": str(item.get("id")),
        "prompt": prompt,
        "model_output": text,
    }
    if ctx is not None:
        if ctx.frames_used:
            row["frames_used"] = dict(ctx.frames_used)
        if ctx.usage:
            row["usage"] = dict(ctx.usage)
        if ctx.media_transforms:
            row["media_transforms"] = list(ctx.media_transforms)
    return row


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """按某字段分组统计正确率。冻结脚本里 by_action / by_choice / by_side 共用此形态。"""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for name, group in sorted(groups.items()):
        count = len(group)
        out[name] = {
            "total": count,
            "accuracy": sum(bool(r.get("correct")) for r in group) / count if count else 0.0,
        }
    return out
