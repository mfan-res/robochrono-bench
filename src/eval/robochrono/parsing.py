#!/usr/bin/env python3
# coding: utf-8
"""模型输出解析。

从 test/ 下八个冻结脚本里搬运而来，行为逐字保留，只做了两件事：

1. 归并重复实现。八个脚本里 ``strip_json_fence`` 完全一致，
   ``parse_model_answer`` 对六个选择题任务行为一致（``planning`` 那版多了一个
   未使用的局部变量 ``valid_ids``，且 ``extract_choice`` 里的
   ``option_id in valid_ids`` 恒为真，两者同源）。真正的差异只有两处：
   ``step_order`` 用 ``choices`` 而非 ``options``，且它的 ``normalize_text``
   保留连字符（选项形如 ``1-3-2-4``）。

2. 新增 BC-02 的思考块剥离与 JSON 兜底，**默认关闭**。关闭时输出与冻结脚本
   逐字节相同，这是 replay 回归的前提。

见 REFACTOR_PLAN.md 的 BC-02 / BC-03。
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

# BC-02：推理型模型即使设了 thinking=disabled 也可能把思考过程混在 content 里。
# 这些标签块在解析前被整体剥掉。
_REASONING_BLOCK = re.compile(
    r"<\s*(think|thinking|reasoning|thought|analysis)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# 只有开标签没有闭标签时（输出被截断），丢弃从开标签到文本末尾之前的部分是不安全的，
# 因为答案可能就在后面。这里只处理「开标签在最前面且后面还有内容」的情况。
_UNCLOSED_REASONING = re.compile(
    r"^\s*<\s*(?:think|thinking|reasoning|thought|analysis)\s*>",
    re.IGNORECASE,
)


def strip_json_fence(text: str) -> str:
    """剥掉 markdown 代码围栏。与冻结脚本逐字一致。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def strip_reasoning_blocks(text: str) -> str:
    """BC-02：剥掉 <think>...</think> 一类的思考块。"""
    cleaned = _REASONING_BLOCK.sub(" ", text)
    if _UNCLOSED_REASONING.match(cleaned):
        # 开标签未闭合：去掉标签本身，保留其后内容
        cleaned = _UNCLOSED_REASONING.sub("", cleaned, count=1)
    return cleaned.strip()


def iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """按出现顺序产出文本中所有可解码的顶层 JSON 对象。"""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            yield data


def first_json_object(text: str) -> dict[str, Any]:
    """取第一个可解码的 JSON 对象。trajectory 的现有行为，逐字保留。"""
    cleaned = strip_json_fence(text)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    for data in iter_json_objects(cleaned):
        return data
    return {}


def last_json_object(text: str) -> dict[str, Any]:
    """BC-02 兜底：取最后一个可解码的 JSON 对象。

    思考块里常出现看似答案的 JSON 片段，最终答案通常在末尾，
    所以兜底时取最后一个而不是第一个。
    """
    cleaned = strip_json_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    found: dict[str, Any] = {}
    for data in iter_json_objects(cleaned):
        found = data
    return found


def normalize_text(text: str, keep_hyphen: bool = False) -> str:
    """小写、去标点、压缩空白。

    ``keep_hyphen`` 对应 step_order 的变体 —— 它的选项是 ``1-3-2-4`` 这样的序列，
    连字符不能被当成标点去掉。
    """
    text = text.lower().strip()
    pattern = r"[^a-z0-9\s-]+" if keep_hyphen else r"[^a-z0-9\s]+"
    text = re.sub(pattern, " ", text)
    return " ".join(text.split())


def options_from_item(item: dict[str, Any], *, null_text_fix: bool = False) -> dict[str, str]:
    """从 ``item.options`` 取 {选项字母: 选项文本}。

    ``null_text_fix`` 即 BC-10，默认关闭以保持与冻结脚本一致。

    冻结脚本写的是 ``str(option.get("text", ""))`` —— 当 ``text`` 为 ``None`` 时
    得到字符串 ``"None"``，``normalize_text`` 后成为 ``"none"``。``left_right`` 和
    ``image_in_video`` 的选项是图片，``text`` 全为 ``None``，于是任何含 "none" 的
    模型输出都会匹配上**第一个选项**。而这两个任务恰好都有 none-option，
    「none of these」正是模型表达该选项最自然的说法，会被系统性误判。

    打开 ``null_text_fix`` 后，``text`` 为 ``None`` 的选项不参与文本匹配。
    """
    options: dict[str, str] = {}
    for option in item.get("options", []):
        if not isinstance(option, dict) or option.get("id") is None:
            continue
        raw = option.get("text", "")
        if null_text_fix and raw is None:
            text = ""
        else:
            text = str(raw)
        options[str(option["id"]).upper()] = text
    return options


def choices_from_item(item: dict[str, Any], *, null_text_fix: bool = False) -> dict[str, str]:
    """step_order 的取法：优先 ``item.choices``，回落到 ``item.options``。"""
    choices = item.get("choices")
    if isinstance(choices, dict):
        return {str(k).upper(): str(v) for k, v in choices.items()}
    return options_from_item(item, null_text_fix=null_text_fix)


def extract_choice(text: str, options: dict[str, str], keep_hyphen: bool = False) -> str | None:
    """从自由文本里抽出一个选项字母。与冻结脚本逻辑一致。

    三级匹配：整串就是字母 → 文本里的孤立大写字母 → 选项文本出现在回答里。
    """
    valid_ids = set(options)
    normalized = str(text).strip().upper()
    if normalized in valid_ids:
        return normalized

    match = re.search(r"\b([A-Z])\b", normalized)
    if match and match.group(1) in valid_ids:
        return match.group(1)

    normalized_text = normalize_text(str(text), keep_hyphen)
    for option_id, option_text in options.items():
        candidate = normalize_text(option_text, keep_hyphen)
        if candidate and candidate in normalized_text:
            return option_id
    return None


_CHOICE_KEYS = ("choice", "answer", "option", "letter")


def _choice_from_payload(data: dict[str, Any], options: dict[str, str], keep_hyphen: bool) -> str | None:
    text = ""
    for key in _CHOICE_KEYS:
        value = data.get(key)
        if value:
            text = str(value)
            break
    return extract_choice(text, options, keep_hyphen)


def parse_choice_answer(
    text: str,
    options: dict[str, str],
    *,
    keep_hyphen: bool = False,
    strip_reasoning: bool = False,
) -> dict[str, Any]:
    """解析选择题回答。

    ``strip_reasoning`` 即 BC-02，默认关闭。关闭时行为与冻结脚本逐字一致。
    打开时先按原逻辑解析；只有原逻辑失败才启用剥离与 JSON 兜底，
    并在 ``parse_recovered`` 标记这一条是靠兜底救回来的。
    """
    baseline = _parse_choice_baseline(text, options, keep_hyphen)
    if baseline["choice"] is not None or not strip_reasoning:
        baseline["parse_recovered"] = False
        baseline["parse_ok"] = baseline["choice"] is not None
        return baseline

    stripped = strip_reasoning_blocks(text)
    recovered = _parse_choice_baseline(stripped, options, keep_hyphen)
    if recovered["choice"] is None:
        payload = last_json_object(stripped)
        if payload:
            recovered = {
                "choice": _choice_from_payload(payload, options, keep_hyphen),
                "parsed": payload,
            }

    recovered["parse_ok"] = recovered["choice"] is not None
    recovered["parse_recovered"] = recovered["choice"] is not None
    return recovered


def _parse_choice_baseline(text: str, options: dict[str, str], keep_hyphen: bool) -> dict[str, Any]:
    cleaned = strip_json_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {"choice": _choice_from_payload(data, options, keep_hyphen), "parsed": data}
    except json.JSONDecodeError:
        pass
    return {"choice": extract_choice(cleaned, options, keep_hyphen), "parsed": cleaned}
