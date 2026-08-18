#!/usr/bin/env python3
# coding: utf-8
"""Time EQA：时间定位任务。

九个任务里唯一按视频分组提问的 —— 一次模型调用回答该 episode 的全部问题，
所以一个 Unit 对应多条结果行。它也是唯一送整段视频的任务，
抽帧策略争议全部集中在这里（见 eval/docs/frame_sampling_investigation.md，BC-09）。

prompt、区间解析、指标口径均从 test/time_eqa_glm_test_multi.py 逐字搬运。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import CallContext, Unit, base_row, text_part, video_part


# --------------------------------------------------------------------------
# 时间解析：逐字搬运
# --------------------------------------------------------------------------


def seconds_to_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, rem = divmod(milliseconds, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_time_value(value: str) -> float:
    value = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value)
    match = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", value)
    if not match:
        raise ValueError(f"Cannot parse time value: {value!r}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


_NUM = r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?"


def _normalize_dashes(text: str) -> str:
    return text.replace("–", "-").replace("—", "-").replace("到", "-")


def parse_interval_text(text: str) -> tuple[float, float]:
    cleaned = _normalize_dashes(text.strip())
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            if "start" in data and "end" in data:
                return parse_time_value(str(data["start"])), parse_time_value(str(data["end"]))
            if "start_time" in data and "end_time" in data:
                return parse_time_value(str(data["start_time"])), parse_time_value(str(data["end_time"]))
            if "answer" in data:
                return parse_interval_text(str(data["answer"]))
    except json.JSONDecodeError:
        pass
    match = re.search(rf"({_NUM})\s*(?:-|,|to)\s*({_NUM})", cleaned, flags=re.I)
    if not match:
        raise ValueError(f"Cannot parse interval from model output: {text!r}")
    return parse_time_value(match.group(1)), parse_time_value(match.group(2))


def parse_interval_row(row: Any) -> tuple[float, float]:
    if isinstance(row, dict):
        if "start" in row and "end" in row:
            return parse_time_value(str(row["start"])), parse_time_value(str(row["end"]))
        if "start_time" in row and "end_time" in row:
            return parse_time_value(str(row["start_time"])), parse_time_value(str(row["end_time"]))
        for key in ("answer", "interval", "timestamp"):
            if key in row:
                return parse_interval_text(str(row[key]))
    if isinstance(row, str):
        return parse_interval_text(row)
    raise ValueError(f"Cannot parse interval row: {row!r}")


def row_id(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    for key in ("id", "question_id", "item_id"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def rows_from_multi_json(data: Any) -> list[Any] | dict[str, Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("answers", "results", "items", "predictions"):
            if isinstance(data.get(key), list):
                return data[key]
        return data
    raise ValueError(f"Cannot parse multi-answer JSON: {data!r}")


def parse_multi_interval_text(text: str, question_ids: list[str]) -> dict[str, dict[str, Any]]:
    """把一次调用返回的多题答案拆回每题。三级回落：整体 JSON → 按 id 的正则行匹配 → 报错。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    predictions: dict[str, dict[str, Any]] = {}
    try:
        rows = rows_from_multi_json(json.loads(cleaned))
        if isinstance(rows, dict):
            for item_id in question_ids:
                if item_id not in rows:
                    continue
                start, end = parse_interval_row(rows[item_id])
                predictions[item_id] = {"pred_start": start, "pred_end": end, "model_answer": rows[item_id]}
            return predictions
        for index, row in enumerate(rows):
            item_id = row_id(row)
            if item_id is None and index < len(question_ids):
                item_id = question_ids[index]
            if item_id not in question_ids:
                continue
            start, end = parse_interval_row(row)
            predictions[item_id] = {"pred_start": start, "pred_end": end, "model_answer": row}
        return predictions
    except json.JSONDecodeError:
        pass

    id_pattern = "|".join(re.escape(i) for i in sorted(question_ids, key=len, reverse=True))
    if id_pattern:
        line_pattern = re.compile(
            rf"({id_pattern}).*?({_NUM})\s*(?:-|,|to)\s*({_NUM})", flags=re.I
        )
        for match in line_pattern.finditer(_normalize_dashes(cleaned)):
            predictions[match.group(1)] = {
                "pred_start": parse_time_value(match.group(2)),
                "pred_end": parse_time_value(match.group(3)),
                "model_answer": match.group(0),
            }
    if predictions:
        return predictions
    raise ValueError(f"Cannot parse multi-answer model output: {text!r}")


# --------------------------------------------------------------------------
# 指标：逐字搬运
# --------------------------------------------------------------------------


def temporal_metrics(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> dict[str, Any]:
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start

    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    gt_duration = max(0.0, gt_end - gt_start)
    pred_center = (pred_start + pred_end) / 2

    return {
        "tIoU": intersection / union if union > 0 else 0.0,
        "center_inside": gt_start <= pred_center <= gt_end,
        "overlap_recall": intersection / gt_duration if gt_duration > 0 else 0.0,
        "pointing": intersection > 0,
        "start_error": pred_start - gt_start,
        "end_error": pred_end - gt_end,
        "abs_start_error": abs(pred_start - gt_start),
        "abs_end_error": abs(pred_end - gt_end),
        "pred_center": pred_center,
        "intersection": intersection,
    }


_EMPTY_METRICS = {
    "pred_start": None,
    "pred_end": None,
    "predicted_answer": None,
    "tIoU": 0.0,
    "center_inside": False,
    "overlap_recall": 0.0,
    "pointing": False,
    "start_error": None,
    "end_error": None,
    "abs_start_error": None,
    "abs_end_error": None,
    "pred_center": None,
    "intersection": 0.0,
}


# --------------------------------------------------------------------------
# 任务
# --------------------------------------------------------------------------


def build_prompt(items: list[dict[str, Any]]) -> str:
    question_lines = []
    for index, item in enumerate(items, 1):
        question_lines.append(
            json.dumps(
                {
                    "index": index,
                    "id": str(item["id"]),
                    "question": str(item.get("Q") or item.get("question")),
                },
                ensure_ascii=False,
            )
        )
    questions = "\n".join(question_lines)
    return f"""You are answering temporal grounding questions about synchronized robot manipulation videos.

The inputs may contain multiple synchronized camera views of the same episode. Use all views together.
For each question below, find the full time interval where the robot performs the queried action.
Return the full action segment, not only the instant of contact, grasp, or release.
Use seconds from the start of the video.

Questions, one JSON object per line:
{questions}

Output JSON only. Do not use Markdown.
Return one answer for every question id, preserving the exact id strings.
Required schema:
{{
  "answers": [
    {{
      "id": "<question id>",
      "start": <start time in seconds>,
      "end": <end time in seconds>,
      "answer": "<HH:MM:SS.mmm-HH:MM:SS.mmm>",
      "reason": "<brief visual reason>"
    }}
  ]
}}
"""


def video_paths_for_item(item: dict[str, Any]) -> list[str]:
    data = item.get("input", {})
    if isinstance(data, dict):
        paths = data.get("video_paths")
        if isinstance(paths, list) and paths:
            return [str(p) for p in paths]
        videos = data.get("videos")
        if isinstance(videos, dict) and videos:
            collected = [str(row["video_path"]) for row in videos.values()
                         if isinstance(row, dict) and row.get("video_path")]
            if collected:
                return collected
        if data.get("video_path"):
            return [str(data["video_path"])]
    raise KeyError(f"item {item.get('id')} has no video path; video_dir fallback is not supported")


class TimeEqaTask:
    """按视频分组：一次调用回答一个 episode 的全部问题。"""

    name = "time"

    def __init__(self, **_flags: Any) -> None:
        # BC-02 对本任务不适用：它不解析选项字母，走的是区间正则回落。
        pass

    def units(self, items: list[dict[str, Any]]) -> list[Unit]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            groups.setdefault(str(item["video_id"]), []).append(item)
        return [Unit(key=video_id, items=group) for video_id, group in groups.items()]

    def parts(self, unit: Unit) -> list[dict[str, Any]]:
        paths = video_paths_for_item(unit.items[0])
        return [video_part(p) for p in paths] + [text_part(build_prompt(unit.items))]

    def rows(self, unit: Unit, text: str, ctx: CallContext) -> list[dict[str, Any]]:
        prompt = build_prompt(unit.items)
        question_ids = [str(i["id"]) for i in unit.items]
        paths = video_paths_for_item(unit.items[0])
        predictions = parse_multi_interval_text(text, question_ids)

        out: list[dict[str, Any]] = []
        for item in unit.items:
            item_id = str(item["id"])
            row = base_row(item, prompt, text, ctx)
            row["video_path"] = paths[0]
            row["video_paths"] = paths
            prediction = predictions.get(item_id)

            if prediction is None:
                row.update(_EMPTY_METRICS)
                row["error"] = f"missing answer for item id {item_id}"
                out.append(row)
                continue

            pred_start = parse_time_value(str(prediction["pred_start"]))
            pred_end = parse_time_value(str(prediction["pred_end"]))
            gt_start = float(item["answer_seconds"]["start"])
            gt_end = float(item["answer_seconds"]["end"])
            row["pred_start"] = pred_start
            row["pred_end"] = pred_end
            row["predicted_answer"] = f"{seconds_to_timestamp(pred_start)}-{seconds_to_timestamp(pred_end)}"
            row["model_answer"] = prediction.get("model_answer")
            row.update(temporal_metrics(pred_start, pred_end, gt_start, gt_end))
            out.append(row)
        return out

    def error_rows(self, unit: Unit, error: str) -> list[dict[str, Any]]:
        prompt = build_prompt(unit.items)
        out = []
        for item in unit.items:
            row = base_row(item, prompt, None, None)
            row.update(_EMPTY_METRICS)
            row["error"] = error
            out.append(row)
        return out

    def summarize(self, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        values = [float(r.get("tIoU", 0.0)) for r in rows]
        answered = [
            r for r in rows
            if isinstance(r.get("pred_start"), (int, float)) and isinstance(r.get("pred_end"), (int, float))
        ]
        overlap_recalls = [float(r.get("overlap_recall", 0.0)) for r in answered]
        abs_start = [float(r.get("abs_start_error", 0.0)) for r in answered]
        abs_end = [float(r.get("abs_end_error", 0.0)) for r in answered]

        def mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        return {
            "total": len(rows),
            "answered": len(answered),
            "errors": sum(1 for r in rows if r.get("error")),
            "mean_tIoU": mean(values),
            "tIoU@0.3": mean([float(v >= 0.3) for v in values]) if values else 0.0,
            "tIoU@0.5": mean([float(v >= 0.5) for v in values]) if values else 0.0,
            "tIoU@0.7": mean([float(v >= 0.7) for v in values]) if values else 0.0,
            "center_inside_acc": (
                sum(bool(r.get("center_inside")) for r in answered) / len(answered) if answered else 0.0
            ),
            "pointing_acc": (
                sum(bool(r.get("pointing")) for r in answered) / len(answered) if answered else 0.0
            ),
            "mean_overlap_recall": mean(overlap_recalls),
            "mean_abs_start_error": mean(abs_start),
            "mean_abs_end_error": mean(abs_end),
            "elapsed_seconds": round(elapsed, 3),
            # BC-03：本任务的「解析失败」= 该题没能从批量回答里取到区间
            "parse_failure_rate": (len(rows) - len(answered)) / len(rows) if rows else 0.0,
        }


def build(**flags: Any) -> TimeEqaTask:
    return TimeEqaTask(**flags)
