#!/usr/bin/env python3
# coding: utf-8
"""Evaluate pickplace Time EQA with Zhipu GLM video models and compute tIoU.

Example:
    export ZHIPUAI_API_KEY="your-api-key"
    python pickplace/time_eqa_glm_test_multi.py \
        --input pickplace/time_eqa_first50.json \
        --video-dir /home/kewei/YWC/egodata/pickplace/video \
        --output pickplace/time_eqa_glm_results.json \
        --limit 5
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any
import requests
from requests import HTTPError

from vlm_api import call_vlm, runtime_config, task_config


API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-5v-turbo"
DEFAULT_INPUT = Path("/home/kewei/YWC/egodata/pickplace/time_eqa_first50.json")
DEFAULT_VIDEO_DIR = Path("/home/kewei/YWC/egodata/pickplace/video")
DEFAULT_OUTPUT = Path("/home/kewei/YWC/egodata/pickplace/time_eqa_glm_results.json")
DEFAULT_VIDEO_EXTS = ".mp4,.webm,.mov,.mkv,.avi"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_checkpoint{output_path.suffix}")


def video_to_api_url(video_path: Path) -> str:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    with video_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_video_path(video_id: str, video_dir: Path, video_exts: tuple[str, ...]) -> Path:
    for ext in video_exts:
        candidate = video_dir / f"{video_id}{ext}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    for ext in video_exts:
        matches = sorted(video_dir.rglob(f"{video_id}{ext}"))
        for match in matches:
            if match.is_file() and match.stat().st_size > 0:
                return match
    raise FileNotFoundError(f"Cannot find video for video_id={video_id} under {video_dir}")


def video_paths_for_item(item: dict[str, Any], video_dir: Path, video_exts: tuple[str, ...]) -> list[Path]:
    input_data = item.get("input", {})
    if isinstance(input_data, dict):
        video_paths = input_data.get("video_paths")
        if isinstance(video_paths, list) and video_paths:
            return [Path(str(path)) for path in video_paths]

        videos = input_data.get("videos")
        if isinstance(videos, dict) and videos:
            paths = [
                Path(str(row["video_path"]))
                for row in videos.values()
                if isinstance(row, dict) and row.get("video_path")
            ]
            if paths:
                return paths

        if input_data.get("video_path"):
            return [Path(str(input_data["video_path"]))]

    return [find_video_path(str(item["video_id"]), video_dir, video_exts)]


def seconds_to_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, rem = divmod(milliseconds, 3600_000)
    minutes, rem = divmod(rem, 60_000)
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


def parse_interval_text(text: str) -> tuple[float, float]:
    cleaned = text.strip()
    cleaned = cleaned.replace("–", "-").replace("—", "-").replace("到", "-")
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

    match = re.search(
        r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?)"
        r"\s*(?:-|,|to)\s*"
        r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?)",
        cleaned,
        flags=re.I,
    )
    if not match:
        raise ValueError(f"Cannot parse interval from model output: {text!r}")
    return parse_time_value(match.group(1)), parse_time_value(match.group(2))


def strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_interval_row(row: Any) -> tuple[float, float]:
    if isinstance(row, dict):
        if "start" in row and "end" in row:
            return parse_time_value(str(row["start"])), parse_time_value(str(row["end"]))
        if "start_time" in row and "end_time" in row:
            return parse_time_value(str(row["start_time"])), parse_time_value(str(row["end_time"]))
        if "answer" in row:
            return parse_interval_text(str(row["answer"]))
        if "interval" in row:
            return parse_interval_text(str(row["interval"]))
        if "timestamp" in row:
            return parse_interval_text(str(row["timestamp"]))
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
    cleaned = strip_json_fence(text)
    predictions: dict[str, dict[str, Any]] = {}

    try:
        data = json.loads(cleaned)
        rows = rows_from_multi_json(data)
        if isinstance(rows, dict):
            for item_id in question_ids:
                if item_id not in rows:
                    continue
                pred_start, pred_end = parse_interval_row(rows[item_id])
                predictions[item_id] = {
                    "pred_start": pred_start,
                    "pred_end": pred_end,
                    "model_answer": rows[item_id],
                }
            return predictions

        for index, row in enumerate(rows):
            item_id = row_id(row)
            if item_id is None and index < len(question_ids):
                item_id = question_ids[index]
            if item_id not in question_ids:
                continue
            pred_start, pred_end = parse_interval_row(row)
            predictions[item_id] = {
                "pred_start": pred_start,
                "pred_end": pred_end,
                "model_answer": row,
            }
        return predictions
    except json.JSONDecodeError:
        pass

    id_pattern = "|".join(re.escape(item_id) for item_id in sorted(question_ids, key=len, reverse=True))
    if id_pattern:
        line_pattern = re.compile(
            rf"({id_pattern}).*?"
            r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?)"
            r"\s*(?:-|,|to)\s*"
            r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?)",
            flags=re.I,
        )
        for match in line_pattern.finditer(cleaned.replace("–", "-").replace("—", "-").replace("到", "-")):
            item_id = match.group(1)
            predictions[item_id] = {
                "pred_start": parse_time_value(match.group(2)),
                "pred_end": parse_time_value(match.group(3)),
                "model_answer": match.group(0),
            }

    if predictions:
        return predictions

    raise ValueError(f"Cannot parse multi-answer model output: {text!r}")


def temporal_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0


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


def build_multi_prompt(items: list[dict[str, Any]]) -> str:
    question_lines = []
    for index, item in enumerate(items, 1):
        item_id = str(item["id"])
        question = str(item.get("Q") or item.get("question"))
        question_lines.append(
            json.dumps({"index": index, "id": item_id, "question": question}, ensure_ascii=False)
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


def call_glm(
    api_key: str,
    video_paths: list[Path],
    prompt: str,
    model: str,
    temperature: float,
    thinking: str,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    content = [
        {"type": "video_url", "video_url": {"url": video_to_api_url(video_path)}}
        for video_path in video_paths
    ]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": temperature,
        "thinking": {"type": thinking},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                wait = min(60, 2 ** attempt)
                print(f"  HTTP {response.status_code}, retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == max_retries:
                raise
            wait = min(60, 2 ** attempt)
            print(f"  connection error (attempt {attempt}/{max_retries}), retrying in {wait}s: {exc}", flush=True)
            time.sleep(wait)
        except HTTPError as exc:
            try:
                detail = json.dumps(response.json(), ensure_ascii=False, indent=2)
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"GLM API request failed: {response.status_code} {response.reason}\n{detail}"
            ) from exc

    raise RuntimeError("GLM API request failed without response")


def video_parts_for_group(video_paths: list[Path], prompt: str) -> list[dict[str, Any]]:
    return [{"type": "video", "path": str(video_path)} for video_path in video_paths] + [
        {"type": "text", "text": prompt}
    ]


def extract_message_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected API response: {response}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def load_existing_results(output_path: Path, checkpoint_path: Path, overwrite: bool) -> dict[str, dict[str, Any]]:
    if overwrite:
        return {}
    source = output_path if output_path.exists() else checkpoint_path
    if not source.exists():
        return {}
    data = load_json(source)
    rows = data.get("results", []) if isinstance(data, dict) else []
    return {str(row["id"]): row for row in rows if isinstance(row, dict) and row.get("id")}


def is_finished(row: dict[str, Any] | None) -> bool:
    return isinstance(row, dict) and isinstance(row.get("pred_start"), (int, float)) and isinstance(row.get("pred_end"), (int, float))


def group_items_by_video(items: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        video_id = str(item["video_id"])
        groups.setdefault(video_id, []).append(item)
    return list(groups.items())


def limit_items_by_video(items: list[dict[str, Any]], video_limit: int | None) -> list[dict[str, Any]]:
    if video_limit is None:
        return items
    if video_limit < 0:
        raise ValueError("--limit must be non-negative")

    limited_items: list[dict[str, Any]] = []
    for _, group_items in group_items_by_video(items)[:video_limit]:
        limited_items.extend(group_items)
    return limited_items


def summarize(results: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    values = [float(row.get("tIoU", 0.0)) for row in results]
    answered_rows = [
        row
        for row in results
        if isinstance(row.get("pred_start"), (int, float)) and isinstance(row.get("pred_end"), (int, float))
    ]
    overlap_recalls = [float(row.get("overlap_recall", 0.0)) for row in answered_rows]
    abs_start_errors = [float(row.get("abs_start_error", 0.0)) for row in answered_rows]
    abs_end_errors = [float(row.get("abs_end_error", 0.0)) for row in answered_rows]
    return {
        "total": len(results),
        "answered": len(answered_rows),
        "errors": sum(1 for row in results if row.get("error")),
        "mean_tIoU": sum(values) / len(values) if values else 0.0,
        "tIoU@0.3": sum(value >= 0.3 for value in values) / len(values) if values else 0.0,
        "tIoU@0.5": sum(value >= 0.5 for value in values) / len(values) if values else 0.0,
        "tIoU@0.7": sum(value >= 0.7 for value in values) / len(values) if values else 0.0,
        "center_inside_acc": (
            sum(bool(row.get("center_inside")) for row in answered_rows) / len(answered_rows)
            if answered_rows
            else 0.0
        ),
        "pointing_acc": (
            sum(bool(row.get("pointing")) for row in answered_rows) / len(answered_rows)
            if answered_rows
            else 0.0
        ),
        "mean_overlap_recall": (
            sum(overlap_recalls) / len(overlap_recalls) if overlap_recalls else 0.0
        ),
        "mean_abs_start_error": (
            sum(abs_start_errors) / len(abs_start_errors) if abs_start_errors else 0.0
        ),
        "mean_abs_end_error": (
            sum(abs_end_errors) / len(abs_end_errors) if abs_end_errors else 0.0
        ),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pickplace Time EQA with Zhipu GLM and compute tIoU.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-exts", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of videos, not questions.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_defaults = task_config(args.config, "time")
    args.input = args.input or Path(task_defaults.get("input") or DEFAULT_INPUT)
    args.video_dir = args.video_dir or Path(task_defaults.get("video_dir") or DEFAULT_VIDEO_DIR)
    args.video_exts = args.video_exts or str(task_defaults.get("video_exts") or DEFAULT_VIDEO_EXTS)
    args.output = args.output or Path(task_defaults.get("output") or DEFAULT_OUTPUT)
    runtime = runtime_config(
        config_path=args.config,
        provider_name=args.provider,
        default_model=DEFAULT_MODEL,
        default_api_url=API_URL,
        default_api_key_env="ZHIPUAI_API_KEY",
        cli_api_key=args.api_key,
        cli_model=args.model,
        cli_temperature=args.temperature,
        cli_thinking=args.thinking,
        cli_timeout=args.timeout,
        cli_max_retries=args.max_retries,
    )

    data = load_json(args.input)
    items = data.get("items", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Input must be a list or contain an `items` list")
    total_input_items = len(items)
    total_input_videos = len(group_items_by_video(items))
    items = limit_items_by_video(items, args.limit)

    video_exts = tuple(ext.strip() for ext in args.video_exts.split(",") if ext.strip())
    checkpoint_path = checkpoint_path_for(args.output)
    results_by_id = load_existing_results(args.output, checkpoint_path, args.overwrite)
    pending = [item for item in items if not is_finished(results_by_id.get(str(item.get("id"))))]

    print(f"Input items: {total_input_items}")
    print(f"Input videos: {total_input_videos}")
    print(f"Selected items: {len(items)}")
    print(f"Selected videos: {len(group_items_by_video(items))}")
    print(f"Existing results: {len(results_by_id)}")
    print(f"Pending: {len(pending)}")
    print(f"Provider: {runtime['provider']}")
    print(f"Model: {runtime['model']}")
    print(f"API URL: {runtime['api_url']}")

    pending_groups = group_items_by_video(pending)
    print(f"Pending videos: {len(pending_groups)}")

    started_at = time.perf_counter()
    for group_index, (video_id, group_items) in enumerate(pending_groups, 1):
        prompt = build_multi_prompt(group_items)
        question_ids = [str(item["id"]) for item in group_items]
        group_started = time.perf_counter()

        print(
            f"[{group_index}/{len(pending_groups)}] {video_id} questions={len(group_items)}",
            flush=True,
        )
        try:
            video_paths = video_paths_for_item(group_items[0], args.video_dir, video_exts)
            print(f"  videos={len(video_paths)}", flush=True)
            for video_path in video_paths:
                print(f"  video_path={video_path}", flush=True)
            raw, model_text = call_vlm(runtime, video_parts_for_group(video_paths, prompt))
            predictions = parse_multi_interval_text(model_text, question_ids)
            group_seconds = round(time.perf_counter() - group_started, 3)

            for item in group_items:
                item_id = str(item["id"])
                question = str(item.get("Q") or item.get("question"))
                gt_start = float(item["answer_seconds"]["start"])
                gt_end = float(item["answer_seconds"]["end"])
                prediction = predictions.get(item_id)

                if prediction is None:
                    result = {
                        **item,
                        "video_path": str(video_paths[0]),
                        "video_paths": [str(path) for path in video_paths],
                        "prompt": prompt,
                        "pred_start": None,
                        "pred_end": None,
                        "predicted_answer": None,
                        "model_output": model_text,
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
                        "error": f"missing answer for item id {item_id}",
                        "timing": {"group_seconds": group_seconds},
                    }
                else:
                    pred_start = parse_time_value(str(prediction["pred_start"]))
                    pred_end = parse_time_value(str(prediction["pred_end"]))
                    metrics = temporal_metrics(pred_start, pred_end, gt_start, gt_end)
                    result = {
                        **item,
                        "video_path": str(video_paths[0]),
                        "video_paths": [str(path) for path in video_paths],
                        "prompt": prompt,
                        "pred_start": pred_start,
                        "pred_end": pred_end,
                        "predicted_answer": f"{seconds_to_timestamp(pred_start)}-{seconds_to_timestamp(pred_end)}",
                        "model_output": model_text,
                        "model_answer": prediction.get("model_answer"),
                        **metrics,
                        "timing": {"group_seconds": group_seconds},
                    }
                results_by_id[item_id] = result
                print(f"  {item_id} {question} tIoU={float(result.get('tIoU', 0.0)):.4f}", flush=True)
        except Exception as exc:
            group_seconds = round(time.perf_counter() - group_started, 3)
            print(f"  group error: {exc}", flush=True)
            for item in group_items:
                item_id = str(item["id"])
                result = {
                    **item,
                    "prompt": prompt,
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
                    "error": str(exc),
                    "timing": {"group_seconds": group_seconds},
                }
                results_by_id[item_id] = result

        results = [results_by_id[str(item["id"])] for item in items if str(item.get("id")) in results_by_id]
        summary = summarize(results, time.perf_counter() - started_at)
        print(f"  mean={summary['mean_tIoU']:.4f} answered={summary['answered']}/{summary['total']}", flush=True)

        checkpoint_data = {
            "input": str(args.input),
            "video_dir": str(args.video_dir),
            "provider": runtime["provider"],
            "model": runtime["model"],
            "api_url": runtime["api_url"],
            "prompt_note": "Each result item contains the exact per-video prompt sent to the model in the `prompt` field.",
            "results": results,
            "summary": summary,
        }
        save_json(checkpoint_path, checkpoint_data)

    results = [results_by_id[str(item["id"])] for item in items if str(item.get("id")) in results_by_id]
    output_data = {
        "input": str(args.input),
        "video_dir": str(args.video_dir),
        "provider": runtime["provider"],
        "model": runtime["model"],
        "api_url": runtime["api_url"],
        "prompt_note": "Each result item contains the exact per-video prompt sent to the model in the `prompt` field.",
        "results": results,
        "summary": summarize(results, time.perf_counter() - started_at),
    }
    save_json(args.output, output_data)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(json.dumps(output_data["summary"], ensure_ascii=False, indent=2))
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
