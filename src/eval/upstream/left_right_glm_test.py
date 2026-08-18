#!/usr/bin/env python3
# coding: utf-8
"""Evaluate left/right gripper-view matching VQA with Zhipu GLM image models.

Example:
    export ZHIPUAI_API_KEY="your-api-key"
    python pickplace/left_right_glm_test.py \
        --input pickplace/left_right_vqa.json \
        --output pickplace/left_right_glm_results.json \
        --limit 10
    python /home/llm/yyywv/test_vlm/workflow/test/time_eqa_glm_test_multi.py --config /home/llm/yyywv/test_vlm/workflow/test/config_test.json --provider local_qwen
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
DEFAULT_INPUT = Path("/home/kewei/YWC/egodata/pickplace/left_right_vqa.json")
DEFAULT_OUTPUT = Path("/home/kewei/YWC/egodata/pickplace/left_right_glm_results.json")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_checkpoint{output_path.suffix}")


def image_to_api_url(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if image_path.stat().st_size <= 0:
        raise ValueError(f"Empty image file: {image_path}")
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def valid_option_ids(item: dict[str, Any]) -> set[str]:
    return {str(option["id"]).upper() for option in item.get("options", [])}


def extract_choice(text: str, item: dict[str, Any]) -> str | None:
    valid_ids = valid_option_ids(item)
    normalized = str(text).strip().upper()
    if normalized in valid_ids:
        return normalized

    match = re.search(r"\b([A-Z])\b", normalized)
    if match and match.group(1) in valid_ids:
        return match.group(1)

    normalized_text = normalize_text(str(text))
    for option in item.get("options", []):
        option_id = str(option["id"]).upper()
        option_text = normalize_text(str(option.get("text", "")))
        if option_text and option_text in normalized_text:
            return option_id
    return None


def parse_model_answer(text: str, item: dict[str, Any]) -> dict[str, Any]:
    cleaned = strip_json_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            choice_text = str(
                data.get("choice")
                or data.get("answer")
                or data.get("option")
                or data.get("letter")
                or ""
            )
            return {
                "choice": extract_choice(choice_text, item),
                "parsed": data,
            }
    except json.JSONDecodeError:
        pass

    return {
        "choice": extract_choice(cleaned, item),
        "parsed": cleaned,
    }


def item_question_text(item: dict[str, Any]) -> str:
    question = str(item.get("Q") or item.get("question") or "")
    return question.split("\nOptions:", 1)[0].strip()


def content_for_item(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    input_data = item.get("input", {})
    head_image_path = Path(str(input_data.get("image_path") or input_data.get("head_image", {}).get("image_path")))
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Head camera image:"},
        {"type": "image", "path": str(head_image_path)},
        {"type": "text", "text": "Candidate options:"},
    ]

    for option in item.get("options", []):
        option_id = str(option["id"]).upper()
        if option.get("is_none_option") or option.get("type") == "none":
            content.append({"type": "text", "text": f"Option {option_id}: {option.get('text')}"})
            continue
        image_path = option.get("image_path")
        if not image_path:
            raise ValueError(f"Option {option_id} in {item.get('id')} has no image_path")
        content.extend(
            [
                {"type": "text", "text": f"Option {option_id} image:"},
                {"type": "image", "path": str(image_path)},
            ]
        )

    content.append({"type": "text", "text": prompt})
    return content


def build_prompt(item: dict[str, Any]) -> str:
    question = item_question_text(item)
    option_ids = ", ".join(sorted(valid_option_ids(item)))
    return f"""You are answering a visual matching question for a robot manipulation episode.

The first image is from the head camera. The following labeled option images are candidate gripper-camera views at the same moment or distractors.
Choose the single option letter that shows the requested gripper camera's view for the head-camera moment.
If none of the listed images match, choose the option labeled "All other options are wrong."

Question:
{question}

Valid option letters: {option_ids}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter>",
  "reason": "<brief visual reason>"
}}
"""


def call_glm(
    api_key: str,
    item: dict[str, Any],
    prompt: str,
    model: str,
    temperature: float,
    thinking: str,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content_for_item(item, prompt),
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


def score_prediction(item: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected_choice = str(item.get("answer") or item.get("A") or "").upper()
    pred_choice = prediction.get("choice")
    return {
        "expected_choice": expected_choice,
        "expected_answer_text": item.get("answer_text"),
        "expected_target_side": item.get("target_side"),
        "pred_choice": pred_choice,
        "correct": pred_choice == expected_choice,
    }


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
    return isinstance(row, dict) and bool(row.get("model_output")) and not row.get("error")


def summarize(results: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    total = len(results)
    answered = [row for row in results if row.get("model_output")]
    by_side: dict[str, list[dict[str, Any]]] = {}
    by_choice: dict[str, list[dict[str, Any]]] = {}
    by_distractor_type: dict[str, int] = {}

    for row in results:
        side = str(row.get("expected_target_side") or "unknown")
        by_side.setdefault(side, []).append(row)
        choice = str(row.get("expected_choice") or "unknown")
        by_choice.setdefault(choice, []).append(row)
        chosen = str(row.get("pred_choice") or "")
        for option in row.get("options", []):
            if str(option.get("id")).upper() == chosen:
                key = str(option.get("distractor_type") or "correct")
                by_distractor_type[key] = by_distractor_type.get(key, 0) + 1
                break

    summary: dict[str, Any] = {
        "total": total,
        "answered": len(answered),
        "errors": sum(1 for row in results if row.get("error")),
        "accuracy": sum(bool(row.get("correct")) for row in results) / total if total else 0.0,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "chosen_option_type_counts": by_distractor_type,
        "by_side": {},
        "by_choice": {},
    }
    for side, rows in sorted(by_side.items()):
        count = len(rows)
        summary["by_side"][side] = {
            "total": count,
            "accuracy": sum(bool(row.get("correct")) for row in rows) / count if count else 0.0,
        }
    for choice, rows in sorted(by_choice.items()):
        count = len(rows)
        summary["by_choice"][choice] = {
            "total": count,
            "accuracy": sum(bool(row.get("correct")) for row in rows) / count if count else 0.0,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate left/right gripper-view VQA with Zhipu GLM.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_defaults = task_config(args.config, "left_right")
    args.input = args.input or Path(task_defaults.get("input") or DEFAULT_INPUT)
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
    if args.limit is not None:
        items = items[: args.limit]

    checkpoint_path = checkpoint_path_for(args.output)
    results_by_id = load_existing_results(args.output, checkpoint_path, args.overwrite)
    pending = [item for item in items if not is_finished(results_by_id.get(str(item.get("id"))))]

    print(f"Input items: {len(items)}")
    print(f"Existing results: {len(results_by_id)}")
    print(f"Pending: {len(pending)}")
    print(f"Provider: {runtime['provider']}")
    print(f"Model: {runtime['model']}")
    print(f"API URL: {runtime['api_url']}")

    started_at = time.perf_counter()
    for index, item in enumerate(pending, 1):
        item_id = str(item["id"])
        prompt = build_prompt(item)
        item_started = time.perf_counter()
        image_count = 1 + sum(1 for option in item.get("options", []) if option.get("image_path"))

        print(f"[{index}/{len(pending)}] {item_id} images={image_count}", flush=True)
        try:
            raw, model_text = call_vlm(runtime, content_for_item(item, prompt))
            prediction = parse_model_answer(model_text, item)
            scores = score_prediction(item, prediction)
            result = {
                **item,
                "prompt": prompt,
                "model_output": model_text,
                "model_prediction": prediction.get("parsed"),
                **scores,
                "timing": {"seconds": round(time.perf_counter() - item_started, 3)},
            }
        except Exception as exc:
            scores = score_prediction(item, {"choice": None, "parsed": None})
            result = {
                **item,
                "prompt": prompt,
                "model_output": None,
                "model_prediction": None,
                **scores,
                "correct": False,
                "error": str(exc),
                "timing": {"seconds": round(time.perf_counter() - item_started, 3)},
            }
            print(f"  error: {exc}", flush=True)

        results_by_id[item_id] = result
        results = [results_by_id[str(item["id"])] for item in items if str(item.get("id")) in results_by_id]
        summary = summarize(results, time.perf_counter() - started_at)
        print(
            f"  correct={bool(result.get('correct'))} "
            f"choice={result.get('pred_choice')} expected={result.get('expected_choice')} "
            f"acc={summary['accuracy']:.4f}",
            flush=True,
        )

        checkpoint_data = {
            "input": str(args.input),
            "provider": runtime["provider"],
            "model": runtime["model"],
            "api_url": runtime["api_url"],
            "prompt_note": "Each result item contains the exact prompt sent to the model in the `prompt` field.",
            "results": results,
            "summary": summary,
        }
        save_json(checkpoint_path, checkpoint_data)

    results = [results_by_id[str(item["id"])] for item in items if str(item.get("id")) in results_by_id]
    output_data = {
        "input": str(args.input),
        "provider": runtime["provider"],
        "model": runtime["model"],
        "api_url": runtime["api_url"],
        "prompt_note": "Each result item contains the exact prompt sent to the model in the `prompt` field.",
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
