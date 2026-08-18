#!/usr/bin/env python3
# coding: utf-8
"""Evaluate image-in-video VQA with VLM video+image models."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from vlm_api import call_vlm, runtime_config, task_config


API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-5v-turbo"
DEFAULT_INPUT = Path("/home/kewei/YWC/egodata/pickplace/image_in_video_vqa.json")
DEFAULT_OUTPUT = Path("/home/kewei/YWC/egodata/pickplace/image_in_video_glm_results.json")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_checkpoint{output_path.suffix}")


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


def resolve_media_path(path_text: str, input_path: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else input_path.parent / path


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
        if option_id in valid_ids and option_text and option_text in normalized_text:
            return option_id
    return None


def parse_model_answer(text: str, item: dict[str, Any]) -> dict[str, Any]:
    cleaned = strip_json_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            choice_text = str(data.get("choice") or data.get("answer") or data.get("option") or data.get("letter") or "")
            return {"choice": extract_choice(choice_text, item), "parsed": data}
    except json.JSONDecodeError:
        pass
    return {"choice": extract_choice(cleaned, item), "parsed": cleaned}


def item_question_text(item: dict[str, Any]) -> str:
    question = str(item.get("Q") or item.get("question") or "")
    return question.split("\nOptions:", 1)[0].strip()


def build_prompt(item: dict[str, Any]) -> str:
    option_ids = ", ".join(sorted(valid_option_ids(item)))
    return f"""You are answering a visual matching question for a robot manipulation video clip.

First inspect the left-eye video clip, then inspect each labeled option image.
Choose the single option letter whose image appears in the video clip.
If none of the listed images appear in the clip, choose the option labeled "All other options are wrong."

Question:
{item_question_text(item)}

Valid option letters: {option_ids}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter>",
  "reason": "<brief visual reason>"
}}
"""


def content_for_item(item: dict[str, Any], prompt: str, input_path: Path) -> list[dict[str, Any]]:
    input_data = item.get("input", {})
    clip_path_text = str(input_data.get("clip_path") or input_data.get("video_path") or "")
    if not clip_path_text:
        raise ValueError(f"Item {item.get('id')} has no input.clip_path")
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Left-eye video clip:"},
        {"type": "video", "path": str(resolve_media_path(clip_path_text, input_path))},
        {"type": "text", "text": "Candidate option images:"},
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
                {"type": "image", "path": str(resolve_media_path(str(image_path), input_path))},
            ]
        )

    content.append({"type": "text", "text": prompt})
    return content


def score_prediction(item: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected_choice = str(item.get("answer") or item.get("A") or "").upper()
    return {
        "expected_choice": expected_choice,
        "expected_answer_text": item.get("answer_text"),
        "expected_category": item.get("answer_category"),
        "pred_choice": prediction.get("choice"),
        "correct": prediction.get("choice") == expected_choice,
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
    by_distractor_type: dict[str, int] = {}
    for row in results:
        chosen = str(row.get("pred_choice") or "")
        for option in row.get("options", []):
            if str(option.get("id")).upper() == chosen:
                key = str(option.get("distractor_type") or "correct")
                by_distractor_type[key] = by_distractor_type.get(key, 0) + 1
                break
    return {
        "total": total,
        "answered": len(answered),
        "errors": sum(1 for row in results if row.get("error")),
        "accuracy": sum(bool(row.get("correct")) for row in results) / total if total else 0.0,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "chosen_option_type_counts": by_distractor_type,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate image-in-video VQA with VLM APIs.")
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

    task_defaults = task_config(args.config, "image_in_video")
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
        image_count = sum(1 for option in item.get("options", []) if option.get("image_path"))
        print(f"[{index}/{len(pending)}] {item_id} video=1 images={image_count}", flush=True)
        try:
            raw, model_text = call_vlm(runtime, content_for_item(item, prompt, args.input))
            prediction = parse_model_answer(model_text, item)
            result = {
                **item,
                "prompt": prompt,
                "model_output": model_text,
                "model_prediction": prediction.get("parsed"),
                **score_prediction(item, prediction),
                "timing": {"seconds": round(time.perf_counter() - item_started, 3)},
            }
        except Exception as exc:
            result = {
                **item,
                "prompt": prompt,
                "model_output": None,
                "model_prediction": None,
                **score_prediction(item, {"choice": None}),
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
        save_json(
            checkpoint_path,
            {
                "input": str(args.input),
                "provider": runtime["provider"],
                "model": runtime["model"],
                "api_url": runtime["api_url"],
                "prompt_note": "Each result item contains the exact prompt sent to the model in the `prompt` field.",
                "results": results,
                "summary": summary,
            },
        )

    results = [results_by_id[str(item["id"])] for item in items if str(item.get("id")) in results_by_id]
    save_json(
        args.output,
        {
            "input": str(args.input),
            "provider": runtime["provider"],
            "model": runtime["model"],
            "api_url": runtime["api_url"],
            "prompt_note": "Each result item contains the exact prompt sent to the model in the `prompt` field.",
            "results": results,
            "summary": summarize(results, time.perf_counter() - started_at),
        },
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
