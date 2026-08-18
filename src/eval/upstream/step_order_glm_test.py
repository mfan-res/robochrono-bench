#!/usr/bin/env python3
# coding: utf-8
"""Evaluate step-order VQA samples with a VLM API.

Example:
    export ZHIPUAI_API_KEY="your-api-key"
    python /home/tianhao/NAS/lzm/egocentric/test/step_order_glm_test.py \
        --input /home/tianhao/NAS/lzm/egocentric/vqa_step_order_left_eye/step_order_vqa.json \
        --output /home/tianhao/NAS/lzm/egocentric/vqa_step_order_left_eye/step_order_glm_results.json \
        --limit 10
"""

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
DEFAULT_INPUT = Path("/home/tianhao/NAS/lzm/egocentric/vqa_step_order_left_eye/step_order_vqa.json")
DEFAULT_OUTPUT = Path("/home/tianhao/NAS/lzm/egocentric/vqa_step_order_left_eye/step_order_glm_results.json")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_checkpoint{output_path.suffix}")


def strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]+", " ", text)
    return " ".join(text.split())


def load_items(input_path: Path) -> list[dict[str, Any]]:
    data = load_json(input_path)
    items = data.get("items", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Input must be a list or contain an `items` list")
    return [item for item in items if isinstance(item, dict)]


def resolve_media_path(path_text: str, input_path: Path) -> Path:
    path = Path(str(path_text))
    if path.exists():
        return path

    input_dir = input_path.parent
    if path.is_absolute():
        parts = path.parts
        if input_dir.name in parts:
            start = parts.index(input_dir.name) + 1
            relocated = input_dir.joinpath(*parts[start:])
            if relocated.exists():
                return relocated
        for anchor in ("initial_images", "images", "montages", "step_order"):
            if anchor in parts:
                relocated = input_dir.joinpath(*parts[parts.index(anchor) :])
                if relocated.exists():
                    return relocated
        return path

    return (input_path.parent / path).resolve()


def image_paths_for_item(item: dict[str, Any], input_path: Path) -> list[Path]:
    paths: list[Path] = []
    for key in ("initial_image", "image"):
        if item.get(key):
            paths.append(resolve_media_path(str(item[key]), input_path))

    input_data = item.get("input", {})
    if isinstance(input_data, dict):
        for key in ("initial_image", "image", "image_path"):
            if input_data.get(key):
                candidate = resolve_media_path(str(input_data[key]), input_path)
                if candidate not in paths:
                    paths.append(candidate)
        image_paths = input_data.get("image_paths")
        if isinstance(image_paths, list):
            for image_path in image_paths:
                candidate = resolve_media_path(str(image_path), input_path)
                if candidate not in paths:
                    paths.append(candidate)

    if len(paths) < 2:
        raise ValueError(f"Cannot find initial image and montage image for item {item.get('id')}")
    for path in paths[:2]:
        if not path.exists():
            raise FileNotFoundError(path)
    return paths[:2]


def choices_for_item(item: dict[str, Any]) -> dict[str, str]:
    choices = item.get("choices")
    if isinstance(choices, dict):
        return {str(key).upper(): str(value) for key, value in choices.items()}

    options = item.get("options")
    if isinstance(options, list):
        return {
            str(option.get("id")).upper(): str(option.get("text"))
            for option in options
            if isinstance(option, dict) and option.get("id") is not None
        }
    return {}


def valid_option_ids(item: dict[str, Any]) -> set[str]:
    return set(choices_for_item(item))


def build_prompt(item: dict[str, Any]) -> str:
    question = str(item.get("Q") or item.get("question") or "").strip()
    choices = choices_for_item(item)
    if choices and "Options:" not in question:
        option_lines = "\n".join(f"{label}. {text}" for label, text in sorted(choices.items()))
        question = f"{question}\nOptions:\n{option_lines}"

    return f"""You are solving a robot manipulation step-order VQA task.

You will receive two images in this order:
1. The initial state image.
2. A montage of shuffled result-state images labeled Image 1, Image 2, etc.

Choose the option whose sequence puts the numbered result-state images in the correct chronological operation order after the initial state.
Choose exactly one option letter from the provided options. Do not invent a new option.

Question:
{question}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter, e.g. A>",
  "reason": "<brief visual reason>"
}}
"""


def image_parts_for_item(image_paths: list[Path], prompt: str) -> list[dict[str, Any]]:
    return [{"type": "image", "path": str(image_path)} for image_path in image_paths] + [
        {"type": "text", "text": prompt}
    ]


def extract_choice(text: str, item: dict[str, Any]) -> str | None:
    valid_ids = valid_option_ids(item)
    normalized = str(text).strip().upper()
    if normalized in valid_ids:
        return normalized

    match = re.search(r"\b([A-Z])\b", normalized)
    if match and match.group(1) in valid_ids:
        return match.group(1)

    normalized_text = normalize_text(str(text))
    choices = choices_for_item(item)
    for option_id, option_text in choices.items():
        if normalize_text(option_text) and normalize_text(option_text) in normalized_text:
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
            return {"choice": extract_choice(choice_text, item), "parsed": data}
    except json.JSONDecodeError:
        pass
    return {"choice": extract_choice(cleaned, item), "parsed": cleaned}


def expected_choice(item: dict[str, Any]) -> str:
    return str(item.get("answer") or item.get("A") or "").upper()


def score_prediction(item: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected = expected_choice(item)
    predicted = prediction.get("choice")
    return {
        "expected_choice": expected,
        "expected_answer_order": item.get("answer_order") or item.get("answer_text"),
        "pred_choice": predicted,
        "correct": predicted == expected,
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
    return {
        "total": total,
        "answered": len(answered),
        "errors": sum(1 for row in results if row.get("error")),
        "accuracy": sum(bool(row.get("correct")) for row in results) / total if total else 0.0,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate step-order VQA with a VLM API.")
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

    task_defaults = task_config(args.config, "step_order")
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

    items = load_items(args.input)
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

        try:
            image_paths = image_paths_for_item(item, args.input)
            print(f"[{index}/{len(pending)}] {item_id} images={len(image_paths)}", flush=True)
            _, model_text = call_vlm(runtime, image_parts_for_item(image_paths, prompt))
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
