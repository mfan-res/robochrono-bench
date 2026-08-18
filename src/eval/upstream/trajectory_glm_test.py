#!/usr/bin/env python3
# coding: utf-8
"""Evaluate trajectory prediction QA with VLM models.

The model is asked to output ordered left/right gripper trajectory points.
The script compares predicted and reference trajectories with:

- Hausdorff distance
- discrete Frechet distance
- Chamfer distance
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_api import call_vlm, runtime_config, task_config


API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-5v-turbo"
DEFAULT_INPUT = Path("/home/kewei/YWC/egodata/pickplace/trajectory_qa_2d_first50_6move.json")
DEFAULT_OUTPUT = Path("/home/kewei/YWC/egodata/pickplace/trajectory_glm_results.json")
MAX_COORDINATE_RETRIES = 1


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_checkpoint{output_path.suffix}")


def image_label_from_key(key: Any) -> str:
    return str(key).split(".")[-1]


def primary_image_key_for_item(item: dict[str, Any], images: dict[Any, Any]) -> Any | None:
    for key in ("prediction_view_label", "primary_view_label", "view_label"):
        if item.get(key) in images:
            return item[key]
    for key in ("prediction_view", "primary_view", "view"):
        value = item.get(key)
        if value in images:
            return value
        label = image_label_from_key(value) if value else ""
        if label in images:
            return label
    for key in ("prediction_image", "main_image", "image"):
        if item.get(key):
            target = str(item[key])
            for image_key, path in images.items():
                if str(path) == target:
                    return image_key
    return None


def image_inputs_for_item(item: dict[str, Any]) -> list[dict[str, str]]:
    images = item.get("images")
    if isinstance(images, dict) and images:
        primary_key = primary_image_key_for_item(item, images)
        ordered_keys: list[Any] = []
        if primary_key is not None:
            ordered_keys.append(primary_key)
        ordered_keys.extend(key for key, _ in sorted(images.items()) if key != primary_key)
        return [
            {
                "label": image_label_from_key(key),
                "path": str(images[key]),
                "role": "primary" if key == primary_key else "context",
            }
            for key in ordered_keys
        ]
    if item.get("image"):
        return [{"label": "main_view", "path": str(item["image"]), "role": "primary"}]
    input_data = item.get("input", {})
    if isinstance(input_data, dict):
        if isinstance(input_data.get("image_paths"), list):
            return [
                {
                    "label": "main_view" if index == 0 else f"context_view_{index}",
                    "path": str(path),
                    "role": "primary" if index == 0 else "context",
                }
                for index, path in enumerate(input_data["image_paths"])
            ]
        if input_data.get("image_path"):
            return [{"label": "main_view", "path": str(input_data["image_path"]), "role": "primary"}]
    raise ValueError(f"Cannot find image paths for item {item.get('id')}")


def image_parts_for_item(image_inputs: list[dict[str, str]], prompt: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for image_input in image_inputs:
        label = image_input["label"]
        if image_input.get("role") == "primary":
            parts.append({"type": "text", "text": "Primary view image:"})
        else:
            parts.append({"type": "text", "text": f"Context view {label}:"})
        parts.append({"type": "image", "path": image_input["path"]})
    parts.append({"type": "text", "text": prompt})
    return parts


def strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def infer_dimension(item: dict[str, Any]) -> int:
    answer = item.get("answer") or item.get("A") or {}
    coordinate_frame = str(answer.get("coordinate_frame", "")).lower() if isinstance(answer, dict) else ""
    if "image" in coordinate_frame or "pixel" in coordinate_frame:
        return 2
    return 3


def expected_trajectory(item: dict[str, Any]) -> dict[str, list[list[float]]]:
    answer = item.get("answer") or item.get("A") or {}
    rows = answer.get("trajectory", []) if isinstance(answer, dict) else []
    left: list[list[float]] = []
    right: list[list[float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "left_gripper_uv" in row:
            l_uv = row.get("left_gripper_uv") or {}
            r_uv = row.get("right_gripper_uv") or {}
            if l_uv.get("valid") and l_uv.get("u") is not None and l_uv.get("v") is not None:
                left.append([float(l_uv["u"]), float(l_uv["v"])])
            if r_uv.get("valid") and r_uv.get("u") is not None and r_uv.get("v") is not None:
                right.append([float(r_uv["u"]), float(r_uv["v"])])
        else:
            if isinstance(row.get("left_gripper_xyz"), list):
                left.append([float(value) for value in row["left_gripper_xyz"]])
            if isinstance(row.get("right_gripper_xyz"), list):
                right.append([float(value) for value in row["right_gripper_xyz"]])
    return {"left_gripper": left, "right_gripper": right}


def path_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(
        math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(points[index], points[index + 1])))
        for index in range(len(points) - 1)
    )


def active_gripper_for_item(item: dict[str, Any], expected: dict[str, list[list[float]]] | None = None) -> str:
    answer = item.get("answer") or item.get("A") or {}
    for source in (item, answer if isinstance(answer, dict) else {}):
        active = source.get("active_gripper") if isinstance(source, dict) else None
        if active in {"left_gripper", "right_gripper", "both", "unknown"}:
            return str(active)
        metadata = source.get("active_gripper_metadata") if isinstance(source, dict) else None
        if isinstance(metadata, dict) and metadata.get("active_gripper") in {
            "left_gripper",
            "right_gripper",
            "both",
            "unknown",
        }:
            return str(metadata["active_gripper"])

    trajectories = expected if expected is not None else expected_trajectory(item)
    left_length = path_length(trajectories.get("left_gripper", []))
    right_length = path_length(trajectories.get("right_gripper", []))
    static_threshold = 0.01 if infer_dimension(item) == 3 else 1.0
    dominance_ratio = 2.0
    if left_length < static_threshold and right_length < static_threshold:
        return "unknown"
    if left_length >= static_threshold and left_length >= right_length * dominance_ratio:
        return "left_gripper"
    if right_length >= static_threshold and right_length >= left_length * dominance_ratio:
        return "right_gripper"
    return "both"


def grippers_to_score(active_gripper: str) -> list[str]:
    if active_gripper == "left_gripper":
        return ["left_gripper"]
    if active_gripper == "right_gripper":
        return ["right_gripper"]
    if active_gripper == "both":
        return ["left_gripper", "right_gripper"]
    return []


def image_size_from_file(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except (FileNotFoundError, OSError, ValueError):
        return None, None


def main_view_image_path_for_item(item: dict[str, Any]) -> Path | None:
    for key in ("prediction_image", "main_image", "image"):
        if item.get(key):
            return Path(str(item[key]))

    images = item.get("images")
    if isinstance(images, dict) and images:
        prediction_view = item.get("prediction_view") or item.get("primary_view")
        prediction_label = str(prediction_view).split(".")[-1] if prediction_view else ""
        for key in (prediction_view, prediction_label):
            if key and images.get(key):
                return Path(str(images[key]))
    return None


def image_size_from_intrinsics(item: dict[str, Any]) -> tuple[int | None, int | None]:
    answer = item.get("answer") or item.get("A") or {}
    if not isinstance(answer, dict):
        return None, None
    intrinsics = answer.get("camera_intrinsics") or {}
    if not isinstance(intrinsics, dict):
        return None, None
    width = intrinsics.get("width")
    height = intrinsics.get("height")
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        return None, None


def image_size_for_item(item: dict[str, Any]) -> tuple[int | None, int | None]:
    image_path = main_view_image_path_for_item(item)
    if image_path is not None:
        width, height = image_size_from_file(image_path)
        if width is not None and height is not None:
            return width, height
    return image_size_from_intrinsics(item)


def build_prompt(question: str, item: dict[str, Any]) -> str:
    dim = infer_dimension(item)
    expected = expected_trajectory(item)
    active_gripper = active_gripper_for_item(item, expected)
    if dim == 2:
        width, height = image_size_for_item(item)
        size_note = ""
        if width is not None and height is not None:
            size_note = (
                f" The main-view image size is {width} pixels wide and {height} pixels high; "
                f"the visible canvas spans 0 <= u < {width} and 0 <= v < {height}. "
                f"Any point with u < 0, u >= {width}, v < 0, or v >= {height} is invalid."
            )
        coordinate_note = (
            "Use image pixel coordinates [u, v] in the main-view image, where u increases right "
            f"and v increases down.{size_note} Coordinates must refer only to the first attached "
            "image / main view, not to a concatenated image, not to a resized image, and not to the "
            "side/wrist views. Do not use normalized coordinates."
        )
        example_point = "[123.4, 256.7]"
    else:
        coordinate_note = "Use 3D camera-frame coordinates in meters [x, y, z]."
        example_point = "[0.12, -0.03, 0.45]"

    scored_grippers = grippers_to_score(active_gripper)
    point_count = sum(len(expected[gripper]) for gripper in scored_grippers)
    if active_gripper in {"left_gripper", "right_gripper"}:
        gripper_instruction = (
            f"Predict ordered key trajectory points only for the active gripper: {active_gripper}."
        )
        count_instruction = f"Return approximately {point_count} {active_gripper} points if visible/available."
        schema_body = f'  "{active_gripper}": [{example_point}]'
    elif active_gripper == "both":
        gripper_instruction = "Both grippers are active. Predict ordered key trajectory points for both grippers."
        count_instruction = (
            f"Return approximately {len(expected['left_gripper'])} left-gripper points and "
            f"{len(expected['right_gripper'])} right-gripper points if visible/available."
        )
        schema_body = f'  "left_gripper": [{example_point}],\n  "right_gripper": [{example_point}]'
    else:
        gripper_instruction = (
            "The active gripper could not be determined from the reference metadata. "
            "Predict the gripper trajectory that is visibly performing the action."
        )
        count_instruction = "Return only the gripper trajectory that is performing the action."
        schema_body = f'  "left_gripper": [{example_point}]'
    return f"""You are evaluating a robot manipulation trajectory prediction task.

Question:
{question}

{gripper_instruction}
{coordinate_note}
{count_instruction}

Output JSON only. Do not use Markdown.
Required schema:
{{
{schema_body}
}}
"""


def coerce_point_list(value: Any, dim: int) -> list[list[float]]:
    if isinstance(value, dict):
        for key in ("points", "trajectory", "coordinates"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list):
        return []
    if len(value) >= dim and all(isinstance(value[index], (int, float)) for index in range(dim)):
        try:
            return [[float(value[index]) for index in range(dim)]]
        except (TypeError, ValueError):
            return []
    points: list[list[float]] = []
    for row in value:
        if isinstance(row, dict):
            if dim == 2:
                values = [row.get("u"), row.get("v")]
            else:
                values = [row.get("x"), row.get("y"), row.get("z")]
        else:
            values = row
        if not isinstance(values, list) or len(values) < dim:
            continue
        try:
            points.append([float(values[index]) for index in range(dim)])
        except (TypeError, ValueError):
            continue
    return points


def parse_first_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_json_fence(text)
    decoder = json.JSONDecoder()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def parse_model_answer(text: str, dim: int) -> dict[str, Any]:
    data = parse_first_json_object(text)
    left = (
        data.get("left_gripper")
        or data.get("left")
        or data.get("left_gripper_points")
        or data.get("left_trajectory")
        or []
    )
    right = (
        data.get("right_gripper")
        or data.get("right")
        or data.get("right_gripper_points")
        or data.get("right_trajectory")
        or []
    )
    return {
        "parsed": data,
        "trajectory": {
            "left_gripper": coerce_point_list(left, dim),
            "right_gripper": coerce_point_list(right, dim),
        },
    }


def out_of_bounds_points(
    prediction: dict[str, Any],
    width: int | None,
    height: int | None,
    grippers: list[str] | None = None,
) -> list[str]:
    if width is None or height is None:
        return []
    trajectory = prediction.get("trajectory", {})
    if not isinstance(trajectory, dict):
        return []
    invalid: list[str] = []
    for gripper in grippers or ["left_gripper", "right_gripper"]:
        points = trajectory.get(gripper, [])
        if not isinstance(points, list):
            continue
        for index, point in enumerate(points):
            if not isinstance(point, list) or len(point) < 2:
                continue
            u, v = point[0], point[1]
            if u < 0 or u >= width or v < 0 or v >= height:
                invalid.append(f"{gripper}[{index}]=[{round(float(u), 3)}, {round(float(v), 3)}]")
    return invalid


def build_retry_prompt(
    original_prompt: str,
    model_text: str,
    invalid_points: list[str],
    width: int,
    height: int,
) -> str:
    examples = ", ".join(invalid_points[:12])
    return f"""{original_prompt}

Your previous JSON answer used invalid 2D pixel coordinates outside the main-view image.
Main-view image size: width={width}, height={height}.
Valid coordinate range: 0 <= u < {width} and 0 <= v < {height}.
Invalid points found: {examples}

Previous answer:
{model_text}

Return a corrected JSON answer only. Every [u, v] point must be inside the valid coordinate range above.
"""


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def directed_hausdorff(a: list[list[float]], b: list[list[float]]) -> float | None:
    if not a or not b:
        return None
    return max(min(euclidean(x, y) for y in b) for x in a)


def hausdorff_distance(a: list[list[float]], b: list[list[float]]) -> float | None:
    forward = directed_hausdorff(a, b)
    backward = directed_hausdorff(b, a)
    if forward is None or backward is None:
        return None
    return max(forward, backward)


def chamfer_distance(a: list[list[float]], b: list[list[float]]) -> float | None:
    if not a or not b:
        return None
    a_to_b = sum(min(euclidean(x, y) for y in b) for x in a) / len(a)
    b_to_a = sum(min(euclidean(y, x) for x in a) for y in b) / len(b)
    return a_to_b + b_to_a


def discrete_frechet_distance(a: list[list[float]], b: list[list[float]]) -> float | None:
    if not a or not b:
        return None
    rows = len(a)
    cols = len(b)
    cache = [[-1.0 for _ in range(cols)] for _ in range(rows)]

    def compute(i: int, j: int) -> float:
        if cache[i][j] >= 0:
            return cache[i][j]
        dist = euclidean(a[i], b[j])
        if i == 0 and j == 0:
            value = dist
        elif i > 0 and j == 0:
            value = max(compute(i - 1, 0), dist)
        elif i == 0 and j > 0:
            value = max(compute(0, j - 1), dist)
        else:
            value = max(min(compute(i - 1, j), compute(i - 1, j - 1), compute(i, j - 1)), dist)
        cache[i][j] = value
        return value

    return compute(rows - 1, cols - 1)


def rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def flatten_points(trajectories: dict[str, list[list[float]]]) -> list[list[float]]:
    return trajectories.get("left_gripper", []) + trajectories.get("right_gripper", [])


def point_cloud_extent(points: list[list[float]]) -> float | None:
    if not points:
        return None
    dim = len(points[0])
    mins = [min(point[index] for point in points) for index in range(dim)]
    maxs = [max(point[index] for point in points) for index in range(dim)]
    return math.sqrt(sum((maxs[index] - mins[index]) ** 2 for index in range(dim)))


def score_tolerance_for_item(item: dict[str, Any], expected: dict[str, list[list[float]]]) -> float:
    if infer_dimension(item) == 2:
        width, height = image_size_for_item(item)
        if width is not None and height is not None:
            return 0.05 * math.hypot(width, height)
        extent = point_cloud_extent(flatten_points(expected))
        return max(10.0, 0.1 * extent) if extent is not None else 50.0

    extent = point_cloud_extent(flatten_points(expected))
    return max(0.02, 0.1 * extent) if extent is not None else 0.05


def distance_to_score(distance: float | None, tolerance: float) -> float | None:
    if distance is None or tolerance <= 0:
        return None
    ratio = float(distance) / tolerance
    return 100.0 / (1.0 + ratio * ratio)


def score_curve(expected: list[list[float]], predicted: list[list[float]], tolerance: float) -> dict[str, Any]:
    hausdorff = hausdorff_distance(expected, predicted)
    discrete_frechet = discrete_frechet_distance(expected, predicted)
    chamfer = chamfer_distance(expected, predicted)
    metric_scores = {
        "hausdorff": rounded(distance_to_score(hausdorff, tolerance)),
        "discrete_frechet": rounded(distance_to_score(discrete_frechet, tolerance)),
        "chamfer": rounded(distance_to_score(chamfer, tolerance)),
    }
    return {
        "expected_points": len(expected),
        "predicted_points": len(predicted),
        "hausdorff": rounded(hausdorff),
        "discrete_frechet": rounded(discrete_frechet),
        "chamfer": rounded(chamfer),
        "metric_scores": metric_scores,
        "score": rounded(mean_available(list(metric_scores.values()))),
    }


def mean_available(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def score_prediction(item: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected = expected_trajectory(item)
    active_gripper = active_gripper_for_item(item, expected)
    scored_grippers = grippers_to_score(active_gripper)
    expected_scored = {gripper: expected.get(gripper, []) for gripper in scored_grippers}
    predicted = prediction.get("trajectory", {})
    if not isinstance(predicted, dict):
        predicted = {}
    score_tolerance = score_tolerance_for_item(item, expected_scored)
    gripper_metrics = {
        gripper: score_curve(expected.get(gripper, []), predicted.get(gripper, []), score_tolerance)
        for gripper in scored_grippers
    }
    width, height = image_size_for_item(item) if infer_dimension(item) == 2 else (None, None)
    invalid_points = out_of_bounds_points(prediction, width, height, scored_grippers)
    mean_metric_scores = {
        "hausdorff": rounded(mean_available([row["metric_scores"]["hausdorff"] for row in gripper_metrics.values()])),
        "discrete_frechet": rounded(
            mean_available([row["metric_scores"]["discrete_frechet"] for row in gripper_metrics.values()])
        ),
        "chamfer": rounded(mean_available([row["metric_scores"]["chamfer"] for row in gripper_metrics.values()])),
    }
    mean_metrics = {
        "hausdorff": rounded(mean_available([row["hausdorff"] for row in gripper_metrics.values()])),
        "discrete_frechet": rounded(mean_available([row["discrete_frechet"] for row in gripper_metrics.values()])),
        "chamfer": rounded(mean_available([row["chamfer"] for row in gripper_metrics.values()])),
    }
    return {
        "dimension": infer_dimension(item),
        "active_gripper": active_gripper,
        "scored_grippers": scored_grippers,
        "image_size": {"width": width, "height": height} if width is not None and height is not None else None,
        "score_mapping": {
            "function": "100 / (1 + (distance / tolerance)^2)",
            "tolerance": rounded(score_tolerance),
            "tolerance_rule": "2D uses 5% of image diagonal; 3D uses max(0.02 m, 10% of GT trajectory extent).",
        },
        "out_of_bounds": {
            "count": len(invalid_points),
            "examples": invalid_points[:12],
        },
        "expected_trajectory": expected,
        "expected_scored_trajectory": expected_scored,
        "predicted_trajectory": predicted,
        "predicted_scored_trajectory": {
            gripper: predicted.get(gripper, [])
            for gripper in scored_grippers
        },
        "gripper_metrics": gripper_metrics,
        "active_gripper_metrics": gripper_metrics.get(active_gripper),
        "mean_metrics": mean_metrics,
        "mean_metric_scores": mean_metric_scores,
        "score": rounded(mean_available(list(mean_metric_scores.values()))),
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
    answered = [row for row in results if row.get("model_output")]
    mean_metrics = [row.get("mean_metrics", {}) for row in answered]
    mean_metric_scores = [row.get("mean_metric_scores", {}) for row in answered]
    return {
        "total": len(results),
        "answered": len(answered),
        "errors": sum(1 for row in results if row.get("error")),
        "mean_hausdorff": rounded(mean_available([row.get("hausdorff") for row in mean_metrics])),
        "mean_discrete_frechet": rounded(mean_available([row.get("discrete_frechet") for row in mean_metrics])),
        "mean_chamfer": rounded(mean_available([row.get("chamfer") for row in mean_metrics])),
        "mean_hausdorff_score": rounded(mean_available([row.get("hausdorff") for row in mean_metric_scores])),
        "mean_discrete_frechet_score": rounded(
            mean_available([row.get("discrete_frechet") for row in mean_metric_scores])
        ),
        "mean_chamfer_score": rounded(mean_available([row.get("chamfer") for row in mean_metric_scores])),
        "mean_score": rounded(mean_available([row.get("score") for row in answered])),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trajectory QA with VLM models.")
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

    task_defaults = task_config(args.config, "trajectory")
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
        question = str(item.get("Q") or item.get("question"))
        image_inputs = image_inputs_for_item(item)
        prompt = build_prompt(question, item)
        item_started = time.perf_counter()

        image_labels = [f"{row['role']}:{row['label']}" for row in image_inputs]
        print(f"[{index}/{len(pending)}] {item_id} images={len(image_inputs)} {image_labels} {question}", flush=True)
        try:
            _, model_text = call_vlm(runtime, image_parts_for_item(image_inputs, prompt))
            prediction = parse_model_answer(model_text, infer_dimension(item))
            model_outputs = [model_text]
            prompts = [prompt]
            if infer_dimension(item) == 2:
                width, height = image_size_for_item(item)
                scored_grippers = grippers_to_score(active_gripper_for_item(item))
                for _ in range(MAX_COORDINATE_RETRIES):
                    invalid_points = out_of_bounds_points(prediction, width, height, scored_grippers)
                    if not invalid_points or width is None or height is None:
                        break
                    retry_prompt = build_retry_prompt(prompt, model_text, invalid_points, width, height)
                    print(f"  retry: {len(invalid_points)} predicted 2D points are out of bounds", flush=True)
                    _, model_text = call_vlm(runtime, image_parts_for_item(image_inputs, retry_prompt))
                    prediction = parse_model_answer(model_text, infer_dimension(item))
                    model_outputs.append(model_text)
                    prompts.append(retry_prompt)
            scores = score_prediction(item, prediction)
            result = {
                **item,
                "prompt": prompt,
                "prompts": prompts,
                "image_inputs": image_inputs,
                "model_output": model_text,
                "model_outputs": model_outputs,
                "model_prediction": prediction.get("parsed"),
                **scores,
                "timing": {"seconds": round(time.perf_counter() - item_started, 3)},
            }
        except Exception as exc:
            result = {
                **item,
                "prompt": prompt,
                "model_output": None,
                "model_prediction": None,
                **score_prediction(item, {"trajectory": {}}),
                "error": str(exc),
                "timing": {"seconds": round(time.perf_counter() - item_started, 3)},
            }
            print(f"  error: {exc}", flush=True)

        results_by_id[item_id] = result
        results = [results_by_id[str(item["id"])] for item in items if str(item.get("id")) in results_by_id]
        summary = summarize(results, time.perf_counter() - started_at)
        mean_metrics = result.get("mean_metrics", {})
        print(
            f"  H={mean_metrics.get('hausdorff')} "
            f"F={mean_metrics.get('discrete_frechet')} "
            f"C={mean_metrics.get('chamfer')} "
            f"score={result.get('score')} "
            f"mean_H={summary['mean_hausdorff']}",
            flush=True,
        )

        save_json(
            checkpoint_path,
            {
                "input": str(args.input),
                "provider": runtime["provider"],
                "model": runtime["model"],
                "api_url": runtime["api_url"],
                "metrics": ["hausdorff", "discrete_frechet", "chamfer"],
                "score_mapping": "score = 100 / (1 + (distance / tolerance)^2)",
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
        "metrics": ["hausdorff", "discrete_frechet", "chamfer"],
        "score_mapping": "score = 100 / (1 + (distance / tolerance)^2)",
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
