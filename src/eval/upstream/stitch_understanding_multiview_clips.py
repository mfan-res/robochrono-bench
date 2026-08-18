#!/usr/bin/env python3
# coding: utf-8
"""Stitch multi-view understanding clips into one video per item.

The script reads an understanding VQA JSON whose items contain
`input.clip_paths` or `input.clips`, stitches the synchronized views
horizontally, and writes a new JSON where `input.clip_path` points to the
stitched single-video clip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_INPUT = Path("/home/kewei/YWC/egodata/pickplace/understanding_multi.json")
DEFAULT_OUTPUT_DIR = Path("/home/kewei/YWC/egodata/pickplace/understanding_multi_clips_stitched")
DEFAULT_OUTPUT_JSON = Path("/home/kewei/YWC/egodata/pickplace/understanding_multi_stitched.json")
DEFAULT_VIEW_ORDER = "left_eye,left_wrist,right_wrist"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_") or "item"


def items_from_data(data: Any) -> list[dict[str, Any]]:
    items = data.get("items", data if isinstance(data, list) else [])
    if not isinstance(items, list):
        raise ValueError("Input must be a list or contain an `items` list")
    return items


def clip_rows_for_item(item: dict[str, Any], view_order: list[str]) -> list[dict[str, Any]]:
    input_data = item.get("input", {})
    clips = input_data.get("clips")
    rows: list[dict[str, Any]] = []
    if isinstance(clips, dict) and clips:
        for view in view_order:
            row = clips.get(view)
            if isinstance(row, dict) and row.get("clip_path"):
                rows.append({"view": view, "clip_path": Path(str(row["clip_path"]))})
        for view, row in clips.items():
            if view in view_order:
                continue
            if isinstance(row, dict) and row.get("clip_path"):
                rows.append({"view": str(view), "clip_path": Path(str(row["clip_path"]))})
        if rows:
            return rows

    clip_paths = input_data.get("clip_paths")
    if isinstance(clip_paths, list) and clip_paths:
        return [
            {
                "view": view_order[index] if index < len(view_order) else f"view_{index}",
                "clip_path": Path(str(path)),
            }
            for index, path in enumerate(clip_paths)
        ]

    if input_data.get("clip_path"):
        return [{"view": "default", "clip_path": Path(str(input_data["clip_path"]))}]

    raise KeyError(f"Item {item.get('id')} has no input.clip_paths/input.clips/input.clip_path")


def open_captures(rows: list[dict[str, Any]]) -> list[cv2.VideoCapture]:
    captures: list[cv2.VideoCapture] = []
    for row in rows:
        path = row["clip_path"]
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open clip: {path}")
        captures.append(cap)
    return captures


def clip_metadata(captures: list[cv2.VideoCapture]) -> dict[str, Any]:
    fps_values = [float(cap.get(cv2.CAP_PROP_FPS)) for cap in captures]
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
    widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in captures]
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in captures]
    if any(fps <= 0 for fps in fps_values) or any(count <= 0 for count in frame_counts):
        raise ValueError(f"Invalid clip metadata: fps={fps_values}, frames={frame_counts}")
    if any(width <= 0 for width in widths) or any(height <= 0 for height in heights):
        raise ValueError(f"Invalid clip dimensions: widths={widths}, heights={heights}")
    return {
        "fps": fps_values[0],
        "frames": min(frame_counts),
        "widths": widths,
        "heights": heights,
        "target_height": min(heights),
    }


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    target_width = max(1, int(round(width * target_height / height)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(output, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def stitch_clips(
    rows: list[dict[str, Any]],
    output_path: Path,
    overwrite: bool,
    add_labels: bool,
) -> dict[str, Any]:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        cap = cv2.VideoCapture(str(output_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {"fps": fps, "frames": frames, "width": width, "height": height, "reused": True}

    captures = open_captures(rows)
    try:
        meta = clip_metadata(captures)
        first_frames: list[np.ndarray] = []
        for cap, row in zip(captures, rows):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read first frame from {row['clip_path']}")
            frame = resize_to_height(frame, int(meta["target_height"]))
            if add_labels:
                frame = draw_label(frame, str(row["view"]))
            first_frames.append(frame)

        stitched_first = np.hstack(first_frames)
        height, width = stitched_first.shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(meta["fps"]), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create stitched video: {output_path}")

        writer.write(stitched_first)
        written = 1
        for _ in range(1, int(meta["frames"])):
            frames: list[np.ndarray] = []
            for cap, row in zip(captures, rows):
                ok, frame = cap.read()
                if not ok:
                    break
                frame = resize_to_height(frame, int(meta["target_height"]))
                if add_labels:
                    frame = draw_label(frame, str(row["view"]))
                frames.append(frame)
            if len(frames) != len(captures):
                break
            writer.write(np.hstack(frames))
            written += 1
        writer.release()
    finally:
        for cap in captures:
            cap.release()

    if written == 0:
        raise RuntimeError(f"No frames written for stitched video: {output_path}")
    return {
        "fps": float(meta["fps"]),
        "frames": written,
        "width": width,
        "height": height,
        "views": [row["view"] for row in rows],
        "source_clip_paths": [str(row["clip_path"]) for row in rows],
        "reused": False,
    }


def output_path_for_item(item: dict[str, Any], output_dir: Path) -> Path:
    video_id = str(item.get("video_id") or "unknown_video")
    item_id = str(item["id"])
    return output_dir / video_id / f"{safe_filename(item_id)}_stitched.mp4"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stitch multi-view understanding clips into one video per item.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--views", default=DEFAULT_VIEW_ORDER, help="Comma-separated view order.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    args = parser.parse_args()

    data = load_json(args.input)
    items = items_from_data(data)
    selected_items = items[: args.limit] if args.limit is not None else items
    view_order = [view.strip() for view in args.views.split(",") if view.strip()]

    updated_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(selected_items, 1):
        item_id = str(item["id"])
        rows = clip_rows_for_item(item, view_order)
        output_path = output_path_for_item(item, args.output_dir)
        print(f"[{index}/{len(selected_items)}] {item_id} views={len(rows)} -> {output_path}", flush=True)
        stitched_info = stitch_clips(rows, output_path, args.overwrite, add_labels=not args.no_labels)

        new_item = json.loads(json.dumps(item, ensure_ascii=False))
        input_data = new_item.setdefault("input", {})
        input_data["stitched_clip_path"] = str(output_path)
        input_data["original_clip_path"] = input_data.get("clip_path")
        input_data["original_clip_paths"] = input_data.get("clip_paths", [])
        input_data["clip_path"] = str(output_path)
        input_data["clip_paths"] = [str(output_path)]
        input_data["clips"] = {
            "stitched": {
                "view": "stitched",
                "clip_path": str(output_path),
                **stitched_info,
            }
        }
        input_data["stitched_clip"] = stitched_info
        updated_by_id[item_id] = new_item

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        output_data = json.loads(json.dumps(data, ensure_ascii=False))
        output_data["items"] = [updated_by_id.get(str(item.get("id")), item) for item in items]
        output_data["stitched_clips_dir"] = str(args.output_dir)
        output_data["stitched_view_order"] = view_order
    else:
        output_data = [updated_by_id.get(str(item.get("id")), item) for item in items]

    save_json(args.output_json, output_data)
    print(f"Items stitched: {len(updated_by_id)}")
    print(f"Output JSON: {args.output_json}")
    print(f"Output dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
