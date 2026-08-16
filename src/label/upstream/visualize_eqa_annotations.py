#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize time-EQA annotations on a video.

Default input:
  video: /home/kewei/YWC/egodata/pickplace/video/file-000.mp4
  json : /home/kewei/YWC/egodata/pickplace/time_eqa_first50_6move.json

Example:
  python visualize_eqa_annotations.py
  python visualize_eqa_annotations.py --show
  python visualize_eqa_annotations.py --output file-000_annotated.mp4
"""

import argparse
import json
import re
from pathlib import Path

import cv2


DEFAULT_VIDEO = "/home/kewei/YWC/egodata/pickplace/video/file-005.mp4"
DEFAULT_ANNOTATIONS = "/home/kewei/YWC/egodata/pickplace/time_eqa_first50_6move.json"

COLORS = {
    "pick": (58, 186, 255),
    "move": (92, 214, 92),
    "place": (255, 153, 77),
    "other": (220, 220, 220),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay time-EQA annotations on the corresponding video."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to input video.")
    parser.add_argument(
        "--annotations",
        default=DEFAULT_ANNOTATIONS,
        help="Path to time-EQA annotation JSON.",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Video id in the annotation JSON. Default: input video stem.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path. Default: ./<video_stem>_eqa_vis.mp4",
    )
    parser.add_argument("--show", action="store_true", help="Preview in an OpenCV window.")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Preview only; do not write an output video.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start visualization from this time in seconds.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Stop visualization at this time in seconds.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most this many frames. Useful for quick tests.",
    )
    return parser.parse_args()


def timestamp(seconds):
    total_ms = int(round(float(seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_time_range(text):
    match = re.match(r"\s*(\d+):(\d+):(\d+(?:\.\d+)?)\s*-\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    h1, m1, s1, h2, m2, s2 = match.groups()
    start = int(h1) * 3600 + int(m1) * 60 + float(s1)
    end = int(h2) * 3600 + int(m2) * 60 + float(s2)
    return start, end


def item_time_range(item):
    answer_seconds = item.get("answer_seconds")
    if isinstance(answer_seconds, dict):
        start = answer_seconds.get("start")
        end = answer_seconds.get("end")
        if start is not None and end is not None:
            return float(start), float(end)

    answer = item.get("A") or item.get("answer")
    if isinstance(answer, str):
        parsed = parse_time_range(answer)
        if parsed is not None:
            return parsed

    evidence = item.get("evidence") or []
    if evidence and isinstance(evidence[0], dict):
        start = evidence[0].get("start")
        end = evidence[0].get("end")
        if start is not None and end is not None:
            return float(start), float(end)

    raise ValueError(f"Cannot find time range for annotation item: {item.get('id')}")


def label_from_item(item):
    metadata = item.get("metadata") or {}
    verbs = metadata.get("main_verbs") or []
    objects = metadata.get("objects") or []
    verb = str(verbs[0]).lower() if verbs else "other"
    obj = str(objects[0]) if objects else ""
    if verb != "other" and obj:
        return f"{verb} {obj}", verb

    narration = str(metadata.get("narration") or "").strip()
    if narration:
        cleaned = narration.strip(".").strip()
        return cleaned, verb

    question = item.get("Q") or item.get("question") or item.get("id") or "annotation"
    return str(question), verb


def load_annotations(annotation_path, video_id):
    with open(annotation_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("items", raw if isinstance(raw, list) else [])
    annotations = []
    for item in items:
        if item.get("video_id") != video_id:
            continue
        start, end = item_time_range(item)
        if end <= start:
            continue
        label, verb = label_from_item(item)
        question = str(item.get("Q") or item.get("question") or "")
        annotations.append(
            {
                "id": str(item.get("id", "")),
                "start": start,
                "end": end,
                "label": label,
                "verb": verb if verb in COLORS else "other",
                "question": question,
            }
        )

    annotations.sort(key=lambda x: (x["start"], x["end"], x["id"]))
    if not annotations:
        raise ValueError(f"No annotations found for video_id={video_id!r}")
    return annotations


def blend_rect(frame, pt1, pt2, color, alpha=0.72):
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def text_size(text, scale=0.55, thickness=1):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]


def fit_text(text, max_width, scale=0.55, thickness=1, min_scale=0.38):
    text = str(text)
    if text_size(text, scale, thickness)[0] <= max_width:
        return text, scale
    while scale > min_scale:
        scale -= 0.03
        if text_size(text, scale, thickness)[0] <= max_width:
            return text, scale

    ellipsis = "..."
    result = text
    while result and text_size(result + ellipsis, min_scale, thickness)[0] > max_width:
        result = result[:-1]
    return result + ellipsis if result else ellipsis, min_scale


def put_text(frame, text, org, color=(255, 255, 255), scale=0.55, thickness=1):
    cv2.putText(
        frame,
        str(text),
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_header(frame, frame_index, fps, video_id, active_annotations):
    h, w = frame.shape[:2]
    panel_h = 92 + max(1, min(len(active_annotations), 3)) * 28
    blend_rect(frame, (0, 0), (w, panel_h), (18, 18, 18), alpha=0.9)

    current_time = frame_index / fps if fps > 0 else 0.0
    put_text(frame, f"{video_id} | frame {frame_index} | {timestamp(current_time)}", (16, 28), scale=0.68)

    if active_annotations:
        put_text(frame, "Active annotation", (16, 58), color=(210, 210, 210), scale=0.5)
        rows = active_annotations[:3]
        for idx, ann in enumerate(rows):
            y = 86 + idx * 28
            color = COLORS.get(ann["verb"], COLORS["other"])
            cv2.rectangle(frame, (16, y - 14), (28, y - 2), color, -1)
            label = f"{ann['label']}  {timestamp(ann['start'])}-{timestamp(ann['end'])}"
            label, scale = fit_text(label, w - 52, scale=0.55)
            put_text(frame, label, (38, y), color=(255, 255, 255), scale=scale)
        if len(active_annotations) > 3:
            put_text(frame, f"+{len(active_annotations) - 3} more", (38, panel_h - 12), color=(210, 210, 210), scale=0.45)
    else:
        put_text(frame, "No active annotation at this frame", (16, 65), color=(190, 190, 190), scale=0.55)


def draw_timeline(frame, annotations, current_time, duration):
    h, w = frame.shape[:2]
    margin_x = 34
    bar_y = h - 46
    bar_h = 16
    usable_w = max(1, w - margin_x * 2)

    blend_rect(frame, (0, h - 82), (w, h), (18, 18, 18), alpha=0.82)
    cv2.rectangle(frame, (margin_x, bar_y), (margin_x + usable_w, bar_y + bar_h), (70, 70, 70), -1)

    for ann in annotations:
        x1 = margin_x + int((ann["start"] / duration) * usable_w)
        x2 = margin_x + int((ann["end"] / duration) * usable_w)
        x1 = max(margin_x, min(margin_x + usable_w, x1))
        x2 = max(margin_x, min(margin_x + usable_w, x2))
        if x2 <= x1:
            x2 = x1 + 1
        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), COLORS.get(ann["verb"], COLORS["other"]), -1)

    marker_x = margin_x + int((current_time / duration) * usable_w)
    marker_x = max(margin_x, min(margin_x + usable_w, marker_x))
    cv2.line(frame, (marker_x, bar_y - 10), (marker_x, bar_y + bar_h + 10), (255, 255, 255), 2)

    put_text(frame, "0", (margin_x, h - 16), color=(210, 210, 210), scale=0.43)
    end_text = timestamp(duration)
    put_text(frame, end_text, (w - margin_x - text_size(end_text, 0.43)[0], h - 16), color=(210, 210, 210), scale=0.43)

    legend_x = margin_x
    for verb in ("pick", "move", "place"):
        color = COLORS[verb]
        cv2.rectangle(frame, (legend_x, h - 74), (legend_x + 12, h - 62), color, -1)
        put_text(frame, verb, (legend_x + 18, h - 62), color=(225, 225, 225), scale=0.43)
        legend_x += 78


def draw_annotations(frame, frame_index, fps, video_id, annotations, duration):
    current_time = frame_index / fps if fps > 0 else 0.0
    active = [ann for ann in annotations if ann["start"] <= current_time <= ann["end"]]
    draw_header(frame, frame_index, fps, video_id, active)
    draw_timeline(frame, annotations, current_time, duration)
    return frame


def make_writer(output_path, fps, width, height):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Cannot open output video writer: {output_path}")
    return writer


def main():
    args = parse_args()
    video_path = Path(args.video)
    annotation_path = Path(args.annotations)
    video_id = args.video_id or video_path.stem
    output_path = Path(args.output) if args.output else Path.cwd() / f"{video_path.stem}_eqa_vis.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    if duration <= 0:
        raise ValueError("Video duration is invalid.")

    annotations = load_annotations(annotation_path, video_id)
    start_frame = max(0, int(round(args.start_time * fps))) if fps > 0 else 0
    end_frame = total_frames
    if args.end_time is not None:
        end_frame = min(end_frame, max(start_frame, int(round(args.end_time * fps))))

    writer = None
    if not args.no_write:
        writer = make_writer(output_path, fps, width, height)

    if args.show:
        cv2.namedWindow("EQA annotation visualization", cv2.WINDOW_NORMAL)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    processed = 0

    print(f"video      : {video_path}")
    print(f"annotations: {annotation_path}")
    print(f"video_id   : {video_id}")
    print(f"segments   : {len(annotations)}")
    if writer is not None:
        print(f"output     : {output_path}")

    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        draw_annotations(frame, frame_index, fps, video_id, annotations, duration)

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("EQA annotation visualization", frame)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key in (27, ord("q")):
                break

        processed += 1
        frame_index += 1
        if args.max_frames is not None and processed >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"processed  : {processed} frames")


if __name__ == "__main__":
    main()
