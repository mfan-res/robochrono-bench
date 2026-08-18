#!/usr/bin/env python3
# coding: utf-8
"""六个选择题任务：understanding / left_right / image_in_video /
planning / planning_2 / step_order。

这六个脚本在冻结版里各约 320~470 行，其中真正不同的只有四处：取媒体、
组 prompt、结果里带哪些 expected_* 字段、汇总里加哪些分组。其余全是样板。

prompt 文本、打分口径、汇总口径均从冻结脚本**逐字搬运**，
只在末尾追加 BC-03 的两个新字段（不改动任何既有字段的定义）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import parsing
from .base import CallContext, Unit, base_row, count_by, image_part, one_item_per_unit, text_part, video_part

# --------------------------------------------------------------------------
# prompt：逐字搬运自冻结脚本
# --------------------------------------------------------------------------


def _question(item: dict[str, Any]) -> str:
    return str(item.get("Q") or item.get("question") or "")


def _question_head(item: dict[str, Any]) -> str:
    """left_right / image_in_video 只取 Options: 之前的部分。"""
    return _question(item).split("\nOptions:", 1)[0].strip()


def prompt_understanding(item: dict[str, Any]) -> str:
    return f"""You are answering a multiple-choice visual understanding question about an egocentric robot manipulation video clip.

The input may contain multiple synchronized camera-view clips of the same moment. Use all views together.
The clips show the video up to the current moment. Choose the option that best matches what is happening right now.
Choose exactly one option letter from the provided options. Do not invent a new action.

Question:
{_question(item)}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter, e.g. A>",
  "reason": "<brief visual reason>"
}}
"""


def prompt_planning(item: dict[str, Any]) -> str:
    return f"""You are observing a robot manipulation task and need to predict the next action.

Look at the current video clip and choose the next action the robot should take.
Choose exactly one option from the provided option letters. Do not invent a new action.

Question:
{_question(item)}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter, e.g. A>",
  "reason": "<brief visual reason>"
}}
"""


def prompt_planning_2(item: dict[str, Any]) -> str:
    return f"""You are observing a robot manipulation task and need to predict the next action.

Look at the current image or synchronized multi-view images and choose the next action the robot should take.
Choose exactly one option from the provided option letters. Do not invent a new action.

Question:
{_question(item)}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter, e.g. A>",
  "reason": "<brief visual reason>"
}}
"""


def prompt_left_right(item: dict[str, Any]) -> str:
    option_ids = ", ".join(sorted(parsing.options_from_item(item)))
    return f"""You are answering a visual matching question for a robot manipulation episode.

The first image is from the head camera. The following labeled option images are candidate gripper-camera views at the same moment or distractors.
Choose the single option letter that shows the requested gripper camera's view for the head-camera moment.
If none of the listed images match, choose the option labeled "All other options are wrong."

Question:
{_question_head(item)}

Valid option letters: {option_ids}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter>",
  "reason": "<brief visual reason>"
}}
"""


def prompt_image_in_video(item: dict[str, Any]) -> str:
    option_ids = ", ".join(sorted(parsing.options_from_item(item)))
    return f"""You are answering a visual matching question for a robot manipulation video clip.

First inspect the left-eye video clip, then inspect each labeled option image.
Choose the single option letter whose image appears in the video clip.
If none of the listed images appear in the clip, choose the option labeled "All other options are wrong."

Question:
{_question_head(item)}

Valid option letters: {option_ids}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter>",
  "reason": "<brief visual reason>"
}}
"""


def prompt_step_order(item: dict[str, Any]) -> str:
    question = _question(item).strip()
    choices = parsing.choices_from_item(item)
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


# --------------------------------------------------------------------------
# 媒体：逐字搬运自冻结脚本的取路径逻辑
# --------------------------------------------------------------------------


def media_video_list(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """planning / understanding：若干视频在前，prompt 在后。"""
    data = item.get("input", {})
    for key in ("clip_paths", "video_paths"):
        value = data.get(key)
        if isinstance(value, list) and value:
            return [video_part(p) for p in value] + [text_part(prompt)]
        if isinstance(value, dict) and value:
            return [video_part(p) for p in value.values()] + [text_part(prompt)]

    clips = data.get("clips")
    if isinstance(clips, dict) and clips:
        paths = [row["clip_path"] for row in clips.values() if isinstance(row, dict) and row.get("clip_path")]
        if paths:
            return [video_part(p) for p in paths] + [text_part(prompt)]

    for key in ("clip_path", "video_path"):
        if data.get(key):
            return [video_part(data[key]), text_part(prompt)]

    raise KeyError(f"item {item.get('id')} has no clip_path/clip_paths or video_path/video_paths")


def media_image_list(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """planning_2：若干图片在前，prompt 在后。"""
    data = item.get("input", {})
    paths = data.get("image_paths")
    if isinstance(paths, list) and paths:
        return [image_part(p) for p in paths] + [text_part(prompt)]

    images = data.get("images")
    if isinstance(images, dict) and images:
        collected = [row["image_path"] for row in images.values() if isinstance(row, dict) and row.get("image_path")]
        if collected:
            return [image_part(p) for p in collected] + [text_part(prompt)]

    return [image_part(data["image_path"]), text_part(prompt)]


def media_head_and_options(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """left_right：head 图 + 逐个带标注的选项图。"""
    data = item.get("input", {})
    head = data.get("image_path") or (data.get("head_image") or {}).get("image_path")
    parts: list[dict[str, Any]] = [
        text_part("Head camera image:"),
        image_part(head),
        text_part("Candidate options:"),
    ]
    parts.extend(_option_image_parts(item))
    parts.append(text_part(prompt))
    return parts


def media_clip_and_options(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """image_in_video：clip 视频 + 逐个带标注的选项图。"""
    data = item.get("input", {})
    clip = data.get("clip_path") or data.get("video_path")
    if not clip:
        raise ValueError(f"item {item.get('id')} has no input.clip_path")
    parts: list[dict[str, Any]] = [
        text_part("Left-eye video clip:"),
        video_part(clip),
        text_part("Candidate option images:"),
    ]
    parts.extend(_option_image_parts(item))
    parts.append(text_part(prompt))
    return parts


def _option_image_parts(item: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for option in item.get("options", []):
        option_id = str(option["id"]).upper()
        if option.get("is_none_option") or option.get("type") == "none":
            parts.append(text_part(f"Option {option_id}: {option.get('text')}"))
            continue
        image_path = option.get("image_path")
        if not image_path:
            raise ValueError(f"option {option_id} in {item.get('id')} has no image_path")
        parts.append(text_part(f"Option {option_id} image:"))
        parts.append(image_part(image_path))
    return parts


def media_step_order(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """step_order：initial 图 + montage 图，共两张。"""
    paths: list[str] = []
    for key in ("initial_image", "image"):
        if item.get(key):
            paths.append(str(item[key]))
    data = item.get("input", {})
    if isinstance(data, dict):
        for key in ("initial_image", "image", "image_path"):
            if data.get(key) and str(data[key]) not in paths:
                paths.append(str(data[key]))
        for path in data.get("image_paths") or []:
            if str(path) not in paths:
                paths.append(str(path))
    if len(paths) < 2:
        raise ValueError(f"cannot find initial image and montage image for item {item.get('id')}")
    return [image_part(p) for p in paths[:2]] + [text_part(prompt)]


# --------------------------------------------------------------------------
# 任务定义
# --------------------------------------------------------------------------


@dataclass
class ChoiceSpec:
    name: str
    prompt: Callable[[dict[str, Any]], str]
    media: Callable[[dict[str, Any], str], list[dict[str, Any]]]
    # 结果字段名 -> 从 item 里取的键（元组表示按序回落）
    extra_fields: dict[str, Any] = field(default_factory=dict)
    summary_groups: tuple[str, ...] = ()
    distractor_counts: bool = False
    use_choices: bool = False


SPECS: dict[str, ChoiceSpec] = {
    "understanding": ChoiceSpec(
        name="understanding",
        prompt=prompt_understanding,
        media=media_video_list,
        extra_fields={
            "expected_answer_text": "answer_text",
            "expected_action": "answer_action",
            "expected_subject": "answer_subject",
            "expected_target": "answer_target",
        },
        summary_groups=("expected_action", "expected_choice"),
    ),
    "planning": ChoiceSpec(
        name="planning",
        prompt=prompt_planning,
        media=media_video_list,
        extra_fields={
            "expected_answer_text": "answer_text",
            "expected_action": "answer_action",
            "expected_subject": "answer_subject",
            "expected_target": "answer_target",
        },
        summary_groups=("expected_action", "expected_choice"),
    ),
    "planning_2": ChoiceSpec(
        name="planning_2",
        prompt=prompt_planning_2,
        media=media_image_list,
        extra_fields={
            "expected_answer_text": "answer_text",
            "expected_action": "answer_action",
            "expected_subject": "answer_subject",
            "expected_target": "answer_target",
        },
        summary_groups=("expected_action", "expected_choice"),
    ),
    "left_right": ChoiceSpec(
        name="left_right",
        prompt=prompt_left_right,
        media=media_head_and_options,
        extra_fields={
            "expected_answer_text": "answer_text",
            "expected_target_side": "target_side",
        },
        summary_groups=("expected_target_side", "expected_choice"),
        distractor_counts=True,
    ),
    "image_in_video": ChoiceSpec(
        name="image_in_video",
        prompt=prompt_image_in_video,
        media=media_clip_and_options,
        extra_fields={
            "expected_answer_text": "answer_text",
            "expected_category": "answer_category",
        },
        distractor_counts=True,
    ),
    "step_order": ChoiceSpec(
        name="step_order",
        prompt=prompt_step_order,
        media=media_step_order,
        extra_fields={"expected_answer_order": ("answer_order", "answer_text")},
        use_choices=True,
    ),
}

# 汇总里 by_* 分组的对外命名，与冻结脚本一致
_GROUP_LABEL = {
    "expected_action": "by_action",
    "expected_choice": "by_choice",
    "expected_target_side": "by_side",
}


class ChoiceTask:
    """六个选择题任务的统一实现，由 ChoiceSpec 参数化。"""

    def __init__(self, spec: ChoiceSpec, *, strip_reasoning: bool = False, null_text_fix: bool = False) -> None:
        self.spec = spec
        self.name = spec.name
        self.strip_reasoning = strip_reasoning     # BC-02
        self.null_text_fix = null_text_fix         # BC-10

    # -- 切分与组装 --------------------------------------------------------

    def units(self, items: list[dict[str, Any]]) -> list[Unit]:
        return one_item_per_unit(items)

    def parts(self, unit: Unit) -> list[dict[str, Any]]:
        item = unit.items[0]
        return self.spec.media(item, self.spec.prompt(item))

    # -- 解析与打分 --------------------------------------------------------

    def _options(self, item: dict[str, Any]) -> dict[str, str]:
        if self.spec.use_choices:
            return parsing.choices_from_item(item, null_text_fix=self.null_text_fix)
        return parsing.options_from_item(item, null_text_fix=self.null_text_fix)

    def rows(self, unit: Unit, text: str, ctx: CallContext) -> list[dict[str, Any]]:
        item = unit.items[0]
        prompt = self.spec.prompt(item)
        prediction = parsing.parse_choice_answer(
            text,
            self._options(item),
            keep_hyphen=self.spec.use_choices,
            strip_reasoning=self.strip_reasoning,
        )
        row = base_row(item, prompt, text, ctx)
        row["model_prediction"] = prediction.get("parsed")
        row.update(self._score(item, prediction.get("choice")))
        # BC-03：新增字段，不改动既有 correct / accuracy 的定义
        row["parse_ok"] = bool(prediction.get("parse_ok"))
        row["parse_recovered"] = bool(prediction.get("parse_recovered"))
        return [row]

    def error_rows(self, unit: Unit, error: str) -> list[dict[str, Any]]:
        item = unit.items[0]
        row = base_row(item, self.spec.prompt(item), None, None)
        row["model_prediction"] = None
        row.update(self._score(item, None))
        row["correct"] = False
        row["error"] = error
        row["parse_ok"] = False
        row["parse_recovered"] = False
        return [row]

    def _score(self, item: dict[str, Any], pred_choice: str | None) -> dict[str, Any]:
        expected = str(item.get("answer") or item.get("A") or "").upper()
        scored: dict[str, Any] = {"expected_choice": expected}
        for out_key, src in self.spec.extra_fields.items():
            keys = src if isinstance(src, tuple) else (src,)
            value = None
            for key in keys:
                value = item.get(key)
                if value:
                    break
            scored[out_key] = value
        scored["pred_choice"] = pred_choice
        scored["correct"] = pred_choice == expected
        if self.spec.distractor_counts:
            # 冻结脚本在汇总阶段从 row["options"] 里回查；我们按 BC-04 不再复制
            # 整条 item 进结果行，所以改在打分时就把这一项记下来。
            # 取值与冻结脚本一致：命中选项的 distractor_type，缺省为 "correct"。
            scored["chosen_distractor_type"] = None
            for option in item.get("options", []):
                if str(option.get("id")).upper() == str(pred_choice or ""):
                    scored["chosen_distractor_type"] = str(option.get("distractor_type") or "correct")
                    break
        return scored

    # -- 汇总 --------------------------------------------------------------

    def summarize(self, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        total = len(rows)
        answered = [r for r in rows if r.get("model_output")]
        summary: dict[str, Any] = {
            "total": total,
            "answered": len(answered),
            "errors": sum(1 for r in rows if r.get("error")),
            "accuracy": sum(bool(r.get("correct")) for r in rows) / total if total else 0.0,
            "elapsed_seconds": round(elapsed, 3),
        }

        if self.spec.distractor_counts:
            summary["chosen_option_type_counts"] = self._distractor_counts(rows)

        for group_key in self.spec.summary_groups:
            summary[_GROUP_LABEL[group_key]] = count_by(rows, group_key)

        # BC-03：解析失败与答错分离。既有 accuracy 定义保持不变。
        parsed = [r for r in rows if r.get("parse_ok")]
        summary["parse_failure_rate"] = (total - len(parsed)) / total if total else 0.0
        summary["accuracy_answered"] = (
            sum(bool(r.get("correct")) for r in parsed) / len(parsed) if parsed else 0.0
        )
        summary["parse_recovered"] = sum(1 for r in rows if r.get("parse_recovered"))
        return summary

    @staticmethod
    def _distractor_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        """统计模型选中的是哪一类干扰项。需要原始 options，从 row 里已无法取得，
        因此改为在打分时记录 —— 见 ``_score`` 里的 chosen_distractor_type。"""
        counts: dict[str, int] = {}
        for row in rows:
            key = row.get("chosen_distractor_type")
            if key:
                counts[str(key)] = counts.get(str(key), 0) + 1
        return counts


def build(name: str, **flags: Any) -> ChoiceTask:
    # 只挑本任务认识的开关。调用方统一把所有 flag 发给所有任务，
    # 由各任务自行取用 —— 这样新增一个开关不必再去改两处「哪个 run 收哪些 flag」
    # 的名单（那种名单漏改一处就是静默失效）。
    return ChoiceTask(
        SPECS[name],
        strip_reasoning=bool(flags.get("strip_reasoning", False)),
        null_text_fix=bool(flags.get("null_text_fix", False)),
    )
