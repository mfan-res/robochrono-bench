#!/usr/bin/env python3
# coding: utf-8
"""Trajectory：夹爪轨迹预测（2D 像素坐标 / 3D 相机坐标两套输入）。

九个任务里唯一的回归任务，也是唯一带自适应重试的 —— 2D 预测点越界时会
带着越界信息重问一次。指标是 Hausdorff / discrete Fréchet / Chamfer 三个距离，
经 100/(1+(d/tol)^2) 映射成 0~100 分。

几何函数、容差规则、prompt 与汇总口径均从 test/trajectory_glm_test.py 逐字搬运。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..parsing import first_json_object
from .base import CallContext, Unit, base_row, image_part, one_item_per_unit, text_part

MAX_COORDINATE_RETRIES = 1


# --------------------------------------------------------------------------
# 几何：逐字搬运
# --------------------------------------------------------------------------


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
    rows, cols = len(a), len(b)
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


def mean_available(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    return sum(finite) / len(finite) if finite else None


def path_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(euclidean(points[i], points[i + 1]) for i in range(len(points) - 1))


def flatten_points(trajectories: dict[str, list[list[float]]]) -> list[list[float]]:
    return trajectories.get("left_gripper", []) + trajectories.get("right_gripper", [])


def point_cloud_extent(points: list[list[float]]) -> float | None:
    if not points:
        return None
    dim = len(points[0])
    mins = [min(p[i] for p in points) for i in range(dim)]
    maxs = [max(p[i] for p in points) for i in range(dim)]
    return math.sqrt(sum((maxs[i] - mins[i]) ** 2 for i in range(dim)))


# --------------------------------------------------------------------------
# item 读取：逐字搬运
# --------------------------------------------------------------------------


def image_label_from_key(key: Any) -> str:
    return str(key).split(".")[-1]


def infer_dimension(item: dict[str, Any]) -> int:
    answer = item.get("answer") or item.get("A") or {}
    frame = str(answer.get("coordinate_frame", "")).lower() if isinstance(answer, dict) else ""
    return 2 if ("image" in frame or "pixel" in frame) else 3


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
                left.append([float(v) for v in row["left_gripper_xyz"]])
            if isinstance(row.get("right_gripper_xyz"), list):
                right.append([float(v) for v in row["right_gripper_xyz"]])
    return {"left_gripper": left, "right_gripper": right}


def active_gripper_for_item(item: dict[str, Any], expected: dict[str, list[list[float]]] | None = None) -> str:
    answer = item.get("answer") or item.get("A") or {}
    for source in (item, answer if isinstance(answer, dict) else {}):
        if not isinstance(source, dict):
            continue
        active = source.get("active_gripper")
        if active in {"left_gripper", "right_gripper", "both", "unknown"}:
            return str(active)
        metadata = source.get("active_gripper_metadata")
        if isinstance(metadata, dict) and metadata.get("active_gripper") in {
            "left_gripper", "right_gripper", "both", "unknown",
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
    if active_gripper in {"left_gripper", "right_gripper"}:
        return [active_gripper]
    if active_gripper == "both":
        return ["left_gripper", "right_gripper"]
    return []


def main_view_image_path_for_item(item: dict[str, Any]) -> Path | None:
    for key in ("prediction_image", "main_image", "image"):
        if item.get(key):
            return Path(str(item[key]))
    images = item.get("images")
    if isinstance(images, dict) and images:
        prediction_view = item.get("prediction_view") or item.get("primary_view")
        label = image_label_from_key(prediction_view) if prediction_view else ""
        for key in (prediction_view, label):
            if key and images.get(key):
                return Path(str(images[key]))
    return None


def image_size_from_file(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except (FileNotFoundError, OSError, ValueError):
        return None, None


def image_size_from_intrinsics(item: dict[str, Any]) -> tuple[int | None, int | None]:
    answer = item.get("answer") or item.get("A") or {}
    if not isinstance(answer, dict):
        return None, None
    intrinsics = answer.get("camera_intrinsics") or {}
    if not isinstance(intrinsics, dict):
        return None, None
    try:
        return int(intrinsics.get("width")), int(intrinsics.get("height"))
    except (TypeError, ValueError):
        return None, None


def image_size_for_item(item: dict[str, Any]) -> tuple[int | None, int | None]:
    path = main_view_image_path_for_item(item)
    if path is not None:
        width, height = image_size_from_file(path)
        if width is not None and height is not None:
            return width, height
    return image_size_from_intrinsics(item)


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
    """主视角在前、上下文视角在后。视角名从数据里读，**不硬编码** ——
    生成器已改为让视角跟随 config 的 views，旧数据是
    left_eye/right_eye/left_wrist，新数据是 left_eye/left_wrist/right_wrist。"""
    images = item.get("images")
    if isinstance(images, dict) and images:
        primary_key = primary_image_key_for_item(item, images)
        ordered: list[Any] = []
        if primary_key is not None:
            ordered.append(primary_key)
        ordered.extend(k for k, _ in sorted(images.items()) if k != primary_key)
        return [
            {
                "label": image_label_from_key(k),
                "path": str(images[k]),
                "role": "primary" if k == primary_key else "context",
            }
            for k in ordered
        ]
    if item.get("image"):
        return [{"label": "main_view", "path": str(item["image"]), "role": "primary"}]
    data = item.get("input", {})
    if isinstance(data, dict):
        if isinstance(data.get("image_paths"), list):
            return [
                {
                    "label": "main_view" if i == 0 else f"context_view_{i}",
                    "path": str(p),
                    "role": "primary" if i == 0 else "context",
                }
                for i, p in enumerate(data["image_paths"])
            ]
        if data.get("image_path"):
            return [{"label": "main_view", "path": str(data["image_path"]), "role": "primary"}]
    raise ValueError(f"Cannot find image paths for item {item.get('id')}")


# --------------------------------------------------------------------------
# prompt / 解析：逐字搬运
# --------------------------------------------------------------------------


def build_prompt(question: str, item: dict[str, Any], example_style: str = "legacy",
                 include_initial_pose: bool = False) -> str:
    """组 prompt。``example_style`` 见 BC-16 候选实验。

    ``legacy``      冻结版写法：schema 里给一个具体数值点，如 ``[[0.12, -0.03, 0.45]]``
    ``placeholder`` 占位符 + 多点：让「照抄示例」在语法上就不成立

    实测冻结写法导致 **3D 上 58% 的预测是逐字抄回示例点**
    （SenseNova 92%、Qwen3-VL-8B 74%）—— 那个 schema 看起来不像格式说明，
    像一个填好的答案。详见报告第五节。
    """
    dim = infer_dimension(item)
    expected = expected_trajectory(item)
    active_gripper = active_gripper_for_item(item, expected)
    # anchor 写法（团队提议）：把 schema 里的示例点换成**这道题真实的初始位姿**。
    # 同时解决两件事：骨架仍是合法 JSON（2B 模型不会掉出 JSON 模式，这是
    # placeholder/caps/zeros 三种写法全部失败的原因），而示例逐题变化、
    # 抄它等于给出一个正确的首点而非错误的固定常数。
    anchor_point: list[float] | None = None
    if example_style in {"anchor", "proposal"} or include_initial_pose:
        _start = expected.get(active_gripper if active_gripper != "both" else "left_gripper") or []
        anchor_point = list(_start[0]) if _start else None

    _scored_pre = grippers_to_score(active_gripper)
    _point_count = sum(len(expected.get(k, [])) for k in _scored_pre) or 10
    _stamps = [f.get("timestamp") for f in ((item.get("answer") or item.get("A") or {})
                                            .get("trajectory") or [])
               if isinstance(f, dict) and isinstance(f.get("timestamp"), (int, float))]
    _duration = (_stamps[-1] - _stamps[0]) if len(_stamps) > 1 else None

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
        example_point = {
            "legacy": "[123.4, 256.7]",
            "placeholder": "[<u1>, <v1>], [<u2>, <v2>], [<u3>, <v3>], ...",
            "caps": "[U1, V1], [U2, V2], [U3, V3], ...",
            # 完整合法 JSON 骨架 + 全零：保住格式锚定，抄了也是 0 分且可检测
            "zeros": ", ".join(["[0.0, 0.0]"] * 10),
            "anchor": ("[%s]" % ", ".join(f"{v:.1f}" for v in anchor_point)
                       if anchor_point else "[123.4, 256.7]"),
            "proposal": (", ".join(["[%s]" % ", ".join(f"{v:.1f}" for v in anchor_point)]
                                   * max(2, _point_count))
                         if anchor_point else "[123.4, 256.7]"),
        }[example_style]
    else:
        coordinate_note = "Use 3D camera-frame coordinates in meters [x, y, z]."
        example_point = {
            "legacy": "[0.12, -0.03, 0.45]",
            "placeholder": "[<x1>, <y1>, <z1>], [<x2>, <y2>, <z2>], [<x3>, <y3>, <z3>], ...",
            "caps": "[X1, Y1, Z1], [X2, Y2, Z2], [X3, Y3, Z3], ...",
            "zeros": ", ".join(["[0.0, 0.0, 0.0]"] * 10),
            "anchor": ("[%s]" % ", ".join(f"{v:.3f}" for v in anchor_point)
                       if anchor_point else "[0.12, -0.03, 0.45]"),
            "proposal": (", ".join(["[%s]" % ", ".join(f"{v:.3f}" for v in anchor_point)]
                                   * max(2, _point_count))
                         if anchor_point else "[0.12, -0.03, 0.45]"),
        }[example_style]

    # 末端初始位姿。真实机器人系统本来就有本体感知，给出它可以锚定坐标系原点与尺度
    # —— 实测「去中心」对齐能让分数涨 7 倍，说明系统性原点偏移是主要误差来源之一。
    # 代价：真值第 1 个点就是初始位姿，等于白送 1/10；原地重复它即可得 4.02 分（3D），
    # 高于当前全部 15 个模型。所以这是个需要权衡的选项，默认关闭。
    pose_note = ""
    if include_initial_pose:
        start = expected.get(active_gripper if active_gripper != "both" else "left_gripper") or []
        if start:
            coords = ", ".join(f"{v:.3f}" for v in start[0])
            pose_note = (
                f"\nThe {active_gripper} is currently at [{coords}] in exactly the coordinate "
                f"system described above. Use it to anchor your predictions; your first point "
                f"should be at or near this position.\n"
            )

    scored = grippers_to_score(active_gripper)
    point_count = sum(len(expected[g]) for g in scored)
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

    # 尖括号写法（placeholder）实测会把 RynnBrain 带进它自己的 `<grasp pose>` 标签模式，
    # 让它改吐散文而不是 JSON。caps 用大写字母占位，避开与模型专有标签的冲突。
    placeholder_note = {
        "legacy": "",
        "placeholder": ("The angle-bracket tokens below are placeholders for the numbers you "
                        "must produce; replace every one of them. Do not copy them literally.\n"),
        "caps": ("The capital letters below are placeholders for the numbers you must produce. "
                 "Replace every one of them with a real coordinate. "
                 "Output must be valid JSON containing only numbers.\n"),
        "zeros": ("The zeros below are placeholders showing the required shape. "
                  "Replace every zero with a real coordinate; do not return zeros.\n"),
        "proposal": (
            (f"The action takes about {_duration:.1f} seconds. " if _duration else "")
            + f"Return exactly {_point_count} points, uniformly spaced in time across it.\n"
            + (f"The {active_gripper} is currently at "
               f"[{', '.join(f'{v:.3f}' for v in anchor_point)}]. This is point 1 of your "
               f"answer -- keep it unchanged. Points 2 to {_point_count} are what you must "
               f"predict; the schema repeats the current position only to show the required "
               f"shape, do NOT return {_point_count} identical points.\n"
               if anchor_point else "")),
        "anchor": ("The single point shown in the schema below is the gripper's CURRENT "
                   "position, i.e. the first point of your answer. Continue the trajectory "
                   "from it — return that point plus the remaining points.\n"),
    }[example_style]

    # proposal 自带更严格的点数说明，旧的 "approximately N" 会与之矛盾
    if example_style == "proposal":
        count_instruction = ""

    return f"""You are evaluating a robot manipulation trajectory prediction task.

Question:
{question}

{gripper_instruction}
{coordinate_note}
{count_instruction}{pose_note}

Output JSON only. Do not use Markdown.
{placeholder_note}Required schema:
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
    if len(value) >= dim and all(isinstance(value[i], (int, float)) for i in range(dim)):
        try:
            return [[float(value[i]) for i in range(dim)]]
        except (TypeError, ValueError):
            return []
    points: list[list[float]] = []
    for row in value:
        if isinstance(row, dict):
            values = [row.get("u"), row.get("v")] if dim == 2 else [row.get("x"), row.get("y"), row.get("z")]
        else:
            values = row
        if not isinstance(values, list) or len(values) < dim:
            continue
        try:
            points.append([float(values[i]) for i in range(dim)])
        except (TypeError, ValueError):
            continue
    return points


def parse_model_answer(text: str, dim: int) -> dict[str, Any]:
    data = first_json_object(text)
    left = (
        data.get("left_gripper") or data.get("left")
        or data.get("left_gripper_points") or data.get("left_trajectory") or []
    )
    right = (
        data.get("right_gripper") or data.get("right")
        or data.get("right_gripper_points") or data.get("right_trajectory") or []
    )
    return {
        "parsed": data,
        "trajectory": {
            "left_gripper": coerce_point_list(left, dim),
            "right_gripper": coerce_point_list(right, dim),
        },
    }


def out_of_bounds_points(
    prediction: dict[str, Any], width: int | None, height: int | None, grippers: list[str] | None = None
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
    original_prompt: str, model_text: str, invalid_points: list[str], width: int, height: int
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


# --------------------------------------------------------------------------
# 打分：逐字搬运
# --------------------------------------------------------------------------


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


def score_curve(
    expected: list[list[float]],
    predicted: list[list[float]],
    tolerance: float,
    zero_unscored: bool = True,
) -> dict[str, Any]:
    """比较两条曲线。空预测的处理见 BC-15。

    ``zero_unscored``（BC-15，默认开启）：真值存在、模型却没给出任何可用点时
    记 **0 分**，而不是 ``None``。

    为什么这不是「宽容」而是「对齐」：``None`` 会被 ``mean_available`` 跳过，
    于是这道题**悄悄退出分母** —— 交白卷或答错维度的模型分数被抬高，
    认真答了但画歪的反而吃亏。选择题（解析不出＝答错，计入 total）与
    time（没输出＝0，计入 total）本来就是这个口径，轨迹是唯一的例外。

    守卫：只在 ``expected`` 非空时才记 0。真值本身为空是**我们的数据问题**，
    不该算到模型头上，那种情况仍然返回 None。
    """
    hausdorff = hausdorff_distance(expected, predicted)
    frechet = discrete_frechet_distance(expected, predicted)
    chamfer = chamfer_distance(expected, predicted)
    metric_scores = {
        "hausdorff": rounded(distance_to_score(hausdorff, tolerance)),
        "discrete_frechet": rounded(distance_to_score(frechet, tolerance)),
        "chamfer": rounded(distance_to_score(chamfer, tolerance)),
    }
    unscored = zero_unscored and bool(expected) and not predicted
    if unscored:
        metric_scores = {key: 0.0 for key in metric_scores}
    # 只在为真时才写这个键：正常行必须与冻结版逐字节一致，
    # 否则「BC 不触发 → 结果不变」这个回归前提就没了。
    return {
        **({"format_failure": True} if unscored else {}),
        "expected_points": len(expected),
        "predicted_points": len(predicted),
        "hausdorff": rounded(hausdorff),
        "discrete_frechet": rounded(frechet),
        "chamfer": rounded(chamfer),
        "metric_scores": metric_scores,
        "score": rounded(mean_available(list(metric_scores.values()))),
    }


def score_prediction(
    item: dict[str, Any], prediction: dict[str, Any], zero_unscored: bool = True
) -> dict[str, Any]:
    expected = expected_trajectory(item)
    active_gripper = active_gripper_for_item(item, expected)
    scored_grippers = grippers_to_score(active_gripper)
    expected_scored = {g: expected.get(g, []) for g in scored_grippers}
    predicted = prediction.get("trajectory", {})
    if not isinstance(predicted, dict):
        predicted = {}
    tolerance = score_tolerance_for_item(item, expected_scored)
    gripper_metrics = {
        g: score_curve(expected.get(g, []), predicted.get(g, []), tolerance,
                       zero_unscored=zero_unscored)
        for g in scored_grippers
    }
    width, height = image_size_for_item(item) if infer_dimension(item) == 2 else (None, None)
    invalid_points = out_of_bounds_points(prediction, width, height, scored_grippers)

    mean_metric_scores = {
        key: rounded(mean_available([row["metric_scores"][key] for row in gripper_metrics.values()]))
        for key in ("hausdorff", "discrete_frechet", "chamfer")
    }
    mean_metrics = {
        key: rounded(mean_available([row[key] for row in gripper_metrics.values()]))
        for key in ("hausdorff", "discrete_frechet", "chamfer")
    }

    return {
        "dimension": infer_dimension(item),
        "active_gripper": active_gripper,
        "scored_grippers": scored_grippers,
        "image_size": {"width": width, "height": height} if width is not None and height is not None else None,
        "score_mapping": {
            "function": "100 / (1 + (distance / tolerance)^2)",
            "tolerance": rounded(tolerance),
            "tolerance_rule": "2D uses 5% of image diagonal; 3D uses max(0.02 m, 10% of GT trajectory extent).",
        },
        "out_of_bounds": {"count": len(invalid_points), "examples": invalid_points[:12]},
        "expected_trajectory": expected,
        "expected_scored_trajectory": expected_scored,
        "predicted_trajectory": predicted,
        "predicted_scored_trajectory": {g: predicted.get(g, []) for g in scored_grippers},
        "gripper_metrics": gripper_metrics,
        "active_gripper_metrics": gripper_metrics.get(active_gripper),
        "mean_metrics": mean_metrics,
        "mean_metric_scores": mean_metric_scores,
        "score": rounded(mean_available(list(mean_metric_scores.values()))),
        # 同上：只在真的失败时才出现，正常行不带这个键
        **({"format_failure": True}
           if any(m.get("format_failure") for m in gripper_metrics.values()) else {}),
    }


# --------------------------------------------------------------------------
# 任务
# --------------------------------------------------------------------------


class TrajectoryTask:
    """2D 与 3D 共用同一实现，维度由 item 的 coordinate_frame 决定。"""

    def __init__(self, name: str = "trajectory", zero_unscored: bool = True,
                 example_style: str = "legacy", include_initial_pose: bool = False,
                 **_flags: Any) -> None:
        self.name = name
        self.example_style = example_style
        self.include_initial_pose = include_initial_pose
        # BC-15：format 不合规（空数组 / 维度错 / 空输出）计 0 分并计入分母。
        # 默认开启 —— 关闭时回到冻结版口径，用于和旧数据对比。
        self.zero_unscored = zero_unscored

    def units(self, items: list[dict[str, Any]]) -> list[Unit]:
        return one_item_per_unit(items)

    def parts(self, unit: Unit) -> list[dict[str, Any]]:
        item = unit.items[0]
        prompt = build_prompt(str(item.get("Q") or item.get("question")), item, self.example_style, self.include_initial_pose)
        return self._parts_with_prompt(item, prompt)

    def _parts_with_prompt(self, item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for image_input in image_inputs_for_item(item):
            if image_input.get("role") == "primary":
                parts.append(text_part("Primary view image:"))
            else:
                parts.append(text_part(f"Context view {image_input['label']}:"))
            parts.append(image_part(image_input["path"]))
        parts.append(text_part(prompt))
        return parts

    def retry_parts(self, unit: Unit, text: str, attempt: int) -> list[dict[str, Any]] | None:
        """2D 预测点越界时重问一次。返回 None 表示不需要重试。"""
        item = unit.items[0]
        if attempt >= MAX_COORDINATE_RETRIES or infer_dimension(item) != 2:
            return None
        width, height = image_size_for_item(item)
        if width is None or height is None:
            return None
        prediction = parse_model_answer(text, 2)
        scored = grippers_to_score(active_gripper_for_item(item))
        invalid = out_of_bounds_points(prediction, width, height, scored)
        if not invalid:
            return None
        prompt = build_prompt(str(item.get("Q") or item.get("question")), item)
        retry_prompt = build_retry_prompt(prompt, text, invalid, width, height)
        return self._parts_with_prompt(item, retry_prompt)

    def rows(self, unit: Unit, text: str, ctx: CallContext) -> list[dict[str, Any]]:
        item = unit.items[0]
        prompt = build_prompt(str(item.get("Q") or item.get("question")), item, self.example_style, self.include_initial_pose)
        prediction = parse_model_answer(text, infer_dimension(item))
        row = base_row(item, prompt, text, ctx)
        row["model_prediction"] = prediction.get("parsed")
        row.update(score_prediction(item, prediction, zero_unscored=self.zero_unscored))
        row["parse_ok"] = bool(prediction.get("parsed"))
        return [row]

    def error_rows(self, unit: Unit, error: str) -> list[dict[str, Any]]:
        item = unit.items[0]
        prompt = build_prompt(str(item.get("Q") or item.get("question")), item)
        row = base_row(item, prompt, None, None)
        row["model_prediction"] = None
        # 执行失败的行不套 BC-15：它没产出任何输出，属于「没跑成」而不是
        # 「答得不合格」，计 0 会把基础设施故障混进模型分数。
        row.update(score_prediction(item, {"trajectory": {}}, zero_unscored=False))
        row["error"] = error
        row["parse_ok"] = False
        return [row]

    def summarize(self, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        answered = [r for r in rows if r.get("model_output")]
        mean_metrics = [r.get("mean_metrics", {}) for r in answered]
        mean_metric_scores = [r.get("mean_metric_scores", {}) for r in answered]

        # BC-15 第二半：分母从 answered 改成**执行成功的全部行**。
        # 只改 score_curve 不够 —— 模型返回空字符串时根本进不了 answered，
        # 那 34 行（RynnBrain trajectory_2D）会照样从分母里消失。
        # 与选择题的 `÷total`、time 的「缺失记 0 计入 total」对齐。
        # 仍然排除带 error 的行：那是没跑成，不是答得不合格。
        if self.zero_unscored:
            counted = [r for r in rows if not r.get("error")]
            scores = [float(r.get("score") or 0.0) for r in counted]
            mean_score = rounded(sum(scores) / len(scores)) if scores else None
        else:
            mean_score = rounded(mean_available([r.get("score") for r in answered]))
        return {
            "total": len(rows),
            "answered": len(answered),
            "errors": sum(1 for r in rows if r.get("error")),
            "mean_hausdorff": rounded(mean_available([m.get("hausdorff") for m in mean_metrics])),
            "mean_discrete_frechet": rounded(mean_available([m.get("discrete_frechet") for m in mean_metrics])),
            "mean_chamfer": rounded(mean_available([m.get("chamfer") for m in mean_metrics])),
            "mean_hausdorff_score": rounded(mean_available([m.get("hausdorff") for m in mean_metric_scores])),
            "mean_discrete_frechet_score": rounded(
                mean_available([m.get("discrete_frechet") for m in mean_metric_scores])
            ),
            "mean_chamfer_score": rounded(mean_available([m.get("chamfer") for m in mean_metric_scores])),
            "mean_score": mean_score,
            # 实际参与打分的行数。可能 < answered —— 一行可以「无 error、
            # parse_ok=True、计入 answered」却仍然算不出 score：模型输出散文里
            # 夹了几个坐标，解析器抠到了、标记成功，但凑不成一条可打分的轨迹。
            # 全量实测 61/1840 行（3.3%）是这种，RynnBrain 的 2D 高达 11%。
            # 不把它显式报出来，mean_score 就是在一个悄悄变小的分母上取的平均，
            # 而看榜的人以为两个模型的打分基数一样。
            "scored": sum(1 for r in answered if r.get("score") is not None),
            # format 不合规的行数：输出了但没给出可用轨迹（空数组 / 维度错 / 空输出）
            "format_failures": sum(
                1 for r in rows
                if not r.get("error") and (r.get("format_failure") or not r.get("model_output"))
            ),
            "zero_unscored": self.zero_unscored,
            "elapsed_seconds": round(elapsed, 3),
            "parse_failure_rate": (
                sum(1 for r in rows if not r.get("parse_ok")) / len(rows) if rows else 0.0
            ),
            # parse_ok 只说「解析器没抛错」，不说「结果可用」。两者的差就是上面那 3.3%。
            "unscored_rate": (
                sum(1 for r in rows if not r.get("error") and r.get("score") is None) / len(rows)
                if rows else 0.0
            ),
        }


def build(name: str = "trajectory", **flags: Any) -> TrajectoryTask:
    # 只挑本任务认识的开关，其余（strip_reasoning 等选择题专用）忽略
    return TrajectoryTask(name, zero_unscored=bool(flags.get("zero_unscored", True)),
                          example_style=str(flags.get("example_style", "legacy")))
