#!/usr/bin/env python3
# coding: utf-8
"""分层提取：把模型「实际表达的答案」取出来，不偏袒任何一种表达方式。

为什么需要这一层
----------------
格式遵循不是我们要测的能力，但它现在深度污染着分数。实测：

    RynnBrain-2B  time      漏答 48%（输出到一半就收尾）
    RynnBrain-2B  轨迹      散文包裹坐标 / 自己训练的 <grasp pose> 标签
    SenseNova     轨迹 2D   95% 用归一化坐标而非像素

试过在**生成阶段**解决（约束解码 + 结构化输出），三轮实验证明不行 ——
它不是剥离了格式变量，而是引入了一个方向因模型而异的新变量：

    Qwen3-VL-8B   无约束 0.4306  →  加约束 0.2265   （退化成零长度区间）
    RynnBrain-2B  无约束 0.0852  →  加约束 0.3050   （漏答被消除）

生成阶段任何干预都会改变模型行为。所以改为：**自由生成，事后提取**。

分层与判据
----------
    L1 严格   合法 JSON 且结构匹配        确定、零成本、覆盖实测 86%
    L2 结构   裸数组 / 圆括号点列          确定、零成本；**仅限有强校验可依的场景**
    L3 模型   LLM 转 JSON                 覆盖剩余 14%，见 extract_llm.py
    都失败 →  记为 format 失败（BC-15 计 0 分）

L2 只保留「能被合理性校验兜住」的提取（轨迹点列：维度、量级、画布范围都可验；
选项原文的最长匹配：无歧义）。**猜测性的启发式一律砍掉** —— 选择题曾用
「答案语境正则」替代冻结版的首字母匹配，结果从「误判成 A」变成「连
"The answer is C." 都拒掉」。规则边界调不完，那类输入交给 L3。

**每一层的候选都必须过合理性校验，不通过就降级。** 这一条是整个设计的关键：
早先那个只抠不验的正则看起来救回 70%，加校验后真实可用只有 20% ——
另外 50% 是把 2D 像素硬切成三元组的假阳性。**假阳性比漏掉更糟**，
它把「明显的格式失败」变成「看似有效、实则完全错误的低分」。

每次提取都记录用了哪一层、以及候选被拒的原因，供审计与偏向检测。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_NESTED_ARRAY = re.compile(r"\[\s*\[[^\[\]]*\](?:\s*,\s*\[[^\[\]]*\])*\s*\]")
_PAREN_POINTS = re.compile(rf"(?:\(\s*{NUMBER}\s*(?:,\s*{NUMBER}\s*){{1,2}}\)\s*,?\s*){{2,}}")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


@dataclass
class Extraction:
    """一次提取的结果与它的来历。``value is None`` 表示没能提出可用答案。"""

    value: Any = None
    layer: str = "none"                       # strict | lenient | model | none
    rejected: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None

    def as_row_fields(self) -> dict[str, Any]:
        """写进结果行的审计字段。用于统计各层占比、检测提取器偏向。"""
        out: dict[str, Any] = {"extract_layer": self.layer}
        if self.rejected:
            out["extract_rejected"] = self.rejected[:5]
        return out


def strip_fence(text: str) -> str:
    return _FENCE.sub("", (text or "").strip()).strip()


_BARE_VALUE = re.compile(r'(:\s*)([A-Za-z][A-Za-z0-9_\-]{0,31})(\s*[,}\]])')
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_SINGLE_QUOTED = re.compile(r"'([^'\\]*)'(\s*[:,}\]])")


def repair_json(text: str) -> str:
    """只做**语法**修复，绝不碰语义。

    实测最常见的一种非法输出是裸的枚举值 —— 模型写 ``{"choice": B, "reason": ...}``，
    字母没加引号。SenseNova 的 understanding 有 267/300 行是这个形态。
    冻结版靠 ``\b([A-Z])\b`` 正则回退救它，而那个取的是全文第一个孤立大写字母，
    只因为 ``"choice"`` 恰好排在 ``"reason"`` 前面才碰巧正确 —— 换个把理由
    写在前面的模型就会静默判错。

    为什么不用 LLM 修这个：实测 DeepSeek 在这类输出上**会绕过 choice 字段自己作答**
    （模型写 `"choice": B`，它读 reason 后返回 D）。80 条里 5 条分歧全是这种。
    提取器替模型答题是最严重的污染。纯语法修复没有这个风险。

    只修三类，都不改变任何已有值：
      裸值加引号   ``{"choice": B}``      → ``{"choice": "B"}``
      去尾逗号     ``[1, 2, ]``           → ``[1, 2]``
      单引号转双   ``{'a': 1}``           → ``{"a": 1}``

    ``true`` / ``false`` / ``null`` 是合法 JSON 字面量，保持原样。
    """
    fixed = _SINGLE_QUOTED.sub(r'"\1"\2', text)
    fixed = _BARE_VALUE.sub(
        lambda m: m.group(1) + (m.group(2) if m.group(2) in ("true", "false", "null")
                                else f'"{m.group(2)}"') + m.group(3),
        fixed,
    )
    return _TRAILING_COMMA.sub(r"\1", fixed)


def json_candidates(text: str) -> list[Any]:
    """从文本里找出所有能独立解析的 JSON 片段，最外层优先。

    原样解析失败时会尝试 :func:`repair_json` 再解析一次 —— 那是纯语法修复。
    """
    cleaned = strip_fence(text)
    out: list[Any] = []
    for candidate in (cleaned, repair_json(cleaned)):
        try:
            out.append(json.loads(candidate))
            break
        except (ValueError, TypeError):
            continue
    # 退一步：找第一个平衡的 {...}
    depth = 0
    start = -1
    for index, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    out.append(json.loads(cleaned[start : index + 1]))
                except ValueError:
                    pass
    return out


# --------------------------------------------------------------------------
# 轨迹
# --------------------------------------------------------------------------


def _point_lists(value: Any) -> list[list[float]]:
    """把任意嵌套结构里的点序列摘出来，不做校验。"""
    if isinstance(value, dict):
        for key in ("points", "trajectory", "coordinates"):
            if key in value:
                return _point_lists(value[key])
        return []
    if not isinstance(value, list) or not value:
        return []
    points: list[list[float]] = []
    for row in value:
        if isinstance(row, (list, tuple)) and all(isinstance(v, (int, float)) for v in row):
            points.append([float(v) for v in row])
    return points


def validate_trajectory(
    points: list[list[float]],
    *,
    dim: int,
    expected_count: int,
    canvas: tuple[int, int] | None = None,
) -> str | None:
    """返回拒绝原因；``None`` 表示通过。

    这些检查专门拦住早先那类假阳性：把 ``<grasp pose> (626,520)`` 这种
    **2D 像素四边形**抠成三元组喂给 3D 任务，数值范围差三个数量级。
    """
    if len(points) < 2:
        return f"点数不足（{len(points)}）"
    widths = {len(p) for p in points}
    if widths != {dim}:
        return f"维度不符（得到 {sorted(widths)}，要求 {dim}）"
    if expected_count and not (expected_count / 4 <= len(points) <= expected_count * 4):
        return f"点数偏离预期过多（{len(points)} vs {expected_count}）"

    magnitude = max(abs(v) for p in points for v in p)
    if dim == 3:
        # 相机系、米。真值范围 x∈[-0.52,0.37] y∈[-0.18,0.29] z∈[0.36,0.74]
        if magnitude > 3.0:
            return f"数值超出米级范围（max={magnitude:.1f}）"
    else:
        if canvas:
            width, height = canvas
            if magnitude <= 1.5:
                return "疑似归一化坐标（全部 ≤1.5），任务要求像素"
            if magnitude > 4 * max(width, height):
                return f"数值远超画布（max={magnitude:.0f}，画布 {width}×{height}）"
    return None


def extract_trajectory(
    text: str,
    *,
    keys: list[str],
    dim: int,
    expected_count: int = 0,
    canvas: tuple[int, int] | None = None,
) -> Extraction:
    """从模型输出里取出 ``keys`` 中某个夹爪的点序列。

    ``keys`` 按优先级给出（通常是 [活动夹爪]）。
    """
    rejected: list[str] = []

    def check(points: list[list[float]], layer: str) -> Extraction | None:
        reason = validate_trajectory(points, dim=dim, expected_count=expected_count, canvas=canvas)
        if reason is None:
            return Extraction(points, layer, rejected)
        rejected.append(f"{layer}: {reason}")
        return None

    # L1 严格：合法 JSON，且键名匹配
    for data in json_candidates(text):
        if not isinstance(data, dict):
            continue
        for key in keys:
            for candidate in (key, key.replace("_gripper", ""), f"{key}_points"):
                if candidate in data:
                    hit = check(_point_lists(data[candidate]), "strict")
                    if hit:
                        return hit

    # L2 宽松：裸数组 / 散文包裹 / 圆括号点列。取点数最多的合法候选。
    best: Extraction | None = None
    for pattern in (_NESTED_ARRAY, _PAREN_POINTS):
        for match in pattern.finditer(text or ""):
            fragment = match.group(0)
            points = _point_lists(_safe_json(fragment))
            if len(points) < 2:
                numbers = [float(x) for x in re.findall(NUMBER, fragment)]
                if len(numbers) >= 2 * dim:
                    points = [numbers[i : i + dim] for i in range(0, len(numbers) - dim + 1, dim)]
            hit = check(points, "lenient")
            if hit and (best is None or len(hit.value) > len(best.value)):
                best = hit
    if best is not None:
        return best

    return Extraction(None, "none", rejected)


def _safe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# 选择题
# --------------------------------------------------------------------------


def extract_choice(text: str, valid_ids: list[str], option_texts: dict[str, str] | None = None) -> Extraction:
    """取出选项字母。

    L2 刻意**不**沿用冻结版的 ``\\b([A-Z])\\b`` —— 那个取全文第一个孤立大写字母，
    英文冠词 "a" 大写后就是合法选项。实测 ``"A robot arm is above..."`` 被判成 A，
    而基线里 32 条输出有 28 条走这条回退路径。这里改为要求字母出现在
    **答案语境**中（answer/choice/option 之后，或独占一行/结尾），
    并且当出现多个互相冲突的候选时**拒绝**而不是取第一个。
    """
    ids = [i.upper() for i in valid_ids]
    rejected: list[str] = []

    # L1 严格：JSON 里的 choice/answer 字段
    for data in json_candidates(text):
        if not isinstance(data, dict):
            continue
        for key in ("choice", "answer", "option", "label"):
            raw = data.get(key)
            if raw is None:
                continue
            token = str(raw).strip().upper()
            if token in ids:
                return Extraction(token, "strict", rejected)
            if option_texts:
                hit = _match_option_text(token, option_texts)
                if hit:
                    return Extraction(hit, "strict", rejected)
            rejected.append(f"strict: 字段 {key}={raw!r} 不是合法选项")

    # L2 已砍掉。冻结版的 `\b([A-Z])\b` 取全文第一个孤立大写字母，
    # 英文冠词 "a" 大写后就是合法选项 —— 实测 "A robot arm is above..." 被判成 A。
    # 我试过用「答案语境」正则替代，结果过度纠正：连 "The answer is C." 都拒掉了。
    # 规则的边界调不完，这类自由文本一律交给 LLM 提取层（L3）。
    #
    # 只保留一条确定性规则：选项原文的**最长**匹配。它没有歧义，
    # 而且能修掉冻结版的子串遮蔽（"move the X" ⊂ "remove the X"，冻结版按 A→F 取第一个）。
    if option_texts:
        hit = _match_option_text(text, option_texts)
        if hit:
            return Extraction(hit, "lenient", rejected)

    return Extraction(None, "none", rejected)


def _match_option_text(text: str, option_texts: dict[str, str]) -> str | None:
    """按文本命中选项，取最长匹配。空文本的选项一律跳过。"""
    normalized = re.sub(r"\s+", " ", (text or "").upper()).strip()
    best_id, best_len = None, 0
    for option_id, option_text in option_texts.items():
        candidate = re.sub(r"\s+", " ", str(option_text or "").upper()).strip()
        if not candidate or candidate == "NONE":
            continue
        if candidate in normalized and len(candidate) > best_len:
            best_id, best_len = option_id.upper(), len(candidate)
    return best_id


# --------------------------------------------------------------------------
# 时间区间
# --------------------------------------------------------------------------


def extract_intervals(text: str, question_ids: list[str]) -> Extraction:
    """取出每个问题 id 的 (start, end)。返回 ``{id: (start, end)}``。

    与冻结版的关键差别：**逐 id 隔离**。冻结版里一行解析失败会抛异常，
    冒泡到组级兜底后给同组全部 6 题写错误行 —— 一个单位后缀吃掉 2% 题量。
    """
    found: dict[str, tuple[float, float]] = {}
    rejected: list[str] = []
    wanted = set(question_ids)

    for data in json_candidates(text):
        rows: list[Any] = []
        if isinstance(data, dict):
            for key in ("answers", "results", "items", "predictions"):
                if isinstance(data.get(key), list):
                    rows = data[key]
                    break
            else:
                rows = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "answer": v}
                        for k, v in data.items() if k in wanted]
        elif isinstance(data, list):
            rows = data
        for index, row in enumerate(rows):
            item_id = _row_id(row) or (question_ids[index] if index < len(question_ids) else None)
            if item_id not in wanted or item_id in found:
                continue
            pair = _interval_from(row)
            if pair is None:
                rejected.append(f"strict: {item_id} 的区间无法解析")
                continue
            found[item_id] = pair
    layer = "strict" if found else "none"

    # L2：对仍然缺失的 id，用 "id ... 数字 - 数字" 的行匹配补齐
    missing = [i for i in question_ids if i not in found]
    if missing:
        pattern = "|".join(re.escape(i) for i in sorted(missing, key=len, reverse=True))
        line = re.compile(rf"({pattern}).*?({NUMBER})\s*(?:-|~|,|to|至)\s*({NUMBER})", re.I | re.S)
        for match in line.finditer(text or ""):
            item_id = match.group(1)
            if item_id in found:
                continue
            try:
                found[item_id] = (float(match.group(2)), float(match.group(3)))
                layer = "lenient" if layer == "none" else layer
            except ValueError:
                rejected.append(f"lenient: {item_id} 的数值无法转换")

    return Extraction(found or None, layer if found else "none", rejected)


def _row_id(row: Any) -> str | None:
    if isinstance(row, dict):
        for key in ("id", "question_id", "item_id"):
            if row.get(key) is not None:
                return str(row[key])
    return None


def _interval_from(row: Any) -> tuple[float, float] | None:
    if isinstance(row, dict):
        for a, b in (("start", "end"), ("start_time", "end_time"), ("from", "to")):
            if a in row and b in row:
                pair = (_seconds(row[a]), _seconds(row[b]))
                return pair if None not in pair else None  # type: ignore[return-value]
        for key in ("answer", "interval", "timestamp", "time"):
            if key in row:
                return _interval_from(str(row[key]))
    if isinstance(row, str):
        match = re.search(rf"({NUMBER})\s*(?:-|~|,|to|至)\s*({NUMBER})", row)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None


def _seconds(value: Any) -> float | None:
    """接受 12.5 / "12.5" / "12.5s" / "00:00:12.500"。冻结版只认前两种。"""
    if isinstance(value, (int, float)):
        return float(value)
    token = str(value).strip().lower().rstrip("s").strip()
    if ":" in token:
        parts = token.split(":")
        try:
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60 + float(part)
            return seconds
        except ValueError:
            return None
    try:
        return float(token)
    except ValueError:
        return None
