#!/usr/bin/env python3
# coding: utf-8
"""标注的语义层 —— 从类别文本到分段记录的全部派生逻辑。

为什么重写这一层、而不重写整个工具
----------------------------------
上游 `video_labeler_timestamp.py` 823 行拆开看是两半：

    纯逻辑 407 行   数据模型、派生、校验、词表解析   ← 已知 bug 全在这里
    GUI    416 行   cv2 窗口、按键、进度条、跳转     ← 无已知问题

而且这一层**可以被证明没写错**：320 份现存标注就是回归语料。
把 `categories.txt` 重放一遍，新实现产出的 narration/objects/main_verbs
必须与现存文件逐字节相同，差异只能是我们声明过要修的 bug。
（`tests/test_label_core_replay.py` 就是干这个的。）

GUI 那 416 行现在没有消费者（我们不标新数据），本机也没有 cv2 与显示器，
改了验证不了 —— 缓做。

修了什么
--------
**B-01 动词后无条件插 "the"**（上游 137-138 行）。词表写 `pick up teapot lid`，
代码产出 `Pick the up teapot lid.`，把介词当成了宾语的第一个词。
只影响词表里第二个词是介词/副词的条目 —— 全量八族只有 tea2 的 3 条中招。

**B-02 `ACTION_VERBS` 硬编码**。上游写死 20 个动词；实测 wash 的数据里有
`main_verbs=['wipe']`，而这份代码的表里**没有 wipe** —— 说明标 wash 时用的
是另一个版本，改过表却没回流。硬编码必然导致这种漂移，改为随词表走。

**B-03 `CLASS_KEYS` 只有 19 个键，却允许 20 个类别**（上游 18-19 行 vs 165 行），
恰好 20 类会 IndexError。

没修、留给 GUI 层的
-------------------
撤销删错段（上游对**排序后**的列表 pop，乱序标注时会删错）—— 那是交互状态管理，
属于 GUI 那 416 行。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_VERSION = "bench-label-core/1"

# 类别行前缀：N1: / C1: / M1: / label1 - / 全角冒号
_PREFIX = re.compile(r"^\s*[A-Za-z]*\d+\s*[:：.\-]\s*")
_ARTICLES = {"a", "an", "the"}

# B-01：这些词跟在动词后面时是**动词短语的一部分**，不是宾语。
# 上游没有这个概念，于是把 "pick up teapot lid" 拆成
# 动词 pick + 宾语 "up teapot lid"，再插冠词成 "pick the up teapot lid"。
_PARTICLES = {"up", "down", "in", "on", "out", "off", "over", "away",
              "aside", "into", "onto", "back", "together", "apart"}


def normalize_class_text(line: str) -> str:
    """去掉编号前缀，压空格。"""
    return " ".join(_PREFIX.sub("", str(line)).split())


def load_categories(text: str) -> list[str]:
    """从词表文件内容解析类别列表，保序去重。"""
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = normalize_class_text(raw)
        if not line or line.startswith("#"):
            continue
        key = line.lower()
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out


def action_verbs(categories: list[str]) -> set[str]:
    """B-02：动词表**从词表本身推导**，不硬编码。

    取每条类别的首词。这样词表里出现 `wipe` / `set` 这类上游没预见的动词时，
    行为自动正确，不会像 wash 那样要改代码、改完又漂移。
    """
    return {c.split()[0].lower() for c in categories if c.split()}


def describe(category: str, verbs: set[str] | None = None) -> tuple[list[str], list[str], str]:
    """类别文本 → (objects, main_verbs, narration)。

    **不再用于产出标注**（D-25 之后段里只存 subtask 引用）。保留它有两个用途：
    出题时按需推导动词/宾语（例如生成规则型干扰项），以及对
    ``segments.before_subtask_id/`` 做回归比对。


    这是上游 `build_segment_description` 的重写。规则：

        首词在动词表内  → main_verbs=[首词]，其余为宾语，narration 补冠词
        首词不在表内    → objects/main_verbs 为空，narration 原样（仅首字母大写 + 句号）

    补冠词时**跳过介词/副词**（B-01）：

        pick up teapot lid  →  Pick up the teapot lid.    （而非 Pick the up teapot lid.）
        pick bowl           →  Pick the bowl.
        pick the bowl       →  Pick the bowl.             （已有冠词则不动）
    """
    cleaned = normalize_class_text(category).rstrip(".")
    if not cleaned:
        return [], [], ""
    words = cleaned.split()
    verbs = verbs if verbs is not None else {words[0].lower()}

    objects: list[str] = []
    main_verbs: list[str] = []
    body = cleaned

    if words[0].lower() in verbs:
        main_verbs = [words[0].lower()]
        rest = words[1:]
        # B-01：介词属于动词短语，连同动词一起留在前面，不进宾语
        lead: list[str] = []
        while rest and rest[0].lower() in _PARTICLES:
            lead.append(rest.pop(0))
        if rest and rest[0].lower() in _ARTICLES:
            article = rest.pop(0)
        else:
            article = "the" if rest else ""
        if rest:
            objects = [" ".join(rest)]
            body = " ".join([words[0], *lead, article, *rest])
        else:
            body = " ".join([words[0], *lead])

    body = " ".join(body.split())
    narration = f" {body[0].upper()}{body[1:]}." if body else ""
    return objects, main_verbs, narration


def format_timestamp(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours, rem = divmod(int(total), 3600)
    minutes, secs = divmod(rem, 60)
    millis = int(round((total - int(total)) * 1000))
    if millis == 1000:                       # 进位，否则会出现 .1000
        secs, millis = secs + 1, 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def frame_to_time(frame: int, fps: float) -> float:
    return round(frame / fps, 3) if fps > 0 else 0.0


@dataclass
class Segment:
    """一个动作分段。

    ``start_frame`` / ``end_frame`` 是权威（人工输入），秒与时间串全部派生。
    ``subtask`` 是**引用**，不是文字 —— 文字只存在于 ``subtasks.json`` 一处（D-25）。
    """
    start_frame: int
    end_frame: int                # 闭区间：该帧本身属于这一段
    subtask: str                  # subtasks.json 里的 id，不是渲染后的字符串
    episode_index: int | None = None

    def as_record(self, index: int, video_stem: str, fps: float) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": f"{re.sub(r'_h264$', '', video_stem)}-{index}",
            "subtask": self.subtask,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            # end 用 end_frame+1：闭区间转半开。
            # ⚠ 这会让相邻段的**秒区间**重叠一帧（P-04，全量 631 处）。
            # 保留上游语义（否则与 320 份现存标注对不上），
            # 但下游判「落在恰好一个段内」必须走帧号，不能走秒。
            "start": frame_to_time(self.start_frame, fps),
            "end": frame_to_time(self.end_frame + 1, fps),
            "start_time": format_timestamp(frame_to_time(self.start_frame, fps)),
            "end_time": format_timestamp(frame_to_time(self.end_frame + 1, fps)),
        }
        if self.episode_index is not None:
            record["episode_index"] = self.episode_index
        return record


def can_add(new: Segment, existing: list[Segment]) -> tuple[bool, str]:
    """能否加入这一段。允许共享边界，拒绝真重叠。

    共享边界（前段 ``end_frame`` == 后段 ``start_frame``）是上游的既有语义，
    保留 —— 动作之间本来就是连续过渡。
    """
    if new.end_frame < new.start_frame:
        return False, "end_frame 小于 start_frame"
    for seg in existing:
        if new.start_frame == seg.end_frame or new.end_frame == seg.start_frame:
            continue                                  # 共享边界，放行
        if new.start_frame <= seg.end_frame and seg.start_frame <= new.end_frame:
            return False, f"与 {seg.start_frame}-{seg.end_frame} 重叠"
    return True, ""


def make_id(text: str) -> str:
    """subtask 的 ID：去掉冠词，不加序号。

    去冠词是因为 ID 是标识符不是句子；不加序号是因为序号会与列表顺序耦合，
    而 **ID 必须稳定** —— 那是它的全部价值（修措辞时所有引用自动跟随）。
    """
    words = [w for w in re.split(r"[^A-Za-z0-9]+", text.strip().rstrip(".").lower()) if w]
    kept = [w for w in words if w not in _ARTICLES]
    return "_".join(kept or words)


def build_subtasks(texts: list[str]) -> list[dict[str, str]]:
    """从若干动作描述生成 subtask 定义（保序去重）。ID 冲突时抛错，不静默合并。"""
    out: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    for text in texts:
        clean = " ".join(str(text).split())
        if not clean or clean in {o["text"] for o in out}:
            continue
        sid = make_id(clean)
        if sid in seen:
            raise ValueError(f"subtask ID 冲突：{clean!r} 与 {seen[sid]!r} 都得到 {sid!r}")
        seen[sid] = clean
        out.append({"id": sid, "text": clean})
    return out


def build_document(video_stem: str, segments: list[Segment], *, fps: float,
                   total_frames: int, subtasks: list[dict[str, str]], video_rel: str,
                   video_sha256: str = "", episode_bounds: list[list[float]] | None = None
                   ) -> dict[str, Any]:
    """产出符合 ``schemas/segments.json`` 的完整文档。

    ``source`` 块是**必填**的 —— 缺了它就无法回答「这份标注用的哪版工具、
    哪份词表、对哪段视频」。wash 的版本漂移正是因为上游没有这一块。
    """
    known = {s["id"] for s in subtasks}
    unknown = {s.subtask for s in segments} - known
    if unknown:
        raise ValueError(f"引用了未定义的 subtask：{sorted(unknown)}")
    ordered = sorted(segments, key=lambda s: (s.start_frame, s.end_frame))
    return {
        "source": {
            "video": video_rel,
            "video_sha256": video_sha256,
            "fps": fps,
            "total_frames": total_frames,
            "tool_version": TOOL_VERSION,
            "subtasks_sha256": hashlib.sha256(
                json.dumps(subtasks, ensure_ascii=False, sort_keys=True)
                .encode("utf-8")).hexdigest()[:16],
            "episode_bounds": episode_bounds,
        },
        "segments": [s.as_record(i + 1, video_stem, fps) for i, s in enumerate(ordered)],
    }
