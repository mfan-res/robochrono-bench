#!/usr/bin/env python3
# coding: utf-8
"""按文件名解析媒体路径。

问题
----
QA JSON 里的媒体路径是**生成机上的路径**，三个新族各不相同：

    airpods   time_video_crop_top/left_eye/file-000.mp4      相对，缺族目录前缀
    express   time_video_crop_top/left_eye/file-000.mp4      前缀 + 子目录实为 left_eye_compress
    tea2      /ssd/yyywv/workflow_outputs/.../file-000.mp4   生成机绝对路径 + 子目录改名

实测 18,635 个媒体引用里 13,515 个（72.5%）在本地解析不到 —— 但**文件都在**，
只是路径对不上。现有的 ``tools/normalize_qa_paths.py`` 是按前缀重写的，
处理不了 ``left_eye`` → ``left_eye_compress`` 这种子目录改名。

做法
----
在**该族自己的目录树内**按文件名建索引，解析时按文件名查。实测按文件名
能找到 99%（airpods 3885/3925、express 3924/3974、tea2 2996/3016）。

三条原则：

1. **只在原路径不存在时才解析。** 原路径能用就一个字不动 ——
   stack_cubes 已经规范化过，不该被二次改写。
2. **重名不猜。** 同名文件出现在多处时**不解析**并记录下来，交人工判断。
   猜错会让模型看到错误的媒体，而且不报错 —— 比解析失败更糟。
3. **不改磁盘上的 JSON。** 在内存里替换。数据集 61 GB 且可重下，
   改写会让本地与远端分叉（下载器已经为此加过一次保护）。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MEDIA_SUFFIXES = {".mp4", ".jpg", ".jpeg", ".png", ".webp", ".avi", ".mov"}
_MEDIA_RE = re.compile(r"\.(?:mp4|jpe?g|png|webp|avi|mov)$", re.I)
# 这些键存的是题干/答案文本，里面可能出现文件名，不该当路径处理
SKIP_KEYS = {"Q", "question", "A", "answer_text", "reason", "prompt"}

# 溯源字段：指向生成机上的原始素材，**从未随数据集发布**，评测也不读它们。
# 单独标出来，免得它们把「可用率」压低而掩盖真正的问题。
# （stack_cubes 曾因此显示 150 个「缺失」，实际评测用的 input.video_paths 是好的。）
PROVENANCE_KEYS = {
    "original_video_path", "original_video_paths",
    "source_video_path", "source_video_paths",
    "prejoined_video_path", "prejoined_video_paths",
    "left_eye", "right_eye", "left_wrist", "right_wrist",
    "cam_color", "wrist_l", "wrist_r",          # pen_inbox / gift_inhand 用的名字
}

# 光按字段名判断不够：``video_path`` 在 time 任务里是评测要用的字段，
# 在 left_right / planning_2 等任务里却指向未发布的原始素材 —— 一刀切会误伤。
# 改为按**路径内容**判断：lerobot 原始视频一律形如
# ``…/videos/observation.images.<view>/…``，实测八个族共 146,573 个引用，
# 无一例外，且本地一个都没有。
PROVENANCE_PATH_MARKERS = ("observation.images",)


def is_provenance_path(value: str) -> bool:
    """这条路径是否指向未随数据集发布的原始素材。"""
    normalized = str(value).replace("\\", "/")
    return any(marker in normalized for marker in PROVENANCE_PATH_MARKERS)


@dataclass
class ResolveStats:
    total: int = 0
    already_ok: int = 0
    resolved: int = 0
    provenance_skipped: int = 0
    ambiguous: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def usable(self) -> int:
        return self.already_ok + self.resolved

    def summary(self) -> str:
        return (f"{self.usable}/{self.total} 可用"
                f"（原本就对 {self.already_ok}，按文件名解析 {self.resolved}）"
                f"　重名未解析 {len(self.ambiguous)}　找不到 {len(self.unresolved)}"
                f"　溯源字段跳过 {self.provenance_skipped}")


class MediaIndex:
    """一个族目录下的 文件名 → 路径 索引。"""

    def __init__(self, roots: list[Path], prefer_group: str = "") -> None:
        self.roots = [Path(r) for r in roots]
        self.prefer_group = prefer_group
        self._index: dict[str, list[Path]] | None = None

    def build(self) -> dict[str, list[Path]]:
        if self._index is not None:
            return self._index
        index: dict[str, list[Path]] = defaultdict(list)
        for root in self.roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.suffix.lower() in MEDIA_SUFFIXES and path.is_file():
                    index[path.name].append(path)
        self._index = dict(index)
        return self._index

    def lookup(self, original: str, prefer_group: str = "") -> tuple[Path | None, int]:
        """按原路径找本地文件。返回 (唯一命中, 候选数)。

        重名不靠猜，靠**逐级加长后缀**消歧。实测三种重名成因都能这样解决：

          同一片段在 planning/ 与 understanding/ 各存一份
              → ``prefer_group`` 用 QA 文件所在的 group 决定取哪份
          ``left_eye`` 与 ``left_eye_compress`` 并存
              → 原路径写的是哪个就取哪个，父目录名足以区分
          多视角同名（left_eye / left_wrist / right_wrist）
              → 同上

        逐级比对原路径的尾部片段（先文件名，再加一级父目录……），
        一旦候选唯一就返回。始终唯一不了才算重名。
        """
        parts = Path(str(original).replace("\\", "/")).parts
        if not parts:
            return None, 0
        hits = self.build().get(parts[-1], [])
        if len(hits) <= 1:
            return (hits[0] if hits else None), len(hits)

        # 逐级加长后缀
        for depth in range(2, min(len(parts), 6) + 1):
            suffix = tuple(parts[-depth:])
            narrowed = [h for h in hits if h.parts[-depth:] == suffix]
            if len(narrowed) == 1:
                return narrowed[0], 1
            if narrowed:
                hits = narrowed

        # 尾部分不开时，用原路径的**中间段**再筛一次。
        # 生成机路径形如 workflow_outputs/time/hand/gift_inhand/time_joined_videos/…，
        # 其中的 `time` 正对应本地的 understanding/<family>/**time**/time_joined_videos/，
        # 而同族下 `time` / `time_understanding` / `image_in_video` / `step_order`
        # 各存一份同名文件 —— 只比尾部分不开，这一段才是判据。
        stage_words = [p for p in parts[:-1]
                       if p in {"time", "time_understanding", "image_in_video",
                                "step_order", "planning", "understanding"}]
        for word in reversed(stage_words):          # 靠近文件名的那一段更具体
            narrowed = [h for h in hits if word in h.parts]
            if len(narrowed) == 1:
                return narrowed[0], 1
            if narrowed:
                hits = narrowed

        # 仍分不开时，用 QA 所在的 group 兜底（planning 的题取 planning 那份）
        if prefer_group:
            same_group = [h for h in hits if f"/{prefer_group}/" in str(h).replace("\\", "/")]
            if len(same_group) == 1:
                return same_group[0], 1
        return None, len(hits)


def resolve_items(
    items: list[dict[str, Any]],
    index: MediaIndex,
    stats: ResolveStats | None = None,
    base: Path | None = None,
) -> list[dict[str, Any]]:
    """就地解析 items 里的媒体路径（返回同一批对象）。"""
    stats = stats if stats is not None else ResolveStats()
    cache: dict[str, str | None] = {}

    def resolve_one(value: str) -> str:
        if value in cache:
            hit = cache[value]
            return hit if hit else value
        stats.total += 1
        if Path(value).exists():
            stats.already_ok += 1
            cache[value] = value
            return value
        # ── 相对 QA 文件所在目录 ────────────────────────────────
        # 放在文件名索引【之前】。文件名索引是 v1 的做法（媒体就散在 QA 目录里），
        # 对新数据既找不到、又危险 —— 新切片叫 `000163-000264.mp4` 这种纯帧号，
        # **跨族必然重名**，索引搜到了也可能给出别的族那一份。
        # 相对 QA 文件解析既准确又可移植，换机器不用重新生成 QA。
        if base is not None:
            near = (base / value).resolve()
            if near.exists():
                stats.resolved += 1
                cache[value] = str(near)
                return str(near)
        name = Path(value.replace("\\", "/")).name
        found, count = index.lookup(value, index.prefer_group)
        if found is not None:
            stats.resolved += 1
            cache[value] = str(found)
            return str(found)
        if count > 1:
            stats.ambiguous.append(f"{name}（{count} 处同名）")
        else:
            stats.unresolved.append(name)
        cache[value] = None
        return value

    def walk(node: Any, key: str | None = None) -> Any:
        if isinstance(node, str):
            if key in SKIP_KEYS or not _MEDIA_RE.search(node):
                return node
            if key in PROVENANCE_KEYS or is_provenance_path(node):
                stats.provenance_skipped += 1
                return node
            return resolve_one(node)
        if isinstance(node, dict):
            return {k: (v if k in SKIP_KEYS else walk(v, k)) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, key) for v in node]
        return node

    for index_, item in enumerate(items):
        items[index_] = walk(item)
    return items


def index_for_qa(qa_path: Path, datasets_root: Path | None = None) -> MediaIndex:
    """给定 QA 文件，推出该族的媒体搜索根。

    QA 布局是 ``<root>/QA/<group>/<family>/<file>.json``。媒体分散在
    ``QA/planning/<family>/`` 与 ``QA/understanding/<family>/`` 两处
    （例如 planning 的题会引用 understanding 下的视频），所以两边都要进索引。
    """
    qa_path = Path(qa_path).resolve()
    parts = qa_path.parts
    # 不能假设 QA 文件就在族目录下 —— 一半的族多嵌了一层（且层名不统一）。
    # 从路径里找 "QA" 这一节，它后面依次是 <group>/<family>。
    try:
        qa_index = len(parts) - 1 - parts[::-1].index("QA")
    except ValueError:
        return MediaIndex([qa_path.parent])
    group = parts[qa_index + 1] if qa_index + 1 < len(parts) else ""
    family = parts[qa_index + 2] if qa_index + 2 < len(parts) else ""
    qa_root = Path(*parts[: qa_index + 1])
    roots = [qa_root / g / family for g in ("planning", "understanding")]
    if datasets_root:
        roots.append(Path(datasets_root))
    # QA 文件所在的 group 用于重名兜底：planning 的题优先取 planning 下那份
    return MediaIndex([r for r in roots if r.exists()], prefer_group=group)
