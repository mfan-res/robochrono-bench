#!/usr/bin/env python3
# coding: utf-8
"""把八个族的异构数据规范化成统一结构。

为什么
------
评测代码里有四处补丁在**运行时**吸收数据布局的不一致（QA 三种深度、
四种路径风格、视角子目录改名、族属性与文档冲突）。问题不在补丁本身，
而在于它们每次运行都要执行，**且失败方式是静默的** ——
`qa_path` 找不到就整族消失、媒体解析不到就整题跳过。这两种都发生过。

所以把「运行时解析」变成「一次性构建」：不一致在构建阶段解决，
留下可审计的记录，评测代码只面对一种结构。

四条原则
--------
1. **原始数据只读。** 61 GB 已全量哈希校验、可重下，改写会与远端分叉。
2. **产物可完全重建。** 删掉重跑得到逐字节相同的结果。
3. **缺失显式。** 解析不到的媒体、定位不到的 QA 都进 manifest 的 issues，
   不让下游「刚好跳过」。
4. **不改题目内容。** 只做定位、命名、路径的规范化，题干选项答案一字不动。

`media[]` 怎么来的
------------------
**直接调用各任务自己的 ``parts()``，只取其中的媒体部分。** 不重新实现一遍
取图逻辑 —— 那样必然与现有行为产生偏差，而这份产物的验收标准恰恰是
「规范化前后发给模型的内容逐字节相同」。用同一段代码产出，才谈得上等价。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import tasks
from .mediaindex import ResolveStats, index_for_qa, resolve_items
from .tasks.base import Unit, load_items

BUILDER_VERSION = "v2"      # v2：planning_2 的预拼接图拆成三张单视角（团队决定，2026-08-16）
RUNS = ("understanding", "left_right", "image_in_video", "time",
        "planning", "planning_2", "step_order")


SPLIT_VIEWS = ("left_eye", "left_wrist", "right_wrist")
_JPEGTRAN = shutil.which("jpegtran")     # 有就走无损裁剪，没有退回 PIL 重编码


def split_prejoined(path: Path, out_dir: Path, views: tuple[str, ...] = SPLIT_VIEWS) -> list[Path]:
    """把横向拼接的多视角图等宽切成单视角图。

    起因：``planning_2`` 里 stack_cubes 收到 **1 张 2880×540 的预拼接图**，
    其余七族收到 **3 张独立图**。这不是画质差异，是输入结构不同 ——
    模型看到「一张宽图」和「三张图」时，视觉 token 的组织方式、
    以及能否区分视角，都不一样。团队决定统一为三张分开。

    边界是干净的：实测拼接图宽度恰为单视角的整数倍（2880 = 960×3），
    x=960 / x=1920 处相邻列的平均像素差为 148，而块内仅 1.2。
    切出来的三张是 960×540，与其余七族的单视角尺寸完全一致。

    **切分逐比特无损** —— 优先用 ``jpegtran -crop``，它在 DCT 系数层面裁剪，
    不解码不重编码。实测切完再拼回与原图**逐像素相同**（最大差 0）。
    这不是吹毛求疵：走 PIL 解码重存一遍是 56.9 dB，虽然肉眼无差，
    但那会让「输入变了吗」这个问题变成需要论证的，而不是显然的。

    没有 jpegtran 时退回 PIL（q95、不做色度下采样），并在返回值里标出来。

    产物写进 ``normalized/_derived/``，与原始数据分开，可随构建重建。
    """
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.width, image.height
    count = len(views)
    if width % count:
        return []                           # 宽度不是整数倍，不猜，交回调用方
    step = width // count
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    for index, view in enumerate(views):
        target = out_dir / f"{path.stem}__{view}{path.suffix}"
        if target.exists():
            produced.append(target)
            continue
        if _JPEGTRAN and path.suffix.lower() in (".jpg", ".jpeg"):
            # 写临时文件再改名：构建中断不会留下半张图冒充成品
            tmp = target.with_suffix(target.suffix + ".part")
            subprocess.run([_JPEGTRAN, "-copy", "none",
                            "-crop", f"{step}x{height}+{index * step}+0",
                            "-outfile", str(tmp), str(path)], check=True)
            os.replace(tmp, target)
        else:
            with Image.open(path) as image:
                image.crop((index * step, 0, (index + 1) * step, height)).save(
                    target, quality=95, subsampling=0)
        produced.append(target)
    return produced


def fingerprint(path: Path) -> dict[str, Any]:
    """给源文件留个指纹，用来判断产物是否过期。

    同时记 ``size``/``mtime_ns``（快，但重新下载会变）和 ``sha256``
    （慢，但只认内容）。校验时先比前两个，不一致再比哈希 ——
    这样重下了同样的文件不会误报过期，而内容真变了一定抓得到。
    """
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest()[:16]}


def fingerprint_matches(recorded: dict[str, Any], path: Path) -> bool:
    if not path.exists():
        return False
    stat = path.stat()
    if (stat.st_size == recorded.get("size")
            and stat.st_mtime_ns == recorded.get("mtime_ns")):
        return True                                  # 快路径命中
    return fingerprint(path).get("sha256") == recorded.get("sha256")


@dataclass
class Freshness:
    """规范化产物相对当前代码与当前数据是否仍然有效。"""
    ok: bool = True
    reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.ok = False
        self.reasons.append(reason)


def check_freshness(datasets_root: Path, out_root: Path | None = None) -> Freshness:
    """产物过期检测。

    重构的目的是让「运行时解析」变成「一次性构建」，但那样就多了一个
    新的失效方式：**产物相对代码或数据过期，而评测照跑不误。**
    实测过 —— 把一个 jsonl 藏起来，评测仍然跑满 300 题，只是 BC-16
    悄悄没生效。这正是重构本该消灭的那类静默失败，不能换个位置再长出来。

    查四件事：
      1. manifest 在不在；
      2. 构建器版本与当前代码一致（升级了 builder 忘了重建）；
      3. 数据里出现了 manifest 没有的族（新族到货了）；
      4. 每个 run 的 jsonl 还在，且其源 QA 文件指纹未变（数据更新了没重建）。
    """
    datasets_root = Path(datasets_root)
    out_root = Path(out_root) if out_root else datasets_root / "normalized"
    state = Freshness()

    manifest_path = out_root / "manifest.json"
    if not manifest_path.exists():
        state.fail(f"没有 {manifest_path} —— 先跑 tools/build_normalized.py")
        return state
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        state.fail(f"manifest.json 解析失败：{exc}")
        return state

    built = manifest.get("builder_version")
    if built != BUILDER_VERSION:
        state.fail(f"构建器版本不符：产物 {built}，代码 {BUILDER_VERSION} —— 需要重建")

    families = manifest.get("families", {})
    qa_root = datasets_root / "QA"
    on_disk = {canonical_family(p.name)
               for group in ("planning", "understanding") if (qa_root / group).exists()
               for p in (qa_root / group).iterdir() if p.is_dir()}
    for extra in sorted(on_disk - set(families)):
        state.fail(f"数据里有族 {extra} 但产物里没有 —— 需要重建")

    for canon, entry in sorted(families.items()):
        for run, run_entry in (entry.get("runs") or {}).items():
            if not (out_root / canon / f"{run}.jsonl").exists():
                state.fail(f"{canon}/{run}.jsonl 缺失 —— 需要重建")
                continue
            source = run_entry.get("source") or {}
            rel = source.get("qa_file")
            if not rel:
                state.fail(f"{canon}/{run} 没有源文件指纹（产物由旧版构建器产出）")
                continue
            if not fingerprint_matches(source, datasets_root / rel):
                state.fail(f"{canon}/{run} 的源 QA 已变动（{rel}）—— 需要重建")
    return state


def canonical_family(name: str) -> str:
    """族名规范化。

    ``Take_out_the_trash`` 与其余八族的全小写不一致，而族名会出现在
    配置、结果目录、报表列名三处。统一为全小写下划线，manifest 记原名。
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def detect_layout(datasets_root: Path, family: str) -> str:
    """QA 相对族目录的深度：flat / nested / scattered。"""
    depths = set()
    for run in RUNS:
        try:
            path = tasks.qa_path(datasets_root, family, run)
        except ValueError:
            continue
        if not path.exists():
            continue
        family_root = Path(datasets_root) / "QA" / tasks.QA_GROUP[run] / family
        depths.add(len(path.relative_to(family_root).parts) - 1)
    if not depths:
        return "unknown"
    return {0: "flat", 1: "nested"}.get(max(depths), "scattered")


def detect_schema_version(items: list[dict[str, Any]], run: str) -> str:
    """``planning`` 有两套不兼容的 input 结构，先标注不合并。

    stack_cubes 用 clips/prejoined_video_path（较早的流水线），
    其余七族用 joined_clip/view_order。哪套是当前意图**尚待数据方确认**，
    猜错会让一边的输入变成错的且不报错，所以只标注。
    """
    if run != "planning" or not items:
        return "n/a"
    keys = set((items[0].get("input") or {}).keys())
    if "joined_clip" in keys or "view_order" in keys:
        return "current"
    if "clips" in keys or "prejoined_video_path" in keys:
        return "legacy"
    return "unknown"


@dataclass
class RunReport:
    items: int = 0
    media: int = 0
    unresolved: int = 0
    ambiguous: int = 0
    schema_version: str = "n/a"
    source: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"items": self.items, "media": self.media,
                "unresolved": self.unresolved, "ambiguous": self.ambiguous,
                "schema_version": self.schema_version,
                # 源文件指纹：让「数据更新了但没重建」可被检测
                "source": self.source}


@dataclass
class BuildReport:
    families: dict[str, Any] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)


def _media_for_units(task: Any, items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """逐 unit 调 ``task.parts()``，把媒体部分按顺序记下来。

    返回 ``{item_id: [{"kind","path","label"}]}``。time 一个 unit 含 6 题，
    它们共用同一段视频，所以同组每题记同一份。
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for unit in task.units(items):
        media: list[dict[str, Any]] = []
        label = ""
        try:
            parts = task.parts(unit)
        except Exception as exc:  # noqa: BLE001  单题取媒体失败不该中断整族
            parts = []
            media.append({"kind": "error", "path": "", "label": f"{type(exc).__name__}: {exc}"})
        for part in parts:
            kind = part.get("type")
            if kind == "text":
                label = str(part.get("text", ""))[:80]   # 紧邻其后的媒体的说明文字
            elif kind in ("image", "video"):
                media.append({"kind": kind, "path": str(part.get("path", "")), "label": label})
                label = ""
        for item in unit.items:
            out[str(item.get("id"))] = media
    return out


def _split_planning2_items(items: list[dict[str, Any]], out_dir: Path,
                           *, dry_run: bool) -> dict[str, Any] | None:
    """BC-16：把 planning_2 里单张的预拼接图换成三张单视角图。

    改的是 ``input.image_paths`` 本身，不是事后修 ``media[]`` —— 因为
    ``media[]`` 是调 ``parts()`` 产出的，改了源头，两边自动一致，
    也就不存在「media 说三张、实际发一张」的分叉。

    只在「该题恰好一张图、且宽度 ≥ 高度×3」时动手。其余七族本来就是
    三张独立图，宽高比不满足，不会被碰 —— 判据来自数据本身而不是族名，
    所以新族进来时行为是可预期的。

    原路径保留在 ``input.prejoined_image_path``，可回查、可回退。
    """
    from PIL import Image

    changed = 0
    skipped: list[str] = []
    for item in items:
        data = item.get("input") or {}
        paths = data.get("image_paths")
        if not (isinstance(paths, list) and len(paths) == 1):
            continue
        path = Path(str(paths[0]))
        if not path.exists():
            continue
        try:
            with Image.open(path) as probe:
                if probe.width < probe.height * len(SPLIT_VIEWS):
                    continue                        # 不是横向拼接，放过
                if probe.width % len(SPLIT_VIEWS):
                    skipped.append(f"{path.name} 宽 {probe.width} 不是 3 的整数倍")
                    continue
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{path.name}: {type(exc).__name__}")
            continue
        if dry_run:
            changed += 1
            continue
        parts = split_prejoined(path, out_dir)
        if len(parts) != len(SPLIT_VIEWS):
            skipped.append(f"{path.name} 切分失败")
            continue
        data["prejoined_image_path"] = str(path)
        data["image_paths"] = [str(p) for p in parts]
        data["view_order"] = list(SPLIT_VIEWS)
        changed += 1
    if not changed and not skipped:
        return None
    out: dict[str, Any] = {"items_split": changed, "views": list(SPLIT_VIEWS),
                           "codec": "jpegtran-lossless" if _JPEGTRAN else "pil-reencode-q95"}
    if skipped:
        out["skipped"] = skipped[:5]
    return out


def normalize_family(
    datasets_root: Path,
    family: str,
    out_root: Path,
    *,
    dry_run: bool = False,
    report: BuildReport | None = None,
) -> dict[str, Any]:
    """规范化一个族。返回该族的 manifest 片段。"""
    report = report if report is not None else BuildReport()
    datasets_root = Path(datasets_root)
    canon = canonical_family(family)
    entry: dict[str, Any] = {
        "original_name": family,
        "layout": detect_layout(datasets_root, family),
        "runs": {},
    }
    target = Path(out_root) / canon
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)

    for run in RUNS:
        try:
            qa_path = tasks.qa_path(datasets_root, family, run)
        except ValueError as exc:
            report.issues.append({"kind": "qa_ambiguous", "family": canon, "run": run,
                                  "detail": str(exc).splitlines()[0]})
            continue
        if not qa_path.exists():
            entry.setdefault("missing_runs", []).append(run)
            continue

        items = load_items(qa_path)
        stats = ResolveStats()
        resolve_items(items, index_for_qa(qa_path), stats, base=qa_path.parent)

        run_report = RunReport(items=len(items), media=stats.total,
                               unresolved=len(stats.unresolved),
                               ambiguous=len(stats.ambiguous),
                               schema_version=detect_schema_version(items, run),
                               source={"qa_file": str(qa_path.relative_to(datasets_root)),
                                       **fingerprint(qa_path)})
        if run == "planning_2":
            # 先改 items，再产 media[] —— 顺序反了就会两边不一致
            split_stats = _split_planning2_items(
                items, Path(out_root) / "_derived" / canon / "planning_2", dry_run=dry_run)
            if split_stats:
                entry.setdefault("derived", {})["planning_2_split"] = split_stats
                report.issues.append({"kind": "planning_2_split", "family": canon, "run": run,
                                      "count": split_stats["items_split"],
                                      "detail": "BC-16：预拼接图拆成三张单视角"})
        media_by_id = _media_for_units(tasks.build(run), items)

        if not dry_run:
            with (target / f"{run}.jsonl").open("w", encoding="utf-8") as handle:
                for index, item in enumerate(items):
                    item_id = str(item.get("id"))
                    handle.write(json.dumps({
                        "id": item_id,
                        "family": canon,
                        "run": run,
                        # 题目内容原样搬运，一个字不改
                        "item": item,
                        # 唯一的媒体入口：由任务自己的 parts() 产出，顺序即发送顺序
                        "media": media_by_id.get(item_id, []),
                        "source": {"qa_file": str(qa_path.relative_to(datasets_root)),
                                   "item_index": index},
                    }, ensure_ascii=False) + "\n")

        entry["runs"][run] = run_report.as_dict()
        if stats.unresolved:
            report.issues.append({"kind": "media_unresolved", "family": canon, "run": run,
                                  "count": len(stats.unresolved),
                                  "examples": sorted(set(stats.unresolved))[:5]})
        if stats.ambiguous:
            report.issues.append({"kind": "media_ambiguous", "family": canon, "run": run,
                                  "count": len(stats.ambiguous),
                                  "examples": sorted(set(stats.ambiguous))[:5]})

    versions = {r["schema_version"] for r in entry["runs"].values()} - {"n/a"}
    if versions:
        entry["schema_version"] = sorted(versions)[0]
    if "missing_runs" in entry:
        report.issues.append({"kind": "runs_missing", "family": canon,
                              "detail": entry["missing_runs"]})
    return entry


def build(
    datasets_root: Path,
    out_root: Path,
    *,
    families: list[str] | None = None,
    dry_run: bool = False,
    source_sha: str = "",
) -> dict[str, Any]:
    """构建全部（或指定）族，写出 manifest。"""
    datasets_root = Path(datasets_root)
    qa_root = datasets_root / "QA"
    found = sorted({p.name for group in ("planning", "understanding")
                    if (qa_root / group).exists()
                    for p in (qa_root / group).iterdir() if p.is_dir()})
    targets = families or found

    report = BuildReport()
    for family in targets:
        report.families[canonical_family(family)] = normalize_family(
            datasets_root, family, out_root, dry_run=dry_run, report=report)

    # 部分重建要**并进**旧 manifest，不能整个覆盖。
    # 覆盖过一次：`--family stack_cubes` 把其余七族从 manifest 里抹掉了，
    # 而那七族的 jsonl 还好端端躺在盘上 —— 产物与索引就此不一致。
    families, issues = dict(report.families), list(report.issues)
    previous_path = Path(out_root) / "manifest.json"
    if previous_path.exists() and set(targets) != set(found):
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if previous.get("builder_version") == BUILDER_VERSION:
            merged = dict(previous.get("families") or {})
            merged.update(families)
            families = merged
            touched = set(report.families)
            issues = [i for i in (previous.get("issues") or [])
                      if i.get("family") not in touched] + issues
            source_sha = source_sha or previous.get("source_sha", "")
        # 构建器版本不同就不并 —— 旧产物本来就该整体重建

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder_version": BUILDER_VERSION,
        # 远端 commit。用于检测「数据更新了但没重建」—— 这次是靠手工比对才发现
        # 远端已推进 7 个提交，应当自动化。
        "source_sha": source_sha,
        "families": families,
        "issues": issues,
    }
    if not dry_run:
        Path(out_root).mkdir(parents=True, exist_ok=True)
        (Path(out_root) / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
