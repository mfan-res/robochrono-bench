#!/usr/bin/env python3
# coding: utf-8
"""汇总与成本估算。

冻结版完全没有这一层 —— 跑完 N 个模型就是 N×9 个互不相干的 JSON，
要出一张对比表只能手工翻文件。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import tasks
from .matrix import Plan, RunSpec
from .tasks.base import load_items


# --------------------------------------------------------------------------
# 成本估算
# --------------------------------------------------------------------------


@dataclass
class Estimate:
    runs: int = 0
    items: int = 0
    calls: int = 0
    media_bytes: int = 0
    missing_media: int = 0


def estimate_run(spec: RunSpec, datasets_root: Path) -> Estimate:
    """统计一个 run 的调用量与媒体体积。

    刻意走任务自己的 ``parts()``，而不是遍历整条 item —— QA JSON 里存了大量
    溯源路径（原始 LeRobot 视频、未发布的 time_joined_videos 等），
    它们从不会被发给模型，算进去既会高估体积，也会误报「文件缺失」。
    """
    qa_path = spec.qa_path(datasets_root)
    items = load_items(qa_path)
    task = tasks.build(spec.run)
    units = task.units(items)

    est = Estimate(runs=1, items=len(items), calls=len(units))
    seen: set[str] = set()
    for unit in units:
        try:
            parts = task.parts(unit)
        except Exception:  # noqa: BLE001  取不到媒体的 unit 由 preflight 负责报告
            continue
        for part in parts:
            if part.get("type") not in {"image", "video"}:
                continue
            path = str(part["path"])
            if path in seen:
                continue
            seen.add(path)
            file_path = Path(path)
            if file_path.exists():
                est.media_bytes += file_path.stat().st_size
            else:
                est.missing_media += 1
    return est


def estimate_matrix(specs: list[RunSpec], datasets_root: Path) -> dict[str, Estimate]:
    """按模型聚合。同一族的媒体会被多个模型重复发送，所以按模型分别累计。"""
    by_model: dict[str, Estimate] = {}
    cache: dict[tuple[str, str], Estimate] = {}

    for spec in specs:
        cache_key = (spec.family, spec.run)
        if cache_key not in cache:
            cache[cache_key] = estimate_run(spec, datasets_root)
        one = cache[cache_key]

        agg = by_model.setdefault(spec.model.name, Estimate())
        agg.runs += 1
        agg.items += one.items
        agg.calls += one.calls
        agg.media_bytes += one.media_bytes
        agg.missing_media += one.missing_media
    return by_model


def format_estimate(by_model: dict[str, Estimate], plan: Plan) -> str:
    kinds = {m.name: m.kind for m in plan.models}
    lines = [
        f"{'model':<38} {'kind':<6} {'runs':>5} {'items':>8} {'calls':>8} {'media':>10}",
        "-" * 82,
    ]
    total = Estimate()
    for name, est in sorted(by_model.items()):
        lines.append(
            f"{name:<38} {kinds.get(name, '?'):<6} {est.runs:>5} {est.items:>8,} "
            f"{est.calls:>8,} {est.media_bytes / 1e9:>9.2f}G"
        )
        total.runs += est.runs
        total.items += est.items
        total.calls += est.calls
        total.media_bytes += est.media_bytes
        total.missing_media += est.missing_media
    lines.append("-" * 82)
    lines.append(
        f"{'合计':<38} {'':<6} {total.runs:>5} {total.items:>8,} "
        f"{total.calls:>8,} {total.media_bytes / 1e9:>9.2f}G"
    )
    api_calls = sum(e.calls for n, e in by_model.items() if kinds.get(n) == "api")
    api_bytes = sum(e.media_bytes for n, e in by_model.items() if kinds.get(n) == "api")
    if api_calls:
        lines.append("")
        lines.append(f"其中付费 API：{api_calls:,} 次调用，媒体 {api_bytes / 1e9:.2f} GB")
        lines.append("注意：媒体字节数不等于计费 token 数，各家换算不同；")
        lines.append("      先用一个族跑真实账单反推单价，再决定是否全量铺开。")
    if total.missing_media:
        lines.append("")
        lines.append(f"警告：{total.missing_media} 个媒体路径在本机不存在")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 结果汇总
# --------------------------------------------------------------------------


def collect(results_dir: Path) -> list[dict[str, Any]]:
    """扫描 results 目录，收集每个 (模型, 族, 任务) 的主指标。"""
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(results_dir.rglob("*.summary.json")):
        run = summary_path.name[: -len(".summary.json")]
        meta_path = summary_path.with_name(f"{run}.meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metric = tasks.PRIMARY_METRIC.get(run, "accuracy")
        rows.append(
            {
                # 用 plan.json 里的模型名，不用 provider 名 —— provider 是接入方式的
                # 标识（local_qwen 同时服务多个 Qwen 模型），拿它当模型名会撞车。
                "model": (meta.get("model_name") or meta.get("provider")
                          or summary_path.parents[2].name),
                "variant": meta.get("frames_variant") or "default",
                "family": meta.get("family") or summary_path.parents[1].name,
                "run": run,
                "metric": metric,
                "value": summary.get(metric),
                "total": summary.get("total"),
                "answered": summary.get("answered"),
                "errors": summary.get("errors"),
                "parse_failure_rate": summary.get("parse_failure_rate"),
                "aborted": summary.get("aborted"),
                "frames": (meta.get("frames") or {}).get("value"),
            }
        )
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    """模型 × 任务 的对比表，每个任务族一张。"""
    if not rows:
        return "（没有找到任何结果）"

    out: list[str] = []
    families = sorted({r["family"] for r in rows})
    for family in families:
        subset = [r for r in rows if r["family"] == family]
        runs = [r for r in tasks.ALL_RUNS if any(x["run"] == r for x in subset)]

        # 抽帧档位必须进入行键。否则 fps=1 与 fps=2 的同一个 (模型, 任务)
        # 会互相覆盖，而我们特意跑两档的结果就只剩一个数、且看不出是哪一档。
        keys = sorted({(r["model"], r["variant"]) for r in subset})
        index = {(r["model"], r["variant"], r["run"]): r for r in subset}
        multi_variant = len({v for _, v in keys}) > 1

        out.append(f"### {family}\n")
        head = "| model | " + ("frames | " if multi_variant else "")
        out.append(head + " | ".join(runs) + " |")
        out.append("| --- | " + ("--- | " if multi_variant else "")
                   + " | ".join("---:" for _ in runs) + " |")
        for model, variant in keys:
            cells = []
            for run in runs:
                row = index.get((model, variant, run))
                if row is None:
                    cells.append("—")
                elif row.get("aborted"):
                    cells.append("aborted")
                elif row.get("value") is None:
                    cells.append("n/a")
                else:
                    mark = "*" if (row.get("total") or 0) < 100 else ""
                    cells.append(f"{row['value']:.4g}{mark}")
            label = f"| {model} | " + (f"{variant} | " if multi_variant else "")
            out.append(label + " | ".join(cells) + " |")
        out.append("")

    out.append("主指标：选择题类 accuracy，trajectory 为 mean_score，time 为 tIoU@0.5。")
    out.append("　　time 不用 mean_tIoU —— 「整段视频都报」就能拿 0.13，不是 0（见 tasks/__init__.py）。")
    out.append("`*` 表示样本量少于 100，置信区间较宽（step_order 每族仅 50 题）。")
    if any(r["variant"] != "default" for r in rows):
        out.append("frames 列是抽帧档位；静态图任务不分档，两行取值相同。")
    return "\n".join(out)


def to_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "variant", "family", "run", "metric", "value", "total",
              "answered", "errors", "parse_failure_rate", "frames", "aborted"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


# --------------------------------------------------------------------------
# 打包回传
# --------------------------------------------------------------------------


def pack(results_dir: Path, output: Path, full: bool = False) -> dict[str, Any]:
    """把结果打成一个 tar.gz，供 scp 回传。

    默认只带第 1 层（汇总）与第 2 层（逐题精简记录），这两层加起来在全矩阵下
    约几百 MB；``--full`` 才带上媒体缓存等大件。
    """
    import tarfile

    output.parent.mkdir(parents=True, exist_ok=True)
    patterns = ["*.summary.json", "*.meta.json", "*.jsonl"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(results_dir.rglob(pattern)))
    if not full:
        # 媒体缓存是本地中间产物，回传没有意义
        files = [f for f in files if ".media_cache" not in f.parts]

    with tarfile.open(output, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(results_dir)))

    return {
        "files": len(files),
        "bytes": output.stat().st_size,
        "output": str(output),
    }
