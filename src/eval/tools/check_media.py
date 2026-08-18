#!/usr/bin/env python3
# coding: utf-8
"""用各评测脚本自己的取媒体函数验证 QA 数据可用。

这是 BC-08 路径规范化的验收判据，也是 preflight 的原型：不调模型、不加载权重，
只回答一个问题 —— 每个任务的每一道题，脚本真的能拿到它要送给模型的媒体文件吗。

刻意复用 test/ 下冻结脚本的函数，而不是重新实现一遍解析逻辑，
这样验的就是真实行为而非我们对行为的理解。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "test"))


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", data) if isinstance(data, dict) else data
    return [i for i in items if isinstance(i, dict)]


def paths_from_parts(parts: list[dict]) -> list[Path]:
    return [Path(str(p["path"])) for p in parts if p.get("type") in {"image", "video"}]


def build_checks(qa: Path) -> list[tuple[str, Path, object]]:
    """返回 (任务名, QA 文件, 取媒体函数)。函数签名统一为 (item, input_path) -> list[Path]。"""
    und = qa / "QA" / "understanding" / "stack_cubes"
    pln = qa / "QA" / "planning" / "stack_cubes"

    import image_in_video_glm_test as iiv
    import left_right_glm_test as lr
    import planning_2_glm_test as p2
    import planning_glm_test as p1
    import step_order_glm_test as so
    import time_eqa_glm_test_multi as teq
    import trajectory_glm_test as traj
    import understanding_glm_test as und_mod

    return [
        ("time", und / "time_vqa.json",
         lambda it, p: teq.video_paths_for_item(it, p.parent, (".mp4", ".webm", ".mov", ".mkv", ".avi"))),
        ("understanding", und / "understanding_vqa.json",
         lambda it, p: und_mod.media_paths_for_item(it, "clip_path")),
        ("left_right", und / "left_right_vqa.json",
         lambda it, p: paths_from_parts(lr.content_for_item(it, ""))),
        ("image_in_video", und / "image_in_video_vqa.json",
         lambda it, p: paths_from_parts(iiv.content_for_item(it, "", p))),
        ("planning", pln / "planning_vqa.json",
         lambda it, p: p1.video_paths_for_item(it)),
        ("planning_2", pln / "planning_2_vqa.json",
         lambda it, p: p2.image_paths_for_item(it)),
        ("step_order", pln / "step_order_vqa.json",
         lambda it, p: so.image_paths_for_item(it, p)),
        ("trajectory_2D", pln / "trajectory_qa_2d.json",
         lambda it, p: [Path(r["path"]) for r in traj.image_inputs_for_item(it)]),
        ("trajectory_3D", pln / "trajectory_qa_3d.json",
         lambda it, p: [Path(r["path"]) for r in traj.image_inputs_for_item(it)]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify every QA item's media resolves on this machine.")
    parser.add_argument("--datasets-root", type=Path, default=REPO / "eval" / "datasets")
    parser.add_argument("--limit", type=int, default=None, help="Only check the first N items per task.")
    args = parser.parse_args()

    print(f"{'task':<16} {'items':>6} {'media':>8} {'missing':>8} {'errors':>7}  status")
    print("-" * 62)

    exit_code = 0
    for name, qa_path, media_fn in build_checks(args.datasets_root):
        if not qa_path.exists():
            print(f"{name:<16} {'-':>6} {'-':>8} {'-':>8} {'-':>7}  QA 文件缺失")
            exit_code = 1
            continue

        items = load_items(qa_path)
        if args.limit:
            items = items[: args.limit]

        n_media = n_missing = n_error = 0
        first_problem = ""
        for item in items:
            try:
                for media in media_fn(item, qa_path):
                    n_media += 1
                    if not media.exists():
                        n_missing += 1
                        if not first_problem:
                            first_problem = f"missing {media}"
            except Exception as exc:  # noqa: BLE001
                n_error += 1
                if not first_problem:
                    first_problem = f"{type(exc).__name__}: {exc}"

        ok = n_missing == 0 and n_error == 0
        exit_code |= 0 if ok else 1
        print(
            f"{name:<16} {len(items):>6} {n_media:>8} {n_missing:>8} {n_error:>7}  "
            f"{'OK' if ok else first_problem[:80]}"
        )

    print("-" * 62)
    print("全部通过" if exit_code == 0 else "存在问题，见上表")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
