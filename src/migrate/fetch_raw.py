#!/usr/bin/env python3
# coding: utf-8
"""从 ``yyyyywv/egocentric`` 抓取本 benchmark 用到的原始录像，写进 ``data/raw/``。

只抓需要的部分
--------------
上游 94 GB / 20+ 个族，我们只用 8 个族；而且 stack_cubes 上游有 800 集，
本 benchmark 只用 50 集。**要哪些集由标注文件名决定** ——
``data/label/<族>/segments/file-XXX_segments.json`` 有哪些，就抓哪些。
这样「原片 ↔ 标注」天然对齐，不会抓来一堆没有标注的集。

顺带抓 ``data/*.parquet``（机器人状态轨迹，2D/3D 轨迹任务的真正来源）
和 ``meta/``（相机分辨率、fps 等，写 meta.json 要用）。

断点续传
--------
每个文件下完立刻改名到位，中途挂掉重跑会跳过已完成的。
用 ``.part`` 临时文件，避免半截文件冒充成品。

meta.json
---------
下完给每族写一份，其中 ``usable`` 记「有效画面区」—— A 组顶部约 40 行
是采集时烧录的时间戳条，永久覆盖画面。**不记下来，下一个人一定按原生分辨率算。**
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# 实测：单连接被限速在约 2.5 MB/s，多连接几乎线性叠加（4 路 ≈ 8 MB/s）。
# 瓶颈是每连接带宽，不是磁盘或 CPU —— 串行下载只用得上 1/4 的可用带宽。
WORKERS = 6
_local = threading.local()


def _session() -> requests.Session:
    """每线程一个 Session —— requests.Session 不保证线程安全。"""
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPO = "yyyyywv/egocentric"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
API = f"https://huggingface.co/api/datasets/{REPO}"

# A 组：采集时烧录了时间戳黑条，顶部约 40 行被永久覆盖（2026-08-17 抽帧实证）
TIMESTAMP_BURNED = {"wash", "tea", "stack_cubes"}
BURNED_ROWS = 40

# 只认 LeRobot 声明的结构。**按名字筛会把派生物混进来** ——
# 上游 gripper/stack_cubes/videos/video_crop_top_10/ 就是一份已裁时间戳的
# 派生视频（960×486，恰好 50 集），躺在「原始数据」仓库里。
# 按结构筛才能挡住它，而且挡掉的东西要记进 meta.json，不静默丢弃。
VIDEO_RE = re.compile(r"videos/observation\.images\.([^/]+)/chunk-\d+/([^/]+)\.mp4$")

# 物理视角名 → 逻辑名。三个采集平台三套命名，但**角色相同**。
# 用 main 而不是 head：八族主视角抽帧比对过，都是「工作台总览」这个角色，
# 但取景明显不同（A 组更远更全，cam_color 最近）。只能确认角色一致，
# 不能确认物理机位一致 —— 叫 head 就是替数据下一个没验证的结论。
VIEW_MAP = {
    "left_eye": "main", "top": "main", "cam_color": "main",
    "right_eye": "main_right",                       # 仅 gripper 有，七个任务都没用
    "left_wrist": "wrist_left", "wrist_L": "wrist_left", "wrist_l": "wrist_left",
    "right_wrist": "wrist_right", "wrist_R": "wrist_right", "wrist_r": "wrist_right",
}


def needed_episodes(family: str) -> list[str]:
    """要哪些集 —— 由标注决定，不多抓。"""
    seg = DATA / "label" / family / "segments"
    return sorted(p.stem.replace("_segments", "") for p in seg.glob("*_segments.json"))


def remote_files() -> list[str]:
    r = requests.get(API, timeout=120)
    r.raise_for_status()
    return [s["rfilename"] for s in r.json().get("siblings", [])]


def fetch(rel: str, dest: Path) -> tuple[bool, int]:
    """返回 (是否新下载, 字节数)。已存在则跳过 —— 断点续跑靠这个。"""
    if dest.exists():
        return False, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件名带线程 id，避免两个线程撞同一个 .part
    tmp = dest.with_suffix(dest.suffix + f".{threading.get_ident()}.part")
    with _session().get(f"{BASE}/{rel}", stream=True, timeout=600) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    os.replace(tmp, dest)
    return True, dest.stat().st_size


def main() -> int:
    families = json.loads((DATA / "families.json").read_text(encoding="utf-8"))["families"]
    print("列远端文件表 …", flush=True)
    files = remote_files()
    grand = 0

    for canon, info in families.items():
        prefix = info["raw"] + "/"
        want = set(needed_episodes(canon))
        out = DATA / "raw" / canon
        mine = [f for f in files if f.startswith(prefix)]

        videos, views_seen, excluded = [], set(), {}
        by_view: dict[str, set[str]] = {}
        for f in mine:
            if not f.endswith(".mp4"):
                continue
            m = VIDEO_RE.search(f)
            if not m:                                   # 不符合声明结构 = 不是原始视频
                key = f[len(prefix):].rsplit("/", 1)[0]
                excluded[key] = excluded.get(key, 0) + 1
                continue
            view, stem = m.group(1), m.group(2)
            if stem in want:
                videos.append(f)
                views_seen.add(view)
                by_view.setdefault(view, set()).add(stem)

        states = [f for f in mine if "/data/" in f and f.endswith(".parquet")
                  and Path(f).stem in want]
        metas = [f for f in mine if "/meta/" in f]

        # 「有任意视角」不等于「可用」——一集缺了某个视角，多视角任务就出不了题。
        # 第一版只统计前者，tea2 因此显示 21/21 齐全，实际有一集缺右腕视角。
        got_eps = {Path(f).stem for f in videos}
        views = sorted(views_seen)
        complete = set.intersection(*by_view.values()) if by_view else set()
        partial = {e: sorted(VIEW_MAP.get(v, v) for v in views if e not in by_view[v])
                   for e in sorted(got_eps - complete)}
        missing = sorted(want - got_eps)
        unmapped = sorted(v for v in views if v not in VIEW_MAP)

        print(f"\n=== {canon} ===  需要 {len(want)} 集，上游有 {len(got_eps)} 集"
              f"，视角 {[VIEW_MAP.get(v, v + '(未映射)') for v in views]}", flush=True)
        if excluded:
            for k, n in sorted(excluded.items()):
                print(f"    ⊘ 排除非原始视频 {k}/  {n} 个", flush=True)
        if unmapped:
            print(f"    ⚠ 视角未在 VIEW_MAP 中：{unmapped}", flush=True)
        if missing:
            print(f"    ⚠ 完全缺失 {len(missing)} 集：{missing[:5]}{' …' if len(missing) > 5 else ''}",
                  flush=True)
        if partial:
            head = list(partial.items())[:3]
            print(f"    ⚠ 视角不全 {len(partial)} 集："
                  f"{', '.join(f'{e}缺{v}' for e, v in head)}"
                  f"{' …' if len(partial) > 3 else ''}", flush=True)

        def one(rel: str) -> tuple[bool, int]:
            try:
                return fetch(rel, out / rel[len(prefix):])
            except Exception as exc:                          # noqa: BLE001
                print(f"    ✗ {rel}: {type(exc).__name__} {exc}", flush=True)
                return False, 0

        total = new = done = 0
        todo = videos + states + metas
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for fresh, size in pool.map(one, todo):
                total += size
                new += int(fresh)
                done += 1
                if done % 30 == 0:
                    print(f"    … {done}/{len(todo)}，{total/1e9:.1f} GB", flush=True)
        grand += total

        info_path = out / "meta" / "info.json"
        native = fps = None
        if info_path.exists():
            meta = json.loads(info_path.read_text(encoding="utf-8"))
            fps = meta.get("fps")
            for key, feat in (meta.get("features") or {}).items():
                if "image" in key and feat.get("shape"):
                    h, w = feat["shape"][0], feat["shape"][1]
                    native = [w, h]
                    break

        burned = canon in TIMESTAMP_BURNED
        (out / "meta.json").write_text(json.dumps({
            "source": {"repo": REPO, "path": info["raw"]},
            "native": native,
            "usable": [native[0], native[1] - BURNED_ROWS] if (native and burned) else native,
            "usable_note": (f"顶部约 {BURNED_ROWS} 行是采集时烧录的时间戳条，永久覆盖画面"
                            if burned else "无时间戳叠加，全画面可用"),
            "timestamp_burned_in": burned,
            "fps": fps,
            # 物理名 → 逻辑名。逻辑名是**角色**（工作台总览 / 左右腕），
            # 不是物理机位 —— 八族主视角取景明显不同，见 views_note。
            "views": {VIEW_MAP.get(v, v): f"observation.images.{v}" for v in views},
            "views_physical": views,
            "views_note": "main 是角色名（工作台总览）。三个采集平台的主视角"
                          "物理机位与取景不同（A 组更远更全，cam_color 最近），"
                          "只确认角色一致，未确认机位一致。",
            # 上游 raw 仓库里混入的派生物，已按 LeRobot 声明结构挡掉
            "excluded": [{"path": k, "count": n,
                          "reason": "不符合 videos/observation.images.<view>/chunk-NNN/ 结构，"
                                    "非原始视频"}
                         for k, n in sorted(excluded.items())],
            # 状态数据有两种打包约定，**暂不统一** —— 现在没有消费者
            # （轨迹任务已搁置）。等真要用时再规范，避免凭想象定接口。
            "states_packing": ("packed" if len(states) == 1 and len(got_eps) > 1
                               else "per_episode"),
            "states_files": len(states),
            "episodes_wanted": len(want),
            # complete = 全部视角都在，才算可用；present 只表示「有任意视角」
            "episodes_complete": len(complete),
            "episodes_present": len(got_eps),
            "episodes_missing": missing,
            "episodes_partial": partial,          # 集 → 缺哪些视角
            "episodes_per_view": {VIEW_MAP.get(v, v): len(by_view[v]) for v in views},
            "incomplete": info.get("raw_incomplete"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"    完成：{len(complete)}/{len(want)} 集可用（全视角齐）"
              f"，{len(videos)} 视频 + {len(states)} 状态，{total/1e9:.2f} GB", flush=True)

    print(f"\n全部完成，共 {grand/1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
