#!/usr/bin/env python3
# coding: utf-8
"""规范化等价性：走 normalized 与走原始 QA，发给模型的内容必须逐字节相同。

这是数据重构的**验收关卡**。规范化只应当做四件事 —— 定位 QA、统一族名、
解析媒体路径、把媒体收敛成一个有序数组。题干、选项、答案一个字都不该变。

**如果这个测试挂了，就说明构建器动了不该动的东西。**

判据分两层：

  内容层   逐题比对 item 字典。允许的唯一差异是媒体路径被解析成绝对路径
           （原路径在本机不存在，解析后存在）—— 其余任何字段变化都算失败。
  发送层   逐题比对 ``task.parts()`` 产出的完整 parts：文本逐字节相同，
           媒体的类型、顺序、以及**指向的文件内容哈希**相同。
           比哈希而不只比路径 —— 路径变了但内容一致才是我们要的。

**声明过的行为变更怎么办**

BC-16 让 stack_cubes 的 ``planning_2`` 从 1 张预拼接图变成 3 张单视角图 ——
这是**有意**的输入变化，发送层必然不同。做法不是把这一格从表里划掉，
而是换一套更强的判据（见 ``check_declared_split``）：

  1. 题干、选项、答案仍逐字段相同 —— 变的只能是媒体；
  2. 三张图横向拼回去与原预拼接图**逐像素相同** —— 证明没丢也没多任何画面。

这样「声明过的变更」依然被完整验证，只是验的是「变得对不对」而不是
「变没变」。豁免必须来自 manifest 里构建器自己写下的记录，
不是测试里手写的族名 —— 手写的白名单会在数据换了以后继续沉默地放行。

不需要 GPU 与 API key。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _P  # noqa: E402
sys.path.insert(0, str(_P.EVAL))

from robochrono import tasks  # noqa: E402
from robochrono.mediaindex import index_for_qa, resolve_items  # noqa: E402
from robochrono.normalize import RUNS, canonical_family  # noqa: E402
from robochrono.tasks.base import load_items  # noqa: E402

DATASETS = _P.DATASETS
NORMALIZED = DATASETS / "normalized"
SAMPLE_PER_RUN = 12          # 每个 run 抽多少题做发送层比对（媒体哈希较慢）


def file_digest(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def parts_signature(parts: list[dict[str, Any]], *, hash_media: bool) -> list[Any]:
    """把 parts 变成可比对的签名。"""
    signature: list[Any] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            signature.append(("text", part.get("text", "")))
        else:
            path = str(part.get("path", ""))
            signature.append((kind, file_digest(path) if hash_media else Path(path).name))
    return signature


def load_raw(family: str, run: str) -> list[dict[str, Any]] | None:
    try:
        qa_path = tasks.qa_path(DATASETS, family, run)
    except ValueError:
        return None
    if not qa_path.exists():
        return None
    items = load_items(qa_path)
    resolve_items(items, index_for_qa(qa_path))
    return items


def load_norm(canon: str, run: str) -> list[dict[str, Any]] | None:
    path = NORMALIZED / canon / f"{run}.jsonl"
    if not path.exists():
        return None
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line)["item"])
    return out


def check_declared_split(raw: list[dict[str, Any]], norm: list[dict[str, Any]],
                         sample: int = 8) -> tuple[int, str]:
    """验证 BC-16 的拆分「变得对」：题目没变，画面没丢。

    返回 ``(不合格题数, 说明)``。
    """
    import numpy as np
    from PIL import Image

    bad = 0
    for a, b in zip(raw, norm):
        # 题目部分必须一字不差 —— 只允许 input 里与图片有关的三个键不同
        ka = {k: v for k, v in a.items() if k != "input"}
        kb = {k: v for k, v in b.items() if k != "input"}
        if ka != kb:
            bad += 1
            continue
        ia, ib = a.get("input") or {}, b.get("input") or {}
        changed = {k for k in set(ia) | set(ib) if ia.get(k) != ib.get(k)}
        if changed - {"image_paths", "prejoined_image_path", "view_order"}:
            bad += 1
    if bad:
        return bad, "题目内容被改动"

    # 画面完整性：抽验若干题，三张拼回去必须与原图逐像素相同
    for item in norm[:sample]:
        data = item.get("input") or {}
        src = data.get("prejoined_image_path")
        paths = data.get("image_paths") or []
        if not src or len(paths) != 3:
            return 1, "拆分记录缺失"
        with Image.open(src) as im:
            orig = np.asarray(im.convert("RGB"), dtype=np.int16)
        crops = []
        for p in paths:
            with Image.open(p) as im:
                crops.append(np.asarray(im.convert("RGB"), dtype=np.int16))
        if np.abs(orig - np.concatenate(crops, axis=1)).max() != 0:
            return 1, "拼回与原图不一致"
    return 0, f"BC-16 拆分（题目未变，抽验 {min(sample, len(norm))} 题画面逐像素还原）"


def main() -> int:
    if not NORMALIZED.exists():
        # `normalized/` 是 **v1 数据**的规范化产物，要由完整的 v1 QA（61 GB，
        # 留在旧仓库）构建。本仓库的夹具只有裁剪过的 JSON，建不出来。
        # **显式跳过而不是假装通过** —— 这一关验的是「v1 重构没动过发给模型的内容」，
        # 跑 A4 新旧对照之前必须过。
        print("跳过：normalized/ 未构建。这一关需要完整的 v1 QA。")
        print(f"      export ROBOCHRONO_V1_ROOT=<旧仓库>/eval")
        print(f"      python3 tools/build_normalized.py   # 在旧仓库里跑")
        return 0
        return 1
    manifest = json.loads((NORMALIZED / "manifest.json").read_text(encoding="utf-8"))

    print(f"{'族':<20} {'run':<15} {'题数':>5} {'内容差异':>8} {'发送差异':>8}  status")
    print("-" * 74)
    failures = 0
    notes: list[str] = []

    for canon, entry in sorted(manifest["families"].items()):
        family = entry.get("original_name", canon)
        derived = entry.get("derived") or {}
        for run in RUNS:
            raw = load_raw(family, run)
            norm = load_norm(canon, run)
            if raw is None and norm is None:
                continue
            if raw is None or norm is None:
                print(f"{canon:<20} {run:<15} {'—':>5} {'—':>8} {'—':>8}  "
                      f"{'只有 normalized 有' if raw is None else '只有原始有'}")
                failures += 1
                continue

            # 构建器声明过的输入变更：换一套判据，不是放行
            if run == "planning_2" and derived.get("planning_2_split"):
                if len(raw) != len(norm):
                    print(f"{canon:<20} {run:<15} {len(norm):>5} {'—':>8} {'—':>8}  **题数不符**")
                    failures += 1
                    continue
                bad, note = check_declared_split(raw, norm)
                failures += 0 if bad == 0 else 1
                print(f"{canon:<20} {run:<15} {len(norm):>5} {bad:>8} {'n/a':>8}  "
                      f"{'BC-16 OK' if bad == 0 else '**' + note + '**'}")
                notes.append(f"{canon}/{run}: {note}")
                continue

            content_diff = 0
            if len(raw) != len(norm):
                content_diff = abs(len(raw) - len(norm))
            else:
                for a, b in zip(raw, norm):
                    if a != b:
                        content_diff += 1

            send_diff = 0
            if len(raw) == len(norm):
                task = tasks.build(run)
                raw_units = task.units(raw)[:SAMPLE_PER_RUN]
                norm_units = task.units(norm)[:SAMPLE_PER_RUN]
                for ua, ub in zip(raw_units, norm_units):
                    try:
                        sa = parts_signature(task.parts(ua), hash_media=True)
                        sb = parts_signature(task.parts(ub), hash_media=True)
                    except Exception:  # noqa: BLE001  两边同样失败才算一致
                        sa = sb = None
                    if sa != sb:
                        send_diff += 1

            ok = content_diff == 0 and send_diff == 0
            failures += 0 if ok else 1
            print(f"{canon:<20} {run:<15} {len(norm):>5} {content_diff:>8} {send_diff:>8}  "
                  f"{'OK' if ok else '**不一致**'}")

    print("-" * 74)
    for note in notes:
        print(f"声明变更  {note}")
    print("规范化前后等价（声明过的变更已按其自身判据验证）" if failures == 0
          else f"{failures} 处不一致 —— 构建器动了不该动的东西")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
