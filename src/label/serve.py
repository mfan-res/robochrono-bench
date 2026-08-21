#!/usr/bin/env python3
# coding: utf-8
"""标注器的后端 —— 静态文件 + 四个 API。

    python src/label/serve.py            # 默认 8000
    python src/label/serve.py --port 8123

在 VS Code Remote / SSH 下会被自动转发到本机，浏览器开 http://localhost:8000。

为什么是网页而不是 OpenCV
--------------------------
上游那 416 行 GUI 依赖 cv2 与显示器，本机两样都没有，改了验证不了。
而网页版有一个**我们自己造出来的有利条件**：``data/source`` 是全帧内编码（D-18）。

那本来是为了让切片能 ``-c copy`` 无损，却顺带解决了网页标注最大的技术难点 ——
普通 GOP 视频里 ``video.currentTime = t`` 会跳到最近的关键帧，精度不可控；
**全帧内每一帧都是关键帧，设到哪就是哪**。加上 `requestVideoFrameCallback`
能拿到精确的 `mediaTime`，帧号权威性有保障。

设计要点
--------
**保存前跑在线检查。** 本意是标注工具与离线校验共用同一份判据 ——
此前 `check_labels.py` 与上游工具各写各的，导致 tea2 显示「21/21 齐全」
而实际只有 20 集可用。

⚠ **但这件事目前只做到了一半**：`review()` 只覆盖 `validate.py` 八类里的三类，
而且是另写的一份，不是 import 过来的。见 `review()` 的说明与 cleanup_checklist 4.3。

**帧号是权威**，秒与时间串由后端按 fps 统一派生（`core.Segment`），
前端只报帧号。这样前端浮点误差进不了数据。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from collections import defaultdict
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from label.core import Segment, build_document  # noqa: E402

DATA = ROOT / "data"
UI = Path(__file__).resolve().parent / "ui"


@lru_cache(maxsize=None)
def family_meta(family: str) -> dict[str, Any]:
    return json.loads((DATA / "raw" / family / "meta.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def subtasks(family: str) -> list[dict[str, str]]:
    path = DATA / "label" / family / "subtasks.json"
    return json.loads(path.read_text(encoding="utf-8"))["subtasks"] if path.exists() else []


@lru_cache(maxsize=None)
def episode_bounds(family: str) -> dict[str, list[list[float]]]:
    """各视频里 episode 的时间区间。查不到返回空 —— **不等于「只有一集」**（D-19）。"""
    tables = list((DATA / "raw" / family / "meta" / "episodes").rglob("*.parquet"))
    if not tables:
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    frame = pq.read_table(tables[0]).to_pandas()
    # ⚠ 必须显式取 videos/ 开头那列：data/file_index 会把「状态打包」误读成「多集打包」
    cols = {k: next((c for c in frame.columns
                     if c.startswith("videos/") and c.endswith(k)), None)
            for k in ("file_index", "from_timestamp", "to_timestamp")}
    if not all(cols.values()):
        return {}
    out: dict[str, list[list[float]]] = defaultdict(list)
    for _, row in frame.iterrows():
        out[f"file-{int(row[cols['file_index']]):03d}"].append(
            [float(row[cols["from_timestamp"]]), float(row[cols["to_timestamp"]])])
    return {k: sorted(v) for k, v in out.items()}


@lru_cache(maxsize=None)
def video_frames(family: str, episode: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames", "-of", "csv=p=0",
         str(DATA / "source" / family / episode / "main.mp4")],
        capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def families() -> list[str]:
    registry = json.loads((DATA / "families.json").read_text(encoding="utf-8"))["families"]
    return sorted(f for f, v in registry.items()
                  if v.get("status") != "excluded" and (DATA / "source" / f).is_dir())


def episodes(family: str) -> list[dict[str, Any]]:
    base = DATA / "source" / family
    out = []
    for path in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("file-")):
        seg = DATA / "label" / family / "segments" / f"{path.name}_segments.json"
        count = len(json.loads(seg.read_text(encoding="utf-8"))["segments"]) if seg.exists() else 0
        out.append({"episode": path.name, "labeled": count})
    return out


def file_version(path: Path) -> str:
    """文件版本标识。存在就用 mtime_ns，不存在用空串 —— 「本来没有」也是一种版本。"""
    return str(path.stat().st_mtime_ns) if path.exists() else ""


def load_episode(family: str, episode: str) -> dict[str, Any]:
    meta = family_meta(family)
    seg = DATA / "label" / family / "segments" / f"{episode}_segments.json"
    existing = json.loads(seg.read_text(encoding="utf-8"))["segments"] if seg.exists() else []
    return {
        "family": family, "episode": episode,
        "version": file_version(seg),
        "fps": meta["fps"],
        "total_frames": video_frames(family, episode),
        "views": [v for v in ("main", "wrist_left", "wrist_right")
                  if (DATA / "source" / family / episode / f"{v}.mp4").exists()],
        "subtasks": subtasks(family),
        "episode_bounds": episode_bounds(family).get(episode),
        "segments": [{"start_frame": s["start_frame"], "end_frame": s["end_frame"],
                      "subtask": s["subtask"]} for s in existing],
    }


def save_episode(payload: dict[str, Any]) -> dict[str, Any]:
    """写盘前先跑校验。**校验不过就不写** —— 与离线校验共用同一套判据。"""
    family, episode = payload["family"], payload["episode"]
    meta = family_meta(family)
    segments = [Segment(int(s["start_frame"]), int(s["end_frame"]), s["subtask"])
                for s in payload["segments"]]
    document = build_document(
        episode, segments, fps=meta["fps"], total_frames=video_frames(family, episode),
        subtasks=subtasks(family), video_rel=f"{family}/{episode}/main.mp4",
        episode_bounds=episode_bounds(family).get(episode))
    document["source"]["tool_version"] = "bench-label-web/1"

    out = DATA / "label" / family / "segments" / f"{episode}_segments.json"

    # 并发覆盖检查：加载之后文件被别处改过就拒绝。
    # **这个项目里「静默覆盖」已经付出过代价** —— P-03 就是出题产物静默
    # 覆盖了人工标注，而且直到几个月后才被发现。宁可拒绝，不要悄悄覆盖。
    current = file_version(out)
    if payload.get("version") is not None and payload["version"] != current \
            and not payload.get("force"):
        return {"ok": False, "stale": True, "version": current, "problems": [
            {"level": "block",
             "text": "文件在你加载之后被别处修改过（另一个标签页？脚本？git checkout？）。"
                     "直接保存会覆盖对方的改动 —— 请重新加载后再标，"
                     "或点强制保存以你手上这份为准。"}]}

    problems = review(document, family)
    blocking = [p for p in problems if p["level"] == "block"]
    if blocking and not payload.get("force"):
        return {"ok": False, "problems": problems}

    out.parent.mkdir(parents=True, exist_ok=True)
    # data/label 是不可再生层，覆写前先留一份。
    # 这只是「改坏了还没提交」时的兜底 —— **长期历史由 git 负责**，
    # 所以 .bak/ 不进仓库（否则每保存一次就多一份带时间戳的副本）。
    if out.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = out.parent / ".bak" / f"{episode}.{stamp}.json"
        backup.parent.mkdir(exist_ok=True)
        shutil.copy2(out, backup)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "problems": problems, "path": str(out.relative_to(ROOT)),
            "forced": bool(payload.get("force")), "version": file_version(out)}


def subtask_usage(family: str) -> dict[str, list[str]]:
    """每个 subtask 被哪些集引用 —— 删除前要看它。"""
    used: dict[str, list[str]] = defaultdict(list)
    for path in sorted((DATA / "label" / family / "segments").glob("*_segments.json")):
        for seg in json.loads(path.read_text(encoding="utf-8"))["segments"]:
            used[seg["subtask"]].append(path.stem.replace("_segments", ""))
    return {k: sorted(set(v)) for k, v in used.items()}


def edit_subtasks(payload: dict[str, Any]) -> dict[str, Any]:
    """新增 / 改措辞 / 删除。

    三条硬规则，都源自 D-25 —— **ID 永不改变**是这套设计的全部价值：

        新增     ID 由 text 派生（去冠词），与现有冲突则拒绝
        改 text  ID 不动 —— 所有引用自动跟随（D-21 那类修正的正确姿势）
        删除     仅当无人引用；被引用则拒绝并列出是哪些集

    **中途新增要警告。** 标到第 20 集才发现某个动作不在列表里，说明前 19 集
    要么把它标成了别的、要么跳过了 —— P-01 就是这么产生的：不是有人做错，
    是流程允许了不一致而没人发现。
    """
    from label.core import make_id

    family = payload["family"]
    path = DATA / "label" / family / "subtasks.json"
    doc = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
           else {"family": family, "version": 1, "subtasks": []})
    items: list[dict[str, str]] = doc["subtasks"]
    usage = subtask_usage(family)
    action = payload["action"]
    notes: list[str] = []

    if action == "add":
        text = " ".join(str(payload["text"]).split())
        if not text:
            return {"ok": False, "error": "文字为空"}
        sid = make_id(text)
        if any(s["id"] == sid for s in items):
            return {"ok": False, "error": f"ID {sid} 已存在"}
        items.append({"id": sid, "text": text})
        labeled = sum(1 for v in usage.values() for _ in v)
        if labeled:
            done = len({e for v in usage.values() for e in v})
            notes.append(f"⚠ 已经标注了 {done} 集才新增这个 subtask —— "
                         f"前面那些集里这个动作可能被标成了别的或被跳过，建议复查")
            doc.setdefault("_added_late", []).append(
                {"id": sid, "after_episodes_labeled": done})

    elif action == "rename":
        target = next((s for s in items if s["id"] == payload["id"]), None)
        if target is None:
            return {"ok": False, "error": f"没有 {payload['id']}"}
        target["text"] = " ".join(str(payload["text"]).split())
        notes.append(f"ID 未变（{target['id']}），"
                     f"{len(usage.get(target['id'], []))} 集的引用自动跟随")

    elif action == "delete":
        sid = payload["id"]
        if usage.get(sid):
            return {"ok": False, "error": f"{sid} 正被 {len(usage[sid])} 集引用："
                                          f"{usage[sid][:5]}{' …' if len(usage[sid]) > 5 else ''}"}
        before = len(items)
        items[:] = [s for s in items if s["id"] != sid]
        if len(items) == before:
            return {"ok": False, "error": f"没有 {sid}"}
    else:
        return {"ok": False, "error": f"未知操作 {action}"}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subtasks.cache_clear()
    return {"ok": True, "subtasks": items, "notes": notes,
            "usage": {s["id"]: len(usage.get(s["id"], [])) for s in items}}


def review(document: dict[str, Any], family: str) -> list[dict[str, str]]:
    """在线版检查。**只覆盖 `validate.py` 八类里的三类**：重叠 / 覆盖 / 歧义。

    ⚠ **这不是 `validate.py` 的同一份代码**，尽管本文件开头与
    `validate.py` / README / AGENTS.md 四处都那么写过。实际没有 import 它。
    在线保存时**拦不下**这五类：污染（出题产物回写，即 P-03 本身）、
    引用（未定义的 subtask id）、派生（start/end 与帧号不自洽）、
    序列（动作讲不通，抓出过 wash 两处真错误）、可疑（零长度段）。
    收敛计划见 `docs/cleanup_checklist.md` 的 4.3。

    已实现的三类判据与 `validate.py` 逐条对齐 —— 尤其**重叠走帧号不走秒**。
    """
    segs = document["segments"]
    bounds = document["source"].get("episode_bounds")
    out: list[dict[str, str]] = []

    if not segs:
        out.append({"level": "block", "text": "还没有任何标注段"})
        return out

    ordered = sorted(segs, key=lambda s: s["start_frame"])
    for a, b in zip(ordered, ordered[1:]):
        if b["start_frame"] < a["end_frame"]:
            out.append({"level": "block",
                        "text": f"帧重叠：{a['subtask']} 与 {b['subtask']}"})

    # 多集打包的视频：整集没标是硬错（P-01 就是这么漏的）
    if bounds and len(bounds) > 1:
        spans = [(s["start"], s["end"]) for s in segs]
        missed = [i for i, (lo, hi) in enumerate(bounds)
                  if not any(lo - 0.5 <= a and b <= hi + 0.5 for a, b in spans)]
        if missed:
            out.append({"level": "block",
                        "text": f"本视频含 {len(bounds)} 个 episode，第 {missed} 个完全没有标注"})

    # 同一 episode 内重复 subtask：**不禁止**（wash 确实洗两个盘子），但要让人知道后果
    per: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in segs:
        idx = 0
        if bounds:
            idx = next((i for i, (lo, hi) in enumerate(bounds) if lo - 0.5 <= s["start"] <= hi + 0.5), 0)
        per[idx][s["subtask"]] += 1
    for idx, tally in per.items():
        dup = {k: v for k, v in tally.items() if v > 1}
        if dup:
            out.append({"level": "warn",
                        "text": f"episode {idx} 内重复：{dup} —— 按动作问时刻的题真值将不唯一（P-05）"})
    return out


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:      # 静音访问日志
        pass

    def _send(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                        # noqa: N802
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            self.path = "/index.html"
            return SimpleHTTPRequestHandler.do_GET(self)
        if url.path == "/api/families":
            return self._send({"families": families()})
        if url.path == "/api/episodes":
            return self._send({"episodes": episodes(query["family"][0])})
        if url.path == "/api/episode":
            return self._send(load_episode(query["family"][0], query["episode"][0]))
        if url.path == "/api/usage":
            fam = query["family"][0]
            used = subtask_usage(fam)
            return self._send({"usage": {s["id"]: len(used.get(s["id"], []))
                                         for s in subtasks(fam)}})
        if url.path.startswith("/media/"):
            return self._serve_media(DATA / "source" / url.path[len("/media/"):])
        return SimpleHTTPRequestHandler.do_GET(self)

    def _serve_media(self, target: Path) -> None:
        """按 HTTP Range 分块发送。

        **不支持 Range 时浏览器必须整段下完才能开始播**，而全帧内视频很大
        （tea2 单集 213 MB，wash 52 MB —— 全帧内是普通 GOP 的 3.3 倍，
        那是为了让切片能 -c copy 无损而付出的代价，D-18）。
        支持 Range 之后浏览器只取当前要播的那一段，拖动进度条也是即时的。
        """
        if not target.is_file():
            return self._send({"error": "not found"}, 404)
        size = target.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        partial_req = rng.startswith("bytes=")
        if partial_req:
            spec = rng[6:].split(",")[0]
            lo, _, hi = spec.partition("-")
            if lo:
                start = min(int(lo), size - 1)
                end = min(int(hi), size - 1) if hi else size - 1
            elif hi:                                  # bytes=-N：最后 N 字节
                start = max(0, size - int(hi))
        length = end - start + 1

        self.send_response(206 if partial_req else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial_req:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with target.open("rb") as fh:                 # 分块写，不把整个文件读进内存
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return                            # 浏览器跳转/关闭是常事，不当错误
                remaining -= len(chunk)

    def do_HEAD(self) -> None:                       # noqa: N802
        url = urlparse(self.path)
        if url.path.startswith("/media/"):
            return self._serve_media(DATA / "source" / url.path[len("/media/"):])
        return SimpleHTTPRequestHandler.do_HEAD(self)

    def do_POST(self) -> None:                       # noqa: N802
        route = urlparse(self.path).path
        if route not in ("/api/save", "/api/subtasks"):
            return self._send({"error": "unknown"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            if route == "/api/subtasks":
                return self._send(edit_subtasks(payload))
            return self._send(save_episode(payload))
        except Exception as exc:                      # noqa: BLE001
            return self._send({"ok": False, "problems": [
                {"level": "block", "text": f"{type(exc).__name__}: {exc}"}]}, 200)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # 必须用 partial 传 directory —— SimpleHTTPRequestHandler.__init__ 会用参数
    # 覆盖同名类属性（默认 os.getcwd()），设类属性不生效。
    handler = partial(Handler, directory=str(UI))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"标注器 → http://localhost:{args.port}")
    print(f"  族：{', '.join(families())}")
    print("  VS Code Remote 会自动转发端口；没自动转发就用 PORTS 面板手动加。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
