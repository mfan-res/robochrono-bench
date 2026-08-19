#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第七步：把 `items.jsonl` 投影成评测端能吃的形状。

    python3 src/vqa/pack.py            # 投影并校验，不写盘
    python3 src/vqa/pack.py --write    # 写 data/vqa/eval/

为什么需要一个投影层
--------------------
④ 的产物走新契约（`prompt` / `truth` / `provenance`），
而 ⑥ 评测读的仍是 v1 的字段（`question` / `options` / `answer` / `input.clip_path`）。

**干净的契约留在 `items.jsonl`，适配消费者是这一步的职责。**
这样评测端一行都不用改，它那六套回归也就仍然有效 ——
改评测端等于同时动了「被测的东西」和「测它的尺子」。

顺手掐掉 v1 的一条泄漏
----------------------
v1 的 time 题在 `input` 里带着 `start` / `end`，而那**恰好等于答案**。
本步不发这两个字段。已确认评测端不读它们 ——
`time_eqa.py` 里对 `start`/`end` 的读取全部是在解析**模型输出**，不是读题目。

媒体路径：相对 QA 文件所在目录
------------------------------
v1 的解析顺序是「先按 CWD 试，不行就**按文件名**在 QA 目录下索引」——
因为 v1 的媒体就散在 QA 目录里。对新数据这条既找不到（素材在 `data/vqa/assets/`）
又危险：新切片叫 `000163-000264.mp4` 这种纯帧号，**跨族必然重名**，
索引搜到了也可能给出别的族那一份。

评测端已在文件名索引**之前**加了一步「相对 QA 文件解析」。
所以这里写相对路径 —— **可移植，换机器不用重生成**。
（一度改成写绝对路径能跑通，但那是绕过问题不是解决问题。）

输出布局：扁平的 `<族>/<题型>.json`
--------------------------------
v1 的布局是 `QA/<组>/<族>/<题型>_vqa.json`，其中那层「组」
（understanding / planning）是历史约定不是结构 —— `time` 归 understanding、
`step_order` 归 planning，纯粹因为当初谁先做。而且 v1 里**一半的族还多嵌了
一层子目录、层名还不统一**，`qa_path()` 为此专门带了一段递归兜底。

新数据不继承这些。评测端的 `qa_path()` 已改成**扁平布局优先、v1 布局兜底**，
所以两边都能读，v1 的夹具与六套回归一行未改。
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "data" / "vqa" / "items.jsonl"
OUT = ROOT / "data" / "vqa" / "eval"




def render(stem: str, options: list[dict[str, Any]]) -> str:
    """题干 + 选项，拼成 v1 的 `question` 形状。

    v1 把渲染后的整段存进 `question` 字段，评测端直接发这一整段。
    我们的 `items.jsonl` 里题干与选项是分开的（单一事实来源），
    在这里拼 —— 拼的规则只存在这一处。
    """
    if not options:
        return stem
    lines = "\n".join(f"{o['id']}. {o['text']}" for o in options)
    return f"{stem}\nOptions:\n{lines}"


def project(item: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    prov = item["provenance"]
    media = item["prompt"]["media"]
    rel = [os.path.relpath(ROOT / m["path"], out_dir) for m in media]

    # 图选项题型：题面媒体与选项图分开挂。评测端读的是
    #   left_right      input.image_path（主视角图）+ options[].image_path
    #   image_in_video  input.clip_path（片段）      + options[].image_path
    opts = [m for m in media if m["role"].startswith("option:")]
    if opts:
        head = next(m for m in media if not m["role"].startswith("option:"))
        head_key = "clip_path" if head["kind"] == "video" else "image_path"
        row: dict[str, Any] = {
            "id": item["id"],
            "video_id": f"{item['family']}/{prov['episode']}",
            "type": item["task"],
            "question": item["prompt"]["stem"],      # 图选项不往题干里拼文字选项
            "answer": item["truth"]["answer"],
            "answer_text": item["truth"]["answer_text"],   # 图选项没有 option text，
            "input": {head_key: os.path.relpath(ROOT / head["path"], out_dir)},
            "options": [{"id": m["role"].split(":")[1],
                         "image_path": os.path.relpath(ROOT / m["path"], out_dir)}
                        for m in opts],
        }
        return row

    # step_order：题面是 N 张带标号的图，选项是文字（排列）。
    # 发 `image_paths` 而不是拼好的宫格 —— 拼图是 v1 的做法，见 BC-16。
    if item["task"] == "step_order":
        steps = sorted((m for m in media if m["role"].startswith("step:")),
                       key=lambda m: int(m["role"].split(":")[1]))
        return {
            "id": item["id"],
            "video_id": f"{item['family']}/{prov['episode']}",
            "type": item["task"],
            "question": item["prompt"]["stem"],
            "answer": item["truth"]["answer"],
            "options": [{"id": o["id"], "text": o["text"]}
                        for o in item["prompt"]["options"]],
            "input": {"image_paths": [os.path.relpath(ROOT / m["path"], out_dir)
                                      for m in steps]},
        }

    key = "video_path" if item["task"] == "time" else "clip_path"
    # time 要多发一个视频时长。**它不是真值** —— 是这段视频的客观属性，
    # 与「动作在第几秒」无关。不给的话「用秒作答」根本无法执行：
    # 模型只看到若干抽帧，会退回输出 [0,1] 的比例（BC-18）。
    extra_input: dict[str, Any] = {}
    if item["task"] == "time":
        seconds = item["provenance"]["recipe"]["clip"].get("seconds")
        if seconds:
            extra_input["video_seconds"] = round(float(seconds), 3)

    # **只发评测端真读的字段。** 逐个核对过源码：
    #   读   id / video_id / type / question / answer / options[id,text] / input
    #   不读 answer_text / correct_option / source_id / provenance
    #   冗余 Q 与 A —— 读法是 `Q or question` / `answer or A`，留一个就够
    #
    # 不读的一个都不发。它们全都能由 `id` 回查 `items.jsonl` 得到
    # （id 形如 `<族>/<集>/<段id>@<题型>`），发一份等于把同一个事实存两处 ——
    # 这个项目已经因为「同一个东西存 11 份」付出过代价（D-25）。
    row: dict[str, Any] = {
        "id": item["id"],
        "video_id": f"{item['family']}/{prov['episode']}",   # time 按它分组，须全局唯一
        "type": item["task"],
        "question": render(item["prompt"]["stem"], item["prompt"]["options"]),
        "answer": item["truth"]["answer"],
        # ⚠ input 里【只放媒体路径】。v1 在这里还放了 start/end，而那等于答案
        "input": {key: rel[0], **({f"{key}s": rel} if len(rel) > 1 else {}),
                  **extra_input},
    }
    if item["prompt"]["options"]:
        row["options"] = [{"id": o["id"], "text": o["text"]}
                          for o in item["prompt"]["options"]]
    if item["task"] == "time":
        row["answer_seconds"] = item["truth"]["extra"]["seconds"]
    return row


def main() -> int:
    if not ITEMS.exists():
        print(f"❌ 没有 {ITEMS.relative_to(ROOT)} —— 先跑 src/vqa/compose.py --write")
        return 1
    items = [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines()]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for i in items:
        groups[(i["family"], i["task"])].append(i)

    print(f"投影 {len(items)} 道题 → {len(groups)} 个文件")
    written: list[tuple[str, str, int, Path]] = []
    problems: list[str] = []

    for (family, task), rows in sorted(groups.items()):
        out_dir = OUT / family
        packed = [project(i, out_dir) for i in rows]

        # ── 出厂检查 ──
        for p, orig in zip(packed, rows):
            for k in ("start", "end", "answer_seconds", "seconds"):
                if k in p["input"]:
                    problems.append(f"{p['id']}: input 里出现了 {k}（可反推答案）")
            for r in [v for v in p["input"].values() if isinstance(v, str)]:
                if not (out_dir / r).resolve().exists():
                    problems.append(f"{p['id']}: 媒体解析不到 {r}")
            if p["question"].count("Options:") > 1:
                problems.append(f"{p['id']}: 题干里重复出现 Options:")
        if task != "time":
            for p in packed:
                if p["answer"] not in {o["id"] for o in p["options"]}:
                    problems.append(f"{p['id']}: answer 不在 options 里")

        written.append((family, task, len(packed), out_dir / f"{task}.json"))
        if "--write" in sys.argv:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{task}.json").write_text(
                json.dumps({"items": packed}, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8")

    by_fam: dict[str, int] = defaultdict(int)
    for f, _, n, _ in written:
        by_fam[f] += n
    print(f"  {dict(by_fam)}")
    print(f"  题型 {dict(Counter(t for _, t, _, _ in written))}")

    if problems:
        print(f"\n❌ {len(problems)} 项不合格：")
        for x in problems[:6]:
            print(f"   {x}")
        return 1
    print("\n出厂检查：input 无真值字段 ✓　媒体可解析 ✓　答案在选项内 ✓")

    if "--write" in sys.argv:
        manifest = {"items": len(items), "files": [str(p.relative_to(ROOT)) for _, _, _, p in written],
                    "fingerprint": json.loads((ROOT / "build" / "plan.json")
                                              .read_text(encoding="utf-8")).get("fingerprint")}
        (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
        size = sum(p.stat().st_size for _, _, _, p in written)
        print(f"已写入 {OUT.relative_to(ROOT)}/（{size / 1e6:.1f} MB，含 manifest）")
    else:
        print("加 --write 写入 data/vqa/eval/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
