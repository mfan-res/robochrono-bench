#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第六步：把计划与素材组装成题目，按 `schemas/item.json` 的契约输出。

    python3 src/vqa/compose.py            # 组装并校验，不写盘
    python3 src/vqa/compose.py --write    # 写 data/vqa/items.jsonl

**这一步不碰 ffmpeg。** 素材在 [5] 已经切好，这里只做组装 ——
所以改题干、改选项顺序、改输出格式都是秒级的，不用重切 61 GB。
这正是 D-05 把 assets 与 compose 分开的目的。

三块分界：能不能反推答案
------------------------
```
prompt      发出去的全部内容。键是【封闭白名单】—— 没验证过的字段进不来
truth       答案，以及任何能反推答案的东西
provenance  溯源与配方，既不是答案也不发送
```

分界线是「能不能反推答案」，不是「看起来像不像答案」。
逐字段核对时发现五个字段名字完全不像答案却能反推 ——
`step_order` 的 `source_order` 排序后就是答案序列、
`time` 的 `input.start/end` 恰等于 `answer_seconds`、
`understanding` 的 `start` 单独就能把答案猜到 0.34–1.00。
**所以 prompt 用白名单而不是黑名单**：黑名单要求每个新字段都有人去验，
白名单让「没验过的进不来」成为默认。

选项顺序：确定性打乱
--------------------
按 `md5(题目id|选项文字)` 排序，**不是随机数**。
整条流水线唯一的要求是「同样输入必得同样一批题」（D-47 为此加了构建自检），
用随机数就得管种子，而种子管漏一次就再也复现不了。

出厂检查
--------
每一道题都过 `schemas/item.json`，另加四条本文件自己的判据：

1. prompt 里不得出现任何真值字段（schema 的封闭白名单已挡，这里再验一次）
2. 每个媒体路径必须真实存在
3. 答案字母在全库上分布均匀 —— 偏了说明打乱有问题
4. 选项文字不得重复 —— 同一道题里两个一样的选项等于少一个选项
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
ASSETS = ROOT / "data" / "vqa" / "assets"
OUT = ROOT / "data" / "vqa" / "items.jsonl"
LETTERS = "ABCDEF"


def shuffled(item_id: str, texts: list[str]) -> list[str]:
    """确定性打乱 —— 同一道题永远得到同一个顺序。"""
    return sorted(texts, key=lambda t: hashlib.md5(f"{item_id}|{t}".encode()).hexdigest())


def as_clock(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


ROLE = {"clip": "clip", "video": "context", "frame": "frame"}


def path_of(m: dict[str, Any]) -> str:
    ext = "jpg" if m["kind"] == "frame" else "mp4"
    return f"data/vqa/assets/{m['key']}.{ext}"


def compose(plan_item: dict[str, Any], media: dict[str, dict[str, Any]]) -> dict[str, Any]:
    task, item_id = plan_item["task"], plan_item["id"]
    prov = plan_item["provenance"]

    blocks: list[dict[str, Any]] = []
    for i, key in enumerate(plan_item["media"]):
        m = media[key]
        # step_order 的三张题面图**必须带序号** —— 题干问的就是
        # 「Image 1 / 2 / 3 的先后」，角色里丢了序号，下游就只能靠数组下标，
        # 那是隐式契约，改一次顺序就静默错位。
        role = f"step:{i + 1}" if task == "step_order" else ROLE[m["kind"]]
        blocks.append({"role": role, "view": m["view"],
                       "kind": "image" if m["kind"] == "frame" else "video",
                       "path": path_of(m)})

    # ── 图选项题型：选项本身是图，按 role=option:A…D 挂进 media ──
    # **顺序打乱与文字题共用同一个函数** —— 打乱规则只存在一处，
    # 否则两种题型的「确定性」会各自漂移。
    if plan_item.get("image_options"):
        order = shuffled(item_id, plan_item["image_options"])
        letter = LETTERS[order.index(plan_item["correct_option"])]
        for i, key in enumerate(order):
            m = media[key]
            blocks.append({"role": f"option:{LETTERS[i]}", "view": m["view"],
                           "kind": "image", "path": path_of(m)})
        return {
            "id": item_id, "family": plan_item["family"], "task": task,
            "group": plan_item["group"],
            "prompt": {"stem": plan_item["stem"],
                       # 图选项的 text 为 null —— 选项内容在 media 里
                       "options": [{"id": LETTERS[i], "text": None}
                                   for i in range(len(order))],
                       "media": blocks},
            "truth": {"answer": letter,
                      # ⚠ answer_text 与 option_text 不是一回事：图选项没有 text，
                      # 而 answer_text 是「left gripper camera view」这类描述。
                      # 全量比对过 5,553 题两者不同，合并会丢真信息。
                      "answer_text": plan_item["answer_text"], "option_text": None,
                      "extra": {"subtask": plan_item["answer_subtask"],
                                "correct_asset": plan_item["correct_option"]}},
            "provenance": prov,
        }

    if task == "time":
        lo, hi = plan_item["answer_seconds"]
        return {
            "id": item_id, "family": plan_item["family"], "task": task,
            "group": plan_item["group"],
            "prompt": {"stem": plan_item["stem"], "options": [], "media": blocks},
            "truth": {"answer": f"{as_clock(lo)}-{as_clock(hi)}",
                      "answer_text": None, "option_text": None,
                      "extra": {"seconds": {"start": lo, "end": hi},
                                "subtask": plan_item["answer_subtask"]}},
            "provenance": prov,
        }

    order = shuffled(item_id, [plan_item["answer_text"], *plan_item["distractors"]])
    letter = LETTERS[order.index(plan_item["answer_text"])]
    return {
        "id": item_id, "family": plan_item["family"], "task": task,
        "group": plan_item["group"],
        "prompt": {"stem": plan_item["stem"],
                   "options": [{"id": LETTERS[i], "text": t} for i, t in enumerate(order)],
                   "media": blocks},
        "truth": {"answer": letter, "answer_text": plan_item["answer_text"],
                  "option_text": plan_item["answer_text"],
                  "extra": {"subtask": plan_item["answer_subtask"]}},
        "provenance": prov,
    }


def main() -> int:
    plan = json.loads((BUILD / "plan.json").read_text(encoding="utf-8"))
    media = {m["key"]: m for m in plan["media"]}
    items = [compose(i, media) for i in plan["items"]]

    print(f"组装 {len(items)} 道题（题库指纹 {plan.get('fingerprint', '?')}）")

    # ── 出厂检查 ──────────────────────────────────────────────
    bad: list[str] = []

    try:
        import jsonschema
        schema = json.loads((ROOT / "src" / "common" / "schemas" / "item.json")
                            .read_text(encoding="utf-8"))
        v = jsonschema.Draft202012Validator(schema)
        fails = [(i["id"], e.message) for i in items for e in list(v.iter_errors(i))[:1]]
        print(f"  ① schema     {len(items) - len(fails)}/{len(items)} 通过")
        bad += [f"schema: {i} — {m}" for i, m in fails[:5]]
    except ImportError:
        print("  ① schema     跳过（没装 jsonschema）")

    TRUTH_WORDS = {"answer", "truth", "seconds", "subtask", "correct", "label"}
    leaks = [i["id"] for i in items
             if TRUTH_WORDS & {k.lower() for k in i["prompt"]}
             or any(TRUTH_WORDS & {k.lower() for k in o} for o in i["prompt"]["options"])]
    print(f"  ② prompt 无真值字段   {'✓' if not leaks else f'✗ {len(leaks)}'}")
    bad += [f"prompt 含真值字段: {x}" for x in leaks[:5]]

    missing = {b["path"] for i in items for b in i["prompt"]["media"]
               if not (ROOT / b["path"]).exists()}
    print(f"  ③ 媒体存在   {'✓' if not missing else f'✗ 缺 {len(missing)}'}")
    bad += [f"媒体缺失: {p}" for p in list(missing)[:5]]

    # 图选项的 text 全是 null，比 text 会全部误报 —— 改比【实际内容】：
    # 文字题比 text，图选项比媒体路径。
    def distinct(i: dict[str, Any]) -> int:
        opts = i["prompt"]["options"]
        if opts and opts[0]["text"] is None:
            return len({m["path"] for m in i["prompt"]["media"]
                        if m["role"].startswith("option:")})
        return len({o["text"] for o in opts})

    dup = [i["id"] for i in items if i["prompt"]["options"]
           and distinct(i) != len(i["prompt"]["options"])]
    print(f"  ④ 选项不重复 {'✓' if not dup else f'✗ {len(dup)}'}")
    bad += [f"选项重复: {x}" for x in dup[:5]]

    pos = Counter(i["truth"]["answer"] for i in items if i["prompt"]["options"])
    n = sum(pos.values())
    spread = (max(pos.values()) - min(pos.values())) / n if pos else 0
    print(f"  ⑤ 答案位置   {dict(sorted(pos.items()))}  极差 {spread:.1%}"
          + ("  ⚠ 偏斜" if spread > 0.03 else "  ✓"))

    # ⑥ 图选项：选项之间是否够不像（D-56 / D-57）。
    # **验的是 plan 实际承诺的门槛**（`plan.json` 的 `option_floor`），
    # 不是这里另算一遍 —— 检查与规则分叉会在合规的题上报警，踩过一次。
    #
    # 两个题型验的边不同，因为它们的构造不同：
    #   image_in_video  四个选项【两两】六条边 —— 规则对称，答案才不会成离群点
    #   left_right      正确项与每条干扰项 —— 2×2 的对称性由构造本身保证
    frames_path = BUILD / "frames.json"
    plan_path = BUILD / "plan.json"
    if frames_path.exists() and plan_path.exists():
        import itertools

        import numpy
        payload = json.loads(frames_path.read_text(encoding="utf-8"))
        promised = json.loads(plan_path.read_text(encoding="utf-8")).get("option_floor", {})
        scale = promised.get("scale", {})
        floors = payload["floors"]
        at = {k: i for i, k in enumerate(payload["order"])}
        vecs = numpy.load(BUILD / "frames_desc.npy").astype("float32")

        def key_of(path: str) -> str | None:
            return path.split("assets/", 1)[1].rsplit(".", 1)[0] if "assets/" in path else None

        def dist(a: str, b: str) -> float:
            d = vecs[at[a]] - vecs[at[b]]
            return float((d @ d / d.size) ** 0.5)

        near, checked = [], 0
        for item in items:
            opts = [m for m in item["prompt"]["media"] if m["role"].startswith("option:")]
            if not opts:
                continue
            keys = {m["role"]: key_of(m["path"]) for m in opts}
            if any(k is None or k not in at for k in keys.values()):
                continue
            views = {k.split("/")[2] for k in keys.values()}
            base = min((floors[f"{item['family']}/{v}"] for v in views
                        if f"{item['family']}/{v}" in floors), default=0.0)
            floor = base * scale.get(item["task"], 1.0)
            ans = f"option:{item['truth']['answer']}"
            pairs = (itertools.combinations(sorted(keys), 2) if item["task"] == "image_in_video"
                     else [(ans, r) for r in sorted(keys) if r != ans])
            checked += 1
            for ra, rb in pairs:
                if dist(keys[ra], keys[rb]) < floor - 1e-6:
                    near.append(f"{item['id']} 的 {ra}/{rb}")
                    break
        print(f"  ⑥ 图选项够不像 {'✓' if not near else f'✗ {len(near)}'}"
              f"　（查了 {checked} 道，按 plan 承诺的门槛）")
        bad += [f"选项之间画面差不足: {x}" for x in near[:5]]

    by_task = Counter(i["task"] for i in items)
    print(f"\n  题型 {dict(by_task)}")
    print(f"  组数 {len({i['group'] for i in items})}"
          f"（time 是一个视频一组，其余一题一组）")

    if bad:
        print(f"\n❌ {len(bad)} 项不合格：")
        for b in bad[:8]:
            print(f"   {b}")
        return 1

    if "--write" in sys.argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", encoding="utf-8") as fh:
            for i in items:
                fh.write(json.dumps(i, ensure_ascii=False) + "\n")
        print(f"\n已写入 {OUT.relative_to(ROOT)}（{OUT.stat().st_size / 1e6:.1f} MB）")
    else:
        print("\n加 --write 写入 data/vqa/items.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
