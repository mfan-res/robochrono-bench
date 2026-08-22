#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第 4 步：决定出哪些题，以及每道题要什么素材。

    python3 src/vqa/plan.py                  # 打印题量与素材统计
    python3 src/vqa/plan.py --write          # 写 build/plan.json
    python3 src/vqa/plan.py --time-repeats keep --cap 5    # 换参数

**这一步不生产任何东西**，只出两张表：

    items    每道题的内容（问谁、答案是什么、用哪几条干扰项、要哪些素材）
    media    去重后的素材需求清单（[5] 照着切）

拆开的理由（D-05）
------------------
把「要什么素材」先枚举出来，重复的就能在生产前合并。
实测旧的 `planning_clips` 8.9 GB 与 `understanding_clips` 是同一批片段的
重复编码 —— 同分辨率同帧数同时长，像素差只有 0.1–0.3 的编码噪声。
理解题与规划题用的**本来就是同一个片段**，只是问法不同。

一条必须守住的规则：**片段不得越过本段段尾**
--------------------------------------------
planning 问「接下来会发生什么」，答案是**下一段**的动作。
片段只要越过本段段尾，露出的就是答案本身。

这不是假想。stack_cubes 被回写进标注层的出题窗口（P-03），四段全部越界 2 秒：

```
                  人工标注           回写的窗口
pick_red_cube     0.00 → 25.70      10.00 → 27.70     ← 段尾之后 +2.00s
place_red_cube   25.70 → 33.05      31.05 → 35.05     ← +2.00s
```

`window_for_segment` 的 `after_window` 参数在 planning 上就是泄漏。
所以这里把它写成**断言**（`assert_no_leak`），而不是一个可配置项 ——
可配置意味着有人会配错，而配错了不报错。

> 起点后移是另一回事，**那是合理的**：段的开头往往还是上一个动作的余波
> （P-06），不后移的话片段里看不到目标动作。起点怎么取由 `--window` 决定，
> 终点则一律不越界。

三个待决项在这里生效
--------------------
都做成参数，人拍板后改默认值即可，不必改代码：

``--time-repeats``  P-05：wash 每集洗两个盘子，「pick the plate 在第几秒」有两个答案。
                    ``skip``（默认）只对不重复的动作出时间题；``keep`` 全出。
``--window``        A1：``raw`` 用标注段原始起止（默认，时长自然但本身泄答案）；
                    ``tail`` 只取段尾若干秒（时长恒定，但看不到动作全过程）。
``--cap``           A3：每族每集最多出几道，``None``（默认）不封顶。
``--none-option``   A7：见下。

A7 ·「都不对」这个选项在旧题里从来不是答案
------------------------------------------
全量统计 `eval/datasets/QA`：**10,787 道题有「All other options are wrong.」，
它是正确答案的有 0 道。**

于是它是白送的一次排除 —— 名义六选一，**实际五选一**，
随机基线不是 16.7% 而是 20%。三种处置：

``off``         （默认）不放这个选项，五选一。基线 20%，与实际难度一致。
``fixed``       放，但永不为答案。复刻 v1，仅供 A4 对照用。
``answerable``  放，并在一部分题里**抽掉正确答案**让它成为真答案。
                这会真正考「能不能说出都不对」，但改变了题型语义，属 ① 能力定义。

默认选 ``off`` 是因为它**不改变任何一道题的实际难度**，只是让写下来的基线
等于真实基线。``answerable`` 需要人拍板。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BUILD = ROOT / "build"

PLAN_VERSION = "1"

# 出题用的视角。三视角都有，但 main 是唯一被标注参照的那个 ——
# 其余视角靠时间对齐，段边界在它们上面未经核验。
# ── 共用地基在 tasks/_base.py ──────────────────────────────────────
# 本文件只做**编排**：读产物 → 遍历段 → 调题型规则 → 去重素材 → 自检 → 报表。
# 常量与判据（含它们的实测依据）、选项构造、帧候选，全部在 _base。
from tasks._base import (  # noqa: E402
    Ctx, DISTRACTORS_PER_QUESTION, MUTUAL_RATIO, RECIPE_VERSION, TASKS, VIEW,
    Looks, cooccurrence,
)
from tasks.image import emit_image_in_video, emit_left_right  # noqa: E402
from tasks.order import emit_step_order  # noqa: E402
from tasks.text import emit_text  # noqa: E402
from tasks.timing import emit_time  # noqa: E402


def fingerprint(items: list[dict[str, Any]]) -> str:
    """一批题的内容指纹。**必须覆盖每个题型真正的「答案与选项」。**

    此前它只算 ``[id, answer_text, *distractors]`` —— 那是文字选项题的形状。
    图选项题的 `answer_text` 是个常量、`distractors` 是空的，`time` 也一样，
    于是**三个题型共 3,904 道（占 38%）完全不在覆盖范围内**。实测：

        把 1,264 道 image_in_video 的选项顺序全反转  → 指纹不变
        把 2,640 道 left_right 的正确答案全换掉      → 指纹不变

    确定性自检对它们形同虚设。这三个都是**后加的题型** ——
    加题型时没人回来看这个函数，而它不报错，只是悄悄少管一块。

    现在按题型取真正的内容：文字题取 answer_text + distractors，
    图选项题取 correct_option + image_options，time 取媒体。
    **新增题型时若两个字段都取不到，这里会显式报错**，不再静默漏过。
    """
    rows: list[list[str]] = []
    for item in items:
        parts = [item["id"], item["task"]]
        text = [item.get("answer_text") or "", *(item.get("distractors") or [])]
        images = [item.get("correct_option") or "", *(item.get("image_options") or [])]
        media = list(item.get("media") or [])
        if len(text) > 1:                 # 文字选项题
            parts += text
        elif len(images) > 1:             # 图选项题
            parts += images
        elif media:                       # time：内容就是那段视频
            parts += media
        else:
            raise AssertionError(
                f"{item['id']}（{item['task']}）没有可指纹的内容 —— "
                "新题型请在 fingerprint() 里声明它的『答案与选项』长什么样。"
                "**不要让它静默漏过**，那正是图选项题曾经的处境。")
        rows.append(parts)
    return hashlib.md5(json.dumps(rows, ensure_ascii=False,
                                  sort_keys=True).encode()).hexdigest()[:12]


# **顺序即 items 的顺序**，改动会让 build/plan.json 的行序变化（内容不变）。
EMITTERS = (emit_text, emit_left_right, emit_image_in_video,
            emit_step_order, emit_time)


def build(index: dict[str, Any], vocab: dict[str, Any],
          window: str, time_repeats: str, cap: int | None,
          none_option: str, looks: "Looks") -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    media: dict[str, dict[str, Any]] = {}       # 内容寻址去重
    skipped: Counter = Counter()

    def need(kind: str, family: str, episode: str,
             start: int | None = None, end: int | None = None,
             view: str = VIEW) -> str:
        """登记一条素材需求，返回它的 key。同样的需求只登记一次。

        `kind` 为 ``frame`` 时抽单帧（图选项题型用），``clip`` 切片，``video`` 整段。
        `view` 默认 main —— 只有 `left_right` 会用到手腕视角。
        """
        key = (f"{family}/{episode}/{view}/{kind}" +
               (f"/{start:06d}-{end:06d}" if start is not None else ""))
        media.setdefault(key, {
            "key": key, "kind": kind, "family": family, "episode": episode,
            "view": view, "start_frame": start, "end_frame": end,
            "source": f"data/source/{family}/{episode}/{view}.mp4",
            "used_by": [],
        })
        return key

    # 图选项题型的帧池：每段一条记录（三个视角同一时刻）
    pool: dict[str, list[dict[str, Any]]] = {}
    for fam in sorted(index):
        rows = []
        for ep in index[fam]["episodes"]:
            for i, seg in enumerate(ep["segments"]):
                rows.append({"family": fam, "episode": ep["episode"], "seg_index": i,
                             "segment_id": seg["id"], "subtask": seg["subtask"],
                             "frame": (seg["start_frame"] + seg["end_frame"]) // 2})
        pool[fam] = rows

    for family in sorted(index):
        entry = index[family]
        fps = entry["fps"]
        texts = {s["id"]: s["text"] for s in vocab[family]["subtasks"]}
        actions = list(texts.values())
        compat = cooccurrence(entry["episodes"])
        # 可借的：别族的真实动作，且其宾语不在本场景里（否则可能碰巧也是真的）
        here = {s["object"] for s in vocab[family]["subtasks"] if s["object"]}
        borrowable = sorted({s["text"] for f, v in vocab.items() if f != family
                             for s in v["subtasks"]
                             if s["object"] and s["object"] not in here}
                            - set(actions))

        for episode in entry["episodes"]:
            segments = episode["segments"]
            ctx = Ctx(family=family, episode=episode, segments=segments, fps=fps,
                      texts=texts, compat=compat, borrowable=borrowable, looks=looks,
                      need=need, items=items, skipped=skipped, window=window,
                      cap=cap, none_option=none_option, time_repeats=time_repeats)
            for emit in EMITTERS:
                emit(ctx)

    for item in items:
        # **`image_options` 也要计入引用。** 图选项题型把选项图放在
        # `image_options` 而不是 `media` 里（compose 需要区分「题面」与「选项」），
        # 漏统计的后果不只是账算错 —— 若哪天按「没被引用」清理素材，
        # 4,170 张里有 1,390 张会被误删。
        for key in list(item["media"]) + list(item.get("image_options", [])):
            media[key]["used_by"].append(item["id"])

    choices = DISTRACTORS_PER_QUESTION + 1 + (none_option != "off")
    return {
        "plan_version": PLAN_VERSION,
        "recipe_version": RECIPE_VERSION,
        "options": {"window": window, "time_repeats": time_repeats, "cap": cap,
                    "view": VIEW, "distractors_per_question": DISTRACTORS_PER_QUESTION,
                    "none_option": none_option,
                    "choices_per_question": choices,
                    "random_baseline": round(1 / choices, 4)},
        "items": items,
        "media": list(media.values()),
        "skipped": dict(skipped),
    }


def report(plan: dict[str, Any], index: dict[str, Any]) -> None:
    by_family: dict[str, Counter] = defaultdict(Counter)
    for item in plan["items"]:
        by_family[item["family"]][item["task"]] += 1

    print(f"{'族':<13}" + "".join(f"{t:>13}" for t in TASKS) + f"{'合计':>8}")
    print("-" * 78)
    total: Counter = Counter()
    for family in sorted(by_family):
        counts = by_family[family]
        total.update(counts)
        print(f"{family:<13}" + "".join(f"{counts[t]:>13}" for t in TASKS)
              + f"{sum(counts.values()):>8}")
    print("-" * 78)
    print(f"{'合计':<13}" + "".join(f"{total[t]:>13}" for t in TASKS)
          + f"{sum(total.values()):>8}")

    from collections import Counter as _C
    kinds = _C(m["kind"] for m in plan["media"])
    clips = [m for m in plan["media"] if m["kind"] == "clip"]
    frames = [m for m in plan["media"] if m["kind"] == "frame"]
    reuse = sum(len(m["used_by"]) for m in clips)
    freuse = sum(len(m["used_by"]) for m in frames)
    print(f"\n素材需求：{dict(kinds)}")
    print(f"  切片被 {reuse} 道题引用 —— 去重省下 {reuse - len(clips)} 次编码"
          f"（understanding / planning / planning_2 / image_in_video 共用同一片段）")
    if frames:
        byview = _C(m["view"] for m in frames)
        print(f"  帧被 {freuse} 次引用，去重后 {len(frames)} 张 —— 省下 {freuse - len(frames)} 次抽帧")
        print(f"    按视角 {dict(byview)}")
        print(f"    **所有选项图同一条抽帧路径、同一套参数** —— 正确图与干扰图之间"
              f"没有图像统计上的差别可辨")

    if plan["skipped"]:
        print("\n没出成的：")
        for why, n in sorted(plan["skipped"].items(), key=lambda kv: -kv[1]):
            print(f"  {n:>6}  {why}")


def main() -> int:
    def arg(name: str, default: str | None = None) -> str | None:
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    window = arg("--window", "raw")
    time_repeats = arg("--time-repeats", "skip")
    none_option = arg("--none-option", "off")
    cap_raw = arg("--cap")
    cap = int(cap_raw) if cap_raw else None

    index = json.loads((BUILD / "index.json").read_text(encoding="utf-8"))["families"]
    frames_path = BUILD / "frames.json"
    if not frames_path.exists():
        print("❌ 缺 build/frames.json —— 图选项题型要靠它判「干扰项在画面上分不分得开」。\n"
              "   先跑 python3 src/vqa/frames.py --write")
        return 1
    looks_payload = json.loads(frames_path.read_text(encoding="utf-8"))
    import numpy
    looks_desc = numpy.load(BUILD / "frames_desc.npy")
    vocab = json.loads((BUILD / "vocab.json").read_text(encoding="utf-8"))["families"]
    # `data/llm_cache/` 三代（v1-vendor / v2 / v3）全部退场，只作留档 ——
    # 干扰项改为一律取自真实标签，不再需要任何生成物（D-37 / D-38）。
    looks = Looks(looks_payload, looks_desc)
    plan = build(index, vocab, window, time_repeats, cap, none_option, looks)

    # 确定性自检：同样输入必须得到同样一批题。
    # **构建两遍比对**，因为这类 bug（遍历 set、用 dict 顺序、掺进时间戳）
    # 不会报错，只会让每次出的题悄悄不同 —— 而下游的盲测结论就此失效。
    again = build(index, vocab, window, time_repeats, cap, none_option, looks)
    fp = [fingerprint(p["items"]) for p in (plan, again)]
    if fp[0] != fp[1]:
        print(f"❌ 构建不确定：两次指纹 {fp[0]} ≠ {fp[1]}")
        print("   同样输入得到了不同的题。常见原因：遍历 set / dict、掺进时间或随机数。")
        return 1
    # 把【实际用的门槛】写进产物 —— 出厂检查照着这个验，
    # 而不是自己另算一遍。检查与规则分叉会在合规的题上报警（踩过）。
    plan["option_floor"] = {
        "percentile_source": "build/frames_floors.json",
        "scale": {"image_in_video": MUTUAL_RATIO, "left_right": 1.0},
        "applies_to": {"image_in_video": "四个选项两两（六条边）",
                       "left_right": "正确项与每条干扰项"},
    }
    plan["fingerprint"] = fp[0]
    opt = plan["options"]
    print(f"指纹 {plan['fingerprint']}（两次构建一致）")
    print(f"配方 {RECIPE_VERSION}   window={window}  time-repeats={time_repeats}  "
          f"cap={cap}  none-option={none_option}")
    print(f"选择题 {opt['choices_per_question']} 选一，随机基线 "
          f"{opt['random_baseline']:.1%}"
          + ("" if none_option != "fixed" else
             "  ⚠ fixed 下「都不对」永不为答案，真实基线其实是 "
             f"{1 / (opt['choices_per_question'] - 1):.1%}") + "\n")
    report(plan, index)

    if "--write" in sys.argv:
        BUILD.mkdir(exist_ok=True)
        out = BUILD / "plan.json"
        out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写入 {out.relative_to(ROOT)}（{out.stat().st_size / 1e6:.1f} MB）")
    else:
        print("\n加 --write 写入 build/plan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
