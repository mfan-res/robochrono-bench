#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第四步：决定出哪些题，以及每道题要什么素材。

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

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BUILD = ROOT / "build"

PLAN_VERSION = "1"
RECIPE_VERSION = "v2.0"

# 出题用的视角。三视角都有，但 main 是唯一被标注参照的那个 ——
# 其余视角靠时间对齐，段边界在它们上面未经核验。
VIEW = "main"

TASKS = ("understanding", "planning", "planning_2", "time")

STEMS = {
    "understanding": "Based on the egocentric video up to now, choose the ONE option "
                     "that best matches what is happening RIGHT NOW?",
    "planning": "Based on the current visual state, what should happen next?",
    "planning_2": "The overall task is {task_name}. Based on the current visual state, "
                  "what should happen next?",
    "time": 'When did the action "{action}" happen?',
}

# 族的自然语言任务名，planning_2 的题干要用
TASK_NAMES = {
    "airpods": "putting the earphones into the airpods case",
    "gift_inhand": "handing over a gift",
    "pen_inbox": "putting a pen into a box",
    "stack_cubes": "stacking cubes",
    "tea": "making tea",
    "tea2": "brewing tea with a teapot",
    "wash": "washing dishes",
}

DISTRACTORS_PER_QUESTION = 4
NEARBY_PER_QUESTION = 2          # 其余用 LLM 的补满，见 pick_distractors
NONE_TEXT = "All other options are wrong."   # v1 的原文，保持一致便于对照


def assert_no_leak(clip: tuple[int, int], segment: dict[str, Any], task: str) -> None:
    """片段不得越过本段段尾。**断言而非配置项** —— 见模块 docstring。"""
    if task.startswith("planning") and clip[1] > segment["end_frame"]:
        raise AssertionError(
            f"{task} 片段 {clip} 越过段尾 {segment['end_frame']}"
            f"（{segment['id']} / {segment['subtask']}）—— 露出的正是答案")


def clip_for(segment: dict[str, Any], mode: str, fps: float) -> tuple[int, int]:
    """片段的起止帧。终点恒为段尾，只有起点随 mode 变。"""
    if mode == "raw":
        return segment["start_frame"], segment["end_frame"]
    if mode == "tail":                       # A1 的备选：时长恒定，消掉时长线索
        span = int(round(3.0 * fps))
        return max(segment["start_frame"], segment["end_frame"] - span), segment["end_frame"]
    raise ValueError(f"未知的 window 模式：{mode}")


def pick_distractors(subtask: str, pool: dict[str, list[str]], nearby: list[str],
                     exclude: set[str]) -> tuple[list[str], dict[str, int]]:
    """取 4 条干扰项：**先 2 条近邻，再用 LLM 的补满**。返回 (文字, 来源计数)。

    近邻 = 本族其它真实动作（对这一段是错的，但确实存在）。
    **这两条不是可选的，是必需的** ——

    只用 LLM 干扰项时实测：**4,115 道题 100% 能靠「哪个选项是真实存在的动作」答对。**
    因为 [3] 的 `not_correct` 判据保证了 LLM 干扰项不等于任何真实动作，
    于是「真动作」与「编的」两类文字完全不相交，跨题一学就会，不用看视频。

    ```
    真实动作   Pick the pen. / Pick the box. / Place the pen.
    只用 LLM   Pick the pen. / Place the box. / Pick the cup. / Pick the tray. / Pick the cloth.
                    ↑ 唯一一个真出现过的
    ```

    这个捷径是判据本身造出来的：判据没错（干扰项确实不该等于真动作），
    错在**只有一个来源**。混两个来源就没有可分的边界了。

    **顺序取而非随机取** —— 出题要确定，随机只在选项打乱那一步。
    """
    out: list[str] = []
    source: dict[str, int] = {"nearby": 0, "llm": 0}

    for text in nearby:
        if len(out) >= NEARBY_PER_QUESTION:
            break
        if text.lower() not in exclude:
            out.append(text)
            exclude.add(text.lower())
            source["nearby"] += 1

    for text in pool.get(subtask, []):
        if len(out) >= DISTRACTORS_PER_QUESTION:
            break
        if text.lower() not in exclude:
            out.append(text)
            exclude.add(text.lower())
            source["llm"] += 1

    # 小族（3 个动作）近邻凑不满 2 条时会多拿 LLM 的。**如实记，不补齐** ——
    # D-03：兜底发生时要留下痕迹，静默兜底正是问题 h 藏那么久的原因。
    return out, source


def build(index: dict[str, Any], vocab: dict[str, Any], pools: dict[str, Any],
          window: str, time_repeats: str, cap: int | None,
          none_option: str) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    media: dict[str, dict[str, Any]] = {}       # 内容寻址去重
    skipped: Counter = Counter()

    def need(kind: str, family: str, episode: str,
             start: int | None = None, end: int | None = None) -> str:
        """登记一条素材需求，返回它的 key。同样的需求只登记一次。"""
        key = (f"{family}/{episode}/{VIEW}/{kind}" +
               (f"/{start:06d}-{end:06d}" if start is not None else ""))
        media.setdefault(key, {
            "key": key, "kind": kind, "family": family, "episode": episode,
            "view": VIEW, "start_frame": start, "end_frame": end,
            "source": f"data/source/{family}/{episode}/{VIEW}.mp4",
            "used_by": [],
        })
        return key

    for family in sorted(index):
        entry = index[family]
        fps = entry["fps"]
        texts = {s["id"]: s["text"] for s in vocab[family]["subtasks"]}
        pool = pools[family]["distractors"]

        for episode in entry["episodes"]:
            segments = episode["segments"]
            per_episode = 0
            # 近邻候选按「同集内出现过」优先 —— 同集的动作在画面上更接近，
            # 比族里另一个八竿子打不着的动作更难排除
            in_episode = list(dict.fromkeys(s["subtask"] for s in segments))
            order = in_episode + [i for i in texts if i not in in_episode]

            for i, segment in enumerate(segments):
                if cap is not None and per_episode >= cap:
                    skipped["cap"] += 1
                    continue
                nxt = segments[i + 1] if i + 1 < len(segments) else None
                # 下一段必须在同一 episode 内 —— 跨 episode 的「接下来」不成立
                if nxt is not None and nxt["episode_index"] != segment["episode_index"]:
                    nxt = None

                base = f"{family}/{episode['episode']}/{segment['id']}"
                clip = clip_for(segment, window, fps)

                for task in ("understanding", "planning", "planning_2"):
                    if task != "understanding" and nxt is None:
                        skipped[f"{task}:段尾无下一段"] += 1
                        continue
                    answer_id = segment["subtask"] if task == "understanding" else nxt["subtask"]
                    answer_text = texts[answer_id]

                    exclude = {answer_text.lower()}
                    if task != "understanding":
                        # planning 的干扰项不能是「当前正在做的动作」—— 那不是预测
                        exclude.add(texts[segment["subtask"]].lower())
                    nearby = [texts[i] for i in order if i != answer_id]
                    options, source = pick_distractors(answer_id, pool, nearby, exclude)
                    if len(options) < DISTRACTORS_PER_QUESTION:
                        skipped[f"{task}:干扰项不足"] += 1
                        continue

                    assert_no_leak(clip, segment, task)
                    items.append({
                        "id": f"{base}@{task}",
                        "family": family, "task": task, "group": f"{base}@{task}",
                        "stem": STEMS[task].format(task_name=TASK_NAMES[family]),
                        "answer_subtask": answer_id, "answer_text": answer_text,
                        "distractors": options,
                        # [6] 照此组装选项。放在 plan 而非 compose，是因为它
                        # 决定题量与基线，属于「出哪些题」而不是「怎么排版」
                        "none_option": NONE_TEXT if none_option != "off" else None,
                        "media": [need("clip", family, episode["episode"], *clip)],
                        "provenance": {
                            "episode": episode["episode"],
                            "segment_id": segment["id"],
                            "subtask": segment["subtask"],
                            "next_subtask": nxt["subtask"] if nxt else None,
                            "recipe": {
                                "version": RECIPE_VERSION,
                                "clip": {"mode": window, "view": VIEW,
                                         "start_frame": clip[0], "end_frame": clip[1],
                                         "seconds": round((clip[1] - clip[0] + 1) / fps, 3)},
                                "distractors": source,
                                "synthetic": False,
                            },
                        },
                    })
                    per_episode += 1

            # ---- time：一集一组，用全长视频 ----
            if not episode["full_video_usable"]:
                skipped["time:该集有未标注的 episode"] += 1
                continue
            group = f"{family}/{episode['episode']}@time"
            asked = 0
            for segment in segments:
                if time_repeats == "skip" and "ambiguous_repeat" in segment["reasons"]:
                    skipped["time:同集内动作重复（P-05）"] += 1
                    continue
                items.append({
                    "id": f"{family}/{episode['episode']}/{segment['id']}@time",
                    "family": family, "task": "time", "group": group,
                    "stem": STEMS["time"].format(action=texts[segment["subtask"]].rstrip(".")),
                    "answer_subtask": segment["subtask"], "answer_text": None,
                    "answer_seconds": [segment["start"], segment["end"]],
                    "distractors": [],                       # time 不是选择题
                    "media": [need("video", family, episode["episode"])],
                    "provenance": {
                        "episode": episode["episode"], "segment_id": segment["id"],
                        "subtask": segment["subtask"], "next_subtask": None,
                        "recipe": {
                            "version": RECIPE_VERSION,
                            "clip": {"mode": "full_video", "view": VIEW,
                                     "start_frame": 0, "end_frame": episode["frames"] - 1,
                                     "seconds": episode["duration"]},
                            "distractors": {}, "synthetic": False,
                        },
                    },
                })
                asked += 1
            if asked == 0:
                skipped["time:整组无可用动作"] += 1

    for item in items:
        for key in item["media"]:
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

    clips = [m for m in plan["media"] if m["kind"] == "clip"]
    videos = [m for m in plan["media"] if m["kind"] == "video"]
    reuse = sum(len(m["used_by"]) for m in clips)
    print(f"\n素材需求：{len(clips)} 段切片 + {len(videos)} 个全长视频")
    print(f"  切片被 {reuse} 道题引用 —— 去重省下 {reuse - len(clips)} 次编码"
          f"（理解题与规划题共用同一片段）")

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
    vocab = json.loads((BUILD / "vocab.json").read_text(encoding="utf-8"))["families"]
    pools = {p.stem: json.loads(p.read_text(encoding="utf-8"))
             for p in (DATA / "llm_cache" / "v2").glob("*.json")}

    plan = build(index, vocab, pools, window, time_repeats, cap, none_option)
    opt = plan["options"]
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
