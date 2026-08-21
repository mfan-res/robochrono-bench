#!/usr/bin/env python3
# coding: utf-8
"""⑤ 验题：题目层面有没有【不看视频就能利用的捷径】。**不调任何模型。**

    python3 src/vqa/tests/test_option_shortcuts.py

为什么这一套值得固化成回归
--------------------------
判断「分数低是模型不行还是题目有问题」时，这两类证据的性质完全不同：

    甲  纯数据可证        选项位置、长度、重复、图像统计 …… 换任何模型都不变
    乙  必须有模型才知道   模型答对多少、会不会掉进某个坑

**只有甲类查干净了，乙类的低分才能归给模型。** 而甲类每次重建题目都会变
（换配方、加题型、改干扰项规则），所以它必须是回归，不是一次性脚本。

反过来也要记住：**甲类只能证伪捷径，不能证明题目做得出来**
（`docs/disclosures.md` 的「盲基线只能证伪捷径」同理）。全绿不等于题目没问题，
只等于「不看视频拿不到分」这一条成立。

判据为什么是 4σ 而不是 3σ
-------------------------
这里一次跑十几个独立检验（6 个题型 × 4 个字母，再加其余各项）。
3σ 在单次检验下是 0.3% 的假警率，但做 24 次之后期望假警 0.06 次 —— 还行；
可一旦以后加题型、加族，检验数会继续涨。**取 4σ 留出余量**，
免得这套回归变成「每次跑都红一两条，然后大家开始忽略它」。

实测基线（2026-08-21，10,178 道题）：最大偏离 2.1σ，八项全过。
唯一接近的是「某个动作特别容易当答案」（wash 的 `Put the bowl.` +3.1σ），
**那条只报不判** —— 它反映的是动作在采集数据里出现频次不均，
而且不可利用：永远答它在 wash 上只有 8.4%，远低于随机的 25%。
"""

from __future__ import annotations

import collections
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "data" / "vqa" / "eval"
BUILD = ROOT / "build"

TEXT_RUNS = ("understanding", "planning", "planning_2", "step_order")
IMAGE_RUNS = ("left_right", "image_in_video")
CHOICE_RUNS = TEXT_RUNS + IMAGE_RUNS
SIGMA = 4.0


def families() -> list[str]:
    return sorted(p.name for p in EVAL.iterdir() if p.is_dir())


def load(run: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for family in families():
        path = EVAL / family / f"{run}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("items") if isinstance(data, dict) else data
        out += [(family, it) for it in items]
    return out


def sigma(hits: int, n: int, p: float) -> float:
    """二项检验的 z 值。n=0 时返回 0（该项没有数据，由调用方决定要不要算通过）。"""
    if not n:
        return 0.0
    return (hits / n - p) / math.sqrt(p * (1 - p) / n)


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def add(self, check: str, detail: str, worst: float, ok: bool | None = None) -> None:
        passed = (abs(worst) < SIGMA) if ok is None else ok
        self.rows.append((check, detail, f"{worst:+.1f}σ" if worst else "—", passed))

    def note(self, check: str, detail: str) -> None:
        """只报不判 —— 属实但不可利用的偏斜。"""
        self.rows.append((check, detail, "只报", True))

    def render(self) -> int:
        width = max(len(r[0]) for r in self.rows)
        print(f"{'检查':<{width}}  {'最大偏离':>8}  说明")
        print("-" * (width + 62))
        for check, detail, worst, ok in self.rows:
            mark = "✓" if ok else "✗"
            print(f"{check:<{width}}  {worst:>8}  {mark} {detail}")
        bad = [r for r in self.rows if not r[3]]
        print("-" * (width + 62))
        if bad:
            print(f"❌ {len(bad)} 项发现可利用的捷径 —— **不要把低分归给模型**，先查这里")
            return 1
        print(f"题目层面未发现可利用捷径（判据 {SIGMA:g}σ）。"
              "\n⚠ 这只说明「不看视频拿不到分」，**不说明题目做得出来**。")
        return 0


def check_letter_balance(rep: Report) -> None:
    """正确答案在 A/B/C/D 上是否均匀 —— 不均匀的话猜位置就能超基线。"""
    worst = 0.0
    for run in CHOICE_RUNS:
        items = load(run)
        if not items:
            continue
        counts = collections.Counter(it["answer"] for _, it in items)
        n = len(items)
        for letter in "ABCD":
            worst = max(worst, abs(sigma(counts[letter], n, 0.25)), key=abs)
    rep.add("正确答案的字母分布", f"{len(CHOICE_RUNS)} 个题型 × 4 个字母", worst)


def check_length_shortcut(rep: Report) -> None:
    """「最长的那个是对的」。

    `plan.build_options` 会按词数接近答案来挑干扰项，所以正确项**既不该**是
    唯一最长、**也不该**是唯一最短。实测两头都只有 6–8%（随机 25%）——
    这是防御生效的证据，不只是「没问题」。
    这里只判**高于** 25%（那才是捷径）；低于是好事。
    """
    worst = 0.0
    detail = ""
    for run in TEXT_RUNS:
        n = longest = 0
        for _, it in load(run):
            texts = {o["id"]: o.get("text") or "" for o in it["options"]}
            if not all(texts.values()):
                continue
            words = {k: len(v.split()) for k, v in texts.items()}
            top = max(words.values())
            n += 1
            if words[it["answer"]] == top and list(words.values()).count(top) == 1:
                longest += 1
        if not n:
            continue
        z = sigma(longest, n, 0.25)
        if z > worst:
            worst, detail = z, f"{run} 的答案唯一最长占 {longest / n:.1%}"
    rep.add("答案是不是最长的那个", detail or "各题型答案长度与干扰项接近", worst)


def check_permutation_balance(rep: Report) -> None:
    """step_order 的正确排列偏向某一种的话，总选那一种就能超基线。"""
    items = load("step_order")
    if not items:
        return
    answers = collections.Counter(
        next(o["text"] for o in it["options"] if o["id"] == it["answer"]) for _, it in items)
    n = len(items)
    kinds = len(answers) or 1
    worst = 0.0
    for _, k in answers.items():
        worst = max(worst, abs(sigma(k, n, 1 / kinds)), key=abs)
    rep.add("step_order 正确排列的分布", f"{kinds} 种排列，各约 {1 / kinds:.1%}", worst)


def check_duplicate_options(rep: Report) -> None:
    """同一题里出现重复选项 = 实际选项数变少，随机基线不再是 25%。"""
    bad = total = 0
    for run in TEXT_RUNS:
        for _, it in load(run):
            texts = [o.get("text") for o in it["options"] if o.get("text")]
            total += 1
            if len(set(texts)) < len(texts):
                bad += 1
    rep.add("同一题里有重复选项", f"{total} 道题，重复 {bad} 道", 0.0, ok=(bad == 0))


def check_side_balance(rep: Report) -> None:
    """left_right 问左/问右应当各半 —— 不然总答同一侧就有偏差。"""
    items = load("left_right")
    if not items:
        return
    left = sum(1 for _, it in items if " left " in it.get("question", ""))
    rep.add("left_right 问左/问右均衡",
            f"问左 {left} / 共 {len(items)}", sigma(left, len(items), 0.5))


def check_borrowed_clash(rep: Report) -> None:
    """借自别族的干扰项**撞上本族真实动作**就可能碰巧是对的（tea 与 tea2 都有 Pour the tea.）。

    判据走 `build/plan.json`，因为只有那里能分清「本族动作」与「借来的」。
    """
    plan_path = BUILD / "plan.json"
    vocab_path = BUILD / "vocab.json"
    if not (plan_path.exists() and vocab_path.exists()):
        rep.add("借来的干扰项撞车", "build/plan.json 或 vocab.json 缺失，跳过", 0.0, ok=True)
        return
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))["families"]
    own = {f: {s["text"] for s in v["subtasks"]} for f, v in vocab.items()}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    borrowed = clash = total = 0
    for it in plan["items"]:
        if it["task"] not in ("understanding", "planning", "planning_2"):
            continue
        mine = own.get(it["family"], set())
        for text in it["distractors"]:
            total += 1
            if text not in mine:
                borrowed += 1
    # 借来的按定义不在本族词表里；这里再正面查一次，防止 build_options 的过滤失效
    for it in plan["items"]:
        if it["task"] not in ("understanding", "planning", "planning_2"):
            continue
        mine = own.get(it["family"], set())
        clash += sum(1 for t in it["distractors"] if t in mine and t == it["answer_text"])
    rep.add("借来的干扰项撞上本族动作",
            f"干扰项 {total} 条，借用 {borrowed} 条（{borrowed / total:.1%}），撞车 {clash} 条",
            0.0, ok=(clash == 0))


def check_image_size(rep: Report) -> None:
    """正确图能不能靠**文件大小**这种粗暴的图像统计认出来。

    `assets.py` 的设计前提是所有帧走同一条 ffmpeg 命令、同一套参数 ——
    「若两者用不同参数抽，图像统计本身就成了线索」。这一项就是在验那个前提。
    """
    worst = 0.0
    detail = ""
    for run in IMAGE_RUNS:
        n = big = 0
        for family, it in load(run):
            sizes: dict[str, int] = {}
            for o in it["options"]:
                rel = o.get("image_path")
                if not rel:
                    continue
                path = (EVAL / family / rel).resolve()
                try:
                    sizes[o["id"]] = path.stat().st_size
                except OSError:
                    pass
            if len(sizes) < len(it["options"]):
                continue
            n += 1
            top = max(sizes.values())
            if sizes.get(it["answer"]) == top and list(sizes.values()).count(top) == 1:
                big += 1
        if not n:
            continue
        z = sigma(big, n, 0.25)
        if abs(z) > abs(worst):
            worst, detail = z, f"{run} 的答案文件最大占 {big / n:.1%}（抽样 {n}）"
    rep.add("正确图能否靠文件大小认出", detail or "未找到图选项素材", worst)


def note_answer_frequency(rep: Report) -> None:
    """某个动作特别容易当答案 —— **只报不判**。

    它反映的是动作在采集数据里出现频次不均，不是设计缺陷，而且不可利用：
    永远答那一个，在该族也只有个位数的正确率，远低于随机的 25%。
    """
    worst = (0.0, "", "", 0, 0, 0)
    for run in ("understanding", "planning"):
        per_family: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for family, it in load(run):
            text = next((o["text"] for o in it["options"] if o["id"] == it["answer"]), None)
            if text:
                per_family[family][text] += 1
        for family, counts in per_family.items():
            n = sum(counts.values())
            kinds = len(counts) or 1
            text, hits = counts.most_common(1)[0]
            z = sigma(hits, n, 1 / kinds)
            if z > worst[0]:
                worst = (z, run, f"{family} 的「{text}」", hits, n, kinds)
    if worst[1]:
        z, run, what, hits, n, kinds = worst
        rep.note("某个动作特别容易当答案",
                 f"最偏 {run}／{what} {hits}/{n} = {hits / n:.1%}"
                 f"（该族 {kinds} 个动作，均匀应为 {1 / kinds:.1%}，{z:+.1f}σ）"
                 f" —— 不可利用，永远答它也只有 {hits / n:.1%}")


def main() -> int:
    if not EVAL.exists():
        print(f"跳过：没有 {EVAL}。先跑 python3 src/vqa/pack.py --write")
        return 0
    rep = Report()
    check_letter_balance(rep)
    check_length_shortcut(rep)
    check_permutation_balance(rep)
    check_duplicate_options(rep)
    check_side_balance(rep)
    check_borrowed_clash(rep)
    check_image_size(rep)
    note_answer_frequency(rep)
    return rep.render()


if __name__ == "__main__":
    raise SystemExit(main())
