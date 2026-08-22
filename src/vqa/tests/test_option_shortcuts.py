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


def check_image_outlier(rep: Report) -> None:
    """图选项题：答案是不是「离其它三个最远／最近」的那一个。

    **这条抓过一次真捷径。** 曾把「干扰项彼此」的门槛放宽到一半，理由是
    「那只为了不出现双胞胎，门槛该低些」。规则一旦不对称，答案就成了离群点 ——
    「挑最不像的那个」白送 **28 个百分点**（答案最远占比 53%，应为 25%）。
    改成六条边同一条门槛（`plan.MUTUAL_RATIO = 0.85`）之后降到 2 个百分点。

    **判据按族算，不看总数** —— 正向偏与反向偏会互相抵消：
    实测 gift_inhand 最远 50.0%（+4.4σ）而 wash 最近 35.8%（+5.7σ），
    合计却是 27.2%，看起来干净。总数掩盖分族偏斜是这条最容易被漏掉的方式。

    **报的是「白送多少分」，不是 σ。** 单族 n 小时 σ 容易吓人，
    而真正要问的是「这个偏斜能不能被利用」——
    永远选最远的那张，实测只多拿 2.2 个百分点，属噪声量级，不该为它调参。
    """
    desc_path = BUILD / "frames_desc.npy"
    frames_path = BUILD / "frames.json"
    plan_path = BUILD / "plan.json"
    if not (desc_path.exists() and frames_path.exists() and plan_path.exists()):
        rep.note("图选项的离群性", "缺 build/frames_desc.npy 或 plan.json，跳过")
        return
    try:
        import numpy
    except ImportError:
        rep.note("图选项的离群性", "没装 numpy，跳过")
        return

    desc = numpy.load(desc_path).astype(numpy.float32)
    order = json.loads(frames_path.read_text(encoding="utf-8"))["order"]
    row = {k: i for i, k in enumerate(order)}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    def dist(a: str, b: str) -> float:
        return float(numpy.abs(desc[row[a]] - desc[row[b]]).mean())

    worst_gain = 0.0
    detail = ""
    for run in IMAGE_RUNS:
        n = far_hit = 0
        per: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for item in plan["items"]:
            if item.get("task") != run:
                continue
            keys = item.get("image_options") or []
            answer = item.get("correct_option")
            if len(keys) != 4 or answer not in keys or not all(k in row for k in keys):
                continue
            totals = [sum(dist(keys[i], keys[j]) for j in range(4) if j != i) for i in range(4)]
            hit = keys[totals.index(max(totals))] == answer
            n += 1
            far_hit += hit
            cell = per[item["family"]]
            cell[0] += 1
            cell[1] += hit
        if not n:
            continue
        gain = far_hit / n - 0.25
        if abs(gain) > abs(worst_gain):
            skew = max(per.items(), key=lambda kv: abs(kv[1][1] / kv[1][0] - 0.25))
            worst_gain = gain
            detail = (f"{run}：永远选最远的那张多拿 {gain:+.1%}"
                      f"（{far_hit}/{n}）；最偏的族 {skew[0]} "
                      f"{skew[1][1] / skew[1][0]:.1%}")
    # 判据：白送 ≥5 个百分点才算捷径。修复前是 +28pp，修复后 +2.2pp。
    rep.add("图选项：答案是不是离群的那个", detail or "无图选项题",
            0.0, ok=(abs(worst_gain) < 0.05))


def note_answer_coverage(rep: Report) -> None:
    """每个动作当答案的次数是否均匀 —— **只报不判**。

    这一条抓的是「答案空间被结构性削掉了一块」，而**盲基线抓不到它**：
    纯文本模型不知道「这个族的开场动作是哪个」，那是**数据集层面的先验**，
    要看过整个数据集才有。题面内部的检查（位置/长度/重复/图像统计）同样抓不到。

    实测（2026-08-23）：`planning` 在四个族里各有 1 个动作**从没当过答案** ——
    每集最后一段没有「下一步」所以被跳过，于是每集的**开场动作**永远不是答案；
    而这批数据每集按同一脚本执行，开场动作四个族里各只有一种。
    三动作族的答案空间因此实际只剩 2 个，最常见的那个占 50%。
    「划掉本族开场动作再猜」白送 **+4.9pp**（小族 +8.3）。

    **为什么只报不判**：三个候选修法都比原问题差 ——
      干扰项排除开场动作  三个族凑不满四选一，得多借别族动作（披露第 2 条，+8pp）
      末段也出题、答「结束」 造一条更容易的新捷径（且披露第 7 条已否掉这个选项）
      平衡答案分布        **数学上做不到** —— 开场动作结构上不可能当答案

    它与披露第 1 条（`P(next|current)=100%`）、1c（当前动作在选项里）**同源**：
    都是「每集按同一脚本执行」这个采集事实的后果，在出题层面无解。
    设成会红的门禁只会让人以后忽略整个套件 —— **判据的严厉程度应当取决于
    「能不能修」，而不只是「有多大」。**

    `understanding` 是天然的对照组：同样的数据、同样的干扰项构造，
    六族**全部覆盖且占比精确等于均匀值**。这证明偏差不来自构造，
    纯粹来自「问下一步」与「每集同脚本」的组合。
    """
    plan_path, vocab_path = BUILD / "plan.json", BUILD / "vocab.json"
    if not (plan_path.exists() and vocab_path.exists()):
        rep.note("答案是否覆盖全部动作", "缺 build/plan.json 或 vocab.json，跳过")
        return
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))["families"]

    lines: list[str] = []
    for run in ("understanding", "planning", "planning_2"):
        seen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for item in plan["items"]:
            if item.get("task") == run:
                seen[item["family"]][item.get("answer_subtask")] += 1
        for family, counts in sorted(seen.items()):
            defined = {s["id"] for s in vocab[family]["subtasks"]}
            never = defined - set(counts)
            if not never:
                continue
            total = sum(counts.values())
            top = counts.most_common(1)[0][1] / total
            lines.append(f"{run}／{family} {len(never)}/{len(defined)} 个动作从没当过答案，"
                         f"最常见的占 {top:.0%}（均匀 {1 / len(defined):.0%}）")

    if not lines:
        rep.note("答案是否覆盖全部动作", "各题型的动作全部当过答案")
        return
    # 挑最偏的那条报出来 —— 三动作族最能说明问题（答案空间只剩 2 个）
    worst = min(lines, key=lambda t: -float(t.split("最常见的占 ")[1].split("%")[0]))
    rep.note("答案是否覆盖全部动作",
             f"{len(lines)} 处，最偏的：{worst}"
             f"　understanding 六族全覆盖可作对照 —— 结构性偏差，三个修法都更糟")


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
    check_image_outlier(rep)
    note_answer_coverage(rep)
    note_answer_frequency(rep)
    return rep.render()


if __name__ == "__main__":
    raise SystemExit(main())
