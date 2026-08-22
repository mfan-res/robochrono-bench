#!/usr/bin/env python3
# coding: utf-8
"""⑤ 验题第一步：盲基线 —— 不给视频，只给题干和选项，看模型能答对多少。

    python3 src/vqa/blind.py --n 20                       # 测题库现状（默认）
    python3 src/vqa/blind.py --n 20 --policy cross,four   # 加测两个候选策略
    python3 src/vqa/blind.py --n 20 --policy all          # 全部 6 个策略（6 倍请求）

**这是判断「题目泄没泄」的唯一硬判据。** 超过 `1 / 选项数` 多少，题就泄了多少。

为什么必须实测而不能推理
------------------------
之前用代数方法分析过一轮，结论是「96% 的 planning 题可以零视频解出」。
**那个数字是上界，不是预期** —— 它假设攻击者能统计整个题库、
反推出每个族的动作表与转移表。被评测的 VLM 一题一请求、零样本，拿不到这些。

但有一部分照样成立：**动作顺序不是秘密，是常识。**
「洗碗：拿刷子 → 拿碗 → 擦碗 → 放碗」任何语言模型都知道，
而 planning_2 的题干直接把任务名告诉了它。这条通路零样本就能用。

代数分不清这两者，实测能。

协议
----
- **一题一请求。** 绝不批量 —— 批量会让模型在一个请求里看到多道题，
  从而自行统计出动作表，那正好是我们想区分开的东西。
- 选项顺序按 `md5(item_id)` 确定性打乱，排除位置线索。
- `temperature=0`，只要一个字母。
- 明确告诉模型视频不可用、必须猜 —— 否则它会拒答，测不到东西。

不测 time
---------
time 是开放作答。判据现在有了 —— `docs/disclosures.md` §10 定的 `tIoU@0.5`，
但**盲基线对它没有意义**：不给视频就报一个时间区间，报什么都是瞎猜，
而「整段视频都报」这个退化解已经能拿 mean_tIoU 0.13（见 `tasks/__init__.py`
的 `DEGENERATE_FLOOR`）。要测的是那条退化下限，不是模型的先验。
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distract import KEYS, MODEL, call_api  # noqa: E402
from vocab import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
OUT = BUILD / "blind.json"

LETTERS = "ABCDEF"
TASKS = ("understanding", "planning", "planning_2")


def digest(text: str, n: int) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % max(1, n)


def options_as_built(item: dict[str, Any], _actions: list[str]) -> list[str]:
    """题库里实际存的那一套 —— 测的就是真正会发给模型的题。

    选项构造的实现在 `tasks/_base.py:build_options`，这一条只读它的产物 ——
    **只有这一条不会分叉。**

    ⚠ 本文件其余策略（`options_cross` / `options_four` / `options_pool`）是
    本地重写，**已经分叉**，别照着注释以为它们等价于出题时的逻辑：

      · 轮转盐：`_base` 用 `md5(f"{item_id}|in")` / `|out`，这里用
        `md5(id)` / `md5(id + "|x")`
      · 可借池：`_base` 是去重排序后的集合，这里是含重复、未排序的列表

    它们本来就是**候选方案**的模拟器，不是现状的复刻 —— 分叉是设计意图，
    但当初的注释把它写成了「不可能分叉」。收敛计划见 cleanup_checklist §4.5。
    """
    return [item["answer_text"], *item["distractors"]]


def options_pool(item: dict[str, Any], actions: list[str],
                 pool: list[dict[str, Any]]) -> list[str]:
    """策略④：统一 5 选项，干扰项只用【场景内词汇】。

    取用顺序：本族其它真实动作 → 重组池（按层，obj 最自然、verb-global 最后）。
    两者都优先取词数与答案接近的，避免「最长的那个是对的」。

    **每个族都是 5 选项**，基线统一 20%，跨族分数可比 ——
    这是策略③（小族减选项）被否掉的原因。
    """
    answer = item["answer_text"]
    n = len(normalize(answer).split())
    real = [a for a in actions if a != answer]
    k = digest(item["id"], max(1, len(real)))
    real = real[k:] + real[:k]                       # 轮转，不同题拿到不同子集
    cands = real + [c["text"] for c in pool if c["text"] != answer]
    cands.sort(key=lambda t: abs(len(normalize(t).split()) - n))   # 稳定排序，只按词数
    return [answer, *cands[:4]]


def options_all_real(item: dict[str, Any], actions: list[str]) -> list[str]:
    """策略③：干扰项全部取自本族真实动作，哈希采样，只排除答案。

    小族填不满 5 个就少给几个 —— **不用编的补齐**。
    补齐会把真实基线抬到 33% 却继续写 20%。
    """
    pool = sorted(a for a in actions if a != item["answer_text"])
    k = digest(item["id"], len(pool))
    return [item["answer_text"], *(pool[k:] + pool[:k])[:4]]


def options_three(item: dict[str, Any], actions: list[str]) -> list[str]:
    """策略⑤：全部用本族真实动作，**所有族一律 3 选项**。

    基线 33%，七个族一致，跨族可比。代价是每道题的区分力下降 ——
    但题量有 5,266 道，统计功效不是瓶颈。
    """
    pool = sorted(a for a in actions if a != item["answer_text"])
    k = digest(item["id"], max(1, len(pool)))
    return [item["answer_text"], *(pool[k:] + pool[:k])[:2]]


def options_four(item: dict[str, Any], actions: list[str],
                 others: list[tuple[str, str]], objects: set[str]) -> list[str]:
    """统一 4 选项：只有两个 3 动作的族需要借，各借 1 条 —— 借得最少的可行方案。"""
    return options_cross(item, actions, others, objects)[:4]


def options_cross(item: dict[str, Any], actions: list[str],
                  others: list[tuple[str, str]], objects: set[str]) -> list[str]:
    """策略⑥：本族真实动作优先，不够就**借别族的真实动作**，一律 5 选项。

    借来的动作在别的族里确实当过答案，所以「这是不是真标签」不再有区分度 ——
    这是唯一能同时满足「统一 5 选项」与「全部选项都是真实标签」的做法。

    **借的动作必须提到本场景没有的物体**，否则可能碰巧也是真的
    （tea 与 tea2 都有 `Pour the tea.`）。
    """
    answer = item["answer_text"]
    n = len(normalize(answer).split())
    inside = [a for a in actions if a != answer]
    k = digest(item["id"], max(1, len(inside)))
    inside = inside[k:] + inside[:k]
    # 借：物体不在本场景里，且文字不与本族任何动作重复
    outside = [t for obj, t in others if obj not in objects and t not in actions]
    j = digest(item["id"] + "|x", max(1, len(outside)))
    outside = outside[j:] + outside[:j]
    outside.sort(key=lambda t: abs(len(normalize(t).split()) - n))
    return [answer, *(inside + outside)[:4]]


POLICIES = {"as_built": options_as_built, "allreal": options_all_real,
            "pool": options_pool, "three": options_three, "cross": options_cross, "four": options_four}

# 让模型回答**选项原文**而不是字母。
# 实测这个模型有极强的位置偏好（四选一时 69% 选 D，几乎从不选 A）——
# 按字母作答时它大部分时候在挑位置而不是读内容，会**低估**泄漏。
TEXT_PROMPT = """You are answering a multiple-choice question taken from a
robot-manipulation video benchmark. **The video is not available to you.**
Answer anyway with your single best guess — do not explain, do not refuse.

{stem}

{options}

Reply with the exact text of the option you choose, and nothing else."""

PROMPT = """You are answering a multiple-choice question taken from a robot-manipulation
video benchmark. **The video is not available to you.** Answer anyway with your
single best guess — do not explain, do not refuse.

{stem}

{options}

Reply with one letter only."""


def ask(job: dict[str, Any], key: str, by_text: bool = False) -> dict[str, Any]:
    body = "\n".join((f"- {t}" if by_text else f"{LETTERS[i]}. {t}")
                      for i, t in enumerate(job["shuffled"]))
    tmpl = TEXT_PROMPT if by_text else PROMPT
    try:
        raw = call_api(tmpl.format(stem=job["stem"], options=body), key,
                       timeout=90, json_mode=False, temperature=0.0)
    except Exception as exc:                                   # noqa: BLE001
        # **一次调用失败只能毁掉一个数据点。** 曾经因为漏接 `IncompleteRead`，
        # 一个偶发的响应截断穿透线程池，把已跑完的几百次结果全带走。
        # 失败计入报告（「N 次没解析出字母」），不静默丢弃。
        return {**job, "picked": None, "error": f"{type(exc).__name__}: {exc}"[:100]}
    if by_text:
        norm = raw.strip().strip('"\'.').lower()
        hit = next((i for i, t in enumerate(job["shuffled"])
                    if t.strip(".").lower() == norm.strip(".")), None)
        if hit is None:      # 宽松兜底：包含关系
            hit = next((i for i, t in enumerate(job["shuffled"])
                        if t.strip(".").lower() in norm), None)
        return {**job, "picked": LETTERS[hit] if hit is not None else None,
                "raw": raw.strip()[:60]}
    got = re.search(r"\b([A-F])\b", raw.strip().upper())
    return {**job, "picked": got.group(1) if got else None, "raw": raw.strip()[:40]}


def main() -> int:
    def arg(name: str, default: str) -> str:
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    per_cell = int(arg("--n", "20"))
    # **默认只测 as_built，也就是题库里真正会发给模型的那一套。**
    # 曾经默认 `both`，而 `both` 走的是 `list(POLICIES)` —— 6 个策略全跑，
    # 裸跑一次 `--n 20` 就是 6 倍 API 请求，且其中 5 套是没在用的候选方案。
    which = arg("--policy", "as_built")
    # 限定题型：加测某一个题型时不必把别的也重跑一遍
    only = set(arg("--tasks", ",".join(TASKS)).split(","))
    by_text = "--by-text" in sys.argv
    policies = list(POLICIES) if which == "all" else which.split(",")
    unknown = [p for p in policies if p not in POLICIES]
    if unknown:
        print(f"❌ 未知策略 {unknown}，可选：{', '.join(POLICIES)} 或 all")
        return 1

    plan = json.loads((BUILD / "plan.json").read_text(encoding="utf-8"))
    vocab = json.loads((BUILD / "vocab.json").read_text(encoding="utf-8"))["families"]
    actions = {f: [s["text"] for s in v["subtasks"]] for f, v in vocab.items()}
    # v3 只有 `pool` 策略用得上，而 `pool` 已不是在用的方案。无条件加载的话，
    # 哪天 `llm_cache/v3/` 归档掉，连测 as_built 都会启动即崩。
    pools = {f: json.loads((ROOT / "data" / "llm_cache" / "v3" / f"{f}.json")
                           .read_text(encoding="utf-8"))["pool"] for f in vocab} \
        if "pool" in policies else {}
    objects = {f: {s["object"] for s in v["subtasks"] if s["object"]}
               for f, v in vocab.items()}
    every = [(s["object"], s["text"]) for v in vocab.values() for s in v["subtasks"]]

    key = next((line.split("=", 1)[1].strip().strip("\"'")
                for line in KEYS.read_text(encoding="utf-8").splitlines()
                if line.startswith("DEEPSEEK_API_KEY=")), None)
    if not key:
        print(f"❌ {KEYS} 里没有 DEEPSEEK_API_KEY")
        return 1

    # 均匀抽样：按 id 哈希取前 N，确定性且与题目内容无关
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in plan["items"]:
        if item["task"] in TASKS and item["task"] in only:
            cells[(item["family"], item["task"])].append(item)
    sample = []
    for (family, task), pool in sorted(cells.items()):
        pool.sort(key=lambda it: hashlib.md5(it["id"].encode()).hexdigest())
        sample += pool[:per_cell]

    jobs = []
    for item in sample:
        for policy in policies:
            fam = item["family"]
            if policy == "pool":
                opts = options_pool(item, actions[fam], pools[fam])
            elif policy in ("cross", "four"):
                opts = POLICIES[policy](item, actions[fam], every, objects[fam])
            else:
                opts = POLICIES[policy](item, actions[fam])
            order = sorted(opts, key=lambda t: hashlib.md5(
                f"{item['id']}|{policy}|{t}".encode()).hexdigest())
            jobs.append({
                "id": item["id"], "family": item["family"], "task": item["task"],
                "policy": policy, "stem": item["stem"], "shuffled": order,
                "n_options": len(order),
                "answer_letter": LETTERS[order.index(item["answer_text"])],
            })

    probe = ask(jobs[0], key, by_text)
    if not probe["picked"]:
        print(f"❌ 自检失败，先修这个再跑全量：{probe.get('error') or probe.get('raw')}")
        return 1
    print(f"自检通过（首题作答 {probe['picked']}）。", end=" ")

    print(f"盲测 {len(jobs)} 次调用（{len(sample)} 道题 × {len(policies)} 套选项）"
          f"，模型 {MODEL}，一题一请求\n")
    done: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool_:
        for n, result in enumerate(pool_.map(lambda j: ask(j, key, by_text), jobs), 1):
            done.append(result)
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}")

    report(done, policies)
    out = BUILD / arg("--out", "blind.json")
    out.write_text(json.dumps(done, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n明细写入 {out.relative_to(ROOT)}")
    return 0


def report(done: list[dict], policies: list[str]) -> None:
    def rate(rows: list[dict]) -> tuple[float, float, int]:
        ok = [r for r in rows if r["picked"]]
        if not ok:
            return 0.0, 0.0, 0
        hit = sum(r["picked"] == r["answer_letter"] for r in ok)
        base = sum(1 / r["n_options"] for r in ok) / len(ok)
        return hit / len(ok), base, len(ok)

    for policy in policies:
        rows = [r for r in done if r["policy"] == policy]
        label = {"as_built": "题库现状（plan.json 实际存的）",
                 "allreal": "策略③（全真实动作，小族减选项）",
                 "pool": "策略④（统一 5 选项，只用场景内词汇）",
                 "three": "策略⑤（全真实动作，一律 3 选项）",
                 "cross": "策略⑥（全真实动作，不够借别族，一律 5 选项）",
                 "four": "统一 4 选项（借得最少）"}[policy]
        print(f"\n【{label}】")
        print(f"  {'族':<13}" + "".join(f"{t:>16}" for t in TASKS))
        for family in sorted({r["family"] for r in rows}):
            line = f"  {family:<13}"
            for task in TASKS:
                acc, base, n = rate([r for r in rows
                                     if r["family"] == family and r["task"] == task])
                line += f"{f'{acc:.0%} / {base:.0%}':>16}"
            print(line)
        print(f"  {'—— 合计':<13}", end="")
        for task in TASKS:
            acc, base, n = rate([r for r in rows if r["task"] == task])
            line = f"{acc:.0%} / {base:.0%}"
            print(f"{line:>16}", end="")
        acc, base, n = rate(rows)
        print(f"\n  全部 {acc:.1%}，随机基线 {base:.1%}，"
              f"高出 {acc - base:+.1%}（{n} 次有效作答）")

    bad = [r for r in done if not r["picked"]]
    if bad:
        print(f"\n⚠ {len(bad)} 次没解析出字母（已排除在统计外）")
    pos = Counter(r["picked"] for r in done if r["picked"])
    print(f"\n作答字母分布（查位置偏好）：{dict(sorted(pos.items()))}")


if __name__ == "__main__":
    raise SystemExit(main())
