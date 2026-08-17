#!/usr/bin/env python3
# coding: utf-8
"""⑤ 验题第一步：盲基线 —— 不给视频，只给题干和选项，看模型能答对多少。

    python3 src/vqa/blind.py --n 20                 # 每族每题型抽 20 道
    python3 src/vqa/blind.py --n 20 --policy both   # 现状与策略③ 两套对照

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
time 是开放作答，而「多少算答对」的容差目前没有定义（属 ⑥）。
判据没定死之前测出来的数不可比，**所以这里显式不测，而不是随便设一个容差**。
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


def options_now(item: dict[str, Any], _actions: list[str]) -> list[str]:
    """现状：答案 + 2 近邻 + 2 LLM。"""
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


POLICIES = {"now": options_now, "allreal": options_all_real, "pool": options_pool}

PROMPT = """You are answering a multiple-choice question taken from a robot-manipulation
video benchmark. **The video is not available to you.** Answer anyway with your
single best guess — do not explain, do not refuse.

{stem}

{options}

Reply with one letter only."""


def ask(job: dict[str, Any], key: str) -> dict[str, Any]:
    body = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(job["shuffled"]))
    try:
        raw = call_api(PROMPT.format(stem=job["stem"], options=body), key,
                       timeout=90, json_mode=False, temperature=0.0)
    except RuntimeError as exc:
        return {**job, "picked": None, "error": str(exc)[:80]}
    got = re.search(r"\b([A-F])\b", raw.strip().upper())
    return {**job, "picked": got.group(1) if got else None, "raw": raw.strip()[:40]}


def main() -> int:
    def arg(name: str, default: str) -> str:
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    per_cell = int(arg("--n", "20"))
    which = arg("--policy", "both")
    # 限定题型：加测某一个题型时不必把别的也重跑一遍
    only = set(arg("--tasks", ",".join(TASKS)).split(","))
    policies = list(POLICIES) if which == "both" else [which]

    plan = json.loads((BUILD / "plan.json").read_text(encoding="utf-8"))
    vocab = json.loads((BUILD / "vocab.json").read_text(encoding="utf-8"))["families"]
    actions = {f: [s["text"] for s in v["subtasks"]] for f, v in vocab.items()}
    pools = {f: json.loads((ROOT / "data" / "llm_cache" / "v3" / f"{f}.json")
                           .read_text(encoding="utf-8"))["pool"] for f in vocab}

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
            opts = (options_pool(item, actions[fam], pools[fam]) if policy == "pool"
                    else POLICIES[policy](item, actions[fam]))
            order = sorted(opts, key=lambda t: hashlib.md5(
                f"{item['id']}|{policy}|{t}".encode()).hexdigest())
            jobs.append({
                "id": item["id"], "family": item["family"], "task": item["task"],
                "policy": policy, "stem": item["stem"], "shuffled": order,
                "n_options": len(order),
                "answer_letter": LETTERS[order.index(item["answer_text"])],
            })

    probe = ask(jobs[0], key)
    if not probe["picked"]:
        print(f"❌ 自检失败，先修这个再跑全量：{probe.get('error') or probe.get('raw')}")
        return 1
    print(f"自检通过（首题作答 {probe['picked']}）。", end=" ")

    print(f"盲测 {len(jobs)} 次调用（{len(sample)} 道题 × {len(policies)} 套选项）"
          f"，模型 {MODEL}，一题一请求\n")
    done: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool_:
        for n, result in enumerate(pool_.map(lambda j: ask(j, key), jobs), 1):
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
        label = {"now": "现状（2 近邻 + 2 LLM）",
                 "allreal": "策略③（全真实动作，小族减选项）",
                 "pool": "策略④（统一 5 选项，只用场景内词汇）"}[policy]
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
