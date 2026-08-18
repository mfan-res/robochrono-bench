#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第二步：把 subtask 展开成出题需要的各种形态。

    python3 src/vqa/vocab.py            # 打印词表并核对
    python3 src/vqa/vocab.py --write    # 同时写 build/vocab.json

段里只存 subtask 的 ID（D-04）。出题要用到的其它形态 —— 展示给模型的文字、
查 LLM 缓存的键、生成规则干扰项要的动词与宾语 —— **全部在这里从
`subtasks.json` 一处派生**，不再各处手写。

派生而非手写，是因为「同一个动作有 11 种表示」这件事本身就是问题的来源：
表示越多越难保持一致，而不一致时不报错。这里只有一个真源，其余是函数。

三个必须核对的点
----------------

**1 · 缓存键要归一化后再查。** `llm_cache` 的键是当年生成时的文字，
与今天的 `subtasks.json` 不完全一致，两种不一致都实测到了：

    gift_inhand / pen_inbox / tea   键无冠词    "move gift"      vs  "move the gift"
    tea2                            键是坏文字   "pick the up kettle"

第二种是标注侧 `build_segment_description` 在动词后机械插 "the" 的化石 ——
**当年就是拿这个坏文字去问的 LLM。** 我们把文字修对了，键反而对不上。
去冠词归一化同时消掉这两类：命中率 27/43 → 42/43。

**2 · 动词/宾语切分错了不会报错**，只会让规则干扰项变得莫名其妙。
所以这一步把切分结果整张表打出来给人看，不要只看通过与否。

**3 · 前缀包含关系要显式列出。** `pick_up_teapot` 是 `pick_up_teapot_lid`
的前缀，任何地方用 `startswith` / 子串匹配都会撞。这里把有包含关系的对
列成 `prefix_hazards`，下游断言用的是精确相等。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BUILD = ROOT / "build"

VOCAB_VERSION = "1"

ARTICLES = re.compile(r"\b(?:the|a|an)\b")
# 动词小品词：`pick up` 是一个动词，`up` 不是宾语的一部分
PARTICLES = {"up", "down", "off", "out", "in", "on", "over", "back", "away"}
# 介词短语：`wipe the bowl with brush` 的宾语是 bowl，`with brush` 是修饰
PREPOSITIONS = {"with", "in", "on", "into", "onto", "to", "from", "at"}


def normalize(text: str) -> str:
    """归一化键：小写、去句号、去冠词、压空格。

    去冠词是必需的而不是保守化选择 —— 见模块 docstring 第 1 点。
    """
    return " ".join(ARTICLES.sub("", text.lower().rstrip(".")).split())


def parse(text: str) -> dict[str, Any]:
    """切成 动词 / 宾语 / 修饰。

    结果只用于**生成规则干扰项**（换动词、换宾语），不进入任何真值。
    切错了题目会变怪，但不会答案错 —— 即便如此也要人看一眼，见 docstring 第 2 点。
    """
    words = normalize(text).split()
    if not words:
        return {"verb": "", "object": "", "modifier": ""}

    verb = [words[0]]
    rest = words[1:]
    if rest and rest[0] in PARTICLES:
        verb.append(rest.pop(0))

    modifier: list[str] = []
    for i, w in enumerate(rest):
        if w in PREPOSITIONS:
            modifier = rest[i:]
            rest = rest[:i]
            break

    return {"verb": " ".join(verb), "object": " ".join(rest), "modifier": " ".join(modifier)}


def cache_keys(family: str) -> dict[str, set[str]]:
    """v1-vendor 各缓存文件里有哪些键（已归一化）。

    **只反映 v1 的历史状况**，用来记录那次漂移有多严重。v2 用 subtask ID 当键，
    不会有查不到的情况（由 `distract.py` 保证每个 ID 都有条目）。
    """
    out: dict[str, set[str]] = {}
    for path in sorted((DATA / "llm_cache" / "v1-vendor" / family).glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = {normalize(k) for k in blob.get("task_category_distractors", {})}
    return out


def build_family(family: str) -> dict[str, Any]:
    subtasks = json.loads(
        (DATA / "label" / family / "subtasks.json").read_text(encoding="utf-8"))["subtasks"]
    caches = cache_keys(family)

    entries = []
    for sub in subtasks:
        key = normalize(sub["text"])
        entries.append({
            "id": sub["id"],
            "text": sub["text"],                    # 展示给模型的（唯一真源）
            "key": key,                             # 查 llm_cache 用的
            **parse(sub["text"]),
            "cache_hit_v1": {name: key in keys for name, keys in caches.items()},
        })

    # 归一化后撞车 = 两个不同动作查同一条缓存。必须为空。
    seen: dict[str, str] = {}
    collisions = []
    for e in entries:
        if e["key"] in seen:
            collisions.append([seen[e["key"]], e["id"], e["key"]])
        seen[e["key"]] = e["id"]

    # 前缀包含：任何 startswith / 子串匹配都会在这些对上撞
    ids = [e["id"] for e in entries]
    hazards = [[a, b] for a in ids for b in ids
               if a != b and (b.startswith(a) or a in b)]

    # 族内近邻：规则干扰项的来源
    by_verb: dict[str, list[str]] = {}
    by_object: dict[str, list[str]] = {}
    for e in entries:
        by_verb.setdefault(e["verb"], []).append(e["id"])
        by_object.setdefault(e["object"], []).append(e["id"])

    return {
        "family": family,
        "subtasks": entries,
        "key_collisions": collisions,
        "prefix_hazards": hazards,
        "same_verb": {k: v for k, v in by_verb.items() if len(v) > 1},
        "same_object": {k: v for k, v in by_object.items() if len(v) > 1},
    }


def main() -> int:
    write = "--write" in sys.argv
    registry = json.loads((DATA / "families.json").read_text(encoding="utf-8"))["families"]
    active = sorted(f for f, v in registry.items()
                    if v.get("status") not in ("excluded", "parked") and (DATA / "source" / f).is_dir())

    vocab = {"vocab_version": VOCAB_VERSION, "families": {}}
    misses: list[tuple[str, str, str]] = []
    hazards: list[tuple[str, list[str]]] = []
    collisions: list[tuple[str, list[str]]] = []

    for family in active:
        entry = build_family(family)
        vocab["families"][family] = entry
        print(f"\n【{family}】")
        print(f"  {'id':<28}{'动词':<10}{'宾语':<18}{'修饰':<14}v1缓存")
        for e in entry["subtasks"]:
            hit = "".join("✓" if v else "✗" for v in e["cache_hit_v1"].values()) or "—"
            print(f"  {e['id']:<28}{e['verb']:<10}{e['object']:<18}{e['modifier']:<14}{hit}")
            misses += [(family, e["id"], name)
                       for name, ok in e["cache_hit_v1"].items() if not ok]
        collisions += [(family, c) for c in entry["key_collisions"]]
        hazards += [(family, h) for h in entry["prefix_hazards"]]

    total = sum(len(v["subtasks"]) for v in vocab["families"].values())
    print(f"\n{'=' * 62}\n{total} 个 subtask，{len(active)} 个族")

    if collisions:
        print(f"\n❌ 归一化后键碰撞 {len(collisions)} 处 —— 两个动作会查到同一条缓存：")
        for fam, c in collisions:
            print(f"   {fam}: {c[0]} 与 {c[1]} 都归一化成「{c[2]}」")
    else:
        print("✅ 归一化后无键碰撞")

    if hazards:
        print(f"\n⚠ 前缀包含 {len(hazards)} 对 —— **任何地方都必须精确相等匹配**：")
        for fam, (a, b) in hazards:
            print(f"   {fam}: 「{a}」是「{b}」的前缀/子串")

    if misses:
        by_fam: dict[str, int] = {}
        for fam, _, _ in misses:
            by_fam[fam] = by_fam.get(fam, 0) + 1
        print(f"\nv1 缓存命中情况（**纯历史记录，出题已不读缓存**）：查不到 {len(misses)} 处 "
              + "、".join(f"{f} {n}" for f, n in sorted(by_fam.items())))
        print("   干扰项自 D-38 起改为从真实标签里挑，`llm_cache` 三代全部退场。"
              "词表变动后这个数只会越来越大，属正常。")

    if write:
        BUILD.mkdir(exist_ok=True)
        out = BUILD / "vocab.json"
        out.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写入 {out.relative_to(ROOT)}")
    else:
        print("\n加 --write 写入 build/vocab.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
