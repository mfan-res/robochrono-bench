#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第三步（改版）：造一批「只用场景内词汇」的干扰项。

    python3 src/vqa/pool.py --dry-run     # 只看候选，不调 API
    python3 src/vqa/pool.py --write       # 逐条语义验证并写 data/llm_cache/v3/

为什么推翻了上一版
------------------
上一版（`distract.py`，产物在 `llm_cache/v2/`）让 LLM 自由生成干扰项。
盲测实测：**不给视频，纯文本模型比随机高 12.1 个百分点（+6.2 个标准误）。**

查出的特征很具体 —— **40% 的干扰项引入了场景里不存在的物体**：

```
stack_cubes  场景只有 red cube / yellow cube      6/6  的干扰项越界
wash         场景只有 bowl brush plate rag       22/27
pen_inbox    场景只有 pen box                     8/13
```

pen_inbox 的干扰项写着 `Pick the cup. / Pick the tray. / Pick the spoon.` ——
**不看视频也知道这个数据集里不会有杯子和托盘。**
而同一批里的 `Place the box.` 就无法从文字上排除，因为盒子确实在场景里。

所以正解不是「换个模型重生成」，是**限定词汇表**：
干扰项只能由**本族真实动作里出现过的词**重组而成。

怎么造
------
**在真实动作的文本上做槽位替换，而不是从零件拼句子。**
从零件拼会引入措辞差异 —— 实测语料里 `with brush` 与 `on the teapot`
的冠词用法就不一致，拼出来的句子会带上可辨识的风格。替换保留原句一切细节。

```
Wipe the bowl with brush.  --换宾语-->  Wipe the plate with brush.
Pick the pen.              --换动词-->  Move the pen.
```

分三层，越靠前越自然，取用时优先靠前的：

``obj``          换宾语，动词不动 —— 最自然，与真实动作几乎无从分辨
``verb``         换成本族出现过的另一个动词
``verb-global``  换成语料里别处出现过的动词（最后才用，可能有点怪）

每条候选都要过语义验证
----------------------
槽位替换会撞出同义词：

```
真实 Place the pen.   重组 Put the pen.    ← 同一件事
真实 Pick the box.    重组 Take the box.   ← 同一件事
```

所以每个候选**要对本族的每一个真实动作都验一遍**，只要与其中任何一个同义就淘汰。
这比 v2 严格 —— 那时只对它所属的那一个正确答案验。
验证器和回归沿用 `distract.py` 的（`src/vqa/tests/test_verifier.py`，八条用例双向覆盖）。
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distract import KEYS, MODEL, call_api, verify_prompt  # noqa: E402
from vocab import normalize, parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BUILD = ROOT / "build"
OUT = DATA / "llm_cache" / "v3"

POOL_VERSION = "1"
BATCH = 50            # 每次验证请求塞多少对，太大响应会截断


def swap(text: str, old: str, new: str) -> str | None:
    """把 text 里的 old 换成 new，保留原句其余一切（含首字母大小写）。"""
    if not old or not new or old == new:
        return None
    pattern = re.compile(rf"\b{re.escape(old)}\b", re.IGNORECASE)
    if not pattern.search(text):
        return None
    out = pattern.sub(new, text, count=1)
    return out[0].upper() + out[1:] if out else None


def candidates(family: str, subtasks: list[dict[str, Any]],
               global_verbs: list[str]) -> list[dict[str, Any]]:
    """按三层造候选。**只用本族出现过的物体。**"""
    objects = sorted({s["object"] for s in subtasks if s["object"]})
    verbs = sorted({s["verb"] for s in subtasks if s["verb"]})
    real = {s["key"] for s in subtasks}
    seen: dict[str, dict[str, Any]] = {}

    def add(text: str | None, tier: str, base: str) -> None:
        if not text:
            return
        key = normalize(text)
        if key in real or key in seen:
            return
        got = parse(text)
        if got["object"] not in objects:      # 换出场景外的物体 = 白做
            return
        # 「用刷子擦刷子」—— 机械可判，不必花 API
        if got["modifier"] and got["object"] in got["modifier"].split():
            return
        seen[key] = {"text": text, "key": key, "tier": tier, "from": base,
                     "words": len(key.split()), **got}

    for sub in subtasks:
        for other in objects:                                  # 一层：换宾语
            add(swap(sub["text"], sub["object"], other), "obj", sub["id"])
        for verb in verbs:                                     # 二层：本族动词
            add(swap(sub["text"], sub["verb"], verb), "verb", sub["id"])
        for verb in global_verbs:                              # 三层：全局动词
            if verb not in verbs:
                add(swap(sub["text"], sub["verb"], verb), "verb-global", sub["id"])

    order = {"obj": 0, "verb": 1, "verb-global": 2}
    return sorted(seen.values(), key=lambda c: (order[c["tier"]], c["key"]))


def sane_prompt(objects: list[str], texts: list[str]) -> str:
    """第二道闸门：这个动作本身在物理上说得通吗。

    语义验证器只判「是不是同一件事」，判不了「这句话本身荒不荒唐」。
    槽位替换必然会造出这类东西：

        Close the red cube.         方块没有盖子
        Wipe the brush with brush.  用刷子擦刷子
        Pick up the water.          水拿不起来

    物体都在场景里，但**动作不成立** —— 模型靠常识就能排除，
    于是又成了一条不看视频也能用的线索。淘汰掉。
    """
    listing = "\n".join(f'{i}. "{t}"' for i, t in enumerate(texts, 1))
    return f"""A robot arm works at a table. The only objects present are:
{", ".join(objects)}.

For each instruction below, decide whether it makes physical sense as something
a person could actually ask this robot to do with these objects.

Answer "nonsense" when the action cannot be performed on that object —
closing something that has no lid, picking up a liquid, wiping an object with
itself, pouring a solid. Answer "ok" otherwise, even if the instruction is
unusual, as long as it is physically possible.

{listing}

Return JSON only: {{"1": "ok", "2": "nonsense", ...}}
"""


def judge(prompt_text: str, key: str, offset: int, out: dict[int, str]) -> None:
    raw = call_api(prompt_text, key)
    got = json.loads(re.sub(r"^```\w*|```$", "", raw.strip(), flags=re.M))
    for k, v in got.items():
        try:
            out[offset + int(k) - 1] = str(v).strip().lower()
        except ValueError:
            continue


def sanity(texts: list[str], objects: list[str], key: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for start in range(0, len(texts), BATCH):
        judge(sane_prompt(objects, texts[start:start + BATCH]), key, start, out)
    return out


def verify(pairs: list[tuple[str, str]], key: str) -> dict[int, str]:
    """批量问「候选是否与某个真实动作说的是同一件事」。"""
    verdicts: dict[int, str] = {}
    for start in range(0, len(pairs), BATCH):
        chunk = pairs[start:start + BATCH]
        raw = call_api(verify_prompt(chunk), key)
        got = json.loads(re.sub(r"^```\w*|```$", "", raw.strip(), flags=re.M))
        for k, v in got.items():
            try:
                verdicts[start + int(k) - 1] = str(v).strip().lower()
            except ValueError:
                continue
    return verdicts


def main() -> int:
    dry = "--dry-run" in sys.argv
    write = "--write" in sys.argv

    vocab = json.loads((BUILD / "vocab.json").read_text(encoding="utf-8"))["families"]
    global_verbs = sorted({s["verb"] for v in vocab.values() for s in v["subtasks"]})

    key = None
    if not dry:
        key = next((line.split("=", 1)[1].strip().strip("\"'")
                    for line in KEYS.read_text(encoding="utf-8").splitlines()
                    if line.startswith("DEEPSEEK_API_KEY=")), None)
        if not key:
            print(f"❌ {KEYS} 里没有 DEEPSEEK_API_KEY")
            return 1

    print(f"{'族':<13}{'真实':>5}{'候选':>5}{'不合理':>7}{'同义':>7}{'留下':>5}   各层留下多少")
    print("-" * 80)
    grand = Counter()

    for family in sorted(vocab):
        subs = vocab[family]["subtasks"]
        cands = candidates(family, subs, global_verbs)

        if dry:
            print(f"{family:<13}{len(subs):>5}{len(cands):>5}{'—':>9}{'—':>5}   "
                  f"{dict(Counter(c['tier'] for c in cands))}")
            for c in cands[:3]:
                print(f"    [{c['tier']:<11}] 「{c['text']}」  由 {c['from']} 变来")
            continue

        objects = sorted({s["object"] for s in subs if s["object"]})
        # 先过合理性（便宜，且能减少后面的交叉验证量）
        sane = sanity([c["text"] for c in cands], objects, key)
        absurd = [c for i, c in enumerate(cands) if sane.get(i, "ok") == "nonsense"]
        cands = [c for i, c in enumerate(cands) if sane.get(i, "ok") != "nonsense"]

        # 每个候选 × 每个真实动作，全部要判「different」
        pairs = [(s["text"], c["text"]) for c in cands for s in subs]
        verdicts = verify(pairs, key)
        kept, dropped = [], []
        for i, c in enumerate(cands):
            same = [subs[j]["text"] for j in range(len(subs))
                    if verdicts.get(i * len(subs) + j, "same") != "different"]
            (dropped if same else kept).append({**c, "same_as": same})

        tiers = Counter(c["tier"] for c in kept)
        grand["real"] += len(subs); grand["cand"] += len(cands) + len(absurd)
        grand["absurd"] += len(absurd); grand["drop"] += len(dropped); grand["keep"] += len(kept)
        print(f"{family:<13}{len(subs):>5}{len(cands) + len(absurd):>5}{len(absurd):>7}"
              f"{len(dropped):>7}{len(kept):>5}   {dict(tiers)}")

        if write:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"{family}.json").write_text(json.dumps({
                "family": family,
                "pool": kept,
                "provenance": {
                    "pool_version": POOL_VERSION, "model": MODEL,
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "scene_objects": objects,
                    "rejected_as_synonym": [
                        {"text": d["text"], "same_as": d["same_as"]} for d in dropped],
                    "rejected_as_nonsense": [d["text"] for d in absurd],
                },
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not dry:
        print("-" * 80)
        print(f"{'合计':<13}{grand['real']:>5}{grand['cand']:>5}{grand['absurd']:>7}"
              f"{grand['drop']:>7}{grand['keep']:>5}")
        short = [f for f in vocab
                 if len(json.loads((OUT / f"{family}.json").read_text(encoding='utf-8'))["pool"]) < 4] \
            if write else []
        if short:
            print(f"\n❌ 候选不足 4 条的族：{short}")
        print(f"\n写入 {OUT.relative_to(ROOT)}" if write else "\n加 --write 写入")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
