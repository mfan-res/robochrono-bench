#!/usr/bin/env python3
# coding: utf-8
"""④ 出题第三步：为每个 subtask 生成一批「像真的但是错的」动作标签。

    python3 src/vqa/distract.py --dry-run      # 只打印提示词与判据，不调 API
    python3 src/vqa/distract.py --write        # 调 API 并写 data/llm_cache/v2/

**这一步产出的是不可再生数据。** 重跑会得到不同的干扰项，也就是另一套题。
所以生成一次、过检查、进 git、冻结。v1 的三个问题见 `data/llm_cache/README.md`。

为什么干扰项质量决定题目质量
----------------------------
六个选项里只有一个对。**如果错的那五个能靠「读起来不像」被排除，
模型不看视频也能答对** —— 那这道题量的就不是视觉理解。

v1 里真的发生了：正确答案是通顺的 `Pick up the kettle.`，
干扰项是 `put the up kettle`（LLM 照着一个带 bug 的键仿写的）。
**没有任何一步会报错**，因为没有任何一步检查过干扰项长什么样。

所以这里的重点不是提示词，是**出厂检查**（`check` 函数）。
提示词决定「大概能拿到什么」，检查决定「什么能出厂」。

六条判据
--------
每条都对应一种「不看视频就能排除」的可能：

1. ``form``       首字母大写、句号结尾、其余小写 —— 与 subtask 文字同款
2. ``not_correct`` 归一化后不等于本族任何一个真实动作（否则它不是错的）
3. ``unique``     同一个 subtask 的干扰项之间不重复
4. ``length``     词数在**它对应的正确答案** ±1 以内 —— 防「最长的那个是对的」
5. ``wellformed`` 不出现 `pick the up` 这类错位小品词（v1 的原病）
6. ``parses``     能切出动词+宾语，即与正确答案结构同型

不合格的**逐条退回重生成**，最多三轮；三轮后仍不足就**显式报缺**，
不静默用规则兜底 —— 静默兜底正是 v1 已知问题 h 的机制（D-03）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab import PARTICLES, normalize, parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BUILD = ROOT / "build"
OUT = DATA / "llm_cache" / "v2"
KEYS = Path.home() / ".config" / "robochrono" / "keys.env"

# 生成模型。**必须不在被评测名单内**（现为 glm / qwen / gemini）——
# 用被评测的模型编干扰项是自利偏差，v1 两个模型都踩了，见 llm_cache/README.md。
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TEMPERATURE = 0.4          # 要一点多样性；反正输出会冻结
PER_SUBTASK = 8            # 出题每题用 4 条，多生成是留给判据淘汰的余量
NEEDED = 4
MAX_ROUNDS = 3

MALFORMED = re.compile(r"\b(?:pick|put|take|set|lift)\s+the\s+(?:" + "|".join(PARTICLES) + r")\b")
FORM = re.compile(r"^[A-Z][a-z]+(?: [a-z]+)*\.$")


def prompt_for(family: str, subtasks: list[dict[str, Any]],
               need: dict[str, int], rejected: dict[str, list[str]]) -> str:
    """批量提示：一次给全族的动作，模型才知道哪些标签是「对的」而必须避开。"""
    listing = "\n".join(f"  {s['id']}\t{s['text']}" for s in subtasks)
    objects = sorted({s["object"] for s in subtasks if s["object"]})
    verbs = sorted({s["verb"] for s in subtasks if s["verb"]})

    # 逐条把目标词数写成数字。只说「同长度」时，模型对 2 词标签一律给 5 词，
    # 于是 pour_water / pour_tea 一条都通不过。
    by_id = {s["id"]: s for s in subtasks}
    ask = "\n".join(
        f'  {sid}\t{n} more, each {len(by_id[sid]["text"].split())} words '
        f'(±1), like "{by_id[sid]["text"]}"'
        for sid, n in need.items())

    # 重试时光复述禁用清单没用 —— wash 实测三轮反复提同样六条。
    # 它缺的是**槽位模板**：告诉它句子的骨架，只换填进去的词。
    retry = ""
    if rejected:
        blocks = []
        for sid, texts in rejected.items():
            if not texts or sid not in need:
                continue
            got = parse(by_id[sid]["text"])
            slots = f'{got["verb"]} the ___' + (
                f' {got["modifier"].split()[0]} ___' if got["modifier"] else "")
            blocks.append(
                f'  {sid}  keep the shape "{slots.capitalize()}." and change what '
                f'goes in the blanks.\n'
                f'    already rejected: ' + ", ".join(f'"{t}"' for t in dict.fromkeys(texts)))
        if blocks:
            retry = ("\nAn automatic check rejected your earlier attempts for these — "
                     "almost all because you named an action the robot really performs.\n"
                     "Do not repeat them. Any everyday object that could sit on this "
                     "workbench is fair game for a blank:\n" + "\n".join(blocks) + "\n")

    forbidden = "\n".join(f'     "{s["text"]}"' for s in subtasks)
    return f"""You are writing distractor options for a robot-manipulation video QA benchmark.
A question shows a video clip and asks what the robot is doing. One option is
correct; the rest are distractors you write. A test-taker must have to watch the
video to tell them apart.

Task family: {family}
The robot really performs these actions (these are the CORRECT labels):
{listing}

Objects present in this scene: {", ".join(objects)}
Verbs used in this scene: {", ".join(verbs)}

Write wrong action labels for:
{ask}
{retry}
Requirements — each one exists because violating it lets someone answer without
watching the video:

1. WRITE IT THE SAME WAY. Capitalized first word, everything else lowercase,
   ending in a period, e.g. "Pick up the teapot lid." A distractor that reads
   differently from the correct answer can be spotted on style alone.
2. HIT THE WORD COUNT given for each label above. Count every word including
   "the". If the target is 3 words, "Pour the water into the teapot." (6 words)
   is rejected — write "Pour the milk." or "Stir the water." instead.
   Otherwise "the longest option is correct" becomes a strategy.
3. NEVER OUTPUT ONE OF THESE — they are the correct answers to other questions
   in this same family, and they are already used as distractors by a different
   mechanism. Your job is to produce labels that are NOT in this list:
{forbidden}
3b. AND NEVER OUTPUT A SYNONYM OR PARAPHRASE of the label you are writing for.
   This is the most common way these get rejected. A distractor must be FALSE
   when the robot performs the action — not merely worded differently. Renaming
   the object does not make it false:
     for "Pick up the teapot."       "Pick up the pot."     is the SAME thing
     for "Pick up the tea bag."      "Pick up the sachet."  is the SAME thing
     for "Pick up the teapot lid."   "Pick up the cover."   is the SAME thing
   Also avoid a more general word for the same object ("container" for a
   kettle) and a different verb for the same motion ("grasp" for "pick up").
   Name a DIFFERENT physical object, or a DIFFERENT motion.
4. PLAUSIBLE HERE. A robot arm at this workstation could believably be asked to
   do it. Prefer the objects listed above; a nearby everyday object (cup, tray,
   spoon, cloth) is fine. Do not reach for objects from another room.
5. GRAMMATICAL. "Pick up the kettle." is right; "Pick the up kettle." is not.
   Particles stay with the verb.
6. VARY THEM. Change the verb, or the object, or both. When the word count is
   small, changing the object of a short verb phrase is usually the only way to
   stay in budget — that is expected, just keep them distinct from each other.

Return JSON only, no prose, no code fence:
{{"<subtask_id>": ["<wrong label>", "..."]}}
"""


def check(text: str, correct: dict[str, Any], family_keys: set[str],
          taken: set[str]) -> str | None:
    """返回失败的判据名，通过则 None。**顺序即诊断顺序**，先报最基本的。"""
    if not FORM.match(text):
        return "form"
    if MALFORMED.search(text.lower()):
        return "wellformed"
    key = normalize(text)
    if key in family_keys:
        return "not_correct"
    if key in taken:
        return "unique"
    if abs(len(key.split()) - len(correct["key"].split())) > 1:
        return "length"
    got = parse(text)
    if not got["verb"] or not got["object"]:
        return "parses"
    return None


def verify_prompt(pairs: list[tuple[str, str]]) -> str:
    """对抗验证：让模型去**证伪**自己刚写的干扰项。

    六条机器判据全是表层的 —— 拼写、长度、结构。它们抓不到这一类：

        正确  Pick up the tea bag.     干扰  Pick up the tea sachet.
        正确  Pick up the teapot.      干扰  Pick up the pot.

    字符串不同，**说的是同一件事**。这种「干扰项其实也对」的题比有捷径的题更糟：
    有捷径只是变简单，答案不唯一是怎么答都可能被判错。

    实测这是首轮生成里最常见的失败方式，而且**只在人读了输出之后才发现** ——
    所以把「读一遍」也自动化掉。
    """
    listing = "\n".join(f'{i}. TRUE: "{c}"   CANDIDATE: "{d}"'
                        for i, (c, d) in enumerate(pairs, 1))
    return f"""A robot performs an action described by the TRUE label. Someone
watching the same video writes the CANDIDATE label instead.

For each pair, answer "same" if the candidate could fairly describe that very
same action, and "different" if it describes something the robot did not do.

Answer "same" when the candidate merely renames things:
  - a different word for the same object      teapot / pot, tea bag / sachet
  - a broader word for the same object        kettle / container
  - a different verb for the same motion      pick up / grasp / lift
  - a different word for the same part        lid / cover

Answer "different" only when a person watching the video would say the candidate
is factually wrong — a different physical object, or a different motion.

{listing}

Return JSON only: {{"1": "same", "2": "different", ...}}
"""


def call_api(prompt: str, key: str, timeout: int = 180) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "response_format": {"type": "json_object"},
    }).encode()
    request = urllib.request.Request(
        API_URL, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"DeepSeek 调用失败：{last}")


def generate(family: str, entry: dict[str, Any], key: str) -> dict[str, Any]:
    subtasks = entry["subtasks"]
    family_keys = {s["key"] for s in subtasks}
    by_id = {s["id"]: s for s in subtasks}

    accepted: dict[str, list[str]] = {s["id"]: [] for s in subtasks}
    rejected: dict[str, list[str]] = {s["id"]: [] for s in subtasks}
    reasons: dict[str, int] = {}
    rounds: list[dict[str, Any]] = []

    for round_no in range(1, MAX_ROUNDS + 1):
        need = {sid: PER_SUBTASK - len(v) for sid, v in accepted.items()
                if len(v) < PER_SUBTASK}
        if not need:
            break
        text = prompt_for(family, subtasks, need,
                          rejected if round_no > 1 else {})
        raw = call_api(text, key)
        try:
            batch = json.loads(re.sub(r"^```\w*|```$", "", raw.strip(), flags=re.M))
        except json.JSONDecodeError:
            rounds.append({"round": round_no, "error": "响应不是 JSON", "raw": raw[:200]})
            continue

        # 第一道：表层判据
        survivors: list[tuple[str, str]] = []
        for sid, candidates in batch.items():
            if sid not in by_id or not isinstance(candidates, list):
                continue
            taken = {normalize(t) for t in accepted[sid]}
            for candidate in candidates:
                if len(accepted[sid]) + sum(1 for s, _ in survivors if s == sid) >= PER_SUBTASK:
                    break
                candidate = str(candidate).strip()
                why = check(candidate, by_id[sid], family_keys, taken)
                if why:
                    reasons[why] = reasons.get(why, 0) + 1
                    rejected[sid].append(candidate)
                else:
                    survivors.append((sid, candidate))
                    taken.add(normalize(candidate))

        # 第二道：语义验证。**批量一次调用**，逐条问会慢且贵。
        verdicts: dict[int, str] = {}
        if survivors:
            pairs = [(by_id[sid]["text"], text) for sid, text in survivors]
            try:
                raw_v = call_api(verify_prompt(pairs), key)
                verdicts = {int(k): str(v).strip().lower()
                            for k, v in json.loads(
                                re.sub(r"^```\w*|```$", "", raw_v.strip(), flags=re.M)).items()}
            except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
                # 验证失败时**不放行** —— 未经验证的干扰项不进不可再生层
                rounds.append({"round": round_no, "verify_error": str(exc)[:120]})
                for sid, text in survivors:
                    rejected[sid].append(text)
                continue

        added = 0
        for i, (sid, text) in enumerate(survivors, 1):
            if verdicts.get(i, "same") != "different":     # 缺答复按「同义」处理
                reasons["synonym"] = reasons.get("synonym", 0) + 1
                rejected[sid].append(text)
            else:
                accepted[sid].append(text)
                added += 1
        rounds.append({"round": round_no, "asked": sum(need.values()),
                       "passed_surface": len(survivors), "accepted": added})

    short = {sid: len(v) for sid, v in accepted.items() if len(v) < NEEDED}
    return {
        "family": family,
        "distractors": accepted,          # 键是 subtask ID，不是文字
        "provenance": {
            "model": MODEL, "temperature": TEMPERATURE,
            "prompt_sha256": hashlib.sha256(
                prompt_for(family, subtasks, {s["id"]: PER_SUBTASK for s in subtasks}, {})
                .encode()).hexdigest()[:16],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "per_subtask_target": PER_SUBTASK, "needed_per_question": NEEDED,
            "rounds": rounds,
            "rejected_by_check": reasons,
            "rejected_texts": {k: v for k, v in rejected.items() if v},
            "short_of_needed": short,     # 空 = 每个动作都够用
        },
    }


def report(family: str, blob: dict[str, Any], entry: dict[str, Any]) -> bool:
    by_id = {s["id"]: s for s in entry["subtasks"]}
    print(f"\n【{family}】")
    ok = True
    for sid, options in blob["distractors"].items():
        correct = by_id[sid]
        n_correct = len(correct["key"].split())
        deltas = [len(normalize(o).split()) - n_correct for o in options]
        span = (f"{min(deltas):+d}…{max(deltas):+d}" if deltas else "—")
        mark = "✓" if len(options) >= NEEDED else "✗"
        if len(options) < NEEDED:
            ok = False
        print(f"  {mark} {sid:<26}{len(options)} 条  词数偏差 {span}")
        print(f"      正确 「{correct['text']}」")
        print(f"      干扰 " + " / ".join(f"「{o}」" for o in options[:4]))
    # 跨 subtask 撞车：两个动作拿到同一条干扰项。不算错（各自都是错的），
    # 但同一道题里若两个来源都取到它就会出现重复选项，[4] 组装时要去重。
    seen: dict[str, list[str]] = {}
    for sid, options in blob["distractors"].items():
        for text in options:
            seen.setdefault(normalize(text), []).append(sid)
    shared = {k: v for k, v in seen.items() if len(v) > 1}
    if shared:
        print(f"  跨 subtask 共用 {len(shared)} 条（[4] 组装时去重）：" +
              "、".join(list(shared)[:3]))

    prov = blob["provenance"]
    if prov["rejected_by_check"]:
        print(f"  判据淘汰：{prov['rejected_by_check']}")
    return ok


def main() -> int:
    write = "--write" in sys.argv
    families = sys.argv[sys.argv.index("--family") + 1].split(",") \
        if "--family" in sys.argv else None

    vocab = json.loads((BUILD / "vocab.json").read_text(encoding="utf-8"))["families"]
    targets = [f for f in sorted(vocab) if not families or f in families]

    if "--dry-run" in sys.argv:
        family = targets[0]
        entry = vocab[family]
        print(prompt_for(family, entry["subtasks"],
                         {s["id"]: PER_SUBTASK for s in entry["subtasks"]}, {}))
        return 0

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and KEYS.exists():
        for line in KEYS.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        print(f"❌ 没有 DEEPSEEK_API_KEY（环境变量或 {KEYS}）")
        return 1

    all_ok = True
    for family in targets:
        blob = generate(family, vocab[family], key)
        all_ok &= report(family, blob, vocab[family])
        if write:
            OUT.mkdir(parents=True, exist_ok=True)
            path = OUT / f"{family}.json"
            path.write_text(json.dumps(blob, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            print(f"  → {path.relative_to(ROOT)}")

    print("\n" + "=" * 62)
    if all_ok:
        print(f"✅ 每个 subtask 都拿到 ≥{NEEDED} 条合格干扰项")
    else:
        print(f"❌ 有 subtask 不足 {NEEDED} 条 —— **不要静默用规则补**，"
              f"先看 provenance.rejected_texts 判断是判据太严还是提示词不清")
    if not write:
        print("加 --write 写入 data/llm_cache/v2/")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
