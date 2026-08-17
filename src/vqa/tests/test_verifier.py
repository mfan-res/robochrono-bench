#!/usr/bin/env python3
# coding: utf-8
"""语义验证器的回归。**需要 API，一次调用。**

    python3 src/vqa/tests/test_verifier.py

`distract.py` 的六条机器判据全是表层的，抓不到「干扰项其实说的是同一件事」。
语义验证器补这个洞，但它本身是个 LLM —— **改提示词就可能悄悄失效**，
而失效的表现是「干扰项看起来都挺好」，不会报错。

所以用例分两半，缺一不可：

    前四条   上一轮真实产出的同义词，验证器必须判 same
    后四条   真正不同的动作，验证器必须判 different

**只测前四条是不够的** —— 一个永远回答 "same" 的验证器也能全过，
但它会把所有干扰项都毙掉。两个方向都要测。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distract import KEYS, call_api, verify_prompt  # noqa: E402

# (正确标签, 候选干扰项, 期望判定)
CASES = [
    # 同义 —— 全部来自首轮真实产出，当时六条表层判据一条都没拦住
    ("Pick up the teapot.",     "Pick up the pot.",          "same"),
    ("Pick up the tea bag.",    "Pick up the tea sachet.",   "same"),
    ("Pick up the teapot lid.", "Pick up the teapot cover.", "same"),
    ("Pick up the kettle.",     "Pick up the container.",    "same"),
    # 不同 —— 换物体、换动作、换方向，都是合格的干扰项
    ("Pour the water.",         "Pour the milk.",            "different"),
    ("Pick up the teapot lid.", "Place the lid on teapot.",  "different"),
    ("Pick up the tea bag.",    "Pick up the sugar bowl.",   "different"),
    ("Pour the tea.",           "Stir the tea.",             "different"),
]


def main() -> int:
    key = next((line.split("=", 1)[1].strip().strip("\"'")
                for line in KEYS.read_text(encoding="utf-8").splitlines()
                if line.startswith("DEEPSEEK_API_KEY=")), None)
    if not key:
        print(f"❌ {KEYS} 里没有 DEEPSEEK_API_KEY")
        return 1

    raw = call_api(verify_prompt([(c, d) for c, d, _ in CASES]), key)
    got = json.loads(re.sub(r"^```\w*|```$", "", raw.strip(), flags=re.M))

    passed = 0
    for i, (correct, candidate, want) in enumerate(CASES, 1):
        verdict = str(got.get(str(i), "?")).lower()
        ok = verdict == want
        passed += ok
        print(f"  {'✓' if ok else '✗'} 「{correct}」 vs 「{candidate}」"
              f"  判 {verdict} / 应为 {want}")

    print(f"\n{passed}/{len(CASES)} 正确")
    if passed < len(CASES):
        print("❌ 验证器退化了。**不要放宽用例** —— "
              "它失效的表现是干扰项看起来都挺好，不会报错。")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
