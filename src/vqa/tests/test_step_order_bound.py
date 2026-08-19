#!/usr/bin/env python3
# coding: utf-8
"""step_order 的盲基线上界 —— 这是回归测试，不是一次性脚本。

    python3 src/vqa/tests/test_step_order_bound.py

为什么算而不是测
----------------
挖掉图之后，step_order 的选项只剩「Image 2 -> Image 1 -> Image 3」这类排列
文字，**不含任何场景信息**。盲着答题只可能靠两种固定偏好：

    1 偏爱某种排列写法（如总选「1 -> 2 -> 3」）
    2 偏爱某个字母位置（实测该模型 4 选 1 时 69% 选 D）

两者的最优策略都可以**穷举**：排列只有 6 种，全部 720 种偏好排序都试一遍；
字母只有 4 个。得到的是「**知道整个数据集的对手能拿到的最好成绩**」——
比 n=360 的一次抽样硬得多。

（另一个现实原因：本地 InternVL 要求至少一张图，纯文字的盲测件在它上面
跑不了，360 次连续失败会触发熔断。）

为什么要当回归跑
----------------
上界不是常数，它取决于**选项打乱之后答案落在哪**。改了打乱规则、改了
帧的选法、换了族，它都会动。写死一个「当时算出来是 28.8%」的数字放进
文档，下次没人会再算。
"""

from __future__ import annotations

import collections
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ITEMS = ROOT / "data" / "vqa" / "items.jsonl"

# 上界超过这个就要停下来看 —— 四选一的随机基线是 25%。
# 5 个百分点的余量：排列只有 6 种、题量一千出头，抽样波动本来就有两三个点。
CEILING = 0.30


def main() -> int:
    if not ITEMS.exists():
        print(f"跳过：缺 {ITEMS.relative_to(ROOT)}（先跑 compose）")
        return 0
    items = [json.loads(line) for line in ITEMS.open(encoding="utf-8")
             if '"step_order"' in line]
    items = [i for i in items if i["task"] == "step_order"]
    if not items:
        print("跳过：还没有 step_order 的题")
        return 0

    n = len(items)
    texts = sorted({o["text"] for i in items for o in i["prompt"]["options"]})
    print(f"step_order {n} 道，四选一，随机基线 25.0%，选项文字 {len(texts)} 种")

    best, best_first = 0.0, None
    for rank in itertools.permutations(range(len(texts))):
        order = [texts[j] for j in rank]
        hit = 0
        for item in items:
            present = {o["text"] for o in item["prompt"]["options"]}
            if next(t for t in order if t in present) == item["truth"]["answer_text"]:
                hit += 1
        if hit / n > best:
            best, best_first = hit / n, order[0]

    pos = collections.Counter(i["truth"]["answer"] for i in items)
    letter = max(pos.values()) / n

    print(f"  按排列文字的最优固定偏好　上界 {best:.1%}（最偏爱「{best_first}」）")
    print(f"  按字母位置的最优固定偏好　上界 {letter:.1%}"
          f"（总选 {max(pos, key=pos.get)}，分布 {dict(sorted(pos.items()))}）")

    worst = max(best, letter)
    if worst > CEILING:
        print(f"\n❌ 盲基线上界 {worst:.1%} 超过 {CEILING:.0%} —— "
              f"不看图就能拿到这个分，题目有捷径")
        return 1
    print(f"\n✓ 两条上界都在 {CEILING:.0%} 以内 —— 不看图没有可乘之机")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
