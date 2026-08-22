# 盲基线的历史轮次（归档，不可再生）

⑤ 验题的盲基线：**不给视频，只给题干和选项**，看纯文本模型能答对多少。
超过 `1 / 选项数` 多少，题就泄了多少。

这 9 份是**当时那版题库**的结果。它们原本躺在 `build/`，而 `build/` 名义上是
可再生产物目录 —— 这批恰恰不可再生（`blind.py` 只写 `build/blind.json` 一个
固定路径，这些名字是当初手工改的），所以搬到这里。

**现状的那一份不在这儿**，在 `build/blind_baseline.json` —— 见下方「怎么判断哪份是活的」。

---

## 各轮

| 文件 | 策略 | 选项 | 条 | 盲基线 | 随机 | 选项集仍与现题库一致 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `blind_ship` | as_built | 4 | 1197 | 20.8% | 25.0% | 24% |
| `blind_after_plates` | as_built | 4 | 1026 | 22.7% | 25.0% | 58% |
| `blind_v4` | as_built | 5 | 1197 | 26.6% | 20.0% | 0% |
| `blind_final` | as_built | 6 | 1197 | 22.2% | 16.7% | 0% |
| `blind_four` | four | 4 | 1197 | 24.7% | 25.0% | 25% |
| `blind_four_text` | four | 4 | 1197 | 23.3% | 25.0% | 25% |
| `blind_cross` | cross | 5 | 1197 | 21.2% | 20.0% | 0% |
| `blind_n57` | pool | 5 | 1197 | 23.6% | 20.0% | 0% |
| `blind_three` | three | 3 | 1197 | 33.8% | 33.3% | 0% |

**选项数不同就不能横着比。** 每一行要跟自己那行的「随机」比，不是跟别行比。
`blind_three` 的 33.8% 看着最高，但它是三选一，随机就是 33.3% —— 它其实只高 +0.5pp。

---

## 哪几份是有出处的证据（别删）

**`blind_ship` + `blind_after_plates` 是一对 —— DEVLOG 的 D-47 就是靠它们发现的。**

```
blind_ship          understanding 24.6%
blind_after_plates  understanding 30.7%     +2.4σ
```

D-46 修完之后重跑，`understanding` 涨了 6 个点，**而且五个没改词表的族也全涨了**
—— 所以不是词表拆分造成的。逐题比对才发现 1,023 道共同题里有 738 道（72%）
选项变了，根因是 D-46 新写的那行遍历了 `set`，**出题每次都不一样**。

这两份单独看都只是一个数字；**成对看才是那条 bug 的证据**。所以哪份都不要单删。

其余 7 份是策略探索的过程：`three` / `cross` / `pool` / `four` 都是「统一选项数」
定稿之前试过的方案，结论见 DEVLOG D-38 与 D-48。

---

## 怎么判断哪份是活的

`as_built` 策略测的是**题库里真正会发给模型的那一套**，所以它的结论
**只对当时那版 `plan.json` 成立**。判据不是文件名里的 `final` / `ship`
（这两个名字骗过人），是**选项集还和现在的题库对不对得上**：

```bash
python3 - <<'PY'
import json, sys
cur = {i["id"]: i for i in json.load(open("build/plan.json"))["items"]}
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "build/blind_baseline.json"))
live = [r for r in rows if r["id"] in cur]
same = sum(sorted(r["shuffled"]) == sorted([cur[r["id"]].get("answer_text", "")]
           + list(cur[r["id"]].get("distractors") or [])) for r in live)
print(f"{len(live)}/{len(rows)} 命中现题库，其中 {same/max(1,len(live)):.0%} 选项集一致")
PY
```

现在只有 `build/blind_baseline.json` 是 100%。上表里最高的 `blind_after_plates`
也只有 58% —— **过半的题选项已经变了，拿它的数字讲现状会讲错。**
