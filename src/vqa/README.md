# src/vqa

出题工具。读 `data/{raw,label}`，产出 `build/` 的中间件与 `data/vqa/`。

**配方跟着代码走 git**，不是外部配置：`plan.py` 的 `RECIPE_VERSION`
与那一批常量（`DISTRACTORS_PER_QUESTION`、`IV_MIN_SEGMENT_GAP` 等）
就是配方本身，改它要和代码一起 review。
（曾计划做成 `recipes/<family>.json`，没有实现，这里的说法一并更正。）

**出题是确定性的**：选项打乱用 `md5(item_id|text)` 排序，无 `random`，
`step_order_seed` 固定。同样输入必得同样输出。
**没有非确定性**：干扰项自 D-38 起一律取自真实标签，`data/llm_cache/` 三代全部退场（只作留档）。

## 八个步骤

```
index → vocab → distract → frames → plan → assets → compose → pack
```

`frames` 是 D-56 新增的，**必须排在 plan 之前**。它抽出候选帧池
（每段中点 × 每个视角）并量出每个族的「视觉余量」，供 plan 判断
图选项的干扰项在**画面上**分不分得开 —— 光靠「来自别集、隔了两段」
这类结构判据不够，人工判无解的题结构上全部合规。

它不增加抽帧量：候选池正好就是 assets 后来要抽的那批，只是提前抽。


## 七步的顺序

```
1 index.py    有什么可用来出题        → build/index.json
2 vocab.py    subtask 展开成各种形态   → build/vocab.json
3 frames.py   候选帧池 + 画面距离下限   → build/frames.json + frames_desc.npy
4 plan.py     出哪些题、要哪些素材      → build/plan.json      ← 只算不写盘
5 assets.py   切片 / 抽帧             → data/vqa/assets/     ← 唯一动 ffmpeg 的一步
6 compose.py  组装并过 item.json 契约  → data/vqa/items.jsonl
7 pack.py     投影成评测端的形状        → data/vqa/eval/
```

**1–4 只读和算**，改参数重跑几十秒；**只有 5 动 ffmpeg**。
所以调题目设计不必重切素材（D-05）。

> 原「第三步」是 `distract.py` / `pool.py` 两代 LLM 干扰项生成器，
> 自 D-38 起干扰项一律取自真实标签，它们已退场；`frames.py` 后来插进来时
> 沿用了 plan 的编号，一度两个文件都叫「第四步」。2026-08-23 重编号。

**改题目设计时**：`src/vqa/tests/test_plan_invariants.py` 锁着全部 10,178 道题的
内容指纹，改了要同步改 `EXPECTED` 并说明改了什么。
