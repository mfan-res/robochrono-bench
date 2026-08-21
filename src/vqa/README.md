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

