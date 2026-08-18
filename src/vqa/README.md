# src/vqa

出题工具。读 `data/{raw,label,llm_cache}` + `recipes/`，产出 `data/vqa/`。

`recipes/<family>.json` 跟代码走 git —— 它决定代码怎么跑，
要和代码一起版本控制、一起 review。

**出题是确定性的**：选项打乱用 `md5(item_id|text)` 排序，无 `random`，
`step_order_seed` 固定。同样输入必得同样输出。
唯一的非确定性是 LLM 干扰项，已冻结在 `data/llm_cache/`。

## 八个步骤

```
index → vocab → distract → frames → plan → assets → compose → pack
```

`frames` 是 D-56 新增的，**必须排在 plan 之前**。它抽出候选帧池
（每段中点 × 每个视角）并量出每个族的「视觉余量」，供 plan 判断
图选项的干扰项在**画面上**分不分得开 —— 光靠「来自别集、隔了两段」
这类结构判据不够，人工判无解的题结构上全部合规。

它不增加抽帧量：候选池正好就是 assets 后来要抽的那批，只是提前抽。

