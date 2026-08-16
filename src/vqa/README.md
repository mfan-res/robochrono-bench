# src/vqa

出题工具。读 `data/{raw,label,llm_cache}` + `recipes/`，产出 `data/vqa/`。

`recipes/<family>.json` 跟代码走 git —— 它决定代码怎么跑，
要和代码一起版本控制、一起 review。

**出题是确定性的**：选项打乱用 `md5(item_id|text)` 排序，无 `random`，
`step_order_seed` 固定。同样输入必得同样输出。
唯一的非确定性是 LLM 干扰项，已冻结在 `data/llm_cache/`。
