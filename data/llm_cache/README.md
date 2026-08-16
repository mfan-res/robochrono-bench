# data/llm_cache

LLM 生成的干扰项缓存。**不可再生，冻结，不要删。**

出题过程本身是完全确定性的（选项打乱用 `md5(item_id|text)` 排序，`seed` 固定），
**唯一的非确定性来源就是这里** —— 重新调 LLM 会得到不同的干扰项，
也就变成了另一套题。

```
<family>.json
{"task_category_distractors": {"pick the bowl": ["grasp the cup", "lift the plate", …]}}
```

已知问题：现有八个族用了**两个不同的模型**生成
（glm-5.2 四族 / gemini-3.5-flash 四族），这是一个跨族混淆变量。
重新生成时应统一。
