# src/migrate

一次性迁移脚本：把旧仓库 `benchmark/` 的数据搬进新结构。

**只读旧仓库，只写新仓库。** 迁移完成后可以删掉这个目录，
但建议保留 —— `data/label/` 与 `data/llm_cache/` 是不可再生的，
这些脚本是它们的来源记录（尤其 `llm_cache` 有四个族是反建出来的）。

## 待办：反建缓存的键格式未验证（2026-08-17 暂缓）

`llm_cache.py` 反建的 8 份文件以 **`answer_text`** 为键，但生成器读缓存时
用的很可能是 **`categories.txt` 里的标签**（`load_category_labels` → `clean_text`）。

两者在多数族里相同，但 **airpods 不同**：

```
categories.txt   pick airpods case
answer_text      pick the airpods case
```

生成器的查表是 `llm_distractor_pool.get(clean_text(correct_text))`，
`correct_text` 具体来自哪个尚未追到底。**所以反建的缓存目前只作留档，
不保证能直接用于重新生成。** 要用之前必须先确认这一点。

原始的 8 份（gift_inhand / pen_inbox / tea / wash）不受影响。
