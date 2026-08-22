# src/migrate

一次性迁移脚本：把旧仓库 `benchmark/` 的数据搬进新结构。

**只读旧仓库，只写新仓库。** 迁移完成后可以删掉这个目录，
但建议保留 —— `data/label/` 与 `data/llm_cache/` 是不可再生的，
这些脚本是它们的来源记录（尤其 `llm_cache` 有四个族是反建出来的）。

## ~~待办：反建缓存的键格式未验证（2026-08-17 暂缓）~~ 已失效（2026-08-21）

> **前提消失了**：干扰项自 D-38 起一律取自真实标签，`data/llm_cache/`
> 三代全部退场，不再用于出题。这条待办无需再跟进。
> 原文按「判断被推翻时不删原文」的惯例留在下面。

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

## 已删：`check_labels.py`（2026-08-23）

v1 的标注核验脚本。它读的 `narration` 字段在 D-04（段里改存 subtask id）之后
就不存在了 —— 后果不是归零而是**假发现**：`dup` 恒为最大值，
「重复动作」一栏对每一集报满。

删之前先把它**独有的三条**移植进了 `src/label/checks.py::check_against_video`
（走 `validate.py --probe-video`）：

```
fps 自洽      元表声称的 fps 与 ffprobe 量出的实际值是否一致
帧号越界      end_frame 是否超出视频总帧数
跨度覆盖率    标注跨度占视频的比例
```

**顺序不能反。** 这三条与 `validate.py` 的「派生」不同：那条核的是
`start` 与 `start_frame` **内部自洽**，而这三条核的是**元表与盘上的文件
对不对得上** —— 元表整个错了的话，内部再自洽也没用。tea2 当年就是这么漏过去的。

回归：`src/label/tests/test_checks.py` 第四节。
