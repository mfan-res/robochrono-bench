# 数据重构方案

> 日期：2026-08-16 ｜ 前置：[数据结构现状](dataset_structure.md)
> 状态：**方案待确认，尚未实现**

---

## 1. 为什么要重构

现在评测代码里有四处补丁，都是在**运行时**吸收数据布局的不一致：

| 补丁 | 吸收什么 |
| --- | --- |
| `tasks.qa_path` 递归查找 | QA 文件三种深度布局 |
| `matrix.derive_family_attrs` | 文档与数据冲突的族属性 |
| `mediaindex` 文件名解析 + 逐级消歧 | 四种路径风格、视角子目录改名 |
| `normalize_qa_paths.py` | 前缀重写（已不够用） |

问题不在于补丁本身，而在于**它们每次运行都要执行，且失败方式是静默的**。
`qa_path` 找不到就整族消失，`mediaindex` 解析不到就整题跳过 ——
这两种都发生过，而且都是我先误判、后来才发现的。

**目标：把「运行时解析」变成「一次性构建」。** 评测代码只面对一种结构，
不一致性在构建阶段被解决并留下可审计的记录。

---

## 2. 设计原则

**① 原始数据只读，永不改动。**
61 GB 已全量哈希校验通过、可重下。改写它会让本地与远端分叉。

> **这条曾被违反，已于 2026-08-16 修复。** 早期的 BC-08 工具把媒体路径
> 就地重写成本机绝对路径，改动了 9 个 QA 文件（全在 stack_cubes）。
> 已从 `.orig` 备份全部还原，`datasets/QA/` 里现在没有任何本机路径。
> 详见 4.3。

**② 构建产物可完全重建。**
删掉重跑应当得到逐字节相同的结果。任何不可重建的手工修补都要记录在案。

**③ 缺失要显式，不要静默。**
构建时解析不到的媒体、定位不到的 QA、字段缺失，都进 manifest 的问题清单，
而不是让下游「刚好跳过」。

**④ 不改变题目内容。**
只做定位、命名、路径的规范化。题干、选项、答案一个字不动 ——
否则就不再是同一个 benchmark 了。

---

## 3. 目标结构

```
eval/datasets/
├── QA/…                          原始数据，只读
├── json/…                        原始数据，只读
└── normalized/                   构建产物，gitignore，可重建
    ├── manifest.json             索引 + 构建记录 + 问题清单
    └── <family>/
        ├── understanding.jsonl
        ├── left_right.jsonl
        ├── image_in_video.jsonl
        ├── time.jsonl
        ├── planning.jsonl
        ├── planning_2.jsonl
        └── step_order.jsonl
```

**为什么用 JSONL 而不是 JSON**：逐行可读、可 diff、可流式处理、
坏一行不影响其余 —— 和 `ResultStore` 同样的理由。原始 JSON 里那些
族级元数据（`source`、`video_dir`、`option_design` 等）移到 manifest。

### 3.1 每行的结构

```jsonc
{
  "id": "file-000-1_understand",
  "family": "stack_cubes",
  "run": "understanding",
  "question": "...",                    // 原样
  "options": [...],                     // 原样
  "answer": "D",                        // 原样
  "media": [                            // ← 规范化的关键
    {"role": "primary",  "view": "left_eye",  "kind": "image", "path": "/abs/…"},
    {"role": "context",  "view": "left_wrist","kind": "image", "path": "/abs/…"},
    {"role": "option_a", "view": null,        "kind": "image", "path": "/abs/…"}
  ],
  "meta": { … },                        // 任务特有字段，原样保留
  "source": {                           // 溯源，便于回查
    "qa_file": "QA/planning/gift_inhand/planning/planning_vqa.json",
    "item_index": 0
  }
}
```

**`media` 是唯一的媒体入口。** 现在每个任务各自去 `input.clip_path`、
`input.video_paths`、`images`、`option_images` 里翻找，六个任务六套逻辑，
其中 `left_right` 与 `image_in_video` 的取图顺序还依赖 dict 迭代序。
统一成一个有序数组，`role` 标明用途，任务代码只读它。

原始字段全部保留在 `meta` 里，不丢信息。

### 3.2 manifest.json

```jsonc
{
  "built_at": "2026-08-16T…",
  "source_sha": "561add0d…",            // 远端仓库 commit，用于判断是否过期
  "builder_version": "v1",
  "families": {
    "stack_cubes": {
      "original_name": "stack_cubes",
      "layout": "flat",                 // flat | nested | scattered
      "schema_version": "legacy",       // legacy | current  ← planning 的两套结构
      "attrs": {"two_handed": true},    // 从数据推导，记下依据
      "runs": {
        "understanding": {"items": 300, "media": 500, "unresolved": 0}
      }
    },
    "take_out_the_trash": {
      "original_name": "Take_out_the_trash",   // ← 族名规范化的映射
      "incomplete": ["understanding 组缺失"]
    }
  },
  "issues": [
    {"kind": "media_unresolved", "family": "tea2", "run": "time", "count": 42,
     "examples": ["file-000.mp4"]},
    {"kind": "media_ambiguous",  "family": "wash", "run": "left_right", "count": 160}
  ]
}
```

`source_sha` 让「数据更新了但没重建」这件事可被检测 —— 今天核验时
发现远端已推进 7 个提交，靠的是手工比对，这应当自动化。

---

## 4. 规范化的六件事

| # | 做什么 | 依据 |
| --- | --- | --- |
| 1 | **定位 QA** | 吸收三种深度布局；多处命中报错不猜 |
| 2 | **族名规范化** | `Take_out_the_trash` → `take_out_the_trash`，manifest 记原名 |
| 3 | **解析媒体路径为绝对路径** | 现有 `mediaindex` 逻辑，构建期执行一次 |
| 4 | **统一媒体入口为 `media[]`** | 消除六套取图逻辑与 dict 迭代序依赖 |
| 5 | **标注 schema 版本** | `planning` 两套结构并存，先标注不合并（见下） |
| 6 | **推导并记录族属性** | `two_handed` 从 `target_side` 推，依据写进 manifest |
| 7 | **拆分 planning_2 的预拼接图** | BC-16 —— 唯一一件**有意改变模型输入**的事，见 4.2 |

### 4.2 关于第 7 件事：它和前六件不是一类

前六件都在原则 ④「不改变题目内容」之内 —— 只做定位、命名、路径。
**第 7 件不是**：它把 stack_cubes 的 planning_2 从「发一张宽图」改成
「发三张图」，模型收到的东西确实变了。

之所以仍然放在构建期做，是因为替代方案更糟：改原始数据违反原则 ①，
运行时切则把一次性的事做成每次都做的事，还会让「输入是什么」
取决于运行时状态。构建期做，产物是可检查的文件。

它带来的额外义务是：**验收判据要跟着换**，不能沿用「逐字节相同」。
`test_normalized_equivalence.py` 对这一格改验两条 —— 题目内容逐字段相同，
且三张拼回原图逐像素相同。豁免读的是 manifest 里构建器写下的记录，
不是测试里手写的族名。

### 4.1 关于 `planning` 的两套 schema

现状：stack_cubes 用 `clip_path`/`clips`/`prejoined_video_path`，
其余七族用 `joined_clip`/`view_order`。

**本轮不合并，只标注。** 理由是**我不知道哪套是数据方的当前意图** ——
猜错会让 stack_cubes 或其余七族的输入变成错的，而且不报错。
先在 manifest 里标 `schema_version`，让 `media[]` 的构建针对两套各写一段，
产物统一。等数据方回复后再决定是否废弃其中一套。

**这也意味着：stack_cubes 是唯一的 legacy 族，而我们所有的回归基线都建在它上面。**
新族投入使用前，需要在一个 `current` 族上重新验证一遍。

---

### 4.3 还原 BC-08 就地改写的 9 个 QA 文件（已完成）

**发现经过**：核对「`question` 里内嵌的选项路径 == `options[].image_path`」时，
八个族里只有 stack_cubes 全部不符。查下去发现它的 `options[].image_path`
是本机绝对路径 `/mnt/public/users/wbcd/…`，而 `question` 里那份还是
生成机路径 `/home/llm/yyywv/…`。**前者是我们自己写进去的。**

BC-08 的 `tools/normalize_qa_paths.py` 就地改写了 9 个文件：

```
QA/planning/stack_cubes/{planning,planning_2,step_order,trajectory_qa_2d,trajectory_qa_3d}_vqa.json
QA/understanding/stack_cubes/{understanding,left_right,image_in_video,time}_vqa.json
```

**为什么现在可以还原**：BC-08 解决的问题（三种路径风格在本机都不可用）
现在由 `mediaindex` 在**构建期**解析，比就地改写干净 ——
还原前后媒体解析数字**完全相同**（stack_cubes 5,220 媒体 / 100 未解析 / 0 重名）。

**结果**：9 个文件已从 `.orig` 还原，`datasets/QA/` 里 0 个本机路径；
`question` 内嵌路径与 `options[].image_path` 全量一致（left_right 3,702 题、
image_in_video 1,851 题，按后缀判等，0 处不符）。
`check_freshness` 精确报出了 stack_cubes 那 7 个 run 需要重建。

**连带修了两个测试**。`test_replay_regression` 与 `test_request_equivalence`
原先喂给比对的是**未解析**的 item —— 之前能过，正是因为 BC-08 把绝对路径
烤进了 QA。还原后差别显出来：冻结实现自己把相对路径解析到 QA 目录，
我们的 `parts()` 则依赖预解析。两个测试现在都先调 `mediaindex` 解析，
**比的才是生产里真正跑的那条路**。六套回归全绿。

---

## 5. 实施顺序

1. ✅ **`robochrono/normalize.py`** —— 构建器。输入原始 `datasets/`，输出 `normalized/`
2. ✅ **`tools/build_normalized.py`** —— CLI，含 `--check` 预演、`--verify` 查过期
3. ✅ **`load_run_items` 显式声明数据源**（原计划写的是「优先规范化、缺了回退」，
   实施时改掉了，理由见 5.1）
4. ✅ **回归验证** —— 五套回归全绿，等价性关卡 56 格全过
5. ⬜ **移除运行时补丁** —— `mediaindex` 从加载路径退成构建期依赖

第 4 步是关卡：**如果规范化改变了发给模型的任何字节，就说明它动了不该动的东西。**

### 5.1 为什么把「自动回退」改成了「显式声明」

原计划第 3 步写的是「优先读 `normalized/`，回退到原始路径」。实施后验了一下，
这个设计**把重构要消灭的静默失败原样搬到了新位置**：

```
把 stack_cubes/planning_2.jsonl 藏起来再加载
  → 照跑 300 题，不报错
  → 每题的图从 3 张变回 1 张，BC-16 悄悄失效
```

题数一样、不报错、分数变了 —— 这是最难发现的一种错。根子在于
**走了哪条路取决于文件在不在**，而两条路给出的东西已经不同了。

改成 `load_run_items(..., source="normalized" | "raw")`：

- `"normalized"`（默认，评测走这条）—— 缺失或过期直接抛 `StaleNormalized`，**不回退**
- `"raw"`（replay 回归走这条）—— 显式要求原始 QA，因为它的基线就是规范化之前的行为

### 5.2 过期检测（`check_freshness`）

计划 §3.2 说 `source_sha` 「让『数据更新了但没重建』可被检测」，
但当时只写了字段、没写检查。现在补上，查四件事：

| 查什么 | 抓什么场景 |
| --- | --- |
| manifest 在不在 | 从没构建过；clone 仓库直接跑（`normalized/` 是 gitignore 的） |
| `builder_version` 与代码一致 | 升级了构建器忘了重建 |
| 数据里有 manifest 没有的族 | 新族到货了（如 `Take_out_the_trash`） |
| 每个 run 的 jsonl 还在，且源 QA 指纹未变 | 数据更新了没重建；只重建了一部分 |

指纹同时记 `size`/`mtime_ns` 和 `sha256`：先比前两个（快），
不一致再比哈希。**重新下载了同样的文件不会误报过期，内容真变了一定抓得到。**

三个入口都接了：加载时自动查（每进程每数据根一次）、
`preflight` 单列一项、`tools/build_normalized.py --verify` 可放 CI。
preflight 里产物过期就跳过逐族数据检查 —— 否则同一条原因会在 56 格上各报一遍。

顺带修了一个相关缺陷：**`--family` 部分重建会把 manifest 里其他族抹掉**
（踩过一次）。现在部分重建并进旧 manifest；构建器版本不同时不并，
因为那种情况本来就该整体重建。

---

## 6. 风险与未决

| 风险 | 应对 |
| --- | --- |
| 规范化悄悄改变模型输入 | 第 4 步用 `test_request_equivalence` 逐字节验证 |
| `media[]` 的 role/view 推导错误 | 构建时统计各 role 的数量分布，与题目类型交叉核对 |
| 数据方后续再改布局 | `source_sha` + `builder_version` 让过期可检测 |
| 12.5% 媒体仍解析不到 | 进 manifest 的 issues，**不静默** |

**需要数据方回复才能推进的**（见现状文档第 6 节）：
`planning` 两套 schema 哪套为准、`left_eye` 与 `left_eye_compress` 的关系、
`in_image` 的语义、9 个 0.05 秒片段、`Take_out_the_trash` 的 understanding 何时上传。

这些不阻塞第 1~4 步 —— 构建器可以先按「保留两套、标注版本」实现。
