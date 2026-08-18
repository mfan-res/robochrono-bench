# 数据格式 v2 设计草案

> 日期：2026-08-16 ｜ 状态：**待讨论**
> 前置：[数据结构现状](dataset_structure.md) ｜ [重构方案](dataset_refactor_plan.md)
>
> 与 v1（`normalized/`）的区别：v1 的目标是「不改变任何字节地吸收布局差异」，
> 所以它保留了原始 item 的全部结构。v2 放弃这个约束，**重排结构**。

---

## 0. 先说三个促成设计的实测发现

设计不是从审美出发的。下面三条是全族全量普查（12,965 题）查出来的事实。

### 0.1 ⚠️ 答案就摆在 `input` 里

`step_order` 的 `input.images[]` 每项带 `source_order`：

```jsonc
"input": {"images": [
  {"display_label": "Image 1", "display_index": 1,  "source_order": 8,
   "description": "wipe plate with rag", "segment_id": "file-000-8", "image_path": "..."},
  {"display_label": "Image 2", "display_index": 2,  "source_order": 7, ...}
]}
```

把 `display_index` 按 `source_order` 排序，得到

```
11-12-3-9-10-6-2-1-8-4-13-7-5
```

而该题的 `answer_order` 逐字符相同。**这就是答案本身。**

同理，`options[]` 里带着 `is_correct` 与 `distractor_type`（取值含 `"correct"`）：

| 任务 | 选项里的泄漏键 |
| --- | --- |
| understanding / planning / planning_2 | `distractor_type` |
| left_right / image_in_video | `distractor_type`, `is_correct` |
| step_order | `is_correct` |

**当前没有泄漏** —— 我逐任务验过，`parts()` 只取 `id`/`text` 和路径，
渲染出的 prompt 里不含这些键。但这是**约定**，不是**保证**：
字段名叫 `input`，任何人往 prompt 里多塞一点上下文就漏了，
而症状是分数变高，没有报错。

> 这是 v2 最主要的动机。不是为了好看。

### 0.2 一个东西五个名字，而同一个名字在不同任务里指不同东西

`wash/understanding` 的这五个字段指向**同一个文件**：

```
clip_path
clip_paths[0]
video_path
video_paths[0]
joined_clip.clip_path
```

但 `video_path` 这个名字在不同任务里含义不同：

| 任务 | `video_path` 指的是 | 是不是评测输入 |
| --- | --- | --- |
| understanding / planning | 那段拼接视频 | **是** |
| time | 拼接的全程视频 | **是** |
| planning_2 | 原始录像 | 否，是溯源 |
| step_order | 原始录像 | 否，是溯源 |

这正是 `mediaindex` 当初必须「按路径内容而不是字段名判断溯源」的原因 ——
那个补丁是在替这个命名问题擦屁股。

### 0.3 题面/答案的冗余 —— 三处里只有两处是真冗余

全族全量比对（不是抽样）：

| 断言 | 结果 |
| --- | --- |
| `Q` == `question` | 12,965 题全部相同 ✅ |
| `A` == `answer` == `correct_option.id` | 12,965 题全部相同 ✅ |
| `answer_text` == `correct_option.text` | **left_right 3,702 题、image_in_video 1,851 题全部不同** ❌ |

第三条**不成立**。这两个任务的选项是图，`text` 全为 `null`（只有「都不对」那项有文字）；
而 `answer_text` 是对正确答案的**文字描述**：

```
left_right       answer=F  answer_text="left gripper camera view"        correct_option.text=null
image_in_video   answer=F  answer_text="the option image that appeared…"  correct_option.text=null
understanding    answer=E  answer_text="pick the brush"                   correct_option.text="pick the brush"
```

它们是两个不同的东西，合并会丢掉真信息。

> 这条最初是我照 understanding 一个样本推的，推错了。
> 它是 §1 原则⑥ 的由来 —— **任何「去重合并」在全量验证之前都只是猜测。**

### 0.5 干扰项是 LLM 生成的，而且八个族用了两个不同的模型

`option_design`（族级字段）记录了选项构成：

```jsonc
{"num_options": 6, "correct": 1, "none": 1,
 "nearby_action_distractors": 2,          // 同视频其他真实动作标签，规则挑
 "generated_wrong_label_distractors": 2,  // ← LLM 生成
 "use_llm_distractors": true, "llm_distractor_model": "gemini-3.5-flash"}
```

全量核对 `distractor_type` 与之吻合。但生成模型按族分成两批：

| 生成干扰项的模型 | 族 | 候选标签数 |
| --- | --- | --- |
| `glm-5.2` | airpods(5) / express(4) / stack_cubes(6) / tea2(8) | 4–8 |
| `gemini-3.5-flash` | gift_inhand(3) / pen_inbox(3) / tea(6) / wash(10) | 3–10 |

两点影响：

1. **干扰项难度取决于哪个模型生成**，这是一个跨族混淆变量。
   且这条分界线**与 Q9 的分辨率分组不是同一条**（分辨率 A 组是 stack_cubes/tea/wash）
   —— 两个正交的混淆源。
2. **候选标签数从 3 到 10 差三倍多。** 只有 3 个候选动作的族，
   干扰项能挑的空间小得多，题天然更容易。

`left_right` / `image_in_video` 的干扰项不过 LLM（按规则挑图：
`scene` / `temporal` / `symmetric` / `same_video_other_category`），
`step_order` 的选项是排列，也是规则生成。

**这两条要问数据方**，见 [清单](questions_for_data_team.md) Q11。

### 0.6 选项与答案的分布是干净的

| 检查 | 结果 |
| --- | --- |
| 「都不对」选项的位置 | 六个位置均匀打散（如 understanding：A 340 / B 311 / C 299 / D 282 / E 312 / F 307） |
| 答案分布 | 六个字母均匀（每任务各字母约 1/6） |
| `text: null` 的选项 | 只出现在图选项任务，数量恰为 题数×5 |

没有「答案偏 A」之类的可利用偏置。

### 0.4 其余结构其实相当整齐

普查结果（全族全量），此前担心的「族间不一致」大多不存在：

| 任务 | 题数 | 选项数 | 带图选项 | 含「都不对」选项 | type 取值 |
| --- | ---: | --- | --- | --- | --- |
| understanding | 1,851 | 全 6 | 0 | 全部 | 单一 |
| left_right | 3,702 | 全 6 | 全 5 | 全部 | 单一 |
| image_in_video | 1,851 | 全 6 | 全 5 | 全部 | 单一 |
| planning | 1,532 | 全 6 | 0 | 全部 | 单一 |
| planning_2 | 1,851 | 全 6 | 0 | 全部 | 单一 |
| step_order | 319 | 全 6 | 0 | **无** | 单一 |

顶层与 `input` 字段的族覆盖也几乎全是 8/8，例外只有三处：

```
planning     stack_cubes 独有 clips / prejoined_video_path / start_frame / end_frame / skipped_existing
             其余七族独有 joined_clip / view_order / source_video_paths
planning_2   stack_cubes 独有 prejoined_video_path
time         joined_video 只有 4 族有；time_view 只有 1 族有
```

**结论：结构层面已经没有未知。** 剩下的都是内容问题（见第 5 节）。

---

## 1. 设计原则

**① 模型能看到的东西，集中在一个地方，且键是封闭白名单。**
不是「我们小心别把答案放进去」，而是「答案在结构上放不进去」。

**② 一个概念一个字段。** 五个名字指同一个文件，就留一个。

**③ 同名必同义。** `video_path` 不能在这个任务里是输入、在那个任务里是溯源。
用 `role` 区分用途，用字段位置区分是否输入。

**④ 不改题目内容。** 题干、选项文字、答案一个字不动 —— 只重排它们的位置。

**⑤ 每个字段的值都是原文的逐字节搬运，不做转述。**
v2 只允许四种操作，每种都可机械验证：

| 操作 | 例 | 怎么保证没走样 |
| --- | --- | --- |
| 原样搬运 | `question` → `prompt.question` | 值逐字节相等 |
| 改名 | `Q` → `prompt.question` | 同上 |
| 去重选一 | 五个路径字段 → `media[].path` | **先全量验等值**，见原则⑥ |
| 路径解析 | `..\a\b.mp4` → `/abs/a/b.mp4` | 解析前后指向的文件内容哈希相同 |

不允许的是「我觉得这两个字段是一回事，所以合并」——
那是我的解释进了数据，不是数据本身。

**⑥ 任何合并都要全量验证，且验证留成常驻测试。**
§0.3 是活的教训：我照一个样本推断 `answer_text == correct_option.text`，
全量一验，left_right 与 image_in_video 全部不等 —— 合并会静默丢掉真信息。
所以每一处去重都配一条断言，**验不过就两个字段都留着**。

已验过的（全族全量 12,965 题）：

```
Q == question                            0 处不等 ✅ 可合并
A == answer == correct_option.id         0 处不等 ✅ 可合并
clip_path == clip_paths[0] == video_path
  == video_paths[0] == joined_clip.clip_path   0 处不等 ✅ 可合并
      （planning 有 250 题无 joined_clip、time 有 868 题无 joined_video —— 是缺失，不是冲突）
answer_text == correct_option.text       5,553 处不等 ❌ 不可合并，两个都留
```

---

## 2. 结构

```jsonc
{
  "id": "file-000_step_order",
  "family": "wash",                    // 规范化后的全小写族名
  "task": "step_order",
  "group": "file-000",                 // 同组共享媒体（time 的 6 题一组），无组时 = id

  // ── 模型能看到的全部内容，仅此一处 ──────────────────────────
  "prompt": {
    "question": "...",                 // 原 Q / question（二者相同）
    "options": [                       // 只留 id 与 text，别的键一律不进
      {"id": "A", "text": "12-11-3-9-..."},
      {"id": "F", "text": "All other options are wrong."}
    ],
    "media": [                         // 有序 = 发送顺序
      {"role": "initial", "view": "multiview", "kind": "image", "path": "/abs/…initial.jpg"},
      {"role": "montage", "view": "multiview", "kind": "image", "path": "/abs/…montage.jpg"}
    ]
  },

  // ── 模型永远看不到 ────────────────────────────────────────
  "truth": {
    "answer": "F",                     // 原 A / answer / correct_option.id（已验三者全量相同）
    "answer_text": "11-12-3-9-...",    // 原 answer_text 原样。**不是** correct_option.text，
                                       // 二者在 left_right / image_in_video 里不同（§0.3）
    "option_text": null,               // 原 correct_option.text 原样，图选项任务里为 null
    "extra": {                         // 各任务特有的富真值，原样搬
      "answer_order": "11-12-3-9-10-6-2-1-8-4-13-7-5",
      "chronological_states": [...]
    }
  },

  // ── 溯源，不参与评测 ──────────────────────────────────────
  "provenance": {
    "source_video": {"left_eye": "/abs/…file-000.mp4", ...},
    "qa_file": "QA/planning/wash/step_order/step_order_vqa.json",
    "item_index": 0,                   // 与上一行合起来可精确定位回原文那一行
    "source_id": "file-000-1",
    "segment": {"start": 6.52, "end": 10.6, "start_frame": 163, "end_frame": 264},
    "derived": {"planning_2_split": "jpegtran-lossless"}    // 构建期做过什么
  }
}
```

**不存 `provenance.raw`（原文全文）。** 原始 QA 那 359 MB 完整躺在只读的
`datasets/QA/` 里，`qa_file` + `item_index` 就能定位回原文那一行，再抄一份没有意义。
原文里真正有价值的三样都已各就各位：`option_design` 进族级 manifest、
分段标注本来就单独在 `datasets/json/`、单题溯源字段就在上面这块。
剩下的全是那些指向同一个文件的冗余路径名。
```

### 2.1 白名单是可执行的，不是文档约定

构建器对 `prompt` 下的键做**封闭校验**，出现白名单以外的键就报错：

```python
PROMPT_KEYS  = {"question", "options", "media"}
OPTION_KEYS  = {"id", "text"}
MEDIA_KEYS   = {"role", "view", "kind", "path"}
```

于是 §0.1 的那类泄漏在结构上不可能发生 —— `source_order`、`is_correct`、
`distractor_type` 连进入 `prompt` 的位置都没有。

配一条测试：对每一题，把 `prompt` 整个序列化成文本，
断言其中不含 `truth` 里的任何标识（答案字母以选项形式出现除外）。

### 2.2 `media` 的 role / view

`role` 说明这个媒体在题里干什么，`view` 说明它是哪个相机。
每个任务的 role 组合是**声明式**的，构建时逐题核对，不符就报错：

| 任务 | 声明的 media 组合 |
| --- | --- |
| understanding | `clip`×1（view=multiview） |
| left_right | `context`×1（view=head）+ `option:A…E`×5 |
| image_in_video | `clip`×1 + `option:A…E`×5 |
| time | `clip`×1（整组共享） |
| planning | `clip`×1 |
| planning_2 | `frame`×3（view=left_eye / left_wrist / right_wrist） |
| step_order | `initial`×1 + `montage`×1 |

这张表就是「输入规格」的单一事实来源。§0.4 里 stack_cubes 那几处独有字段，
在这张表下自动消失 —— 它们描述的是同一个 `clip`，只是走了不同的字段名。

**顺带解决**：`left_right` / `image_in_video` 取选项图现在依赖 dict/list 迭代序，
改成由 `role: "option:A"` 显式绑定，顺序不再是隐含约定。

### 2.3 `group` 让 Time EQA 的组结构显式

`time` 是 6 题共用一段视频。现在这个分组是 `task.units()` 在运行时推的，
而已知的 P0 缺陷「一行坏了整组清零」正出在这里。
把 `group` 写进数据，分组就成了可检查的事实而不是推导结果。

---

## 3. 怎么保证「改对了」

旧的验收判据是「新旧两条路 `parts()` 输出逐字节相同」。v2 主动重排结构，
这条判据自然失效。替代的是三条，**都是可执行的**：

**① 迁移期一次性对照（做完即弃）。**
对全部 12,965 题，比对 v1 与 v2 各自产出的媒体**文件内容哈希序列**。
允许差异只有一处：`planning_2` 的 role/view 显式化不改变文件。
这不是永久约束，只是迁移当次的安全网 —— 跑一次，确认，然后删掉。

**② 白名单校验（永久）。** §2.1，构建期强制。

**③ role 覆盖校验（永久）。** §2.2 的声明表，逐题核对数量与组合。
数据方以后改了结构，这里会立刻报错而不是静默少发一张图。

**④ 字段等值断言（永久）。** 原则⑥ 那张表里的每一条，做成常驻测试。
不只是这次验一遍 —— 数据方以后让 `Q` 和 `question` 分了岔，
或者某族的 `clip_path` 不再等于 `video_path`，这里会立刻报错，
而不是让 v2 悄悄按其中一个继续构建。

再加上已有的 `check_freshness` 过期检测，五条合起来覆盖的是
「结构对不对、值有没有走样」而不是「字节变没变」—— 这正是重排结构之后该有的判据。

---

## 4. 与 v1 的关系

v1（`normalized/`）不删。它是「不改一个字节」的那一版，
留着当 §3① 的对照基准和历史结果的回查依据。
v2 写到 `datasets/v2/`，构建器共用同一套 QA 定位与媒体解析代码。

确认 v2 无误、且用 v2 重跑过一轮完整结果之后，再决定 v1 的去留。

---

## 5. 仍然未知的部分（都是内容问题，不阻塞本设计）

结构已无未知。以下六条在 [给数据方的清单](questions_for_data_team.md) 里等回复：

| # | 问题 | 对 v2 的影响 |
| --- | --- | --- |
| Q2 余项 | express 的 time_vqa 引用未压缩版 | 只换 `media[].path`，结构不变 |
| Q3 | 2D 轨迹 `in_image: false` 的语义 | 轨迹任务已搁置，v2 暂不含 |
| Q4 | 10 个时长异常的片段 | 只影响那 10 题 |
| Q5 | `Take_out_the_trash` 缺 understanding 组 | 新族到货后重建即可 |
| Q6 | 四族引用不存在的 `*_full_time_joined_views.mp4` | 380 个未解析路径，全在溯源字段，不影响输入 |
| Q8 | wash 的 time 用了未裁时间戳的三视角视频 | 只换 `media[].path` |
| Q9 | A/B 两组分辨率对齐 | 数据方处理中，只换文件不换结构 |

**待你拍板的设计问题**（下一节讨论）：

1. **轨迹任务（trajectory_2D / 3D）要不要进 v2？** 全局有 14 个 trajectory QA 文件。
   指标已判定无效并搁置，但数据还在。进 v2 = 保留重启的可能；不进 = 结构更干净。
2. **`provenance.raw` 存全文会让文件变大**（原始 QA 共 359 MB）。
   全存？还是只存 `qa_file` + `item_index`，需要时回原文件查？
3. **结果文件（`results/*.jsonl`）要不要一起改？** 你说「评测的格式也要大动」，
   但那是另一套结构，建议这一轮先定数据、下一轮定结果，避免两边同时动。
