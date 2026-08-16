# v2 字段归属表

> 日期：2026-08-16 ｜ 前置：[v2 设计草案](schema_v2_proposal.md)
> 状态：**待评审**。评审通过后，构建器照此实现，没有自由发挥余地。
>
> 覆盖原始数据里全部 **64 个不同字段名**（顶层 38 + `input` 36），
> 八族 12,965 题全量核对。每条都有**依据**，不是按字段名猜的。

---

## 0. 四个去处

| 去处 | 含义 | 模型能看到 |
| --- | --- | --- |
| `prompt` | 发给模型的全部内容，键是封闭白名单 | **是** |
| `truth` | 答案，以及**任何能反推答案的东西** | 否 |
| `provenance` | 溯源与题目参数，既不是答案也不发送 | 否 |
| 丢弃 | 与另一字段全量等值的冗余 | — |

**分界线是「能不能反推答案」，不是「看起来像不像答案」。**
下面有五个字段名字完全不像答案，但实测能反推，因此归入 `truth`。

---

## 1. 会泄答案的字段（必须离开 `prompt`）

这是本表最要紧的一节。这些字段现在都躺在 `input` 或 `options` 里 ——
**当前没有泄漏**（`parts()` 只取 `id`/`text` 和路径），但那是约定不是保证。

| 字段 | 任务 | 怎么反推出答案 | 实测 |
| --- | --- | --- | --- |
| `input.images[].source_order` | step_order | `display_index` 按 `source_order` 排序 = `answer_order` | 逐字符相同 |
| `input.start` / `input.end` | time | == `answer_seconds` | 1,859 题全部相等 |
| `input.start` / `input.end` | image_in_video | == `answer_seconds` | 1,851 题全部相等 |
| `input.start` | **understanding / planning** | 绝对时间位置几乎唯一决定动作标签 | 见 §1.1，准确率最高到 **1.000** |
| `target_side` | left_right | 唯一决定正确选项的 `view`（left→left_wrist、right→right_wrist） | 3,702 题无例外 |
| `options[].is_correct` | left_right / image_in_video / step_order | 直接标出正确项 | — |
| `options[].distractor_type` | understanding / planning / planning_2 / left_right / image_in_video | 取值含 `"correct"` | — |
| `chronological_states` | step_order | 正确时序状态序列 | — |
| `evidence` / `metadata` | time | 含 `start`/`end`、`original_start`/`original_end` | — |
| `next_source_id` / `next_time_eqa` | planning / planning_2 | 「下一动作」正是所问 | — |
| `source_time_eqa` | 五个任务 | 含本片段的时间真值 | — |

> 这就是为什么 `prompt` 要用**封闭白名单**而不是黑名单：
> 上面这张表我能列出来，是因为逐字段验过；下一个字段进来时未必有人验。
> 白名单让「没验过的字段进不了 prompt」成为默认。

### 1.1 `start` / `end` 在 understanding / planning 里也泄漏 —— 一次被推翻的推理

**我原先的判断是错的。** 当时写的是：

> 片段本来就是按 `[start, end]` 切出来的，模型看到的视频就是这一段，
> 告诉它绝对时间没有给出视频里看不到的信息 —— 归 `provenance`。

这只排除了「直接读出答案」，没排除**统计相关**。实测方法：
只用 `start`（或时长）这一个数字去猜答案标签，按 `video_id` 分组切分
（同一视频不跨训练/测试，避免循环排列造成的假象），训练集按分位数分 8 箱，
每箱取最常见标签，与「永远猜最常见标签」的基线比。三个随机种子取均值。

**understanding**

| 族 | 基线 | 只看时长 | 增益 | 只看起始位置 | 增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| airpods | 0.200 | 0.354 | +0.154 | **0.821** | +0.621 |
| express | 0.250 | 0.646 | +0.396 | **0.807** | +0.557 |
| gift_inhand | 0.333 | 0.578 | +0.244 | **0.933** | +0.600 |
| pen_inbox | 0.333 | 0.625 | +0.292 | **0.868** | +0.535 |
| stack_cubes | 0.167 | 0.483 | +0.316 | 0.344 | +0.177 |
| tea | 0.167 | 0.615 | +0.449 | **0.782** | +0.615 |
| tea2 | 0.125 | 0.236 | +0.111 | 0.604 | +0.479 |
| wash | 0.155 | 0.341 | +0.185 | 0.390 | +0.235 |

**planning**

| 族 | 基线 | 只看时长 | 增益 | 只看起始位置 | 增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| airpods | 0.250 | 0.321 | +0.071 | **0.981** | +0.731 |
| express | 0.333 | 0.757 | +0.424 | 0.812 | +0.479 |
| gift_inhand | 0.500 | 0.550 | +0.050 | **0.967** | +0.467 |
| pen_inbox | 0.500 | 0.667 | +0.167 | **1.000** | +0.500 |
| stack_cubes | 0.200 | 0.483 | +0.283 | 0.371 | +0.171 |
| tea | 0.200 | 0.626 | +0.426 | 0.779 | +0.579 |
| tea2 | 0.143 | 0.175 | +0.032 | 0.579 | +0.437 |
| wash | 0.168 | 0.317 | +0.149 | 0.406 | +0.238 |

`start` 在 pen_inbox 的 planning 上达到 **1.000** —— 完全确定。
原因不难理解：这些任务是**脚本化**的，每次录制里同一个动作发生在
大致相同的时刻，所以「第几秒」几乎就等于「在做哪一步」。

**结论：`start` / `end` / `start_frame` / `end_frame` / `frames` 一律归 `truth`，
所有任务无例外。** 不再按任务分流 —— 分流规则要靠逐条论证维持，而这次论证就错了。

### 1.2 但时长这一条不是字段问题，是**数据本身的捷径**

核实过：片段的实际时长**恰好等于** `end - start`（ffprobe 抽验，差 0.000）。

也就是说 —— **模型不需要这个字段就能拿到时长，量一下视频就有。**
把 `start`/`end` 藏进 `truth` 解决不了它。

| 泄漏 | 藏字段能解决吗 |
| --- | --- |
| `start`（绝对位置） | **能** —— 片段里看不出它在原视频的哪一段 |
| `end - start`（时长） | **不能** —— 就是视频长度，删字段也还在 |

时长带来的增益在 understanding 上是 +0.11 ~ +0.45（八族全部为正）。
**这意味着 understanding / planning 的分数里，有一部分来自「片段多长」
而不是「视频里在做什么」。**

需要说明的是，零样本 VLM 不会先验地知道「4 秒 = pick、7 秒 = wipe」这套映射，
所以这不是它今天就能利用的漏洞。但它是一条真实存在的捷径：
任何在本 benchmark 上做过拟合/微调的模型都能吃到，
而且它说明**片段时长与答案相关**，这一点在报告里必须写明。

**这条超出了 v2 的范围**（v2 只重排结构，不改切片），已单列为待办：
考虑把片段统一补/截到相同时长，或在报告里给出「时长基线」作为对照。

---

## 2. 顶层 38 个字段

| 字段 | 出现于 | 去处 | 依据 |
| --- | --- | --- | --- |
| `id` | 全部 7 | `id` | 唯一标识，全族无重复（已验） |
| `Q` | 全部 7 | **丢弃** | `== question`，12,965 题 0 处不符 |
| `question` | 全部 7 | `prompt.stem` + `prompt.options` | 字符串里已内嵌选项；拆成题干+结构化选项后可**逐字节还原**（understanding/planning/planning_2/step_order 全部还原成功）。left_right/image_in_video 只发题干，选项是图 |
| `A` | 全部 7 | **丢弃** | `== answer`，0 处不符 |
| `answer` | 全部 7 | `truth.answer` | — |
| `answer_text` | 全部 7 | `truth.answer_text` | **不可**与 `correct_option.text` 合并：left_right 3,702 + image_in_video 1,851 题不等（图选项的 text 为 null，而 answer_text 是文字描述） |
| `correct_option` | 5 个选择任务 | `.id` 丢弃；`.text` → `truth.option_text`；其余键 → `truth.option_meta` | `.id == answer` 10,787 题 0 处不符 |
| `options` | 6 个选择任务 | `id`/`text` → `prompt.options`；**其余键 → `truth.option_meta`** | 其余键含 `is_correct`/`distractor_type`，见 §1 |
| `choices` | step_order | **丢弃** | `== {options[].id: text}`，319 题 0 处不符 |
| `type` | 全部 7 | **丢弃** | 与任务名一一对应（left_right 的值是 `left_right_gripper_view`，映射固定），由 `task` 决定 |
| `video_id` | 全部 7 | `provenance.video_id` | — |
| `source_id` | 全部 7 | `provenance.source_id` | — |
| `input` | 全部 7 | 容器 | 内部字段见 §3 |
| `answer_action` | understanding/time/planning/planning_2 | `truth.extra.action` | 答案的动词部分 |
| `answer_objects` | 同上 | `truth.extra.objects` | 答案的宾语部分 |
| `answer_category` | image_in_video | `truth.extra.category` | — |
| `answer_seconds` | image_in_video/time | `truth.extra.seconds` | — |
| `answer_order` | step_order | `truth.extra.order` | — |
| `chronological_states` | step_order | `truth.extra.states` | 泄答案，见 §1 |
| `num_choices_to_order` | step_order | `provenance.params.num_choices` | 只是数量，不泄答案 |
| `evidence` | time | `truth.extra.evidence` | 含 start/end，见 §1 |
| `metadata` | time | `truth.extra.metadata` | 含 original_start/end，见 §1 |
| `target_side` | left_right | `truth.extra.target_side` | 泄答案，见 §1 |
| `timestamp` | left_right | `provenance.timestamp` | 只在顶层有（`input` 无同名字段），无合并对象 |
| `timestamp_key` | left_right | `provenance.timestamp_key` | 同上 |
| `next_source_id` | planning/planning_2 | `truth.extra.next_source_id` | 泄答案，见 §1 |
| `next_time_eqa` | planning/planning_2 | `truth.extra.next` | 同上 |
| `source_time_eqa` | 5 个任务 | `truth.extra.source_time_eqa` | 同上 |
| `task_name` | planning_2 | **丢弃** | `== input.task_name`，1,851 题 0 处不符；且其值已内嵌在 `question`（"The overall task is wash. …"） |
| `task` | step_order | `provenance.params.variant` | 取值 `step_order_with_initial_state` |
| `segments_path` | step_order | **丢弃** | `== input.segments_path`，0 处不符 |
| `image` | step_order | **丢弃** | `== input.image`，0 处不符 |
| `initial_image` | step_order | **丢弃** | `== input.initial_image`，0 处不符 |
| `images` | step_order | **丢弃** | `== input.images`，0 处不符（`input.images` 归 `truth`，见 §3） |
| `video_path` | step_order | **丢弃** | `== input.video_path`，0 处不符 |
| `video_paths` | step_order | **丢弃** | 同上，归入 `provenance.source_video` |
| `view` | image_in_video | `provenance.view` | 单视角标识 |
| `views` | step_order | **丢弃** | `== input.views`，0 处不符 |

---

## 3. `input` 的 36 个字段

### 3.1 媒体路径 —— 归 `prompt.media`，但只留一个名字

**五名一物**（understanding / planning）。全量验证 0 处不等值：

```
clip_path == clip_paths[0] == video_path == video_paths[0] == joined_clip.clip_path
```

| 字段 | 出现于 | 去处 | 依据 |
| --- | --- | --- | --- |
| `clip_path` | understanding/image_in_video/planning | `prompt.media[role=clip].path` | 保留这一个 |
| `clip_paths` | 同上 | **丢弃** | `[0] == clip_path`，0 处不符 |
| `video_path` | 6 个任务 | understanding/image_in_video/planning/time → **丢弃**（== clip_path）；planning_2/step_order → `provenance.source_video`（是原始录像，不是输入） | 同名不同义，见草案 §0.2 |
| `video_paths` | 6 个任务 | 同上 | — |
| `joined_clip` | understanding/planning | **丢弃** | `.clip_path == clip_path`；planning 有 250 题无此字段（缺失，非冲突） |
| `joined_video` | time | **丢弃** | `.clip_path == video_path`；868 题无此字段 |
| `videos` | time | `provenance.per_view_video` | 各视角裁顶视频，**不是**评测输入（评测用拼接版） |
| `image_path` | left_right/planning_2/step_order | left_right → `prompt.media[role=context]`；planning_2 → `prompt.media[role=frame]`；step_order → `prompt.media[role=montage]` | 三个任务里语义不同，靠 role 区分 |
| `image_paths` | planning_2/step_order | planning_2 → `prompt.media[role=frame]`×3；step_order → **丢弃** | planning_2：`[0] == image_path == images{}[].image_path`（1,851 题 0 处不符）。step_order：`== [initial_image, image]`，0 处不符 |
| `images` | planning_2/step_order | planning_2 → **丢弃**（同上）；**step_order → `truth.extra.state_order`** | step_order 的 `images` 是含 `source_order` 的元数据列表，**泄答案**，见 §1 |
| `image` | step_order | **丢弃** | `== image_path`，319 题 0 处不符 |
| `initial_image` | step_order | `prompt.media[role=initial]` | — |
| `head_image` | left_right | **丢弃** | `.image_path == image_path`，3,702 题 0 处不符 |
| `clips` | planning（仅 stack_cubes） | **丢弃** | 预拼接流水线残留，其 `clip_path` 与主字段同值 |
| `prejoined_video_path` | planning/planning_2（仅 stack_cubes） | `provenance.prejoined` | 非评测输入 |
| `views` | step_order | `prompt.media[].view` 的取值来源 | `== view_order`（其余任务用后者） |
| `view_order` | understanding/time/planning | `prompt.media[].view` 的取值来源 | — |
| `time_view` | time（仅 1 族） | `provenance.view` | — |

### 3.2 时间与帧 —— 按是否泄答案分流

| 字段 | 出现于 | 去处 | 依据 |
| --- | --- | --- | --- |
| `start` / `end` | **全部 4 个任务** | **`truth.extra.segment`** | time/image_in_video：== `answer_seconds`，全量相等。understanding/planning：`start` 单独就能把答案标签猜到 0.34~1.00（基线 0.17~0.50），见 §1.1 |
| `start_frame` / `end_frame` | image_in_video/planning | **`truth.extra.segment`** | 与 start/end 同信息，不同单位 |
| `frames` | understanding/image_in_video/planning | **`truth.extra.segment`** | `frames / fps` = 时长，同样可反推 |
| `fps` | understanding/image_in_video/planning | `provenance.clip_meta` | 采集帧率，按视频而非按动作变化，不携带单题答案信息 |
| `frame_index` | planning_2 | `provenance.frame_index` | 取帧位置，不泄「下一动作」 |
| `timestamp` / `actual_timestamp` / `timestamp_key` | planning_2 | `provenance.timestamp*` | 同上；与顶层同名字段**不重复**（planning_2 顶层无这些字段） |
| `crop_top_applied` / `crop_top_fraction` | image_in_video/time | `provenance.crop` | 是否裁掉了时间戳条（Q7）。**重要**：它记录了防作弊处理是否生效，必须保留 |

### 3.3 溯源路径 —— 全部归 `provenance`

| 字段 | 出现于 | 去处 | 说明 |
| --- | --- | --- | --- |
| `source_video_paths` | understanding/time/planning | `provenance.source_video` | 指向原始 LeRobot 录像，本机多数不存在 |
| `source_video_path` | image_in_video/planning | `provenance.source_video` | 单数形式，同义 |
| `original_video_paths` | time | `provenance.original_video` | — |
| `segments_path` | step_order | `provenance.segments_path` | 指向 `datasets/json/<族>/*_segments.json`，真值来源 |
| `task_name` | planning_2 | `provenance.task_name` | 其值已内嵌进 `question` |
| `skipped_existing` | planning（仅 stack_cubes） | **丢弃** | 生成流水线的内部标记，恒为 `true`，与题目无关 |

---

## 4. `prompt` 的最终形态

64 个字段里，**只有 3 类进 `prompt`**：

```jsonc
"prompt": {
  "stem": "Based on the current visual state, what should happen next?",
  "options": [{"id": "A", "text": "move the plate"}, …],     // text 可为 null（图选项任务）
  "media":   [{"role": "clip", "view": "multiview", "kind": "video", "path": "/abs/…"}]
}
```

白名单（构建期强制，出现别的键就报错）：

```python
PROMPT_KEYS = {"stem", "options", "media"}
OPTION_KEYS = {"id", "text"}
MEDIA_KEYS  = {"role", "view", "kind", "path"}
```

### 4.1 各任务的 media 声明表

| 任务 | media 组合 | 来源字段 |
| --- | --- | --- |
| understanding | `clip`×1 (multiview) | `input.clip_path` |
| left_right | `context`×1 (head) + `option:A…E`×5 | `input.image_path` + `options[].image_path` |
| image_in_video | `clip`×1 + `option:A…E`×5 | `input.clip_path` + `options[].image_path` |
| time | `clip`×1（整组共享） | `input.video_path` |
| planning | `clip`×1 | `input.clip_path` |
| planning_2 | `frame`×3 (left_eye/left_wrist/right_wrist) | `input.image_paths`（stack_cubes 经 BC-16 拆分） |
| step_order | `initial`×1 + `montage`×1 | `input.initial_image` + `input.image` |

构建期逐题核对数量与组合，不符报错。
**`option:A…E` 的绑定是显式的** —— 现在 `left_right` 取选项图依赖列表迭代序，
改为按 `role` 绑定后，顺序不再是隐含约定。

---

## 5. 汇总

| 去处 | 字段数 | 占比 |
| --- | ---: | ---: |
| `prompt` | 3 类（stem / options / media） | — |
| `truth` | 18 | 28% |
| `provenance` | 25 | 39% |
| 丢弃（已验证等值的冗余） | 18 | 28% |
| 容器（`input`） | 1 | — |

**18 个丢弃项，每一个都有全量等值验证支撑**，没有一个是「看起来一样所以丢」。
这些断言全部做成常驻测试（草案 §3④）—— 数据方以后让某两个字段分了岔，
构建器会报错，而不是悄悄按其中一个继续。

---

## 6. 评审时请重点看这几条

1. **§1 那十条泄漏字段的判定** —— 尤其 `target_side` 和 `input.start/end`，
   它们名字不像答案，是实测出来的。漏判一条，白名单就形同虚设。
2. **`answer_text` 与 `correct_option.text` 不合并** —— 这是我先前判错、
   全量验证纠正的一条。
3. **§1.2 的时长捷径** —— 这不是字段归属问题，删字段解决不了。
   understanding / planning 的分数里有一部分来自「片段多长」。
   **超出 v2 范围，但必须在报告里写明，或做统一时长处理。**
4. **`type` 丢弃** —— 假设它与任务名一一对应。八族全量成立，但新族可能带来新取值。

> **本表原有第 3 条已被推翻并删除。** 那一条声称
> understanding/planning 的 `start`/`end` 不泄漏，是全表唯一一条靠推理下的判断，
> 实测后发现错得很彻底（`start` 最高能把答案猜到 1.000）。
> 现已改为「一律归 `truth`，所有任务无例外」——
> 分流规则要靠逐条论证维持，而这次论证就错了。
