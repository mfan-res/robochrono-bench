# 数据生产流程

> 日期：2026-08-17
> 依据：读 `data/*.py`（6,124 行）、`label/video_labeler_timestamp.py`（850 行）、
> `yyyyywv/egocentric` 原始集、八族全量 QA 元数据，加上抽帧实测。
> 标了**实证**的是看过画面或跑过统计的；标**推断**的是从代码或元数据推的。

---

## 总览

```
① 采集 ──▶ ② 标注 ──▶ ③ 出题 ──▶ ④ 评测
   录像        分段         切片+选项      打分
   .mp4       .json        _vqa.json     results
   .parquet   .txt         + 素材
```

每一步的产物都是下一步的输入，**而每一步都引入了一些下游不知道的东西**。
本文按阶段列出「产出什么」和「埋下了什么」。

---

## ① 采集

**产物**：LeRobot v3.0 格式，托管在 `yyyyywv/egocentric`（94 GB / 7,219 文件）。

```
<平台>/<族>/
├── videos/observation.images.<view>/chunk-000/file-XXX.mp4
├── data/chunk-000/file-XXX.parquet        机器人状态/动作轨迹
└── meta/{info,stats,tasks,episodes}
```

三个采集平台：`gimme` / `gripper` / `hand`，20+ 个族，本 benchmark 用了 8 个。

### 埋下的东西

**a. 时间戳被烧进画面，永久覆盖 RGB。**（实证）

```
2026-07-14 11:49:58.142464 | epoch_ns=1784000998142462163 |
```

顶部约 40 行黑条，带 epoch 纳秒 —— 多相机同步用。
`meta/info.json` 写明相机原生就是 `[540, 960, 3]`，**黑条在这 540 行之内**，
所以是覆盖画面而非外加边框。原始集里就有，任何地方都找不回被盖住的画面。

只有 `gripper` 平台那批有（wash / tea / stack_cubes）。抽帧逐一确认过八个族。

**b. 两套采集设备，参数不同。**（实证）

| | 族 | 分辨率 | fps |
| --- | --- | --- | --- |
| A 组 | wash / tea / stack_cubes | 960×540 | 25 / 20 |
| B 组 | airpods / express / tea2 / gift_inhand / pen_inbox | 640×480 | 20 / 30 |

分辨率差 1.6–1.7 倍，fps 有 20/25/30 三种。**这条分界线贯穿后续所有任务**，
也和「有没有时间戳」是同一条线。

**c. 视角命名至少四套。**

```
gripper   left_eye / left_wrist / right_eye / right_wrist
gimme     top / wrist_L / wrist_R
hand      cam_color / wrist_l / wrist_r
```

`right_eye` 这个第四视角只有 gripper 有，且七个任务一个都没用。

---

## ② 标注

**工具**：`video_labeler_timestamp.py`，交互式 OpenCV 界面。
按 README 的要求，**只标一个视角（第一视角）**，其余视角靠时间对齐。

**它不改视频** —— 只在 `cv2.imshow` 的预览上画字，输出只有 JSON。（实证：读了 850 行，
无 `VideoWriter`、无 `imwrite`）

**产物一：分段标注** `<族>/file-XXX_segments.json`

```jsonc
{"segments": [{"id": "file-000-1", "start": 6.52, "end": 10.6,
               "start_frame": 163, "end_frame": 264,
               "objects": ["brush"], "main_verbs": ["pick"],
               "narration": " Pick the brush."}]}
```

**七个任务的真值全部源自这里。** 本地有 320 份，八族齐全。

**产物二：类别词表** `<任务>.txt`，人手写，一个族一份，列出该任务可能出现的所有动作。

### 埋下的东西

**d. 两套词表，措辞不一致。**（实证）

| 来源 | 例 |
| --- | --- |
| `narration`（逐段打字） | ` Pick the airpods case.` → `pick the airpods case` |
| `categories.txt`（任务级手写） | `pick airpods case` |

**八族里 6 族不一致，差异 100% 集中在冠词 "the"。**
只有 stack_cubes 和 wash 恰好一致（那两份 txt 写全了 "the"）。

这个差异在阶段③会引爆，见 h。

**e. narration 到 answer_text 中间还有一层加工。**
tea2 出现 `pick up kettle` → `pick the up kettle` —— 像是机械地在第二个词前插 "the"，
不像人写的。具体在哪一步加工的没追。

**f. 标注可能多于成题。** tea2 有 21 份标注但只出了 20 个视频的题 ——
`file-020` 因缺某视角视频被跳过，记在 `num_missing_media_skipped`。

---

## ③ 出题

两个生成器，共 6,124 行：

| 脚本 | 产出任务 |
| --- | --- |
| `time_unders_workflow.py` | time, understanding, left_right, image_in_video |
| `planning_workflow.py` | planning, planning_2, step_order, trajectory |

### 3.1 媒体加工

按 segment 的 `start`/`end` 从原片切：

| 产物目录 | 内容 | 供 |
| --- | --- | --- |
| `understanding_clips/` | 单段动作的三视角拼接视频 | understanding |
| `planning_clips/` | 同上，另一套 | planning |
| `image_in_video_clips/` | 单视角片段 | image_in_video |
| `time_joined_videos/` | **全长**三视角拼接 | time / step_order |
| `time_video_crop_top/` | **全长**逐视角，裁掉顶部时间戳 | time |
| `planning_2_frames/` | 抽帧 | planning_2 |
| `option_images/` | 选项图 | left_right / image_in_video |
| `initial_images/` + `montages/` | 初始态图 + 结果态蒙太奇 | step_order |

**裁时间戳由 `crop_time_video_top` 控制，只对 A 组三族开** —— 精准，B 组本来就没有黑条。

**总量 61 GB**，其中 `time_joined_videos` 26 GB + `time_video_crop_top` 13 GB
是同一批内容的两种形态。**派生物约为原始录像的 4 倍。**

### 3.2 选项怎么来（六选一）

以 understanding 为例，每题 6 个选项：

```
1 个  correct                  = clean_text(narration)
2 个  nearby_action            = 同一视频里相邻动作段的 narration
2 个  generated_wrong_label    = 见下
1 个  none                     = "All other options are wrong."
```

`generated_wrong_label` 有**两个来源**，代码里是这样选的：

```python
generated_pool = llm_distractor_pool.get(clean_text(correct_text), [])   # ① 查 LLM 缓存
...
if len(generated_pool) < generated_count:                                 # ② 不够就规则兜底
    for text in generated_label_fallbacks(...):
        add_unique(generated_pool, text)
```

**① LLM 生成**：调外部大模型，按 `categories.txt` 里的每个标签生成 6 条候选，
缓存进 `llm_distractors.json`。

**② 规则兜底**：换动词 × 换宾语（`pick/place/move/put/remove/open/close` ×
该族出现过的所有物体），不够再加通用句（`hold the target object`、
`wait without manipulating any object` 等）。

**两种来源产出的选项标的是同一个 `distractor_type`。** 下游无法区分。

最后按 `md5(item_id|text)` 排序打乱 —— 确定性，同样输入必得同样顺序。

### 3.3 各任务的差异

| 任务 | 输入 | 选项 |
| --- | --- | --- |
| understanding | 1 段拼接视频 | 6 个文字选项 |
| time | 1 段全长视频，**一次问完该视频的全部动作**（3–13 题） | 无选项，答时间区间 |
| left_right | 头部相机图 + 5 张候选腕部相机图 | 图选项，`text` 为 null |
| image_in_video | 1 段视频 + 5 张候选图 | 同上 |
| planning | 1 段"上一动作"视频 | 6 个文字选项 |
| planning_2 | 3 张当前帧 | 6 个文字选项，题干含 `task_name` |
| step_order | 初始态图 + 结果态蒙太奇 | 6 个排列 |

step_order 的候选排列用 `random.Random(42)` 生成 —— 固定种子，确定性。

### 埋下的东西

**g. `question` 字段里已经嵌好了渲染后的选项。**（实证）

```
"Based on the current visual state, what should happen next?\nOptions:\nA. move the plate\nB. ..."
```

评测端三个任务直接用整串，另两个要 `_question_head()` 剥掉。契约没写在任何地方。

**h. ⚠ LLM 缓存的键与查表用的文本不匹配，六族静默退回规则生成。**（实证）

缓存以 **`categories.txt` 的标签**为键（`load_category_labels` → `clean_text`），
查表用的是 **`correct_text`（来自 narration）**。d 说的那个 "the" 差异，
在这里变成查不中：

| 族 | 键对得上 | 干扰项引用 | 命中缓存 | 命中率 |
| --- | --- | ---: | ---: | ---: |
| gift_inhand | ✗ | 180 | 0 | **0%** |
| pen_inbox | ✗ | 300 | 0 | **0%** |
| tea | ✗ | 468 | 0 | **0%** |
| wash | ✓ | 1,034 | 1,034 | **100%** |

要么全中要么全不中，正是键匹配与否的直接后果。tea 的例子：

```
缓存里存着   'close teapot lid' → ['align teapot lid', 'place lid on table', 'wipe teapot lid', …]
实际题目     answer_text = 'open the teapot lid'          ← 多了 "the"，查不中
实际用的     ['pick the teapot lid', 'put the teapot lid'] ← 规则拼的，换动词
```

**LLM 被调用了、结果被缓存了，然后一次都没用上。**

后果：**八族里至多两族（wash、可能还有 stack_cubes）真的用了 LLM 干扰项，
其余六族是规则生成的。** LLM 编的干扰项更贴近正确答案（"align teapot lid" vs
规则拼的 "put the teapot lid"），所以 **wash 的题实质上比其他族难**，
而这与模型能力无关。

这也更正了此前的判断 —— 曾以为跨族混淆来自「两个不同的 LLM」，
实际上更大的混淆是「用没用上 LLM」。

**i. 片段时长与答案相关。**（实证）

片段按 `[start, end]` 切，时长恰好等于 `end - start`。而动作类型和时长相关，
所以**只看片段多长就能把答案猜得远高于基线**：

| 任务 | 基线 | 只看时长 |
| --- | ---: | ---: |
| understanding（八族全为正） | 0.13–0.33 | 0.24–0.65 |
| planning（五族明显） | 0.17–0.50 | 0.32–0.76 |

这不是字段泄漏（删字段解决不了）—— 模型量一下视频长度就知道。
是**切片方式本身带来的捷径**。

**j. 缺失媒体静默跳过。** 找不到所需视频就跳过该段，记进
`num_missing_media_skipped`，不中断生成。好处是鲁棒，代价是题量少了不显眼。

**k. stack_cubes 的 planning_2 给的是一张预拼接宽图**，其余七族是三张独立图 ——
输入结构不同，不只是画质。（评测侧已用 BC-16 拆开）

---

## ④ 评测

`test/`（冻结）与我们的 `eval/`。这一层的问题另有记录（BC-01 ~ BC-16）。

---

## 确定性分析：什么能重现，什么不能

出题过程**没有用 `random` 模块**（step_order 除外，但它固定 `seed=42`）。
所有"随机"都是 `sorted(key=md5(item_id|text))`。

| 输入 | 可再生 | 说明 |
| --- | --- | --- |
| 原始录像 | 否 | 要重下；express 上游只有 3–5 集，拿不回 |
| 分段标注 | 否 | 要重标 |
| 类别 txt | 否 | 文件已丢，但内容存在 `option_design.category_labels` |
| LLM 干扰项缓存 | **否** | 重调 LLM 会得到另一套；且四族的缓存文件从未上传 |
| 生成配置 | 是 | 在 QA 顶层字段 + `data/*_config.json` |
| **61 GB 素材 + 全部题目** | **是** | 给定上面全部，逐字节可重现 |

**所以真正需要保管的只有前四项**，加起来不到 25 GB（其中录像 23 GB）。
现在守着的 61 GB 派生物，理论上是可以扔掉重建的 —— 前提是前四项齐全。

---

## 按阶段归位的问题清单

| # | 阶段 | 问题 | 能否补救 |
| --- | --- | --- | --- |
| a | 采集 | 时间戳烧进画面 | **不能**，画面已被覆盖 |
| b | 采集 | 两套设备，分辨率/fps 不同 | 只能下采样对齐，且非原生 |
| c | 采集 | 视角命名四套 | 加映射表 |
| d | 标注 | 两套词表措辞不一（"the"） | 统一措辞后重出题 |
| e | 标注 | narration 到 answer_text 有加工 | 需追查 |
| f | 标注 | 标注多于成题 | 已记录，正常 |
| g | 出题 | question 内嵌选项 | 结构化后可解 |
| **h** | **出题** | **LLM 缓存键不匹配，六族静默退回规则** | **修 d 后重出题** |
| i | 出题 | 片段时长泄答案 | 统一时长重切 |
| j | 出题 | 缺失媒体静默跳过 | 已记录 |
| k | 出题 | stack_cubes planning_2 预拼接 | 已用 BC-16 拆开 |

**h 是其中最严重的** —— 它不是数据不整齐，是**六个族的题目难度与设计意图不符**，
而且从头到尾没有任何报错。
