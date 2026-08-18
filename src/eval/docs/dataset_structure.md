# ROBOCHRONO 数据结构现状

> 日期：2026-08-16 ｜ 数据：`yyyyywv/ROBOCHRONO` 全量 64.7 GB / 42,122 文件（哈希全量校验通过）
> 目的：为重构做准备。**本文只描述现状，不含改动。**

---

## 0. 先看清有几层 —— 目录名会骗人

`datasets/QA/` 有 61 GB，很容易以为那是数据集本体。**它不是。**
那 61 GB 几乎全是**为出题而生成的素材**（切片、抽帧、蒙太奇、拼接视频）。

```
① 原始采集    LeRobot 机器人录像                              ✗ 本地没有
              observation.images.{left_eye,left_wrist,right_wrist}/…
                  │  QA 里的 source_video_paths 指向它
                  │  → 这就是 380 个「未解析媒体」的来源
                  ▼
② 分段标注    datasets/json/<族>/file-XXX_segments.json       ✓ 759 KB
              {id, start, end, start_frame, end_frame,
               objects, main_verbs, narration}
              ★ 七个任务的真值全部源自这里
                  │  数据方的出题流水线：切片 / 抽帧 / 拼接 /
                  │  套题干模板 / 调 LLM 生成干扰项
                  ▼
③ 题目+素材   datasets/QA/…                                   ✓ 61 GB
              *_vqa.json  题干、选项、答案、出题参数
              媒体        planning_clips/ planning_2_frames/
                          time_joined_videos/ montages/ option_images/…
                  │
                  ▼
④ 评测        robochrono/                                      ← 我们在这层
              渲染 prompt → 调模型 → 打分
```

**所以「数据」和「题目」本来就是分开的，只是分界线在数据方那边**，
而我们只拿到了 ②③ 两层。本地真正意义上的「数据」只有 759 KB 的标注。

这也解释了几件反直觉的事：

- 为什么 `source_video_paths` 永远解析不到 —— 它指向第①层，我们没有
- 为什么 `segments_path` 指向 `datasets/json/` —— 那是第②层，真值来源
- 为什么改不了「片段时长泄答案」—— 切片在第③层已经完成，重切要回到第①层

### 0.0 一个 QA 文件里混着五种东西

这才是真正该拆的地方（v2 的三分就是在拆它）：

| 混在一起的 | 属于 | v2 去处 |
| --- | --- | --- |
| `question` / `options` / `answer` | 题目 | `prompt` + `truth` |
| `option_design` / `seed` / `num_skipped` / `skipped_existing` | 出题流水线记账 | manifest / `provenance` |
| `segments_path` | 指回第②层 | `provenance` |
| `clip_path` / `image_paths` / … | 第③层素材 | `prompt.media` |
| `source_video_paths` | 指回第①层（本机无） | `provenance` |

---

## 0. 规模

| 族 | understanding | left_right | image_in_video | time | planning | planning_2 | step_order | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wash | 517 | 1034 | 517 | 517 | 477 | 517 | 40 | **3,619** |
| stack_cubes | 300 | 600 | 300 | 300 | 250 | 300 | 50 | 2,100 |
| tea | 234 | 468 | 234 | 234 | 195 | 234 | 39 | 1,638 |
| airpods | 200 | 400 | 200 | 200 | 160 | 200 | 40 | 1,400 |
| express | 200 | 400 | 200 | 200 | 150 | 200 | 50 | 1,400 |
| tea2 | 160 | 320 | 160 | 168 | 140 | 160 | 20 | 1,128 |
| pen_inbox | 150 | 300 | 150 | 150 | 100 | 150 | 50 | 1,050 |
| gift_inhand | 90 | 180 | 90 | 90 | 60 | 90 | 30 | 630 |
| **合计** | | | | | | | | **12,965** |

trajectory_2D / trajectory_3D 另有 6 个族有数据（gift_inhand、pen_inbox 没有），
但该指标已于 2026-08-16 暂时放弃，未计入上表。

**一处需要更正的早期判断：** 我曾报告「只有 4 个族有 QA，其余 4 个只有素材」。
那是错的 —— 那 4 个族的 QA **多嵌了一层子目录**，我只在族目录顶层找过。
实际题量是 6,028 的两倍多。

---

## 0.1 版本核验（2026-08-16）

远端仓库 `yyyyywv/ROBOCHRONO` 当前 `sha = 561add0d`，`lastModified 2026-08-16T08:50Z`。
我们的全量下载完成于 08-15。核验结果：

```
远端 43,872 文件      本地 42,140
本地缺少 1,750        全部属于新族 Take_out_the_trash（仅 planning/）
本地多出 18           全部是我们自己产生的，非污染
需补下载 0.17 GB
```

**已有 8 个族一个文件都没变** —— 08-13 到 08-16 之间的 7 个提交全部在加新族，
所以既有的分析、实验与回归基线**不受影响，不需要重来**。

本地多出的 18 个文件：9 个 `*.json.orig`（BC-08 路径规范化备份）、
8 个 `*.singleframe.jpg`（BC-13 单帧兜底产物）、1 个 `README-HF.md`。

### 新族 Take_out_the_trash：尚不可用

```
planning/Take_out_the_trash        1,866 文件  ✅
understanding/Take_out_the_trash   不存在      ❌
```

没有 understanding 就没有 understanding / left_right / image_in_video / time
四个任务。提交标题为 `Take_out_the_trash planning QA` 及 `(part 2)`~`(part 4)`，
且编号乱序（part 4 早于 part 2 推送），**上传应当仍在进行中**。
现在补下载可能拿到中间状态，建议等它完整后再取。

### 族命名也不一致

`Take_out_the_trash` 用了**下划线 + 大小写混合**，其余八个族全部小写
（`stack_cubes`、`gift_inhand`、`pen_inbox`、`tea2`…）。这会影响：
配置文件里的族名、结果目录名、报表里的列名。**重构时应当规范化为统一形式**
（建议全小写下划线：`take_out_the_trash`），并保留一张原名映射表。

---

## 1. 三个轴的不一致

### 1.1 QA 文件的深度：三种布局

| 布局 | 族 | QA 位置 |
| --- | --- | --- |
| **A 扁平** | airpods, express, stack_cubes, tea2 | 直接在族目录下 |
| **B 单层** | gift_inhand, pen_inbox | `planning/`、`time/`、`time_understanding/` |
| **C 分散** | tea, wash | 再拆一层：`step_order/`、`image_in_video/` 各自独立 |

具体：

```
A   QA/planning/stack_cubes/planning_vqa.json
B   QA/planning/gift_inhand/planning/planning_vqa.json
    QA/understanding/gift_inhand/time/time_vqa.json
    QA/understanding/gift_inhand/time_understanding/left_right_vqa.json
C   QA/planning/wash/step_order/step_order_vqa.json
    QA/understanding/wash/image_in_video/image_in_video_vqa.json
```

中间那一层的名字**不统一**：`planning` / `time` / `time_understanding` /
`image_in_video` / `step_order` 都出现过。

**后果：** 按固定布局拼路径会让 B、C 两类共 4 个族**整族静默消失** ——
matrix 只记一句「QA 文件缺失」然后跳过，漏掉 6,937 题。

### 1.2 媒体目录的分组：视角/episode 分子目录 vs 平铺

```
扁平族（airpods 等）
    planning_2_frames/left_eye/          按视角分
    planning_clips/file-000/             按 episode 分
    understanding_clips/file-000/
    image_in_video_vqa/option_images/

嵌套族（gift_inhand, pen_inbox）
    planning/planning_2_frames/          全部平铺，无子目录
    planning/planning_clips/
    time_understanding/understanding_clips/
```

### 1.3 视角子目录的命名：至少四套

| 目录 | 命名变体 |
| --- | --- |
| `trajectory_first_frames/` | `top, wrist_L, wrist_R`（airpods/express/tea2）｜ `left_eye, left_wrist, right_eye`（stack_cubes） |
| `planning_2_frames/` | `left_eye, left_wrist, right_wrist` ｜ `prejoined`（stack_cubes）｜ 无子目录 |
| `time_video_crop_top/` | `left_eye` ｜ `left_eye_compress` ｜ `left_wrist, right_wrist` |

`left_eye` 与 `left_eye_compress` 在 express、tea2 里**同时存在**，内容不同。

---

## 2. JSON 里的路径风格：四种

| 风格 | 例 | 族 |
| --- | --- | --- |
| 本机绝对路径 | `/mnt/public/users/wbcd/.../file-000.mp4` | stack_cubes（已跑过 `normalize_qa_paths.py`） |
| 相对路径 | `time_video_crop_top/left_eye/file-000.mp4` | airpods, express |
| 生成机绝对路径 | `/ssd/yyywv/workflow_outputs/.../file-000.mp4` | tea2 |
| **Windows 反斜杠** | `workflow_outputs\time_understanding\...` | wash, pen_inbox |

现有的 `tools/normalize_qa_paths.py` 是**按前缀重写**的，处理不了
`left_eye` → `left_eye_compress` 这类子目录改名。

### 2.1 媒体解析实测

按「原路径尾部逐级加长」在族目录内消歧后：

```
39,733 个媒体引用（已排除 145,053 个溯源字段）
    原本就能解析     5,120   （只有 stack_cubes）
    按文件名解析回来  29,664
    重名未解析        3,049
    找不到            1,900
    可用率           87.5%
```

**溯源字段**（`original_video_path`、`source_video_paths`、`left_eye`、
`right_wrist` 等）指向生成机上的原始素材，从未随数据集发布，评测也不读它们。
它们曾让 stack_cubes 显示 150 个「缺失」，实为虚警。

---

## 3. item 字段结构：主输入一致，附加字段有差异

> ⚠️ **本节曾被我写错。** 早先报告「`planning` 有两套不兼容的结构」，
> 那是过度解读 —— 逐字段核对后，**评测实际使用的主输入字段八族完全一致**，
> 差异只在附加字段上。以下是更正后的准确描述。

顶层字段在八个族之间一致。`understanding` 与 `step_order` 零差异。
三个任务有附加字段差异：

| 任务 | 主输入（八族一致） | stack_cubes 独有 | 其余七族独有 |
| --- | --- | --- | --- |
| `planning` | `clip_path`, `clip_paths` | `clips`, `prejoined_video_path`, `end_frame`, `start_frame`, `skipped_existing`, `source_video_path` | `joined_clip`, `view_order`, `source_video_paths` |
| `planning_2` | `image_path`, `image_paths`, `images` | `prejoined_video_path` | — |
| `time` | `video_paths` | `time_view` | `joined_video`（wash/tea/gift_inhand/pen_inbox） |

**stack_cubes 由较早版本的流水线产出**，这一点仍然成立（附加字段的差异是证据），
但它**不影响评测取到的输入** —— 主输入字段名一致。

### 3.1 `planning` 与 `planning_2` 是两种不同的任务，不是两个版本

数据方确认：两者都要测。

```
planning     输入上一段动作的视频，预测下一动作      input 全是 mp4
planning_2   输入帧（图片），预测下一动作            input 含 image_path / frame_index
```

实测八族一致：`planning` 的 input 只有视频，`planning_2` 有 7 张图 + 7 段视频。

### 3.2 time 的视频源有三种，不是两种

```
stack_cubes / airpods / express        time_video_crop_top/left_eye/
tea2                                   time_video_crop_top/left_eye_compress/
gift_inhand / pen_inbox / tea / wash   *_full_time_joined_views.mp4   ← 多视角拼接，不走 crop_top
```

后四族用的是**多视角拼接视频**，与前四族不是一类输入。

### 3.3 ⚠️ 输入规格沿「族」系统性分裂成两批

全族逐任务核对媒体规格后发现：**八个族分成两组，每个任务都是同一条分界线。**

| 任务 | A 组（stack_cubes / tea / wash） | B 组（airpods / express / tea2 / gift_inhand / pen_inbox） | 像素比 |
| --- | --- | --- | ---: |
| understanding | 2880×540 | 1920×480 | 1.7× |
| left_right | 960×540 | 640×480 | 1.7× |
| image_in_video | 960×486 | 640×480 | 1.6× |
| time | 960×486 | 640×480 | 1.6× |
| planning | 2880×540 | 1920×480 | 1.7× |
| step_order | 960×**222** | 960×**282** | **0.8×** |

前五个任务 A 组都更大（1.6~1.7 倍像素），只有 step_order 反过来。
单视角尺寸是 960×540（A）与 640×480（B），三视角拼接后成为 2880 与 1920。

**这不是偶发，是两批数据用了不同的采集分辨率。** 跨族分数因此不可直接比较，
且影响全部七个任务 —— 我们此前只在 stack_cubes 上验证，完全看不到这一层。

#### 决定：A 组向 B 组对齐（2026-08-16，**已决策，尚未实施**）

数据方决定把 A 组降采样到 B 组规格，而不是反过来上采样 ——
上采样不会凭空造出信息，只会让「两组等价」这件事变成假象。

**成本**（全量扫描 A 组三族所得）：

| | 文件数 | 体积 | 降采样后估计 |
| --- | ---: | ---: | ---: |
| 视频 | 3,153 | 18.62 GB | |
| 图片 | 14,495 | 1.62 GB | |
| **合计** | **17,648** | **20.24 GB** | **≈ 12 GB** |

**实施前必须先解决的两点**（这也是这条还没动手的原因）：

1. **二次压缩。** A 组现有文件已经是有损编码，重新转码是第二代压缩。
   视频尤其明显 —— 3,153 个视频重编码会引入 B 组没有的压缩伪影。
   等于用「分辨率一致」换来了「编码代数不一致」。
   如果能从原始采集重新导出，就没有这个问题；只有本地转码这一条路时，
   应当在 manifest 里记下这是第几代编码。
2. **降采样 ≠ 原生。** B 组的 640×480 是相机原生采集，A 组降下来的 640×480
   是 960×540 重采样的结果 —— 后者带插值平滑，也会因宽高比不同
   （16:9 → 4:3）需要裁剪或加边。两者在像素数上一致，
   在**图像统计特性上并不一致**，模型对此是敏感的。
   所以对齐之后仍不能声称两组「输入等价」，只能说「视觉预算一致」。

结论：**这条对齐能消除「预算差 1.7 倍」这个最大的不可比因素，值得做**，
但不要把它写成「两组从此可比」。step_order 那条反向的（0.8×）要单独处理。

### 3.4 ⚠️ planning_2 的输入形态不同，不只是分辨率

```
stack_cubes   1 张图   2880×540           ← 三视角预拼接成一张
其余七族      3 张图   640×480 / 960×540   ← 三张独立图分别给出
```

**这是输入结构的差异，不是画质差异** —— 模型看到的东西组织方式就不一样。

> 这一条同时更正 3.2 节的一处判断。早先我写「附加字段不影响评测取到的输入」，
> 并据此认为 stack_cubes 独有的 `prejoined_video_path` 无关紧要。
> **那个判断是错的** —— 它正是预拼接流程的产物，而预拼接改变了 planning_2 的输入形态。

#### 决定：拆成三张（2026-08-16，**已实施** = BC-16）

构建期用 `jpegtran -crop` 在 DCT 系数层面切，逐比特无损（拼回去与原图
最大像素差 0），原始数据不动。300 题 → 900 张 960×540，与其余七族的
单视角尺寸一致。触发条件是「恰好一张图且宽 ≥ 高×3」，来自数据本身而非族名。

细节与验证方式见 [`REFACTOR_PLAN.md` 的 BC-16](../../REFACTOR_PLAN.md)。

---

## 4. 与文档不符之处

`test/testproject.md` 第 1.1 节写：

> 只对双手均参与的任务进行测试。当前双手任务覆盖：`pick cube`。

据此 `plan.json` 只给 stack_cubes 标了 `two_handed`。但实测**八个族的
left_right 全是左右各半的完整双手数据**：

| 族 | 题数 | `target_side` |
| --- | ---: | --- |
| wash | 1034 | left 517 / right 517 |
| stack_cubes | 600 | left 300 / right 300 |
| tea | 468 | left 234 / right 234 |
| （其余五族同样左右各半） | | |

照文档跑会静默丢掉 7 个族的 left_right，共 3,002 题。
已改为**从 `target_side` 字段推导**，文档与数据冲突时以数据为准。

---

## 5. 已知的数据缺陷

| 缺陷 | 规模 | 说明 |
| --- | --- | --- |
| 极短片段 | 9 个 0.05 秒 + 1 个 0.4 秒 | 文件名标注数秒，实际 0.05 秒；本地报 `nframes interval [2,1]`，API 报 `video too short` |
| 2D 真值出画 | 31% 的点 / 72% 的题 | 数据标了 `in_image: false`，但 eval 只读了 `valid`，同时 prompt 又说「画面外无效」并有重问机制 |
| 短于 2 秒的片段 | 10/300（stack_cubes image_in_video） | qwen API 拒收，已用 BC-13 补长处理 |

---

## 6. 向数据方确认的结果（2026-08-16 已回复部分）

| # | 问题 | 状态 |
| --- | --- | --- |
| 1 | `planning` 两套 schema 哪套为准 | ✅ **已澄清** —— 不是两套版本；`planning`（视频输入）与 `planning_2`（帧输入）是两种任务，都要测。附加字段差异不影响主输入 |
| 2 | `left_eye` vs `left_eye_compress` | ✅ **已答**：`_compress` 是原视频的压缩版，**只在测 time 时使用**。⚠️ 但实测 **express 的 time_vqa 引用的是 `left_eye` 而非压缩版**，与该口径不符，待确认是否应切换 |
| 3 | `in_image: false` 的语义 | ⏳ 待回复（trajectory 已搁置，重启前必须解决） |
| 4 | 10 个极短片段 | ⏳ 待回复（9 个切片异常 + 1 个源标注即为 50 毫秒） |
| 5 | `Take_out_the_trash` 的 understanding | ⏳ 待回复 |

## 7. 当前的应对（临时补丁，重构后应当移除）

| 补丁 | 位置 | 作用 |
| --- | --- | --- |
| QA 递归定位 | `tasks.qa_path` | 吸收三种深度布局；多处命中时报错而非静默取第一个 |
| 族属性推导 | `matrix.derive_family_attrs` | 从 `target_side` 推 `two_handed` |
| 媒体路径解析 | `robochrono/mediaindex.py` | 按文件名 + 尾部逐级消歧；溯源字段单独跳过 |
| 审计工具 | `tools/audit_media.py` | 只读，报告各族可用率 |

这些是为了让现有数据能跑起来。**重构成统一结构后，它们应当退化成一次性的
构建步骤，而不是每次运行时的解析逻辑。**
