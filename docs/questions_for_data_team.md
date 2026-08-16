# 给数据方的问题清单

> 日期：2026-08-16 ｜ 提问方：评测框架（`eval/`）
> 数据版本：`yyyyywv/ROBOCHRONO` @ `561add0d`（2026-08-16T08:50Z）
>
> 背景：我们把全量数据（8 个族 12,965 题）接入自动化评测时，
> 发现若干需要你们确认的地方。**每条都标了「不确认会怎样」**，
> 方便你们判断优先级。

---

## Q1 · `planning` 的两种输入 —— 已答，无需改动

**你们的答复**：`planning` 与 `planning_2` 是**两种不同的任务，都要测**。

```
planning     输入上一段动作的视频，预测下一动作
planning_2   输入帧（图片），预测下一动作
```

**我们的核对结果与此一致**，且更正了我们先前的一处误判 ——
早先我们报告「planning 有两套不兼容的 schema」，逐字段核对后发现
**评测实际使用的主输入字段八族完全一致**（`planning` 用 `clip_path`/`clip_paths`，
`planning_2` 用 `image_path`/`images`）。差异只在附加字段上：

| 任务 | stack_cubes 独有 | 其余七族独有 |
| --- | --- | --- |
| `planning` | `clips`, `prejoined_video_path`, `end_frame`, `start_frame` | `joined_clip`, `view_order`, `source_video_paths` |
| `planning_2` | `prejoined_video_path` | — |
| `time` | `time_view` | `joined_video` |

这些附加字段不影响评测取到的输入。**此条无需你们改动**，
只是说明 stack_cubes 由较早版本的流水线产出。

---

## Q2 · `left_eye_compress` —— 已答，但 express 的引用与口径不符

**你们的答复**：`_compress` 是原视频的压缩版，**只在测 time 时使用**。

**余下的不一致**：实测八个族的 `time_vqa` 引用如下 ——

| 族 | `time_vqa` 引用的目录 | 本地存在 |
| --- | --- | --- |
| **express** | **`left_eye`（未压缩）** | `left_eye`, `left_eye_compress` 都有 |
| tea2 | `left_eye_compress` ✅ | 两者都有 |
| stack_cubes / airpods | `left_eye` | 只有 `left_eye`（没有压缩版） |

按「只在测 time 时用压缩版」的口径，**express 的 time_vqa 应当引用
`left_eye_compress` 而非 `left_eye`**，因为它两个版本都有。
stack_cubes / airpods 没有压缩版，无从选择。

**希望得到**：
1. express 的 time_vqa 是否应改为引用压缩版？
2. 其余族是否也要补压缩版？还是压缩只是为了解决 express/tea2 的特定问题
   （比如原视频过大），其他族不需要？

---

## Q3 · 2D trajectory 真值里的 `in_image: false` 是什么语义？

**现象**：`trajectory_qa_2d` 的真值点带两个独立标志：

```json
"right_gripper_uv": {"u": 979.206, "v": 524.960, "valid": true, "in_image": false}
"left_gripper_uv":  {"u": -172.766, "v": 104.369, "valid": true, "in_image": false}
```

统计：**31% 的点标着 `in_image: false`，涉及 72% 的题**。
画布是 960×540，而 u 的范围是 −447 ~ 979。

**冲突**：评测的 prompt 明确告诉模型
「`0 <= u < 960`，越界无效」，并且有一个重问机制会在模型给出越界点时打回重答。
也就是说 —— **我们一边把画面外的点当标准答案，一边惩罚给出画面外点的模型**。

**不确认会怎样**：越守规矩的模型在这 72% 的题上越吃亏。
（该指标已因其他原因暂时搁置，但重启前必须先解决这一条。）

**希望得到**：标 `in_image` 是希望评测方过滤掉这些点，还是仅作信息、
真值应当包含画面外的投影位置？

---

## Q4 · 10 个片段的实际时长与文件名标注严重不符（两种成因）

全量扫描 `planning_clips/` 下所有 mp4，短于 1 秒的共 10 个：

**成因一：切片生成异常（9 个，全在 `stack_cubes`）**
文件名标注数秒，实际只有 0.050 秒 —— 相差最多 313 倍。

```
0.050s  QA/planning/stack_cubes/planning_clips/file-002/file-002_file-002-3_42.250_57.900_prejoined.mp4
            标注 42.250→57.900 = 15.65 秒，实际 0.050 秒（313×）
0.050s  QA/planning/stack_cubes/planning_clips/file-049/file-049_file-049-3_31.650_39.200_prejoined.mp4
            标注 7.55 秒，实际 0.050 秒（151×）
0.050s  QA/planning/stack_cubes/planning_clips/file-049/file-049_file-049-1_10.000_17.900_prejoined.mp4
0.050s  QA/planning/stack_cubes/planning_clips/file-000/file-000_file-000-3_file-000-4_move_58.000_63.950_prejoined.mp4
0.050s  QA/planning/stack_cubes/planning_clips/file-002/file-002_file-002-3_file-002-4_move_56.900_60.500_prejoined.mp4
0.050s  QA/planning/stack_cubes/planning_clips/file-049/file-049_file-049-1_file-049-2_move_16.900_20.650_prejoined.mp4
0.050s  QA/planning/stack_cubes/planning_clips/file-049/file-049_file-049-2_19.650_23.650_prejoined.mp4
0.050s  QA/planning/stack_cubes/planning_clips/file-049/file-049_file-049-3_38.200_42.500_prejoined.mp4
0.400s  QA/planning/stack_cubes/planning_clips/file-046/file-046_file-046-3_file-046-4_move_38.350_42.900_prejoined.mp4
            标注 4.55 秒，实际 0.400 秒（11×）
```

**成因二：源标注的动作段本身就极短（1 个，`pen_inbox`）**

```
0.050s  QA/planning/pen_inbox/planning/planning_clips/file-037/file-037_file-037-2_13.600_13.650_joined_views.mp4
            标注 13.600→13.650 = 0.05 秒，与实际一致 —— 切片没问题，是上游动作分段给出了 50 毫秒的窗口
```

**后果**：本地模型报 `nframes should in interval [2, 1]`，
API 报 `The video modality input does not meet the requirements: video file is too short`。
这些题对所有模型都是硬失败。我们加了单帧兜底让流程不中断，
但那救不回不存在的画面。

**希望得到**：
1. 前 9 个能否重新切片？（看起来是生成流水线的问题）
2. 最后 1 个 —— 50 毫秒的动作段是否应当从数据集里剔除？

## Q5 · `Take_out_the_trash` 只上传了 planning，understanding 何时到位？

**现象**：08-16 的 7 个提交新增了这个族，但：

```
planning/Take_out_the_trash        1,866 文件  ✅
understanding/Take_out_the_trash   不存在      ❌
```

没有 understanding 组，就没有 understanding / left_right / image_in_video / time
四个任务。提交标题为 `… planning QA` 及 `(part 2)`~`(part 4)`，
且推送顺序乱序（part 4 早于 part 2），看起来仍在进行中。

另外这个族名用了**下划线 + 大小写混合**，其余八个族全部小写
（`stack_cubes`、`gift_inhand`、`pen_inbox`…）。族名会出现在配置、
结果目录、报表列名里。

**希望得到**：understanding 组的预计时间？族名能否统一为全小写
（`take_out_the_trash`）？

---

## Q6 · 四个族引用了不存在的 `*_full_time_joined_views.mp4`

**现象**：`stack_cubes`、`tea2`、`airpods`、`express` 的 QA 在
`input.video_path` / `video_paths` / `clip_path` 里引用了形如

```
file-000/file-000_file-000_0.000_full_time_joined_views.mp4
```

的文件，但**这四个族的数据里根本没有这类文件** —— 本地没有，
远端仓库也没有。核对远端全量文件表，`full_time_joined_views`
只存在于另外四个族：

```
gift_inhand   90 个      tea    195 个
pen_inbox    150 个      wash   200 个
```

**推测**：扁平族的 QA 生成时套用了 nested 族的模板，引用了它们没有的文件类型。

**不确认会怎样**：这四个族约 380 个唯一路径（数千次引用）解析不到。
好在评测实际用的主输入（`clip_path` 指向 `planning_clips/`、
`video_paths` 指向 `time_video_crop_top/`）是另外的字段，
所以**当前不阻塞评测** —— 但它让「媒体可用率」这个健康指标失真。

**希望得到**：这些引用是多余的（可以忽略），还是这四个族也应当有拼接视频？

---

## Q7 · 两份 `time_joined_videos` —— 已答，是有意设计

**你们的答复**：`step_order` 的版本裁掉了视频顶部的时间戳条，
防止模型按时间顺序作弊；其他位置保留原始时间戳。

**我们已验证该区分真实存在**（以 `wash/file-000` 为例）：

| 目录 | 分辨率 | 体积 | 时间戳 |
| --- | --- | ---: | --- |
| `planning/`、`time_understanding/` | 2880×**540** | 102 MB | 保留 |
| `step_order/`、`image_in_video/` | 2880×**486** | 76 MB | **已裁** |
| `time/` | 960×486 | 25 MB | 已裁 + 单视角 |

`image_in_video` 也是裁过的（你们只提了 step_order，数据里两者一致）。

**评测侧已按原路径忠实解析**，不做猜测 —— QA 引用哪个目录段就取哪一份。
**此条无需你们改动。** 但由此发现了 Q8。

---

## Q8 · wash 的 time 任务用了**未裁时间戳**的三视角视频，其余七族都不是

**背景**：感谢你们澄清 Q7 —— `step_order/time_joined_videos/` 是裁掉了
视频顶部时间戳条的版本，防止模型读数作弊；其他位置保留原始时间戳。
我们已验证这个区分真实存在：

| 目录 | 分辨率 | 时间戳 |
| --- | --- | --- |
| `planning/`、`time_understanding/` | 2880×**540** | 保留 |
| `step_order/`、`image_in_video/` | 2880×**486** | **已裁** |
| `time/` | 960×486 | 已裁 + 单视角 |

**问题在 time 任务上**。八个族的 `time_vqa` 实际拿到的视频：

| 族 | 分辨率 | 视角 | 时间戳 |
| --- | --- | --- | --- |
| stack_cubes、tea | 960×486 | 单 | 裁掉 ✅ |
| airpods、express、tea2、gift_inhand、pen_inbox | 640×480 | 单 | 裁掉 ✅ |
| **wash** | **2880×540** | **三视角拼接** | **保留** ❌ |

原因是两个 nested 族的 QA 引用了不同的目录段：

```
tea    的 time_vqa 引用   workflow_outputs/time/gripper/tea/time_joined_videos/…            → 裁过
wash   的 time_vqa 引用   workflow_outputs\time_understanding\gripper\wash\time_joined_videos\…  → 未裁
```

**不确认会怎样**，两层影响：

1. **可能作弊** —— time 任务问的正是「动作发生在第几秒」，而 wash 的画面上
   带着时间戳。这恰恰是你们裁剪 step_order 时要防的那种情况。
2. **输入不可比** —— 2880×540 三视角 vs 640×480 单视角，视觉预算差约 6 倍。
   wash 的 time 分数与其余七族不在同一基准上。

**希望得到**：wash 的 `time_vqa` 是否应改为引用 `time/time_joined_videos/`
（与 tea 一致）？看起来只需改一个路径段。

---

## Q9 · 八个族的采集分辨率分成两批，差 1.6~1.7 倍 —— 已决定：A 组向 B 组对齐

全族逐任务核对媒体规格后发现，**八个族分成两组，每个任务都是同一条分界线**：

| 任务 | A 组（stack_cubes / tea / wash） | B 组（其余五族） | 像素比 |
| --- | --- | --- | ---: |
| understanding | 2880×540 | 1920×480 | 1.7× |
| left_right | 960×540 | 640×480 | 1.7× |
| image_in_video | 960×486 | 640×480 | 1.6× |
| time | 960×486 | 640×480 | 1.6× |
| planning | 2880×540 | 1920×480 | 1.7× |
| step_order | 960×**222** | 960×**282** | **0.8×** |

前五个任务 A 组都更大，只有 step_order 反过来。跨族分数因此不可直接比较。

**决定**：A 组降采样到 B 组规格（不上采样 —— 上采样不造信息，只造等价的假象）。

**成本**：A 组 3,153 个视频（18.62 GB）+ 14,495 张图（1.62 GB）
= 17,648 文件 / 20.24 GB，降采样后约 12 GB。

**请你们确认两件事**：

1. **能否从原始采集重新导出，而不是本地转码？** A 组现有文件已是有损编码，
   我们本地转码就是第二代压缩 —— 3,153 个视频重编码会引入 B 组没有的伪影。
   等于用「分辨率一致」换来「编码代数不一致」。从源头导出就没有这个问题。
2. **step_order 那条反向的（A 960×222 vs B 960×282）怎么处理？**
   它和其余六个任务方向相反，按同一条规则对齐会把 B 组也动了。

另外要说明：对齐之后我们**不会**声称两组输入等价。B 组的 640×480 是相机
原生采集，A 组降下来的是 960×540 重采样（带插值平滑，且 16:9→4:3 还要
裁剪或补边）。像素数一致，图像统计特性不一致。能说的只是「视觉预算一致」。

---

## Q10 · planning_2 里 stack_cubes 是一张预拼接图，其余七族是三张 —— 已决定：拆成三张，评测侧已实施

```
stack_cubes   1 张图   2880×540           ← 三视角预拼接成一张
其余七族      3 张图   640×480 / 960×540   ← 三张独立图分别给出
```

**决定**：统一为三张分开。**评测侧已在构建期实现（BC-16），不需要你们改动数据。**

做法：`jpegtran -crop` 在 DCT 系数层面裁剪，不解码不重编码 ——
三张拼回去与原图**逐像素相同**（最大差 0）。300 题 → 900 张 960×540，
与其余七族单视角尺寸一致。原始数据只读不动，产物可随构建重建。

**仅供参考**：如果你们后续重新生成 stack_cubes，直接产出三张独立图即可，
我们这一层就可以去掉。触发条件写的是「恰好一张图且宽 ≥ 高×3」，
不是族名，所以数据换了之后行为是可预期的。

---

## Q11 · 干扰项由 LLM 生成，且八个族用了两个不同的模型

核对 `option_design`（族级字段）后确认，understanding / planning / planning_2
三个任务的每题 6 个选项构成是：

```
1 正确 + 2 nearby_action（同视频其他真实动作标签，规则挑）
       + 2 generated_wrong_label（LLM 生成）+ 1「都不对」
```

全量核对 `distractor_type` 与之吻合。但**生成模型按族分成两批**：

| 生成干扰项的模型 | 族（括号内为候选标签数） |
| --- | --- |
| `glm-5.2` | airpods(5)、express(4)、stack_cubes(6)、tea2(8) |
| `gemini-3.5-flash` | gift_inhand(3)、pen_inbox(3)、tea(6)、wash(10) |

**两个问题**：

1. **两个模型是有意的还是分批生成时换了？** 干扰项难度取决于生成模型，
   这会成为跨族分数的混淆变量。而且这条分界线**和 Q9 的分辨率分组不重合**
   （分辨率 A 组是 stack_cubes / tea / wash），所以是两个正交的混淆源。
   如果不是有意的，建议用同一个模型重新生成一遍干扰项。

2. **候选标签数从 3 到 10 差三倍多。** gift_inhand 和 pen_inbox 只有 3 个候选动作，
   干扰项能挑的空间小得多，题天然更容易。这是任务本身简单，还是标签集不完整？

另外想确认一件事：**`datasets/json/<族>/file-XXX_segments.json` 里的分段标注
（`start`/`end`/`objects`/`main_verbs`/`narration`）是人工标注还是模型产出？**
所有任务的真值都源自这里，我们需要在报告里写清楚真值的来源。

---

## 附：另外两处我们已自行处理，仅告知

**A. 八个族的 `left_right` 都是完整双手数据。**
`test/testproject.md` 写「只对双手均参与的任务测试，当前覆盖 pick cube」，
但实测八个族的 `target_side` 都是左右各半（wash 1034 题、tea 468 题…）。
我们已改为从数据推导，**不再依据该文档**。如果文档描述才是意图，请告知。

**B. QA 文件与媒体的目录布局在族之间不一致。**
QA 有三种深度（族目录下 / 多一层 / 多两层且分散），中间层名称不统一
（`planning`/`time`/`time_understanding`/`image_in_video`/`step_order`）；
媒体路径有四种风格（本机绝对 / 相对 / 生成机绝对 / **Windows 反斜杠**）；
视角子目录命名至少四套（`top,wrist_L,wrist_R` vs `left_eye,left_wrist,right_eye` vs `prejoined`）。

我们正在评测侧做一层规范化来吸收这些差异，**不需要你们改动**。
但如果后续生成能统一布局，这一层就可以去掉。
