# Trajectory prompt 改造：设计、实验与结果

> 日期：2026-08-14 ｜ 数据：stack_cubes ｜ 每轮 40 题
> 目的：搞清楚 3D trajectory 分数为何集体趴在噪声带里，以及能否靠改 prompt 修好
>
> ## ⚠️ 结论（2026-08-16）：团队决定暂时放弃 trajectory_2D / trajectory_3D
>
> 本文记录的全部实验均未能让指标变得可用。保留它是为了让后来者知道
> **哪些路已经走过、为什么不通**，而不是作为可采纳的方案。
>
> 除本文的 prompt 实验外，另有两项决定性证据：
> - 3D：15 个模型中 11 个跑不赢盲基线；完整相似变换校准后最好也只有 4.50/100
> - 2D：真值 31% 的点标着 `in_image: false`（画面外），涉及 72% 的题，
>   而 prompt 告诉模型画面外无效并有重问机制 —— 答对的模型反被惩罚

---

## 0. 起因

3D trajectory 上 **879 条预测里 512 条（58%）是逐字抄回 prompt 里的示例点**：

| 模型 | 2D 抄示例 | 3D 抄示例 |
| --- | ---: | ---: |
| SenseNova-SI-1.1-InternVL3-2B | 0% | **92%** |
| Qwen3-VL-8B-Instruct | 12% | **74%** |
| RynnBrain-2B | 0% | 6% |

示例点是 `[0.12, -0.03, 0.45]`，恰好落在工作区中心附近。它是个**「幸运常数」——
比模型真实预测更接近真值**，所以现有 3D 分数里含有**抄袭红利，抄得越多的模型得分越高**。

这解释了几个此前无法理解的现象：预测轨迹跨度中位数为 0（抄来的是单点，构不成轨迹）；
InternVL3.5-8B(0.081)、InternVL3.5-30B(0.082)、Cosmos3-edge(0.080) 三个不同模型
得分几乎逐位相同；3D 整体比 2D 差一个数量级（2D 示例是像素坐标，只有 4% 被抄）。

---

## 1. 模型收到的输入（冻结版，未改动）

以 `file-000-1_trajectory_3d` 为例。**输入 = 三张静止图 + 一段文字**，按此顺序：

```
[1] 文本  "Primary view image:"
[2] 图片  file-000-1_left_eye.jpg      960×540   70 KB   ← 主视角，2D 坐标以它为准
[3] 文本  "Context view left_wrist:"
[4] 图片  file-000-1_left_wrist.jpg    960×540   80 KB
[5] 文本  "Context view right_eye:"
[6] 图片  file-000-1_right_eye.jpg     960×540   58 KB
[7] 文本  （任务 prompt，见下）
```

三张图**全部取自动作起始帧**（本例 t=10.0s），是静止图不是视频。
真值覆盖 10.0s–27.7s，共 10 个点，**按时间等间隔线性插值**采样。

### 1.1 文字部分（3D，逐字）

```
You are evaluating a robot manipulation trajectory prediction task.

Question:
pick the red cube. You are given synchronized images from three camera views:
left_eye, right_eye, left_wrist. Use all views as context, but predict the key
**3D** trajectory points (in meters) needed to complete this task from the main
viewpoint (left_eye) onward.

Predict ordered key trajectory points only for the active gripper: right_gripper.
Use 3D camera-frame coordinates in meters [x, y, z].
Return approximately 10 right_gripper points if visible/available.

Output JSON only. Do not use Markdown.
Required schema:
{
  "right_gripper": [[0.12, -0.03, 0.45]]
}
```

### 1.2 ⚠️ 3D 缺失的信息（数据里都有，只是没给模型）

| 信息 | prompt 里 | 数据里 |
| --- | --- | --- |
| 末端初始位姿 | ❌ | ✅ 真值第 1 点 |
| 轴向约定（x right / y down / z forward） | ❌ | ✅ `answer.axis_convention` |
| 坐标系名（`camera_opencv`） | ❌ | ✅ `answer.coordinate_frame` |
| 相机内参 | ❌ | ✅ `K = [[681.34,0,490.70],[0,680.39,286.65],[0,0,1]]` |
| 相机外参 | ❌ | ✅ `answer.extrinsic`（world→camera） |
| 动作时长 | ❌ | ✅ 时间戳（各题 1.3s ~ 22.2s，中位 4.0s） |

**3D 只说了一句 "Use 3D camera-frame coordinates in meters"。**
是三个视角里的哪个相机、坐标轴朝哪、原点在哪 —— 一个字都没有。

对照 2D，同一处说明是：

```
Use image pixel coordinates [u, v] in the main-view image, where u increases right
and v increases down. The main-view image size is 960 pixels wide and 540 pixels
high; the visible canvas spans 0 <= u < 960 and 0 <= v < 540. Any point with
u < 0, u >= 960, v < 0, or v >= 540 is invalid. Coordinates must refer only to the
first attached image / main view, not to a concatenated image, not to a resized
image, and not to the side/wrist views. Do not use normalized coordinates.
```

轴向、范围、参照哪张图、禁用归一化 —— 全都说了。
**这是 2D 与 3D 差一个数量级的直接原因之一。**

在这种信息匮乏下，schema 里那个 `[0.12, -0.03, 0.45]` 是 prompt 中关于
「这个坐标系长什么样」的**唯一线索** —— 抄它某种意义上是理性行为。

---

## 2. 六种 prompt 写法

只改 schema 示例与说明文字，**图片输入完全不变**。

### 2.1 `legacy`（冻结版）

```
Required schema:
{
  "right_gripper": [[0.12, -0.03, 0.45]]
}
```

### 2.2 `placeholder` —— 尖括号占位

```
The angle-bracket tokens below are placeholders for the numbers you must produce;
replace every one of them. Do not copy them literally.
Required schema:
{
  "right_gripper": [[<x1>, <y1>, <z1>], [<x2>, <y2>, <z2>], [<x3>, <y3>, <z3>], ...]
}
```

意图：让「照抄」在语法上就不成立，同时用多点暗示这是一条轨迹。

### 2.3 `caps` —— 大写字母占位

```
The capital letters below are placeholders for the numbers you must produce.
Replace every one of them with a real coordinate. Output must be valid JSON
containing only numbers.
Required schema:
{
  "right_gripper": [[X1, Y1, Z1], [X2, Y2, Z2], [X3, Y3, Z3], ...]
}
```

意图：验证 placeholder 的失败是否由尖括号与模型专有标签冲突导致。

### 2.4 `zeros` —— 全零合法 JSON

```
The zeros below are placeholders showing the required shape. Replace every zero
with a real coordinate; do not return zeros.
Required schema:
{
  "right_gripper": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], ... 共 10 个]
}
```

意图：保住完整合法的 JSON 骨架（格式锚定），同时让照抄得 0 分且可检测。

### 2.5 `legacy + pose` —— 保留原示例，另加一句告知初始位姿

```
Return approximately 10 right_gripper points if visible/available.
The right_gripper is currently at [0.303, 0.148, 0.422] in exactly the coordinate
system described above. Use it to anchor your predictions; your first point should
be at or near this position.

Required schema:
{
  "right_gripper": [[0.12, -0.03, 0.45]]        ← 假常数仍在
}
```

意图：给坐标系锚点，但不动 schema。

### 2.6 `anchor` —— schema 示例换成本题真实初始位姿

```
The single point shown in the schema below is the gripper's CURRENT position, i.e.
the first point of your answer. Continue the trajectory from it — return that point
plus the remaining points.
Required schema:
{
  "right_gripper": [[0.303, 0.148, 0.422]]      ← 逐题变化的真实起点
}
```

意图：示例逐题变化，**抄它等于给出一个正确的首点**，「照抄」这个行为本身失去意义；
同时保住完整合法 JSON 骨架。

### 2.7 `proposal` —— anchor + 时长 + 严格点数

```
Predict ordered key trajectory points only for the active gripper: right_gripper.
Use 3D camera-frame coordinates in meters [x, y, z].

Output JSON only. Do not use Markdown.
The action takes about 17.7 seconds. Return exactly 10 points, uniformly spaced in
time across it.
The right_gripper is currently at [0.303, 0.148, 0.422]. This is point 1 of your
answer -- keep it unchanged. Points 2 to 10 are what you must predict; the schema
repeats the current position only to show the required shape, do NOT return 10
identical points.
Required schema:
{
  "right_gripper": [[0.303, 0.148, 0.422], [0.303, 0.148, 0.422], ... 共 10 个]
}
```

四处叠加：真实锚点、重复 N 次（合法 JSON + 形状明确）、告知时长与均分、
点数由 "approximately" 改为 "exactly"。

---

## 3. 实验结果（3D，各 40 题）

括号内为「解析出的点数中位 / 退化数（≤1 点）」。

| 写法 | RynnBrain-2B | SenseNova-2B | Qwen3-VL-8B |
| --- | --- | --- | --- |
| `legacy` | 0.19（9 点，15/40） | 0.56（1 点，36/40） | 0.48（1 点，33/40） |
| `placeholder` | **0.00（40/40 全废）** | 0.13（10 点） | 0.30（10 点） |
| `caps` | **0.00（40/40 全废）** | 0.12（10 点） | 0.26（10 点） |
| `zeros` | **0.00（40/40 全废）** | **0.00（40/40 全废）** | 0.30（10 点） |
| `legacy + pose` | **1.75** | 1.57（1 点，40/40） | 0.67（10 点，0/40） |
| `anchor` | **0.03（全废）** | 1.99（1 点，40/40） | **1.71**（10 点，4/40） |
| **`proposal`** | **0.00（40/40 全废）** | 1.50（9 点，5/40） | **2.54（10 点，0/40）** |

照抄率：所有非 `legacy` 写法均降到 **0/40**。

---

## 4. 三条机制性结论

### 4.1 照抄可以彻底消除，但代价不对称

Qwen3-VL-8B 从 0.48 提升到 **2.54**（5.3 倍），退化率 33/40 → **0/40**，
点数 1 → 10 —— 它第一次真的在预测轨迹。

**但同样的改动让 RynnBrain-2B 从 0.19 掉到 0.00。**

### 4.2 具体示例对 2B 模型起「格式锚定」作用

`placeholder` / `caps` / `zeros` / `anchor` / `proposal` 五种写法，
**RynnBrain-2B 一次都没通过**。它的输出退化成散文或自己训练的专有格式：

```
'The robot should grasp the red cube at `right_gripper`: [[0.82, 0.55, 1.6], ...×9]'
'The robot should grasp the red cube at point <grasp pose> (626,520), (875,570),
 (861,640), (612,590) </grasp pose>'
'The robot should move its right gripper to `[0.50, 0.52, 0.88]`.'
```

坐标常常是好的，但丢了 JSON 花括号，`first_json_object` 找不到 `{` 就返回空。

**曾假设是尖括号与 RynnBrain 的 `<grasp pose>` 标签冲突 —— 不成立**，
`caps` 没有尖括号照样全废。真实机制是：legacy 那个具体示例给了 2B 模型一个
**可以照着填的完整 JSON 骨架**；占位符让骨架不完整，小模型就退回自然输出模式。

**也试过写宽松解析从散文里捞裸数组 —— 只能救回 10/50**，
其余 40 条是模型确实没给出可用轨迹，不是解析问题。

### 4.3 分数上升不等于变好，反之亦然

两处都出现过，都需要看原始输出才能判断：

- 改掉示例后 SenseNova 从 0.58 掉到 0.13。**不是变差了** ——
  原来那 0.58 是抄「幸运常数」赚的，0.13 才是它的真实水平。
- `anchor` 写法下 SenseNova 拿 **1.99，高于 Qwen 真实预测的 1.71**。
  但看原始输出，它 40/40 只是把给的锚点原样吐回：

  ```
  给的锚点 [0.303, 0.148, 0.422]  →  {"right_gripper": [[0.303, 0.148, 0.422]]}
  给的锚点 [-0.461, 0.151, 0.453] →  {"left_gripper": [[0.0, 0.0, 0.0]]}   ← 连手都认错
  ```

  **指标在奖励复读机。** 打分排除第 0 点后降到 1.08，排名才正过来。

---

## 5. 为什么最终不建议改 prompt

**① 会引入新的系统性偏差。** 照此重跑全量，榜单会出现「大模型普遍上升、
小模型普遍归零」—— 那不是能力差异，是格式适应性差异。

**② 改完 3D 仍然无效。** `proposal` 下 Qwen3-VL-8B 的位置分 1.88，
仍低于两条平凡基线：

| 基线 | 位置分 |
| --- | ---: |
| 锚点原地重复 | 2.42 |
| 首尾直线 | 8.84 |
| Qwen3-VL-8B（proposal，最好的本地模型） | **1.88** |

**③ 有独立证据表明形状本身就不对。** 对 Qwen3-VL-8B 的非抄袭预测逐级做最优对齐：

| 对齐方式 | 平均分 |
| --- | ---: |
| 原样 | 0.30 |
| 去中心（消原点差异） | **2.17** |
| 去中心 + 缩放 | **4.14** |
| 去中心 + 缩放 + 旋转（完整相似变换） | 4.50 |
| 对照：真值自身 | 100.00 |

「去中心」单独就涨 7 倍 —— 原点乱猜确实是主要误差来源之一；
旋转几乎无增益 —— 坐标轴朝向不是矛盾。
**但完整校准后也只有 4.50/100。**

所以 3D 低分是「题目没问清楚」和「模型确实做不好」两者叠加，前者可修，后者不可。

---

## 6. 唯一建议采纳的 prompt 改动

**把数据里已有的信息补进 3D prompt** —— 末端初始位姿 + 轴向约定。

理由：

1. **不改变格式，只补充信息**，不会触发 2B 模型的格式崩溃
2. **`legacy + pose` 那一行就是证据**：RynnBrain 在这个写法下从 0.19 涨到 **1.75**，
   是它全部七种写法里最好的一次
3. 有对齐实验支撑 —— 去中心单独涨 7 倍
4. **零数据成本**，`answer.axis_convention` 与真值第 1 点本来就在 JSON 里

**若采纳，必须同时改打分：排除第 0 点。** 否则「把锚点原样吐回」成为高分策略
（实测 SenseNova 因此拿到 1.99，高于 Qwen 的真实预测 1.71）。
排除后 SenseNova 降到 1.08，Qwen 基本不变。

---

## 7. 代码与复现

所有写法均已实现，**默认关闭**，默认 prompt 与冻结版 `test/trajectory_glm_test.py`
**逐字节一致**（`tests/test_request_equivalence.py` 每轮验证）。

```bash
# prompt 写法 A/B：照抄率、点数、分数
python tools/prompt_example_ab.py --model Qwen3-VL-8B-Instruct --dim 3D -n 40 \
       --styles legacy proposal

# 加 --pose 1 测「另加一句告知初始位姿」
python tools/prompt_example_ab.py --model RynnBrain-2B --dim 3D -n 40 \
       --styles legacy --pose 0 1

# 坐标系对齐探测：区分「形状对位置错」与「没信号」
python tools/frame_alignment_probe.py --model Qwen3-VL-8B-Instruct -n 40

# 提案口径评测（含排除第 0 点、位置分/位移分双指标、统一容差）
python tools/proposal_eval.py --model Qwen3-VL-8B-Instruct --dim 3D -n 40
```

代码开关：`TrajectoryTask(example_style=..., include_initial_pose=...)`，
`example_style ∈ {legacy, placeholder, caps, zeros, anchor, proposal}`，默认 `legacy`。

---

## 8. 数据可信度说明

同一写法在不同轮次有小幅波动（Qwen `legacy` 三轮分别 0.48 / 0.47 / 0.48，
照抄 31 / 29 / 31），各轮跑在不同 GPU 上。**40 题样本量下，0.1 以内的差异
不应解读为有意义。** 本文的关键结论均为数量级差异（0.00 vs 2.54），不受此影响。

另需与数据方核对一处：你们表中 SenseNova-SI-1.1 的 `trajectory_3D = 0.1956`
与 RynnBrain-2B 的 `trajectory_3D = 0.1956` **逐位相同**，
我们独立测得 SenseNova 为 **0.4694**，怀疑结果文件被覆盖或串了。
