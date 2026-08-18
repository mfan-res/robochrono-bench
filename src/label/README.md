# src/label —— 标注这一段

这个目录负责 benchmark 七段流水线里的第 ③ 段：**把机器人视频切成带标签的动作分段**。

```
① 能力定义 → ② 采集 → ③ 标注 → ④ 出题 → ⑤ 验题 → ⑥ 评测 → ⑦ 解读
                        ↑ 你在这
```

七个评测任务的真值**全部**源自这里产出的分段。标注错一处，题就错一片，
而且**错得不报错** —— 这是这个模块所有设计的出发点。

---

## 先跑起来

```bash
cd /path/to/bench

python3 src/label/serve.py --port 8000      # 标注器，浏览器开 localhost:8000
python3 src/label/validate.py               # 对现有 320 份跑六条检查
python3 src/label/tests/test_core_replay.py # 重写回归（不需要视频）
```

在 VS Code Remote / SSH 下端口会自动转发；没自动转发就用 PORTS 面板手动加。

**依赖**：`numpy` `pyarrow` `jsonschema` + `ffmpeg`（`ffprobe`）。**不需要 cv2。**

---

## 数据长什么样

```
data/
├── source/<族>/<集>/main.mp4        规范化后的视频（全帧内，逐帧可定位）
└── label/<族>/
    ├── subtasks.json                动作定义：45 个 ID + 文字（八族合计）
    ├── segments/<集>_segments.json  标注产物 ← 这个模块的输出
    ├── segments/.bak/               保存前的即时备份（不进 git）
    └── categories.deprecated.txt    旧词表，留档不用
```

一份标注：

```jsonc
{
  "source": {                                   // 溯源，缺一不可
    "video": "tea2/file-000/main.mp4", "fps": 30, "total_frames": 10212,
    "tool_version": "bench-label-web/1",
    "subtasks_sha256": "63b53973da900758",
    "episode_bounds": [[0, 115.67], [115.67, 230.73], [230.73, 340.40]]
  },
  "segments": [
    {"id": "file-000@f000277", "subtask": "pick_up_teapot_lid",
     "start_frame": 277, "end_frame": 427,      // ← 权威
     "start": 9.233, "end": 14.267,             // ← 派生，由 fps 换算
     "start_time": "00:00:09.233", "end_time": "00:00:14.267"}
  ]
}
```

契约在 `src/common/schemas/segments.json`，320 份现存标注全部通过。

---

## 三个设计决定，请不要绕过

### 1 · 段只存 subtask 的 **ID**，不存文字

同一个动作在流水线各处曾有 **11 种表示**（词表、narration、main_verbs、objects、
题目里的 answer_text、选项文字、题干内嵌、干扰项缓存的键……）。
表示越多，保持一致越难，而不一致时通常不报错。
根因是系统里没有「动作」这个实体，每层存的都是渲染后的字符串而非引用。

所以现在文字只存在于 `subtasks.json` 一处，段里只存 `subtask: "pick_up_teapot_lid"`。

**ID 一旦分配就永不改变** —— 这是它的全部价值。改措辞只动 `subtasks.json` 的 `text`，
所有引用自动跟随。（顺带：`pick_up_teapot` 是 `pick_up_teapot_lid` 的前缀，
**任何地方都要精确相等匹配，不能用 startswith / 子串**。）

### 2 · 帧号是权威，秒是派生

`start_frame` / `end_frame` 由人给出；`start` / `end` / `*_time` 全由后端按 fps 算。
前端只报帧号，浮点误差进不了数据。

**判断「某时刻落在哪个段内」必须走帧号。** 因为 `end = (end_frame + 1) / fps`
（闭区间转半开）会让相邻段的**秒区间**重叠一帧 —— 全量 631 处。
帧号层是干净的，秒层不是。

### 3 · 校验器一套代码，两处使用

`validate.py` 既在标注器保存前跑，也对存量离线跑。**必须是同一份判据** ——
之前离线脚本和标注工具各写各的，结果 tea2 显示「21/21 齐全」而实际只有 20 集可用。

---

## 现在的状态

| 项 | 状态 |
| --- | --- |
| `core.py` 语义层 | ✅ 重写完成，回归三条链路全绿 |
| `validate.py` 六条检查 | ✅ 对 320 份跑通 |
| `serve.py` + `ui/` 网页标注器 | ✅ 可用（播放/逐帧/打点/选动作/管理 subtask/保存） |
| `schemas/segments.json` | ✅ 320/320 通过 |
| 上游 OpenCV 工具 | 📦 原样存在 `upstream/`，**已不使用**，仅供比对 |

`validate.py` 当前报 **40 条**，全部来自同一件事：

```
wash   40 条  每集重复动作 —— 洗两个盘子，同一动作做两遍（P-05，处理中）
```

> 64 → 42：tea2 与 express 移出（D-42）。42 → 40：删掉 pen_inbox 的零长度段（D-44）。

**这些不是 bug，是待决策项。** 详见根目录 `DEVLOG.md` 的「问题记录」P-01 / P-05。

---

## 已知的坑（都踩过，别再踩）

**多集打包的视频。** LeRobot 按体积打包，一个 mp4 里可能装 2–3 个 episode
（tea2：21 个视频装了 53 集）。这个信息只在 `meta/episodes` 元表里，
界面上看不出来，所以容易只标第一集。
标注器现在会在时间轴上画 episode 分界线，且**整集未标就不让保存**。

边界从 `data/raw/<族>/meta/episodes/*.parquet` 读。⚠ 那个 parquet 里有多个
`*_file_index` 列，**必须显式取 `videos/` 开头的那一列** —— 取错会把
「状态打包成一个 parquet」误读成「一个视频装 40 轮」。
⚠ 这张元表本身不完整（tea/wash 只记了 10 集而实际有 39/40），
**查不到 ≠ 只有一集**，这个区别要保留。

**出题产物回写。** stack_cubes 的标注一度存的是出题窗口计算的结果，
而非人工标注本身（pick 起点后移 10 秒、place 越界 2 秒、另有合成的「move」段）。
已从 `metadata.original_*` 还原为人工原件。schema 现在**显式拒绝**
`metadata` / `window_type` / `original_*` 出现在段里。

窗口计算本身是**出题的合理设计** —— 标注段的开头往往还是上一个动作的余波，
不后移的话片段里看不到目标动作。问题只在于结果被写回了标注层。

**标注是「分段」不是「时刻标注」。** 段与段严丝合缝，所以每段的开头往往是
上一个动作的余波（例如 `pick_yellow_cube` 标了 33–57 秒，但机械臂到 56 秒才碰到黄方块）。
**出题时抽帧不能取段的开头。**

---

## 想改点什么？

给 AI coding agent 的详细约束、验证命令和陷阱清单在 **`AGENTS.md`**（同目录）。

改完请务必跑：

```bash
python3 src/label/tests/test_core_replay.py   # 语义没走样
python3 src/label/validate.py                 # 应当仍是 40 条，不多不少
```

**回归的判据不是「输出相同」，是「每一处差异都被声明过」。** 冒出未声明的差异，
就是改坏了。

更多背景（29 条决策、6 条问题记录、19 条被推翻的判断）见根目录 `DEVLOG.md`。
那里面记着每个决定的理由，以及我们判断错过又纠正的地方 ——
**只留结论的话，下一个人会重新踩一遍。**
