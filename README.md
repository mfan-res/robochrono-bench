# bench

机器人 VLM 评测基准。顶层按**东西的种类**分，第二层按**流水线阶段**分。

```
raw ──┐
label ├──▶ src/vqa ──▶ data/vqa ──▶ src/eval ──▶ runs
llm_cache ┘                ▲
recipes ───────────────────┘
```

| 目录 | 装什么 | 可再生 |
| --- | --- | --- |
| `data/` | 全部数据 | 见各子目录 |
| `src/` | 全部代码 | — |
| `runs/` | 每次评测的产物 | 是（重跑） |
| `models/` | 模型权重（建议软链到外部） | 是（重下） |
| `docs/` | 文档 | — |

`src/<stage>` 与 `data/<stage>` 同名是刻意的：一眼看出谁产出谁。
`data/raw/` 没有对应的 `src`，因为它是下载来的输入。

## 四个层间契约

目录整齐不等于结构稳定。**稳定来自「约定被写成机器能检查的东西，并在边界上执行」。**
schema 放在 `src/common/schemas/`，每个边界配一个 `--verify`：

| 边界 | schema | 检查什么 |
| --- | --- | --- |
| `data/raw` | `raw.json` | LeRobot 布局、meta.json 必填项 |
| `data/label` | `segments.json` | 分段结构 |
| `data/vqa` | `item.json` | 三分结构、prompt 白名单、role 声明表 |
| `runs` | `row.json` | 结果行结构 |
