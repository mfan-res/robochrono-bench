# src/eval

评测工具。读 `data/vqa/` + `models/`，产出 `runs/<run_id>/`。

```
robochrono/    实现
reference/     ★ 冻结的原始评测脚本 + 其录制的基线，**永不修改**
configs/       provider / plan / environments
tests/         回归
```

`reference/` 是六套回归的比对基准。它和它的 baselines 放在一起，
让回归自成一体 —— 换路径、换机器都不影响。
