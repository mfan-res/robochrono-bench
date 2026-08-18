# coding: utf-8
"""回归夹具的位置。**六个测试统一从这里取路径,不各写各的。**

搬进本仓库时把 `eval/datasets/QA`（141 MB，不含 trajectory）与
`eval/results/baseline` 换成了**裁剪过的夹具**：

```
fixtures/QA/…        只保留 baseline 引用到的那些题（9 个任务共 130 道，1.6 MB）
fixtures/baseline/   录下来的模型输出，原样搬（2.2 MB）
```

回归本来就是「拿录好的模型输出重放，比对新旧实现的打分」——
它只读 baseline 里出现过的那些 id，整份 QA 里其余 2,570 道从来没被用到。
裁剪之后 3.8 MB 进 git，六套回归在本仓库内自足，不依赖旧仓库还在不在。

**完整的 v1 QA 与媒体（61 GB）留在旧仓库**，A4 新旧对照要跑真模型时才需要，
路径由 `ROBOCHRONO_V1_ROOT` 指定。
"""

from __future__ import annotations

import os
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
FIXTURES = EVAL / "fixtures"

BASELINE = FIXTURES / "baseline"
QA = FIXTURES / "QA"
DATASETS = FIXTURES

# 完整的 v1 数据（QA 媒体 61 GB、模型权重 34 GB）留在旧仓库，不进本仓库。
# 需要它的只有两件事：`test_request_equivalence`（要组装含媒体的真实请求）
# 与 A4 新旧对照（要跑真模型）。
V1_ROOT = Path(os.environ.get(
    "ROBOCHRONO_V1_ROOT",
    "/mnt/public/users/wbcd/workspace/michael/benchmark/eval"))
V1_QA = V1_ROOT / "datasets" / "QA"


def qa_root(need_media: bool = False) -> Path:
    """QA 根目录。要媒体就得用旧仓库的完整数据。

    **找不到时返回 None 让调用方显式跳过，不要假装通过** ——
    「媒体缺失所以这条没测」和「测了并通过」是两回事。
    """
    if not need_media:
        return QA
    return V1_QA if V1_QA.exists() else None
