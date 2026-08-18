#!/usr/bin/env python3
# coding: utf-8
"""按模型分派到不同的 python 环境。

15 个模型没有一个 transformers 版本能全覆盖 —— InternVL 系只能在 4.57.6 上跑，
Cosmos3-Edge 只能在 5.x 上跑，两者互斥。所以矩阵不能在单一解释器里跑完。

分派粒度是**模型**，不是进程内：

    dispatch
      ├─ groupA 的解释器  python -m robochrono matrix --models A B C ...
      └─ groupB 的解释器  python -m robochrono matrix --models D E ...

矩阵本来就是 model-major 调度（一个模型跑完再换下一个），所以按模型切一刀
完全不打乱原有编排。顺带得到两个好处：一套环境装坏了不影响另一套，
某个模型把进程搞崩了也不会带走整轮。

刻意**不用** ``multiprocessing.set_executable``：跨解释器 spawn 依赖两边
python 版本与 pickle 协议一致，很脆，而且报错难查。起子进程简单可靠。

结果都写进同一个 results 目录 —— 各模型的目录互不重叠，不会打架。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_environments(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in config.items() if not str(k).startswith("_")}


def resolve_interpreter(env_config: dict[str, Any], env_name: str, eval_root: Path) -> str:
    """取该环境的解释器路径。留空表示用当前解释器。"""
    envs = env_config.get("envs", {})
    if env_name not in envs:
        raise ValueError(f"environments.json 里没有环境 {env_name!r}；已有：{sorted(envs)}")
    raw = str(envs[env_name].get("python") or "").strip()
    if not raw:
        return sys.executable
    path = Path(raw)
    if not path.is_absolute():
        # 只做词法拼接，**不能 resolve** —— venv 的 bin/python 是指向基础
        # 解释器的符号链接，解析后 sys.prefix 会指向基础环境，venv 的
        # site-packages 完全不生效，B 组会静默用上 A 组的包。
        path = Path(os.path.abspath(eval_root / path))
    if not path.exists():
        raise FileNotFoundError(
            f"环境 {env_name} 的解释器不存在：{path}\n"
            f"先跑 bash tools/setup_envs.sh {env_name}"
        )
    return str(path)


def group_models(env_config: dict[str, Any], model_names: list[str]) -> dict[str, list[str]]:
    """把模型按环境分组，保持 plan 里的原始顺序。"""
    mapping = env_config.get("models", {})
    default = str(env_config.get("default_env", "groupA"))
    groups: dict[str, list[str]] = {}
    for name in model_names:
        groups.setdefault(str(mapping.get(name, default)), []).append(name)
    return groups


def run(
    env_config: dict[str, Any],
    model_names: list[str],
    passthrough: list[str],
    *,
    eval_root: Path,
    dry_run: bool = False,
) -> int:
    """逐个环境起子进程。返回失败的环境数。"""
    groups = group_models(env_config, model_names)
    if not groups:
        print("没有要跑的模型")
        return 0

    print(f"{len(model_names)} 个模型分到 {len(groups)} 套环境：")
    for env_name, names in groups.items():
        print(f"  {env_name:<10} {', '.join(names)}")
    print()

    failures = 0
    for env_name, names in groups.items():
        try:
            interpreter = resolve_interpreter(env_config, env_name, eval_root)
        except (ValueError, FileNotFoundError) as exc:
            print(f"[{env_name}] 跳过：{exc}")
            failures += 1
            continue

        cmd = [interpreter, "-m", "robochrono", *passthrough, "--models", *names]
        print(f"{'='*70}\n[{env_name}] {interpreter}\n  {' '.join(cmd[3:])}\n{'='*70}", flush=True)
        if dry_run:
            continue

        result = subprocess.run(cmd, cwd=str(eval_root))
        if result.returncode != 0:
            print(f"[{env_name}] 退出码 {result.returncode}", flush=True)
            failures += 1

    return failures
