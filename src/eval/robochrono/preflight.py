#!/usr/bin/env python3
# coding: utf-8
"""开跑前自检。

目的很朴素：**不要跑了三小时才发现 key 没配**。整个矩阵可能要跑几天，
任何能在第 0 分钟发现的问题都不该留到第 180 分钟。

检查分三档：
  FAIL  会导致跑不起来，必须修
  WARN  能跑但结果可能不对或不完整，需要知情
  OK    通过

退出码非零表示存在 FAIL。
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import tasks
from .matrix import Plan, RunSpec
from .tasks.base import load_items

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class Check:
    level: str
    name: str
    detail: str


# --------------------------------------------------------------------------
# 环境
# --------------------------------------------------------------------------

REQUIRED_PACKAGES = {
    "torch": "2.6.0",
    "transformers": "4.57.6",
    "decord": None,
    "cv2": None,
    "PIL": None,
    "numpy": None,
    "requests": None,
    "qwen_vl_utils": None,
}


def check_environment() -> list[Check]:
    out: list[Check] = []
    for module, expected in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(module)
        except ImportError:
            out.append(Check(FAIL, f"依赖 {module}", "未安装，见 eval/setup_env.sh"))
            continue
        version = getattr(mod, "__version__", "?")
        if expected and not str(version).startswith(expected):
            out.append(Check(
                WARN, f"依赖 {module}",
                f"装的是 {version}，实测选定的是 {expected}"
                + ("（4.51 会因 dtype= 报错，5.x 会因 meta tensor 加载失败）"
                   if module == "transformers" else ""),
            ))
        else:
            out.append(Check(OK, f"依赖 {module}", str(version)))

    out.append(
        Check(OK, "ffmpeg", shutil.which("ffmpeg") or "")
        if shutil.which("ffmpeg")
        else Check(WARN, "ffmpeg", "未找到；BC-11 的媒体压缩将不可用")
    )
    return out


def check_gpu() -> list[Check]:
    try:
        import torch
    except ImportError:
        return [Check(FAIL, "GPU", "torch 未安装")]
    if not torch.cuda.is_available():
        return [Check(WARN, "GPU", "CUDA 不可用；只能跑 API provider")]
    count = torch.cuda.device_count()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    name = torch.cuda.get_device_properties(0).name
    return [Check(OK, "GPU", f"{count} × {name} {total:.0f} GiB")]


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------


def check_normalized(datasets_root: Path) -> list[Check]:
    """规范化产物是否相对当前代码与当前数据仍然有效。

    放在 check_data 之前单独查一次 —— 否则同一条过期原因会在 56 个
    (族, 任务) 上各报一遍，把真正的问题淹掉。
    """
    from .normalize import check_freshness

    state = check_freshness(datasets_root)
    if state.ok:
        return [Check(OK, "规范化产物", "最新")]
    head = state.reasons[0]
    if len(state.reasons) > 1:
        head += f"（另有 {len(state.reasons) - 1} 项）"
    return [Check(FAIL, "规范化产物", head[:78]),
            Check(FAIL, "  → 重建", "python tools/build_normalized.py")]


def check_data(specs: list[RunSpec], datasets_root: Path, sample: int = 3) -> list[Check]:
    """抽样验证媒体可解析。用任务自己的 parts()，验的是真实行为。"""
    out: list[Check] = []
    checked: set[tuple[str, str]] = set()

    for spec in specs:
        key = (spec.family, spec.run)
        if key in checked:
            continue
        checked.add(key)

        qa_path = spec.qa_path(datasets_root)
        if not qa_path.exists():
            out.append(Check(FAIL, f"数据 {spec.family}/{spec.run}", f"QA 文件缺失 {qa_path}"))
            continue

        try:
            items = tasks.load_run_items(datasets_root, spec.family, spec.run)
        except Exception as exc:  # noqa: BLE001
            out.append(Check(FAIL, f"数据 {spec.family}/{spec.run}", f"QA 无法解析：{exc}"))
            continue

        task = tasks.build(spec.run)
        units = task.units(items)
        missing = 0
        error = ""
        for unit in units[:sample]:
            try:
                for part in task.parts(unit):
                    if part.get("type") in {"image", "video"} and not Path(part["path"]).exists():
                        missing += 1
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                break

        if error:
            out.append(Check(FAIL, f"数据 {spec.family}/{spec.run}", error[:70]))
        elif missing:
            out.append(Check(FAIL, f"数据 {spec.family}/{spec.run}",
                             f"抽样 {sample} 个 unit 中有 {missing} 个媒体文件缺失"))
        else:
            out.append(Check(OK, f"数据 {spec.family}/{spec.run}",
                             f"{len(items)} 题 / {len(units)} 次调用"))
    return out


# --------------------------------------------------------------------------
# 模型与密钥
# --------------------------------------------------------------------------


def check_models(plan: Plan, config: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    providers = config.get("providers", {})

    for model in plan.models:
        provider = providers.get(model.provider)
        if provider is None:
            out.append(Check(FAIL, f"模型 {model.name}",
                             f"providers.json 里没有 {model.provider}"))
            continue

        if model.kind == "api":
            env_name = provider.get("api_key_env", "")
            if env_name.startswith("sk-") or len(env_name) > 40:
                out.append(Check(FAIL, f"模型 {model.name}",
                                 "api_key_env 里似乎填的是 key 本身，而不是环境变量名"))
            elif not os.getenv(env_name):
                out.append(Check(FAIL, f"模型 {model.name}",
                                 f"环境变量 {env_name} 未设置"))
            else:
                budget = provider.get("max_request_bytes", 0)
                note = f"key 已配；请求体预算 {budget/1e6:.0f} MB" if budget else \
                       "key 已配；未设 max_request_bytes，大视频会 413"
                out.append(Check(OK if budget else WARN, f"模型 {model.name}", note))
            continue

        path = Path(model.model or provider.get("model", ""))
        if not path.is_absolute():
            # 与 runtime_config 一致：相对路径以 eval/ 为基准，不是 cwd
            path = (Path(__file__).resolve().parents[1] / path).resolve()
        if not path.exists():
            out.append(Check(FAIL, f"模型 {model.name}", f"权重目录不存在 {path}"))
            continue

        detail = f"{sum(f.stat().st_size for f in path.rglob('*') if f.is_file())/1e9:.2f} GB"
        config_json = path / "config.json"
        if config_json.exists():
            model_config = json.loads(config_json.read_text(encoding="utf-8"))

            # 架构必须要么被 transformers 注册，要么带 trust_remote_code 的自定义代码。
            # Cosmos3-Edge 两者皆无 —— 它要 NVIDIA 的 cosmos-framework（从 GitHub 源码装），
            # 用标准 transformers 加载会报 "does not recognize this architecture"。
            # 这类问题不在开跑前拦住，就会在跑到一半时整个模型全军覆没。
            model_type = str(model_config.get("model_type", ""))
            has_remote_code = bool(model_config.get("auto_map")) or any(path.glob("*.py"))
            if model_type and not has_remote_code:
                try:
                    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

                    if model_type not in CONFIG_MAPPING_NAMES:
                        out.append(Check(
                            FAIL, f"模型 {model.name}",
                            f"{detail}；架构 `{model_type}` 未被 transformers 注册，"
                            "且目录内无自定义代码 —— 需要该模型自己的推理库",
                        ))
                        continue
                except ImportError:
                    pass

            auto_map = model_config.get("auto_map", {})
            if any("--" in str(v) for v in auto_map.values()):
                out.append(Check(WARN, f"模型 {model.name}",
                                 f"{detail}；auto_map 含跨仓库引用，"
                                 "带点号的 repo id 会导致加载失败，"
                                 "跑 eval/tools/patch_local_model.py"))
                continue
        out.append(Check(OK, f"模型 {model.name}", detail))
    return out


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def run_all(plan: Plan, specs: list[RunSpec], datasets_root: Path,
            config: dict[str, Any], skipped: list[tuple[str, str]]) -> tuple[list[Check], int]:
    normalized = check_normalized(datasets_root)
    checks = check_environment() + check_gpu() + check_models(plan, config) + normalized
    # 产物过期就别再逐族验数据了 —— 那些报错全都是同一个原因的回声
    if all(c.level != FAIL for c in normalized):
        checks += check_data(specs, datasets_root)
    return checks, sum(1 for c in checks if c.level == FAIL)


def format_checks(checks: list[Check], skipped: list[tuple[str, str]]) -> str:
    lines = [f"{'':6} {'检查项':<34} 说明", "-" * 92]
    for check in checks:
        marker = {OK: "  OK  ", WARN: " WARN ", FAIL: " FAIL "}[check.level]
        lines.append(f"{marker} {check.name:<34} {check.detail}")
    lines.append("-" * 92)

    fails = sum(1 for c in checks if c.level == FAIL)
    warns = sum(1 for c in checks if c.level == WARN)
    lines.append(f"{len(checks)} 项检查：{len(checks)-fails-warns} 通过，{warns} 警告，{fails} 失败")

    if skipped:
        lines.append("")
        lines.append(f"矩阵中将跳过 {len(skipped)} 个组合：")
        for key, reason in skipped[:10]:
            lines.append(f"  {key}: {reason}")
        if len(skipped) > 10:
            lines.append(f"  ... 另有 {len(skipped)-10} 个")

    lines.append("")
    lines.append("可以开跑" if fails == 0 else "存在 FAIL，请先修复再开跑")
    return "\n".join(lines)
