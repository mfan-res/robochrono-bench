#!/usr/bin/env python3
# coding: utf-8
"""结果 meta 里写的执行参数，必须真的进了请求。**不调任何模型、不联网。**

    python3 src/eval/tests/test_meta_matches_request.py

为什么专门守这一条
------------------
四天里同一个模式出现了**三次**，每次都让一整批结果变成「不是我们以为的那个实验」：

    D-60/61  抽帧档位   _prepare 算出 32 帧、写进 meta，但多卡 worker 自建
                        runtime 拿不到 —— time 实跑 8 帧，meta 写 32
    D-66     frames 派生键 frames_by_run 覆盖时只写 mode/value，
                        runtime 顶层的 num_segments 没跟着改，一份 runtime 两个说法
    D-66     生成参数   thinking: disabled 卡在 send_thinking 默认 False 上从没发出去；
                        max_new_tokens 只出现在四个本地 adapter 里，
                        openai_compatible 的 payload 根本没有 max_tokens
                        —— 实测 reasoning 占 completion 的 99%

**三次都不是测试报红发现的**，是人分别去读代码、跑核对脚本、看 token 用量才发现的。
六套回归当时全绿 —— 它们比的是 `parts()` 与打分，**不含执行参数**。

这类缺陷的共同特征：**结果看起来完全正常，只是不是你以为的那个实验。**
而全量矩阵是 5 条路径 × 10,178 题、约 29 GPU 小时 + 40 M token，
藏一个进去就是整批数据的解释都错了，且看不出来。

判据
----
`matrix_run._meta()` 往结果里写什么，这里就核什么：

    generation.temperature      → 请求体里必须有对应项且相等
    generation.max_new_tokens   → 同上（OpenAI 方言是 max_tokens）
    generation.thinking         → 同上（各家参数名不同，由 thinking_param 声明）
    frames                      → 多卡路径下发给 worker 的档位必须与它一致

**「记了但没发」和「发了但记错」都算失败。** 前者是我们踩过的三次，
后者更隐蔽 —— meta 是事后唯一的凭据，它错了就没有别的地方可查。

不覆盖什么
----------
本地 adapter 没有「请求体」可截，参数直接进 `model.generate(...)`。
这里只核到 runtime 层（各 adapter 都从 runtime 取值，见 vlm_api 的
`request_local_*`），**再往下要靠 A-1 那种逐 run 核对 `frames_used` 的手段**。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import paths as _P  # noqa: E402

from robochrono import matrix, matrix_run, pool, vlm_api  # noqa: E402

CONFIG = _P.EVAL / "configs" / "providers.json"
PLAN = _P.EVAL / "configs" / "plan.json"


class Captured(Exception):
    """截下 payload 之后不真发请求 —— 这个测试不联网。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


def capture_payload(provider: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """走完整的 `call_vlm` 路径，截下将要发出的请求体。返回 (runtime, payload)。"""
    # 塞一个假 key —— **这个测试不联网**，请求在 requests.post 那一层就被截住了。
    # 不这么做的话，没配 keys.env 的机器上这条回归会以「缺密钥」失败，
    # 而它要验的东西跟密钥无关。
    import os
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    env_name = cfg["providers"][provider].get("api_key_env") or "ZHIPUAI_API_KEY"
    os.environ.setdefault(env_name, "test-key-not-used")

    runtime = vlm_api.runtime_config(config_path=CONFIG, provider_name=provider,
                                     default_model="", cli_model=None)
    # 关掉媒体预处理：它要真文件，而这里只核参数
    runtime["max_request_bytes"] = 0
    runtime["min_video_seconds"] = 0.0

    real_post = vlm_api.requests.post

    def fake_post(url: str, **kwargs: Any):  # noqa: ANN401
        raise Captured(kwargs.get("json") or {})

    vlm_api.requests.post = fake_post
    try:
        vlm_api.call_vlm(runtime, [{"type": "text", "text": "ping"}], {})
    except Captured as hit:
        return runtime, hit.payload
    finally:
        vlm_api.requests.post = real_post
    raise AssertionError(f"{provider}: 没有截到请求体（call_vlm 没走到 HTTP 这一层？）")


def check_api_provider(provider: str) -> list[str]:
    """核一个 API provider：meta 会记的每一项，请求体里都要有且相等。"""
    runtime, payload = capture_payload(provider)
    bad: list[str] = []

    if payload.get("temperature") != runtime["temperature"]:
        bad.append(f"temperature: meta 记 {runtime['temperature']}，"
                   f"请求体是 {payload.get('temperature')!r}")

    # OpenAI 方言叫 max_tokens。**这一项曾经完全没发**（D-66）。
    if payload.get("max_tokens") != runtime["max_new_tokens"]:
        bad.append(f"max_new_tokens: meta 记 {runtime['max_new_tokens']}，"
                   f"请求体的 max_tokens 是 {payload.get('max_tokens')!r}")

    # thinking 的参数名各家不同，由 provider 的 thinking_param 声明。
    # 没声明 = 不发 —— 那时 meta 里的 thinking 只是意图不是事实，必须报出来。
    wanted = runtime.get("thinking")
    param = runtime.get("thinking_param")
    if param:
        got = payload.get(str(param))
        if got != (wanted == "enabled"):
            bad.append(f"thinking: meta 记 {wanted!r}，"
                       f"请求体的 {param} 是 {got!r}")
    elif runtime.get("send_thinking"):
        got = (payload.get("thinking") or {}).get("type")
        if got != wanted:
            bad.append(f"thinking: meta 记 {wanted!r}，请求体的 thinking.type 是 {got!r}")
    else:
        bad.append(f"thinking: meta 会记 {wanted!r}，但既没有 thinking_param "
                   f"也没有 send_thinking —— **这一项根本没发出去**，"
                   f"服务端按自己的默认走")
    return bad


def check_pool_frames() -> list[str]:
    """多卡路径：下发给 worker 的抽帧档位必须与 `_prepare` 算出的一致（D-61）。

    这条曾经断在 `_run_local_pool` 里一句 `_, _, store, items, task = _prepare(...)`
    —— runtime 被丢掉，worker 自建一份，而那份不读 `frames_by_run`。
    """
    bad: list[str] = []
    plan = matrix.load_plan(PLAN)
    root = _P.EVAL.parent.parent / "data" / "vqa" / "eval"
    if not root.exists():
        return ["（跳过多卡档位核对：没有 data/vqa/eval）"]
    specs, _ = matrix.expand(plan, root)
    local = [s for s in specs if s.model.is_local]
    seen: set[str] = set()
    for spec in local:
        if spec.run in seen:
            continue
        seen.add(spec.run)
        runtime = vlm_api.runtime_config(config_path=CONFIG,
                                         provider_name=spec.model.provider,
                                         default_model="", cli_model=spec.model.model)
        by_run = (matrix_run._providers_cfg(CONFIG).get("frames_by_run", {}) or {}).get(spec.run)
        if by_run:
            runtime = dict(runtime)
            runtime["frames"] = dict(by_run)
        runtime = matrix_run._apply_frames(runtime, spec)
        item = pool.WorkItem(spec.key, spec.run, "k", [],
                             frames=runtime.get("frames"),
                             align_fps=bool(runtime.get("align_fps_to_segments", False)))
        if item.frames != runtime["frames"]:
            bad.append(f"{spec.run}: WorkItem 带的档位 {item.frames} "
                       f"≠ _prepare 算出的 {runtime['frames']}")
        # meta 记的就是 runtime["frames"]，所以两者一致即可
    return bad


def main() -> int:
    print("核对：结果 meta 里写的执行参数，是不是真的进了请求\n")
    failures = 0

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    api_providers = [name for name, p in config["providers"].items()
                     if p.get("type") in {"openai_compatible", "glm", "qwen"}]
    for provider in sorted(api_providers):
        bad = check_api_provider(provider)
        mark = "✓" if not bad else "✗"
        print(f"{mark} {provider}")
        for line in bad:
            print(f"    {line}")
        failures += bool(bad)

    bad = check_pool_frames()
    real = [b for b in bad if not b.startswith("（")]
    print(f"{'✓' if not real else '✗'} 多卡路径的抽帧档位下发")
    for line in bad:
        print(f"    {line}")
    failures += bool(real)

    print()
    if failures:
        print(f"❌ {failures} 处「meta 说的」与「实际发的」对不上。"
              "\n   **不要拿这批结果下结论** —— 它记录的实验条件不是真实发生的那个。")
        return 1
    print("meta 与请求一致。\n"
          "⚠ 只核到请求体这一层：本地 adapter 没有请求体，参数直接进 model.generate()，"
          "\n   那一层要靠逐 run 核对结果里的 frames_used。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
