#!/usr/bin/env python3
# coding: utf-8
"""Small VLM API adapter for workflow evaluation scripts.

Supported provider types:
  - openai_compatible: GLM, Qwen compatible-mode, Kimi, Ark, vLLM, and similar chat/completions APIs
  - gemini: Gemini generateContent REST API
  - local_qwen: local Qwen-VL weights loaded by transformers
  - local_transformers_vlm: local image-text-to-text weights loaded by transformers
  - local_cosmos_transformers: local Cosmos3 Edge HF weights loaded by transformers
  - local_internvl: local InternVL weights loaded by transformers
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from requests import HTTPError


DEFAULT_GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
_LOCAL_QWEN_MODEL: Any | None = None
_LOCAL_QWEN_PROCESSOR: Any | None = None
_LOCAL_QWEN_MODEL_NAME: str | None = None
_LOCAL_TRANSFORMERS_VLM_MODELS: dict[str, tuple[Any, Any]] = {}
_LOCAL_COSMOS_TRANSFORMERS_MODELS: dict[str, tuple[Any, Any]] = {}
_LOCAL_INTERNVL_MODEL: Any | None = None
_LOCAL_INTERNVL_TOKENIZER: Any | None = None
_LOCAL_INTERNVL_MODEL_NAME: str | None = None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return {key: value for key, value in data.items() if not str(key).startswith("_")}


def task_config(config_path: Path | None, task_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    tasks = config.get("tasks", {})
    if not isinstance(tasks, dict):
        return {}
    task = tasks.get(task_name, {})
    if not isinstance(task, dict):
        raise ValueError(f"config.tasks.{task_name} must be an object")
    return task


def media_mime(path: Path, media_type: str) -> str:
    guessed = mimetypes.guess_type(str(path))[0]
    if guessed:
        return guessed
    if media_type == "image":
        return "image/jpeg"
    if media_type == "video":
        return "video/mp4"
    return "application/octet-stream"


def encode_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty media file: {path}")
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def media_url(path: Path, media_type: str, media_url_format: str) -> str:
    encoded = encode_file(path)
    if media_url_format == "data_url":
        return f"data:{media_mime(path, media_type)};base64,{encoded}"
    if media_url_format == "base64":
        return encoded
    raise ValueError(f"Unsupported media_url_format for OpenAI-compatible API: {media_url_format}")


def selected_provider_config(config: dict[str, Any], provider_name: str | None) -> tuple[str, dict[str, Any]]:
    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("config.providers must be an object")
    selected = provider_name or config.get("provider") or "glm"
    if selected in providers:
        provider = providers[selected]
        if not isinstance(provider, dict):
            raise ValueError(f"config.providers.{selected} must be an object")
        return str(selected), provider
    if not providers and selected == "glm":
        return "glm", {
            "type": "openai_compatible",
            "api_url": DEFAULT_GLM_API_URL,
            "api_key_env": "ZHIPUAI_API_KEY",
            "model": "glm-5v-turbo",
            "media_url_format": "base64",
        }
    raise ValueError(f"Unknown provider {selected!r}; available providers: {sorted(providers)}")


_FRAME_MODES = {"fps", "uniform", "native"}


def resolve_frames(provider: dict[str, Any], defaults: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    """BC-09：抽帧策略解析，杜绝静默覆盖。

    冻结版里 ``video_sample_fps`` 和 ``num_segments`` 可以同时存在，而
    ``internvl_frame_indices`` 在 ``sample_fps > 0`` 时直接 return —— ``num_segments``
    静默失效且没有任何警告。config_test.json 里两个 InternVL provider 都写着
    ``num_segments: 4`` + ``video_sample_fps: 1.0``，那个 4 从未生效过。

    新写法用互斥的 ``frames`` 声明：

        frames: {mode: fps,     value: 1.0}   # 每秒抽几帧
        frames: {mode: uniform, value: 32}    # 全片均分成几段，各取中点
        frames: {mode: native,  value: null}  # 交给下游默认

    ``mode: fps`` 下的实际帧数由 engine 按每个视频的时长在运行时算出
    （``round(时长 × fps)``），不再是静态配置值。

    旧字段仍然支持以便平滑迁移，但同时出现时会告警；``strict_frames: true``
    时直接报错。
    """
    spec = provider.get("frames", defaults.get("frames"))
    if spec is not None:
        if not isinstance(spec, dict) or spec.get("mode") not in _FRAME_MODES:
            raise ValueError(f"frames must be an object with mode in {sorted(_FRAME_MODES)}, got {spec!r}")
        for legacy in ("video_sample_fps", "num_segments"):
            if legacy in provider:
                raise ValueError(
                    f"provider sets both `frames` and legacy `{legacy}`; remove the legacy field"
                )
        mode = str(spec["mode"])
        value = spec.get("value")
        return {
            "mode": mode,
            "value": value,
            # 下游 adapter 仍读这两个字段
            "video_sample_fps": float(value) if mode == "fps" and value else 0.0,
            "num_segments": int(value) if mode == "uniform" and value else 1,
        }

    has_fps = "video_sample_fps" in provider
    has_seg = "num_segments" in provider
    sample_fps = float(provider.get("video_sample_fps", defaults.get("video_sample_fps", 0.0)))
    num_segments = int(provider.get("num_segments", defaults.get("num_segments", 16)))

    if has_fps and has_seg and sample_fps > 0:
        message = (
            f"provider sets both video_sample_fps={sample_fps} and num_segments={num_segments}; "
            f"num_segments is silently ignored when video_sample_fps > 0. "
            f"Use the mutually exclusive `frames` field instead."
        )
        if strict:
            raise ValueError(message)
        print(f"  [warn] {message}", flush=True)

    return {
        "mode": "fps" if sample_fps > 0 else "uniform",
        "value": sample_fps if sample_fps > 0 else num_segments,
        "video_sample_fps": sample_fps,
        "num_segments": num_segments,
    }


def runtime_config(
    config_path: Path | None,
    provider_name: str | None,
    default_model: str,
    default_api_url: str = DEFAULT_GLM_API_URL,
    default_api_key_env: str = "ZHIPUAI_API_KEY",
    cli_api_key: str | None = None,
    cli_model: str | None = None,
    cli_temperature: float | None = None,
    cli_thinking: str | None = None,
    cli_timeout: int | None = None,
    cli_max_retries: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("config.defaults must be an object")
    provider_label, provider = selected_provider_config(config, provider_name)

    api_key_env = str(provider.get("api_key_env") or default_api_key_env)
    api_key = cli_api_key or provider.get("api_key") or os.getenv(api_key_env)
    provider_type = str(provider.get("type") or "openai_compatible")
    model_env = provider.get("model_env")
    model_from_env = os.getenv(str(model_env)) if model_env else None
    model = cli_model or model_from_env or provider.get("model") or default_model
    api_url_env = provider.get("api_url_env")
    api_url_from_env = os.getenv(str(api_url_env)) if api_url_env else None
    local_provider = provider_type in {
        "local_qwen",
        "local_transformers_vlm",
        "local_cosmos_transformers",
        "local_internvl",
        "local_cosmos_framework",
    }
    api_url = str(api_url_from_env or provider.get("api_url") or ("" if local_provider else default_api_url))
    temperature = cli_temperature if cli_temperature is not None else provider.get("temperature", defaults.get("temperature", 0.0))
    thinking = cli_thinking if cli_thinking is not None else provider.get("thinking", defaults.get("thinking", "disabled"))
    timeout = cli_timeout if cli_timeout is not None else provider.get("timeout", defaults.get("timeout", 180))
    max_retries = cli_max_retries if cli_max_retries is not None else provider.get("max_retries", defaults.get("max_retries", 3))

    require_api_key = bool(provider.get("require_api_key", not local_provider))
    if not local_provider and require_api_key and not api_key:
        raise ValueError(f"Missing API key. Set {api_key_env} or pass --api-key.")

    # 本地权重的相对路径以「配置文件所在目录的上一级」（即 eval/）为基准，
    # 而不是当前工作目录 —— 否则从 eval/ 里运行会得到 eval/eval/models/...
    if local_provider and model and config_path is not None:
        model_path = Path(str(model))
        if not model_path.is_absolute():
            model = str((Path(config_path).resolve().parent.parent / model_path).resolve())

    frames = resolve_frames(provider, defaults, strict=bool(config.get("strict_frames", False)))

    max_new_tokens = int(provider.get("max_new_tokens", defaults.get("max_new_tokens", 256)))

    # BC-01：推理参数统一。默认关闭 —— 关闭时与冻结脚本逐字节一致，这是 replay 回归的前提。
    # 打开后对本轮 15 个模型的实际影响只有 max_new_tokens 256→1024
    # （走 local_internvl 的三个模型），temperature 本来就都是 0。
    if bool(config.get("unify_generation", False)):
        temperature = 0.0
        thinking = "disabled"
        max_new_tokens = int(config.get("unified_max_new_tokens", 1024))

    return {
        "provider": provider_label,
        "type": provider_type,
        "api_key": str(api_key or ""),
        "api_key_env": api_key_env,
        "require_api_key": require_api_key,
        "model": str(model),
        "api_url": api_url,
        "temperature": float(temperature),
        "thinking": str(thinking),
        "timeout": int(timeout),
        "max_retries": int(max_retries),
        # 并发只对 API provider 生效；本地模型的并行由 GPU worker 池负责，两者不叠加。
        "concurrency": int(provider.get("concurrency", defaults.get("concurrency", 1))),
        "rate_limit": float(provider.get("rate_limit", defaults.get("rate_limit", 0.0))),
        "media_url_format": str(provider.get("media_url_format") or defaults.get("media_url_format") or ("base64" if provider_label == "glm" else "data_url")),
        "send_thinking": bool(provider.get("send_thinking", False)),
        "extra_payload": provider.get("extra_payload", {}),
        "extra_headers": provider.get("extra_headers", {}),
        "max_new_tokens": max_new_tokens,
        "num_segments": frames["num_segments"],
        "video_sample_fps": frames["video_sample_fps"],
        "frames": frames,
        "device_map": provider.get("device_map", defaults.get("device_map")),
        # BC-11：请求体预算。默认 0 = 不做任何媒体处理，超限就让它 413 并如实记录。
        "max_request_bytes": int(provider.get("max_request_bytes", defaults.get("max_request_bytes", 0))),
        "media_cache_dir": str(provider.get("media_cache_dir", defaults.get("media_cache_dir", ""))),
        # BC-13：服务端要求的视频最短时长，0 表示不处理
        "min_video_seconds": float(provider.get("min_video_seconds", defaults.get("min_video_seconds", 0.0))),
        # 加载后覆盖 LLM 的注意力实现。留空 = 不干预，保持模型代码的默认选择。
        "attn_implementation": str(provider.get("attn_implementation", defaults.get("attn_implementation", ""))),
        "max_image_tiles": int(provider.get("max_image_tiles", defaults.get("max_image_tiles", 12))),
        "max_video_tiles": int(provider.get("max_video_tiles", defaults.get("max_video_tiles", 1))),
        "use_flash_attn": bool(provider.get("use_flash_attn", defaults.get("use_flash_attn", True))),
        "cuda_visible_devices": str(provider.get("cuda_visible_devices", defaults.get("cuda_visible_devices", ""))),
        "cosmos_parallelism_preset": str(provider.get("parallelism_preset", defaults.get("parallelism_preset", "latency"))),
        "cosmos_no_guardrails": bool(provider.get("no_guardrails", defaults.get("no_guardrails", True))),
        "cosmos_output_dir": str(provider.get("cosmos_output_dir", defaults.get("cosmos_output_dir", ""))),
        "cosmos_multi_vision_field": str(provider.get("multi_vision_field", defaults.get("multi_vision_field", "vision_paths"))),
        "system_prompt": str(
            provider.get(
                "system_prompt",
                defaults.get(
                    "system_prompt",
                    "You are a strict JSON API. Return only JSON. Do not include markdown or analysis outside JSON.",
                ),
            )
        ),
    }


def openai_content(parts: list[dict[str, Any]], media_url_format: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        part_type = part.get("type")
        if part_type == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif part_type in {"image", "video"}:
            path = Path(str(part["path"]))
            url_key = "image_url" if part_type == "image" else "video_url"
            content.append({"type": url_key, url_key: {"url": media_url(path, part_type, media_url_format)}})
        else:
            raise ValueError(f"Unsupported content part type: {part_type}")
    return content


def gemini_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for part in parts:
        part_type = part.get("type")
        if part_type == "text":
            output.append({"text": str(part.get("text", ""))})
        elif part_type in {"image", "video"}:
            path = Path(str(part["path"]))
            output.append(
                {
                    "inline_data": {
                        "mime_type": media_mime(path, part_type),
                        "data": encode_file(path),
                    }
                }
            )
        else:
            raise ValueError(f"Unsupported content part type: {part_type}")
    return output


def local_qwen_content(
    parts: list[dict[str, Any]], frames: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """组装 qwen_vl_utils 的 messages 内容。

    BC-09：抽帧档位必须显式传给 ``qwen_vl_utils``，否则它走自己的默认值
    （``FPS=2.0``），我们的 fps 协议对这条路径就完全不生效 ——
    而这条路径覆盖 5 个模型（Qwen3-VL ×3、RynnBrain ×2、Cosmos-Reason2）。
    冻结版调 ``process_vision_info(messages)`` 时一个覆盖参数都没传。

    ``qwen_vl_utils`` 读的是视频元素里的 ``fps`` / ``nframes``（二者互斥）。
    """
    frames = frames or {}
    mode = frames.get("mode")
    video_extra: dict[str, Any] = {}
    if mode == "fps" and frames.get("value"):
        video_extra["fps"] = float(frames["value"])
    elif mode == "uniform" and frames.get("value"):
        # nframes 必须是 FRAME_FACTOR(=2) 的倍数，否则库内部会报错
        n = max(2, int(frames["value"]))
        video_extra["nframes"] = n - (n % 2)

    content: list[dict[str, Any]] = []
    for part in parts:
        part_type = part.get("type")
        if part_type == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif part_type == "image":
            content.append({"type": "image", "image": str(part["path"])})
        elif part_type == "video":
            path = str(part["path"])
            # qwen_vl_utils 的 FRAME_FACTOR=2（它做时间维度合并），单帧视频会算出
            # nframes=0 并抛 "nframes should in interval [2, 1]"。
            # stack_cubes 的 planning 里有 8 个 clip 的时间窗只有 0.05 秒 = 1 帧
            # （start/end 相差 0.05），属于生成侧的数据问题。
            # 单帧视频在信息量上就是一张图，按图片送最忠实，也避免整条 run 失败。
            frame = _single_frame_fallback(path)
            if frame is not None:
                content.append({"type": "image", "image": frame})
            else:
                content.append({"type": "video", "video": path, **video_extra})
        else:
            raise ValueError(f"Unsupported content part type for local_qwen: {part_type}")
    return content


def _single_frame_fallback(path: str) -> str | None:
    """单帧视频返回抽出来的图片路径，否则返回 None。

    抽出的帧缓存在视频旁边，只做一次。
    """
    try:
        from decord import VideoReader, cpu

        video = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        if len(video) >= 2:
            return None
        cache = Path(str(path)).with_suffix(".singleframe.jpg")
        if not cache.exists():
            from PIL import Image

            Image.fromarray(video[0].asnumpy()).save(cache, quality=95)
        return str(cache)
    except Exception:  # noqa: BLE001
        return None


def load_local_qwen(model_name: str) -> tuple[Any, Any]:
    global _LOCAL_QWEN_MODEL, _LOCAL_QWEN_PROCESSOR, _LOCAL_QWEN_MODEL_NAME
    if (
        _LOCAL_QWEN_MODEL is not None
        and _LOCAL_QWEN_PROCESSOR is not None
        and _LOCAL_QWEN_MODEL_NAME == model_name
    ):
        return _LOCAL_QWEN_MODEL, _LOCAL_QWEN_PROCESSOR

    import torch

    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.uint8

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading local Qwen model: {model_name}", flush=True)
    _LOCAL_QWEN_PROCESSOR = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    _LOCAL_QWEN_MODEL = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    _LOCAL_QWEN_MODEL.eval()
    _LOCAL_QWEN_MODEL_NAME = model_name
    return _LOCAL_QWEN_MODEL, _LOCAL_QWEN_PROCESSOR


def request_local_qwen(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    local_model, processor = load_local_qwen(runtime["model"])
    messages = [
        {"role": "system", "content": runtime["system_prompt"]},
        {"role": "user", "content": local_qwen_content(parts, runtime.get("frames"))},
    ]
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}

    # BC-09：记录 qwen_vl_utils 实际抽了多少帧
    for index, video in enumerate(video_inputs or []):
        tensor = video[0] if isinstance(video, (tuple, list)) else video
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 1:
            runtime.setdefault("_frames_used", {})[f"video{index}"] = int(shape[0])

    if video_inputs and isinstance(video_inputs[0], tuple):
        videos = []
        video_metadata = []
        for video_input in video_inputs:
            video, metadata = video_input
            videos.append(video)
            video_metadata.append(metadata)
        video_inputs = videos
        video_kwargs["video_metadata"] = video_metadata

    device = next(local_model.parameters()).device
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(device)

    do_sample = runtime["temperature"] > 0
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": runtime["max_new_tokens"],
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = runtime["temperature"]
    eos_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    if eos_token_id is not None:
        generate_kwargs["pad_token_id"] = eos_token_id


    with torch.no_grad():
        output_ids = local_model.generate(**inputs, **generate_kwargs)
    new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return {
        "choices": [{"message": {"content": output_text}}],
        "model": runtime["model"],
        "local_weights": runtime["model"],
    }


def load_local_transformers_vlm(model_name: str) -> tuple[Any, Any]:
    cached = _LOCAL_TRANSFORMERS_VLM_MODELS.get(model_name)
    if cached is not None:
        return cached

    import torch

    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.uint8

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = Path(model_name).expanduser()
    resolved_model = str(model_path) if model_path.exists() else model_name
    print(f"Loading local transformers VLM: {resolved_model}", flush=True)
    processor = AutoProcessor.from_pretrained(resolved_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        resolved_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    _LOCAL_TRANSFORMERS_VLM_MODELS[model_name] = (model, processor)
    return model, processor


def request_local_transformers_vlm(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    local_model, processor = load_local_transformers_vlm(runtime["model"])
    messages = [
        {"role": "system", "content": runtime["system_prompt"]},
        {"role": "user", "content": local_qwen_content(parts, runtime.get("frames"))},
    ]
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}

    # BC-09：记录 qwen_vl_utils 实际抽了多少帧
    for index, video in enumerate(video_inputs or []):
        tensor = video[0] if isinstance(video, (tuple, list)) else video
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 1:
            runtime.setdefault("_frames_used", {})[f"video{index}"] = int(shape[0])

    if video_inputs and isinstance(video_inputs[0], tuple):
        videos = []
        video_metadata = []
        for video_input in video_inputs:
            video, metadata = video_input
            videos.append(video)
            video_metadata.append(metadata)
        video_inputs = videos
        video_kwargs["video_metadata"] = video_metadata

    device = next(local_model.parameters()).device
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(device)

    do_sample = runtime["temperature"] > 0
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": runtime["max_new_tokens"],
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = runtime["temperature"]
    eos_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    if eos_token_id is not None:
        generate_kwargs["pad_token_id"] = eos_token_id


    with torch.no_grad():
        output_ids = local_model.generate(**inputs, **generate_kwargs)
    new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return {
        "choices": [{"message": {"content": output_text}}],
        "model": runtime["model"],
        "local_weights": runtime["model"],
    }


def local_cosmos_content(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        part_type = part.get("type")
        if part_type == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif part_type in {"image", "video"}:
            path = Path(str(part["path"])).expanduser()
            resolved = path if path.is_absolute() else path.resolve()
            content.append({"type": str(part_type), "url": str(resolved)})
        else:
            raise ValueError(f"Unsupported content part type for local_cosmos_transformers: {part_type}")
    return content


def load_local_cosmos_transformers(model_name: str) -> tuple[Any, Any]:
    cached = _LOCAL_COSMOS_TRANSFORMERS_MODELS.get(model_name)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = Path(model_name).expanduser()
    resolved_model = str(model_path) if model_path.exists() else model_name
    print(f"Loading local Cosmos3 Edge transformers model: {resolved_model}", flush=True)
    processor = AutoProcessor.from_pretrained(resolved_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        resolved_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    _LOCAL_COSMOS_TRANSFORMERS_MODELS[model_name] = (model, processor)
    return model, processor


def _cosmos_frame_kwargs(runtime: dict[str, Any]) -> dict[str, Any]:
    """把 runtime 的 frames 声明翻成 Cosmos3EdgeProcessor 认识的参数。

    实测（transformers 5.15.0，17.70 秒的 clip）：

        processor 自带默认           35 帧   ← 等同 fps=2
        {"fps": 1.0}                 17 帧
        {"fps": 2.0}                 35 帧
        {"num_frames": 8, "fps": None}   8 帧

    ``num_frames`` 与 ``fps`` 在这个 processor 里是互斥的，只给 num_frames
    会因为自带的 fps=2 还在而报 ValueError，所以必须同时把 fps 置空。

    注意 processor 的默认是 fps=2，不是「不抽帧」—— 所以 ``mode: native``
    对这个模型来说实际等于 fps=2，报告里的 default 档要这样理解。
    """
    frames = runtime.get("frames") or {}
    mode = str(frames.get("mode") or "")
    value = frames.get("value")
    if mode == "fps" and value:
        return {"fps": float(value)}
    if mode == "uniform" and value:
        return {"num_frames": int(value), "fps": None}
    return {}


def request_local_cosmos_transformers(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    local_model, processor = load_local_cosmos_transformers(runtime["model"])
    messages = [
        {"role": "system", "content": runtime["system_prompt"]},
        {"role": "user", "content": local_cosmos_content(parts)},
    ]

    # BC-09：抽帧档位必须传给 processor。不传的话它用自带默认 `fps=2`，
    # 于是 fps=1 与 fps=2 两档跑出**逐位相同**的结果 —— 协议形同虚设，
    # 而且因为不报错，只有对比两档的数值才看得出来。
    #
    # 走 `processor_kwargs` 而不是 `**kwargs`：两者实测等价，但前者是
    # apply_chat_template 签名里正式声明的通道，且不会刷「kwargs 应该放进
    # processor_kwargs」的警告。
    processor_kwargs = _cosmos_frame_kwargs(runtime)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **({"processor_kwargs": processor_kwargs} if processor_kwargs else {}),
    )

    # video_grid_thw 的时间维就是实际用到的帧数，直接读，不用反推
    grid = inputs.get("video_grid_thw")
    if grid is not None:
        for index, row in enumerate(grid):
            runtime.setdefault("_frames_used", {})[f"video{index}"] = int(row[0])

    device = getattr(local_model, "device", None) or next(local_model.parameters()).device
    inputs = inputs.to(device)

    do_sample = runtime["temperature"] > 0
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": runtime["max_new_tokens"],
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = runtime["temperature"]
    eos_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    if eos_token_id is not None:
        generate_kwargs["pad_token_id"] = eos_token_id


    with torch.no_grad():
        output_ids = local_model.generate(**inputs, **generate_kwargs)
    new_ids = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs["input_ids"], output_ids)
    ]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return {
        "choices": [{"message": {"content": output_text}}],
        "model": runtime["model"],
        "local_weights": runtime["model"],
    }


def cosmos_part_prompt_and_media(parts: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    prompt_chunks: list[str] = []
    media: list[dict[str, str]] = []
    pending_label = ""
    for part in parts:
        part_type = part.get("type")
        if part_type == "text":
            text = str(part.get("text", "")).strip()
            if text:
                prompt_chunks.append(text)
                pending_label = text
        elif part_type in {"image", "video"}:
            path = Path(str(part["path"])).expanduser()
            media.append(
                {
                    "type": str(part_type),
                    "path": str(path if path.is_absolute() else path.resolve()),
                    "label": pending_label,
                }
            )
            pending_label = ""
        else:
            raise ValueError(f"Unsupported content part type for local_cosmos_framework: {part_type}")
    return "\n\n".join(prompt_chunks), media


def make_cosmos_contact_sheet(images: list[dict[str, str]], output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not images:
        raise ValueError("Cannot build Cosmos contact sheet with no images")
    thumb_w, thumb_h = 384, 288
    label_h = 52
    columns = min(3, len(images))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, item in enumerate(images):
        source = Image.open(item["path"]).convert("RGB")
        source.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        image_x = x + (thumb_w - source.width) // 2
        image_y = y + label_h + (thumb_h - source.height) // 2
        label = item.get("label") or f"Image {index + 1}"
        draw.text((x + 8, y + 8), label[:90], fill=(0, 0, 0), font=font)
        sheet.paste(source, (image_x, image_y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def cosmos_vision_payload(media: list[dict[str, str]], work_dir: Path, multi_field: str) -> dict[str, Any]:
    if not media:
        return {}
    videos = [item for item in media if item["type"] == "video"]
    images = [item for item in media if item["type"] == "image"]
    paths = [item["path"] for item in videos + images]
    if len(paths) > 1:
        return {multi_field: paths}
    if videos:
        return {"vision_path": videos[0]["path"]}
    if len(images) == 1:
        return {"vision_path": images[0]["path"]}
    output_path = work_dir / "cosmos_contact_sheet.jpg"
    make_cosmos_contact_sheet(images, output_path)
    return {"vision_path": str(output_path)}


def request_local_cosmos_framework(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    prompt, media = cosmos_part_prompt_and_media(parts)
    if runtime.get("system_prompt"):
        prompt = f"{runtime['system_prompt']}\n\n{prompt}".strip()

    output_root = Path(runtime.get("cosmos_output_dir") or tempfile.mkdtemp(prefix="cosmos_framework_out_"))
    work_dir = output_root / f"request_{int(time.time() * 1000)}"
    work_dir.mkdir(parents=True, exist_ok=True)
    sample_path = work_dir / "sample.json"
    sample: dict[str, Any] = {
        "model_mode": "reasoner",
        "prompt": prompt,
    }
    sample.update(cosmos_vision_payload(media, work_dir, runtime["cosmos_multi_vision_field"]))
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        "python",
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset",
        runtime["cosmos_parallelism_preset"],
        "-i",
        str(sample_path),
        "-o",
        str(work_dir / "outputs"),
        "--checkpoint-path",
        runtime["model"],
        "--seed",
        "0",
    ]
    if runtime.get("cosmos_no_guardrails", True):
        cmd.append("--no-guardrails")

    env = os.environ.copy()
    if runtime.get("cuda_visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = runtime["cuda_visible_devices"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=runtime["timeout"], env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "cosmos-framework inference failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout={result.stdout[-2000:]}\n"
            f"stderr={result.stderr[-2000:]}"
        )

    outputs = sorted((work_dir / "outputs").rglob("reasoner_text.txt"))
    if not outputs:
        sample_outputs = sorted((work_dir / "outputs").rglob("sample_outputs.json"))
        raise RuntimeError(
            "cosmos-framework inference finished but reasoner_text.txt was not found. "
            f"sample_outputs={sample_outputs}"
        )
    output_text = outputs[0].read_text(encoding="utf-8").strip()
    return {
        "choices": [{"message": {"content": output_text}}],
        "model": runtime["model"],
        "local_weights": runtime["model"],
        "cosmos_sample": str(sample_path),
        "cosmos_output": str(outputs[0]),
    }


def build_internvl_transform(input_size: int) -> Any:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: set[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess_internvl(
    image: Any,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Any]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_internvl_image(image_path: Path, input_size: int, max_num: int) -> Any:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    images = dynamic_preprocess_internvl(
        image,
        image_size=input_size,
        max_num=max_num,
        use_thumbnail=True,
    )
    transform = build_internvl_transform(input_size)
    return torch.stack([transform(tile) for tile in images])


def internvl_frame_indices(video_path: Path, num_segments: int, sample_fps: float = 0.0) -> list[int]:
    from decord import VideoReader, cpu

    video = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    max_frame = len(video) - 1
    if max_frame < 0:
        raise ValueError(f"Empty video file: {video_path}")
    if sample_fps > 0:
        source_fps = float(video.get_avg_fps())
        if source_fps <= 0:
            raise ValueError(f"Cannot determine FPS for video file: {video_path}")
        duration = (max_frame + 1) / source_fps
        frame_count = max(1, int(duration * sample_fps))
        return [
            min(max_frame, int(round(index * source_fps / sample_fps)))
            for index in range(frame_count)
        ]
    if num_segments <= 1:
        return [max_frame // 2]
    seg_size = max_frame / num_segments
    return [
        min(max_frame, int((seg_size / 2) + round(seg_size * idx)))
        for idx in range(num_segments)
    ]


def load_internvl_video(
    video_path: Path,
    input_size: int,
    max_num: int,
    num_segments: int,
    sample_fps: float,
) -> tuple[Any, list[int]]:
    import torch
    from decord import VideoReader, cpu
    from PIL import Image

    video = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    transform = build_internvl_transform(input_size)
    pixel_values_list = []
    num_patches_list = []
    for frame_index in internvl_frame_indices(video_path, num_segments, sample_fps):
        image = Image.fromarray(video[frame_index].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess_internvl(
            image,
            image_size=input_size,
            max_num=max_num,
            use_thumbnail=True,
        )
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
    return torch.cat(pixel_values_list), num_patches_list


def _dtype_kwargs(dtype: Any) -> list[dict[str, Any]]:
    """transformers 4.56 起 ``torch_dtype`` 改名为 ``dtype``。

    冻结代码只用了新名字，导致在 4.51 上这个 kwarg 会被透传给模型构造函数并报
    ``unexpected keyword argument 'dtype'``。这里两种写法都试，
    以兼容各模型官方要求的不同 transformers 版本
    （RynnBrain-2B 要 4.57.1，RynnBrain1.1-2B 要 5.2.0）。
    """
    return [{"dtype": dtype}, {"torch_dtype": dtype}]


def load_local_internvl(
    model_name: str,
    use_flash_attn: bool,
    cuda_visible_devices: str = "",
    device_map: Any = None,
    attn_implementation: str = "",
) -> tuple[Any, Any]:
    global _LOCAL_INTERNVL_MODEL, _LOCAL_INTERNVL_TOKENIZER, _LOCAL_INTERNVL_MODEL_NAME
    if (
        _LOCAL_INTERNVL_MODEL is not None
        and _LOCAL_INTERNVL_TOKENIZER is not None
        and _LOCAL_INTERNVL_MODEL_NAME == model_name
    ):
        return _LOCAL_INTERNVL_MODEL, _LOCAL_INTERNVL_TOKENIZER

    if cuda_visible_devices:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible_devices)

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local InternVL model path does not exist: {model_name}. "
            "Pass --model /path/to/InternVL3.5-8B or set INTERNVL_MODEL_PATH."
        )

    print(f"Loading local InternVL model: {model_path}", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # 冻结版把 device_map 写死成 {"": 0}，单卡以外无法部署。改为可配置：
    # 单卡传 {"": 0}，跨卡传 "auto"（权重按层切分；注意这只聚合权重容量，
    # 不解决单个大张量的分配 —— 见 eval/docs/frame_sampling_investigation.md）。
    if device_map is None:
        device_map = {"": 0} if torch.cuda.is_available() else None

    last_error: Exception | None = None
    for dtype_kwarg in _dtype_kwargs(dtype):
        for extra in ({"use_flash_attn": use_flash_attn}, {}):
            try:
                _LOCAL_INTERNVL_MODEL = AutoModel.from_pretrained(
                    str(model_path),
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map=device_map,
                    **dtype_kwarg,
                    **extra,
                ).eval()
                last_error = None
                break
            except TypeError as exc:
                last_error = exc
        if last_error is None:
            break
    if last_error is not None:
        raise last_error
    # InternVL 的模型代码把 LLM 的注意力实现写死成二选一：
    #     config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
    # 而 'eager' 会把 num_heads × seq² 的注意力矩阵完整物化成 fp32 —— 这正是
    # Time EQA OOM 的根因。第三个选项 'sdpa' 被漏掉了：它同样不物化该矩阵，
    # 且 PyTorch 自带、无需编译安装 flash-attn。
    #
    # 构造函数里那行会覆盖预设的 config，所以只能加载后再改；
    # transformers 4.57 在 forward 时按 config 分发注意力实现，因此有效。
    #
    # 实测（SenseNova-2B，24 GiB 4090）：
    #     eager  32 帧 12.55 GiB，48 帧起 OOM
    #     sdpa   32 帧  6.31 GiB，160 帧 15.98 GiB，显存随帧数线性增长
    # 12 道真实题目上 eager 与 sdpa 的输出逐字节相同。
    if attn_implementation:
        language_model = getattr(_LOCAL_INTERNVL_MODEL, "language_model", None)
        if language_model is not None:
            language_model.config._attn_implementation = attn_implementation
            for module in language_model.modules():
                module_config = getattr(module, "config", None)
                if module_config is not None and hasattr(module_config, "_attn_implementation"):
                    module_config._attn_implementation = attn_implementation
            print(f"  LLM attn_implementation -> {attn_implementation}", flush=True)

    _LOCAL_INTERNVL_TOKENIZER = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
        fix_mistral_regex=True,
    )
    _LOCAL_INTERNVL_MODEL_NAME = model_name
    return _LOCAL_INTERNVL_MODEL, _LOCAL_INTERNVL_TOKENIZER


def video_duration_seconds(path: Path) -> float:
    from decord import VideoReader, cpu

    video = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    fps = float(video.get_avg_fps())
    if fps <= 0:
        raise ValueError(f"Cannot determine FPS for video file: {path}")
    return len(video) / fps


def frames_for_video(path: Path, runtime: dict[str, Any]) -> tuple[int, float]:
    """BC-09：把抽帧策略换算成本次调用的 (num_segments, sample_fps)。

    ``mode: fps`` 时按每个视频的时长在**运行时**算出帧数
    （``round(时长 × fps)``），走均分分支 —— 这样与按 fps 采样的 adapter
    看到的帧数一致，也是团队定的对齐口径。冻结版是静态配置值，
    且 fps 分支会静默架空 num_segments。

    ``align_fps_to_segments`` 默认关闭，保持与冻结脚本一致（直接用 fps 分支）。
    BC-09 的协议定下来之后打开它即可，无需改代码。
    """
    frames = runtime.get("frames") or {}
    mode = frames.get("mode")

    if mode == "fps" and runtime.get("align_fps_to_segments"):
        fps = float(frames.get("value") or 0.0)
        if fps > 0:
            n = max(1, round(video_duration_seconds(path) * fps))
            return n, 0.0

    if mode == "uniform":
        return int(frames.get("value") or runtime["num_segments"]), 0.0

    return int(runtime["num_segments"]), float(runtime["video_sample_fps"])


def request_local_internvl(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    local_model, tokenizer = load_local_internvl(
        runtime["model"],
        runtime["use_flash_attn"],
        runtime.get("cuda_visible_devices", ""),
        runtime.get("device_map"),
        runtime.get("attn_implementation", ""),
    )
    input_size = int(getattr(getattr(local_model, "config", None), "force_image_size", 448) or 448)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pixel_values_list = []
    num_patches_list = []
    prompt_chunks: list[str] = []
    text_chunks: list[str] = []
    media_index = 1

    for part in parts:
        part_type = part.get("type")
        if part_type == "text":
            text_chunks.append(str(part.get("text", "")))
        elif part_type == "image":
            path = Path(str(part["path"]))
            pixel_values = load_internvl_image(path, input_size, runtime["max_image_tiles"])
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            prompt_chunks.append(f"Image{media_index}: <image>\n")
            media_index += 1
        elif part_type == "video":
            path = Path(str(part["path"]))
            num_segments, sample_fps = frames_for_video(path, runtime)
            video_pixels, frame_patches = load_internvl_video(
                path,
                input_size,
                runtime["max_video_tiles"],
                num_segments,
                sample_fps,
            )
            # BC-09：记录实际抽了多少帧。冻结版没有这个信息，事后无法复盘。
            runtime.setdefault("_frames_used", {})[path.name] = len(frame_patches)
            pixel_values_list.append(video_pixels)
            num_patches_list.extend(frame_patches)
            prompt_chunks.extend(
                f"Video{media_index} Frame{i + 1}: <image>\n"
                for i in range(len(frame_patches))
            )
            media_index += 1
        else:
            raise ValueError(f"Unsupported content part type for local_internvl: {part_type}")

    if not pixel_values_list:
        raise ValueError("local_internvl requires at least one image or video part")

    device = next(local_model.parameters()).device
    pixel_values = torch.cat(pixel_values_list).to(dtype).to(device)
    user_prompt = "\n".join(chunk for chunk in text_chunks if chunk)
    question = "".join(prompt_chunks) + user_prompt
    if runtime.get("system_prompt"):
        question = f"{runtime['system_prompt']}\n\n{question}"

    do_sample = runtime["temperature"] > 0
    generation_config: dict[str, Any] = {
        "max_new_tokens": runtime["max_new_tokens"],
        "do_sample": do_sample,
    }
    if do_sample:
        generation_config["temperature"] = runtime["temperature"]


    with torch.no_grad():
        output_text = local_model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
    if isinstance(output_text, tuple):
        output_text = output_text[0]
    return {
        "choices": [{"message": {"content": str(output_text).strip()}}],
        "model": runtime["model"],
        "local_weights": runtime["model"],
    }


def response_text(response: dict[str, Any], provider_type: str = "openai_compatible") -> str:
    if provider_type == "gemini":
        try:
            parts = response["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected Gemini API response: {response}") from exc
        return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected chat/completions API response: {response}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def request_replay(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    """replay provider：不调模型，返回事先录好的输出。

    用于回归比对与离线重算指标 —— 指标口径改了之后可以拿历史输出重新打分，
    不必重新花钱/占卡跑一遍。录制内容由 runtime["replay_table"] 提供，
    键是 unit key（多数任务是 item id，time 是 video_id）。
    """
    key = str(runtime.get("replay_key") or "")
    table = runtime.get("replay_table") or {}
    if key not in table:
        raise KeyError(f"replay: no recorded output for key {key!r}")
    return {"choices": [{"message": {"content": table[key]}}], "_replay": True}


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    """重试等待时长。

    服务端给了 ``Retry-After`` 就听它的 —— 它比我们的猜测准。
    否则指数退避 **加抖动**：并发路径下多个线程往往同时撞上 429，
    没有抖动它们会锁步重试、再次同时撞上，等于没退避。
    """
    if retry_after:
        try:
            # 只支持 delta-seconds 形式；HTTP-date 形式罕见，落到指数退避即可
            return max(0.0, min(120.0, float(retry_after)))
        except ValueError:
            pass
    return min(60.0, 2 ** attempt) * random.uniform(0.5, 1.5)


def request_json(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    provider_type = runtime["type"]
    if provider_type == "replay":
        return request_replay(runtime, parts)
    if provider_type == "local_qwen":
        return request_local_qwen(runtime, parts)
    if provider_type == "local_transformers_vlm":
        return request_local_transformers_vlm(runtime, parts)
    if provider_type == "local_cosmos_transformers":
        return request_local_cosmos_transformers(runtime, parts)
    if provider_type == "local_internvl":
        return request_local_internvl(runtime, parts)
    if provider_type == "local_cosmos_framework":
        return request_local_cosmos_framework(runtime, parts)
    if provider_type == "gemini":
        api_url = runtime["api_url"].format(model=runtime["model"])
        separator = "&" if "?" in api_url else "?"
        if "key=" not in api_url:
            api_url = f"{api_url}{separator}key={runtime['api_key']}"
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": gemini_parts(parts)}],
            "generationConfig": {"temperature": runtime["temperature"]},
        }
        payload.update(runtime.get("extra_payload") or {})
        headers = {"Content-Type": "application/json", **(runtime.get("extra_headers") or {})}
    elif provider_type in {"openai_compatible", "glm", "qwen"}:
        api_url = runtime["api_url"]
        payload = {
            "model": runtime["model"],
            "messages": [{"role": "user", "content": openai_content(parts, runtime["media_url_format"])}],
            "temperature": runtime["temperature"],
            **(runtime.get("extra_payload") or {}),
        }
        if runtime.get("send_thinking") and runtime.get("thinking") is not None:
            payload.setdefault("thinking", {"type": runtime["thinking"]})
        headers = {
            "Authorization": f"Bearer {runtime['api_key']}",
            "Content-Type": "application/json",
            **(runtime.get("extra_headers") or {}),
        }
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")

    for attempt in range(1, runtime["max_retries"] + 1):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=runtime["timeout"])
            if response.status_code in {429, 500, 502, 503, 504} and attempt < runtime["max_retries"]:
                wait = _backoff(attempt, response.headers.get("Retry-After"))
                print(f"  HTTP {response.status_code}, retrying in {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == runtime["max_retries"]:
                raise
            wait = _backoff(attempt)
            print(f"  connection error (attempt {attempt}/{runtime['max_retries']}), retrying in {wait:.1f}s: {exc}", flush=True)
            time.sleep(wait)
        except HTTPError as exc:
            try:
                detail = json.dumps(response.json(), ensure_ascii=False, indent=2)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"VLM API request failed: {response.status_code} {response.reason}\n{detail}") from exc

    raise RuntimeError("VLM API request failed without response")


def call_vlm(
    runtime: dict[str, Any],
    parts: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """调用模型。

    ``meta`` 若给出，会被填入本次调用的可观测信息（BC-09 的管道改造）：

      frames_used  各视频实际抽了多少帧。本地 adapter 直接记录；
                   API provider 拿不到，留空。
      usage        服务端返回的 token 用量。冻结版把整个原始响应丢弃了，
                   导致事后既无法核对成本，也无法反推服务端抽了多少帧。
    """
    provider_type = runtime["type"]
    runtime["_frames_used"] = {}


    # BC-11：远程 provider 有请求体上限，超了会 413。
    # BC-13：部分服务端还有视频最短时长约束，短片段会被 400 拒收。
    # 两者都只在配置里显式给了阈值时才处理。
    media_transforms: list[dict[str, Any]] = []
    budget = int(runtime.get("max_request_bytes") or 0)
    min_seconds = float(runtime.get("min_video_seconds") or 0.0)
    if (budget > 0 or min_seconds > 0) and provider_type in {"openai_compatible", "gemini", "glm", "qwen"}:
        from .media_prep import prepare_parts

        cache_dir = Path(runtime.get("media_cache_dir") or ".media_cache")
        parts, media_transforms = prepare_parts(
            parts, budget or (1 << 62), cache_dir, min_video_seconds=min_seconds)

    raw = request_json(runtime, parts)
    text = response_text(raw, provider_type)

    if meta is not None:
        if media_transforms:
            meta["media_transforms"] = media_transforms
        frames = runtime.pop("_frames_used", {}) or {}
        if frames:
            meta["frames_used"] = frames
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if isinstance(usage, dict):
            meta["usage"] = usage
    runtime.pop("_frames_used", None)
    return raw, text
