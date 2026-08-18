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
        "media_url_format": str(provider.get("media_url_format") or defaults.get("media_url_format") or ("base64" if provider_label == "glm" else "data_url")),
        "send_thinking": bool(provider.get("send_thinking", False)),
        "extra_payload": provider.get("extra_payload", {}),
        "extra_headers": provider.get("extra_headers", {}),
        "max_new_tokens": int(provider.get("max_new_tokens", defaults.get("max_new_tokens", 256))),
        "num_segments": int(provider.get("num_segments", defaults.get("num_segments", 16))),
        "video_sample_fps": float(provider.get("video_sample_fps", defaults.get("video_sample_fps", 0.0))),
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


def local_qwen_content(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        part_type = part.get("type")
        if part_type == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif part_type == "image":
            content.append({"type": "image", "image": str(part["path"])})
        elif part_type == "video":
            content.append({"type": "video", "video": str(part["path"])})
        else:
            raise ValueError(f"Unsupported content part type for local_qwen: {part_type}")
    return content


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
        {"role": "user", "content": local_qwen_content(parts)},
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
        {"role": "user", "content": local_qwen_content(parts)},
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


def request_local_cosmos_transformers(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    local_model, processor = load_local_cosmos_transformers(runtime["model"])
    messages = [
        {"role": "system", "content": runtime["system_prompt"]},
        {"role": "user", "content": local_cosmos_content(parts)},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
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


def load_local_internvl(model_name: str, use_flash_attn: bool, cuda_visible_devices: str = "") -> tuple[Any, Any]:
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
    try:
        _LOCAL_INTERNVL_MODEL = AutoModel.from_pretrained(
            str(model_path),
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
            device_map={"": 0} if torch.cuda.is_available() else None,
        ).eval()
    except TypeError:
        _LOCAL_INTERNVL_MODEL = AutoModel.from_pretrained(
            str(model_path),
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": 0} if torch.cuda.is_available() else None,
        ).eval()
    _LOCAL_INTERNVL_TOKENIZER = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
        fix_mistral_regex=True,
    )
    _LOCAL_INTERNVL_MODEL_NAME = model_name
    return _LOCAL_INTERNVL_MODEL, _LOCAL_INTERNVL_TOKENIZER


def request_local_internvl(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    local_model, tokenizer = load_local_internvl(
        runtime["model"],
        runtime["use_flash_attn"],
        runtime.get("cuda_visible_devices", ""),
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
            video_pixels, frame_patches = load_internvl_video(
                path,
                input_size,
                runtime["max_video_tiles"],
                runtime["num_segments"],
                runtime["video_sample_fps"],
            )
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


def request_json(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> dict[str, Any]:
    provider_type = runtime["type"]
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
                wait = min(60, 2 ** attempt)
                print(f"  HTTP {response.status_code}, retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == runtime["max_retries"]:
                raise
            wait = min(60, 2 ** attempt)
            print(f"  connection error (attempt {attempt}/{runtime['max_retries']}), retrying in {wait}s: {exc}", flush=True)
            time.sleep(wait)
        except HTTPError as exc:
            try:
                detail = json.dumps(response.json(), ensure_ascii=False, indent=2)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"VLM API request failed: {response.status_code} {response.reason}\n{detail}") from exc

    raise RuntimeError("VLM API request failed without response")


def call_vlm(runtime: dict[str, Any], parts: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    raw = request_json(runtime, parts)
    return raw, response_text(raw, runtime["type"])
