#!/usr/bin/env python3
# coding: utf-8
"""L3：用一个纯文本 LLM 把自由输出转成结构化答案。

只在 L1/L2 都失败时调用 —— 实测本地结果 11,510 行里 L1 命中 86%，
需要 L3 的只有 14%，而且高度集中：Qwen3-VL 全部 100% 命中不用调，
SenseNova 的 understanding 只有 11~18% 命中。

选 DeepSeek 的理由：**它不在被测的 15 个模型里**，没有「自己提取自己」的
利益冲突。提取是纯文本任务，不需要视觉能力。

三条必须守住的约束
------------------
**① 可复现。** LLM 有随机性，同一条输入两次提取可能不同，那会打破
   ``test_replay_regression`` 的「相同输入 → 相同分数」保证。
   对策是**内容寻址缓存**：键为 (输出文本, 任务类型, 提取器版本) 的哈希，
   命中即返回，同一条输出全生命周期只提取一次。缓存随结果一起交付、可审计。

**② 不许凭空造答案。** 模型没给答案时，提取器"帮忙"编一个是最危险的失败模式 ——
   比漏掉更糟，因为它把「明显的格式失败」变成「看似有效的低分」。
   对策有两道：prompt 里明确要求没有就返回 null；返回值仍然过 ``extract`` 里
   那套合理性校验（维度、量级、画布范围）。那套校验实测拦下过 25 条假阳性。

**③ 可检测偏向。** 提取器可能偏袒某些输出格式，那样只是把污染从生成端
   搬到提取端。对策是逐模型统计 L1 命中率与 L3 成功率，两个数都进报告。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .extract import Extraction, validate_trajectory

# 提取器版本。**改动 prompt 或解析逻辑时必须递增** —— 它是缓存键的一部分，
# 不递增会让旧缓存冒充新行为。
EXTRACTOR_VERSION = "v1"

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_KEY_ENV = "DEEPSEEK_API_KEY"


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------


class ExtractionCache:
    """内容寻址的 JSONL 缓存。同一条模型输出只提取一次。

    用 JSONL 而不是单个 JSON：可追加、可并发写、坏了一行不影响其余
    （与 ResultStore 同样的理由）。
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._memory: dict[str, Any] = {}
        self._loaded = False

    @staticmethod
    def key(text: str, kind: str, extra: str = "") -> str:
        raw = f"{EXTRACTOR_VERSION}\x00{kind}\x00{extra}\x00{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # 尾部残行，与 ResultStore 同样处理
            self._memory[row["key"]] = row.get("value")

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            self._load()
            if key in self._memory:
                return True, self._memory[key]
        return False, None

    def put(self, key: str, value: Any, meta: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._load()
            self._memory[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "value": value, **(meta or {})},
                                        ensure_ascii=False) + "\n")

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._load()
            return {"entries": len(self._memory),
                    "null_results": sum(1 for v in self._memory.values() if v is None)}


# --------------------------------------------------------------------------
# 提取器
# --------------------------------------------------------------------------


@dataclass
class LlmExtractor:
    """把自由文本转成结构化答案。``api_key`` 为空时只读缓存、不发请求。"""

    cache: ExtractionCache
    model: str = DEFAULT_MODEL
    api_url: str = DEFAULT_API_URL
    api_key: str = ""
    timeout: int = 60
    max_retries: int = 3
    offline: bool = False        # True 时缓存未命中直接返回 none，用于离线复算

    @classmethod
    def from_env(cls, cache_path: Path, **kwargs: Any) -> "LlmExtractor":
        return cls(cache=ExtractionCache(cache_path),
                   api_key=os.getenv(DEFAULT_KEY_ENV, ""), **kwargs)

    # -- 三种任务 ----------------------------------------------------------

    def trajectory(self, text: str, *, gripper: str, dim: int,
                   expected_count: int, canvas: tuple[int, int] | None = None) -> Extraction:
        kind = f"trajectory{dim}d"
        extra = f"{gripper}|{expected_count}|{canvas}"
        unit = "meters in the camera frame" if dim == 3 else "pixels in the main-view image"
        shape = "[x, y, z]" if dim == 3 else "[u, v]"
        instruction = (
            f"Extract the predicted trajectory for the {gripper} from the text below.\n"
            f"Coordinates are {unit}; each point is {shape}.\n"
            f"About {expected_count} points are expected.\n\n"
            "Return ONLY this JSON, nothing else:\n"
            '{"points": [[...], [...]]}\n'
            'If the text does not contain a trajectory for that gripper, return {"points": null}.\n'
            "Do NOT invent coordinates. Do NOT convert between units or coordinate systems.\n"
            "Copy the numbers exactly as they appear."
        )

        def parse(payload: Any) -> Any:
            points = (payload or {}).get("points")
            if not isinstance(points, list):
                return None
            clean = [[float(v) for v in p] for p in points
                     if isinstance(p, list) and all(isinstance(v, (int, float)) for v in p)]
            reason = validate_trajectory(clean, dim=dim, expected_count=expected_count, canvas=canvas)
            return None if reason else clean

        return self._run(text, kind, extra, instruction, parse)

    def choice(self, text: str, *, valid_ids: list[str], option_texts: dict[str, str]) -> Extraction:
        ids = [i.upper() for i in valid_ids]
        listing = "\n".join(f"  {k}: {v or '(image option, no text)'}"
                            for k, v in option_texts.items())
        instruction = (
            "Extract which option the text below selects.\n"
            f"Valid option ids: {', '.join(ids)}\n"
            f"Options:\n{listing}\n\n"
            'Return ONLY this JSON: {"choice": "<id>"}\n'
            'If the text does not clearly select one option, return {"choice": null}.\n'
            "Do NOT guess. If two different options are mentioned with equal weight, return null."
        )

        def parse(payload: Any) -> Any:
            token = str((payload or {}).get("choice") or "").strip().upper()
            return token if token in ids else None

        return self._run(text, "choice", "|".join(ids), instruction, parse)

    def intervals(self, text: str, *, question_ids: list[str]) -> Extraction:
        listing = "\n".join(f"  {i}" for i in question_ids)
        instruction = (
            "Extract the predicted time interval for each question id from the text below.\n"
            f"Question ids:\n{listing}\n\n"
            "Return ONLY this JSON, seconds as numbers:\n"
            '{"intervals": {"<id>": [start, end]}}\n'
            "Omit any id the text does not answer. Do NOT invent times.\n"
            "Convert HH:MM:SS.mmm to seconds; strip any unit suffix such as 's'."
        )

        def parse(payload: Any) -> Any:
            raw = (payload or {}).get("intervals")
            if not isinstance(raw, dict):
                return None
            out: dict[str, tuple[float, float]] = {}
            for item_id, pair in raw.items():
                if item_id in question_ids and isinstance(pair, list) and len(pair) == 2:
                    try:
                        out[item_id] = (float(pair[0]), float(pair[1]))
                    except (TypeError, ValueError):
                        continue
            return out or None

        return self._run(text, "intervals", "|".join(question_ids), instruction, parse)

    # -- 内部 --------------------------------------------------------------

    def _run(self, text: str, kind: str, extra: str, instruction: str, parse: Any) -> Extraction:
        key = ExtractionCache.key(text, kind, extra)
        hit, cached = self.cache.get(key)
        if hit:
            return Extraction(cached, "model_cached" if cached is not None else "none",
                              [] if cached is not None else ["cached: 提取器判定无可用答案"])

        if self.offline or not self.api_key:
            reason = "offline: 缓存未命中且未启用在线提取" if self.offline else \
                     f"未设置 {DEFAULT_KEY_ENV}，跳过 LLM 提取"
            return Extraction(None, "none", [reason])

        try:
            payload = self._call(instruction, text)
        except Exception as exc:  # noqa: BLE001  提取失败不该让整题失败
            return Extraction(None, "none", [f"model: 调用失败 {type(exc).__name__}: {exc}"])

        value = parse(payload)
        self.cache.put(key, value, {"kind": kind, "model": self.model})
        if value is None:
            return Extraction(None, "none", ["model: 提取器判定无可用答案或未通过校验"])
        return Extraction(value, "model", [])

    def _call(self, instruction: str, text: str) -> Any:
        import requests

        body = {
            "model": self.model,
            "temperature": 0.0,            # 提取必须尽可能确定
            "messages": [
                {"role": "system",
                 "content": "You extract structured data from text. You never invent values. "
                            "You output only JSON."},
                {"role": "user", "content": f"{instruction}\n\n--- TEXT ---\n{text}"},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(self.api_url, headers=headers, json=body,
                                         timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    import random
                    import time
                    time.sleep(min(30.0, 2 ** attempt) * random.uniform(0.5, 1.5))
                    continue
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return json.loads(_strip_fence(content))
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt == self.max_retries:
                    raise
        raise last if last else RuntimeError("unreachable")


def _strip_fence(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.M).strip()
