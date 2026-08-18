#!/usr/bin/env python3
# coding: utf-8
"""修复 trust_remote_code 模型在本地加载时的 auto_map 跨仓库引用。

问题
----
部分模型的 config.json 把 auto_map 写成跨仓库引用形式：

    "AutoModel": "sensenova/SenseNova-SI-1.1-InternVL3-2B--modeling_internvl_chat.InternVLChatModel"

`repo_id--module.Class` 这个语法让 transformers 按 **HF repo id** 而不是本地路径
去建动态模块目录（~/.cache/huggingface/modules/transformers_modules/<repo_id>/）。
当 repo id 里含点号（版本号 1.1）时，Python 把它当成包分隔符，导入直接失败：

    ModuleNotFoundError: No module named 'transformers_modules.sensenova.SenseNova-SI-1'

改本地目录名解决不了，因为模块路径来自 auto_map 而非我们传入的路径。

修法
----
把 auto_map 里的 `repo_id--` 前缀去掉，改成纯本地引用。模型目录里本来就有那些
.py 文件，去掉前缀后 transformers 会直接从本地目录加载。

顺带清掉可能已经生成的坏缓存目录。

默认 dry-run，加 --apply 才写入，原文件备份为 config.json.orig。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"


def main() -> int:
    parser = argparse.ArgumentParser(description="Strip cross-repo prefixes from a local model's auto_map.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry-run).")
    args = parser.parse_args()

    config_path = args.model_dir / "config.json"
    if not config_path.exists():
        print(f"没有找到 {config_path}", file=sys.stderr)
        return 1

    config = json.loads(config_path.read_text(encoding="utf-8"))
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict):
        print("config.json 里没有 auto_map，无需处理。")
        return 0

    changed: list[tuple[str, str, str]] = []
    repo_ids: set[str] = set()
    for key, value in list(auto_map.items()):
        if isinstance(value, str) and "--" in value:
            repo_id, local_ref = value.split("--", 1)
            repo_ids.add(repo_id)
            auto_map[key] = local_ref
            changed.append((key, value, local_ref))

    if not changed:
        print("auto_map 已经是本地引用形式，无需处理。")
        return 0

    print(f"模型目录: {args.model_dir}")
    for key, old, new in changed:
        print(f"  {key}")
        print(f"    - {old}")
        print(f"    + {new}")

    # 确认被引用的 .py 文件确实在本地
    missing = []
    for _key, _old, new in changed:
        module = new.split(".")[0]
        if not (args.model_dir / f"{module}.py").exists():
            missing.append(f"{module}.py")
    if missing:
        print(f"\n警告：本地缺少被引用的模块文件 {sorted(set(missing))}，改完仍会加载失败。", file=sys.stderr)
        return 1

    stale = [CACHE_ROOT / rid for rid in repo_ids if (CACHE_ROOT / rid).exists()]
    if stale:
        print("\n将清理的坏缓存目录:")
        for path in stale:
            print(f"  {path}")

    if not args.apply:
        print("\ndry-run，未写入。确认无误后加 --apply。")
        return 0

    backup = config_path.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy2(config_path, backup)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)

    print(f"\n已写入，原文件备份为 {backup.name}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
