#!/usr/bin/env python3
# coding: utf-8
"""同一模型、同一批题，在两个 transformers 版本上跑，逐字节比对输出。

要回答的问题：**换 transformers 版本会不会改变分数？**

这不是学术好奇 —— 15 个模型没法用同一个版本（InternVL 只能 4.57.6，
Cosmos3-Edge 只能 5.x），所以矩阵里的模型注定跑在不同版本上。
如果版本会改变输出，那这份 benchmark 的跨模型对比就多了一个混淆变量，
必须在报告里写明；如果不会，就可以放心分环境。

做法：挑一个两个版本都能加载的模型（RynnBrain-2B / Qwen3-VL 都行，
它们是原生 Qwen3VL 架构，不走 trust_remote_code），用**完全相同**的
provider 配置、抽帧档位、生成参数各跑一遍，比 model_output 原文。

    python tools/version_compare.py --model RynnBrain-2B --runs understanding time --groups 10

前提：两套环境已建好（bash tools/setup_envs.sh），
或用 --python-a / --python-b 指定任意两个解释器。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from robochrono import dispatch  # noqa: E402
from robochrono.store import ResultStore  # noqa: E402


def interpreter_for(env_name: str, override: str | None) -> str:
    if override:
        return override
    config = dispatch.load_environments(EVAL_ROOT / "configs/environments.json")
    return dispatch.resolve_interpreter(config, env_name, EVAL_ROOT)


def version_of(interpreter: str) -> str:
    result = subprocess.run(
        [interpreter, "-c", "import transformers,torch;print(transformers.__version__,torch.__version__)"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() or f"(查不到: {result.stderr.strip()[:60]})"


def run_side(interpreter: str, model: str, runs: list[str], groups: int, out_dir: Path) -> int:
    cmd = [
        interpreter, "-m", "robochrono",
        "--results-dir", str(out_dir),
        "matrix", "--models", model, "--gpus", "1",
        "--limit-groups", str(groups), "--overwrite",
    ]
    print(f"  $ {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd, cwd=str(EVAL_ROOT)).returncode


def compare(dir_a: Path, dir_b: Path) -> int:
    """比对两侧结果。返回有差异的 run 数。"""
    files_a = {f.relative_to(dir_a): f for f in dir_a.rglob("*.jsonl")}
    files_b = {f.relative_to(dir_b): f for f in dir_b.rglob("*.jsonl")}

    only_a = sorted(str(k) for k in files_a.keys() - files_b.keys())
    only_b = sorted(str(k) for k in files_b.keys() - files_a.keys())
    for name in only_a:
        print(f"  只有 A 侧有：{name}")
    for name in only_b:
        print(f"  只有 B 侧有：{name}")

    differing = len(only_a) + len(only_b)
    print(f"\n{'run':<34} {'题数':>5} {'输出相同':>8} {'判定相同':>8}  结论")
    print("-" * 74)

    for key in sorted(files_a.keys() & files_b.keys()):
        rows_a = {str(r["id"]): r for r in ResultStore(files_a[key], meta={}).final_rows()}
        rows_b = {str(r["id"]): r for r in ResultStore(files_b[key], meta={}).final_rows()}
        shared = sorted(rows_a.keys() & rows_b.keys())
        if not shared:
            continue

        same_output = sum(1 for i in shared
                          if rows_a[i].get("model_output") == rows_b[i].get("model_output"))
        # 判分字段：选择题是 correct，轨迹/时间是各自的指标字段
        def verdict(row: dict) -> tuple:
            return tuple(row.get(k) for k in ("correct", "tIoU", "l2", "score") if k in row)
        same_verdict = sum(1 for i in shared if verdict(rows_a[i]) == verdict(rows_b[i]))

        # 标签取 "档位/run"，模型名和任务族对所有行都一样，截掉只会让各行看起来相同
        parts = key.parts
        label = f"{parts[-2]}/{Path(parts[-1]).stem}" if len(parts) >= 2 else str(key)
        note = "一致" if same_output == len(shared) else (
            "判定一致但文本不同" if same_verdict == len(shared) else "**分数会变**")
        if same_output != len(shared):
            differing += 1
        print(f"{label:<34} {len(shared):>5} {same_output:>8} {same_verdict:>8}  {note}")

        # 差异样例，方便判断是随机性还是系统性
        if same_output != len(shared):
            for i in shared:
                if rows_a[i].get("model_output") != rows_b[i].get("model_output"):
                    print(f"      id={i}")
                    print(f"        A: {str(rows_a[i].get('model_output'))[:90]!r}")
                    print(f"        B: {str(rows_b[i].get('model_output'))[:90]!r}")
                    break

    print("-" * 74)
    return differing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RynnBrain-2B")
    parser.add_argument("--runs", nargs="+", default=None, help="默认跑 plan 里该模型的全部 run")
    parser.add_argument("--groups", type=int, default=10, help="每个 run 取前 N 个 unit")
    parser.add_argument("--python-a", default=None)
    parser.add_argument("--python-b", default=None)
    parser.add_argument("--out", type=Path, default=EVAL_ROOT / "results/vercmp")
    parser.add_argument("--compare-only", action="store_true", help="跳过执行，只比对已有结果")
    args = parser.parse_args()

    python_a = interpreter_for("groupA", args.python_a)
    python_b = interpreter_for("groupB", args.python_b)
    dir_a, dir_b = args.out / "A", args.out / "B"

    print(f"模型 {args.model}，每个 run 取前 {args.groups} 个 unit\n")
    print(f"  A: {python_a}\n     {version_of(python_a)}")
    print(f"  B: {python_b}\n     {version_of(python_b)}\n")

    if not args.compare_only:
        for label, interpreter, out_dir in (("A", python_a, dir_a), ("B", python_b, dir_b)):
            print(f"── 跑 {label} 侧 ──")
            code = run_side(interpreter, args.model, args.runs or [], args.groups, out_dir)
            if code != 0:
                print(f"  {label} 侧退出码 {code}")

    differing = compare(dir_a, dir_b)
    print("两个版本输出完全一致" if differing == 0
          else f"{differing} 个 run 存在差异 —— 版本会影响结果，报告里必须写明")
    return 1 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
