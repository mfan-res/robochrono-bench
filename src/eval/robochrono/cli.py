#!/usr/bin/env python3
# coding: utf-8
"""命令行入口。

    python -m robochrono list
    python -m robochrono run --provider <name> --family stack_cubes --runs understanding
    python -m robochrono export --results-dir <dir>

冻结版是八条互不相同的命令、每条一套参数；这里是一条命令、一套参数、
一个 (模型 × 任务族 × 任务) 的矩阵。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import dispatch, engine, matrix, matrix_run, pool, preflight, report, tasks
from .store import ResultStore
from .mediaindex import ResolveStats, index_for_qa, resolve_items
from .tasks.base import load_items
from .vlm_api import runtime_config


def _providers_cfg() -> dict:
    """读 providers.json。按题型覆盖抽帧数要用（见该文件的 _frames_by_run_note）。"""
    import json as _json
    from pathlib import Path as _Path
    return _json.loads((_Path(__file__).resolve().parents[1] / "configs" / "providers.json")
                       .read_text(encoding="utf-8"))

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/providers.json"
DEFAULT_PLAN = Path(__file__).resolve().parents[1] / "configs/plan.json"
DEFAULT_ENVIRONMENTS = Path(__file__).resolve().parents[1] / "configs/environments.json"
EVAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = Path(__file__).resolve().parents[1] / "datasets"
DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results"
KEYS_FILE = Path.home() / ".config/robochrono/keys.env"


def load_keys(path: Path = KEYS_FILE) -> None:
    """从仓库之外的 keys.env 读 API key，避免密钥进 git。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip())


def cmd_list(args: argparse.Namespace) -> int:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("runs:")
    for run in tasks.ALL_RUNS:
        print(f"  {run:<16} {tasks.QA_GROUP[run]:<14} {tasks.QA_FILENAME[run]:<26} "
              f"主指标 {tasks.PRIMARY_METRIC[run]}")
    print("\nproviders:")
    for name, provider in config.get("providers", {}).items():
        print(f"  {name:<38} {provider.get('type')}")
    root = Path(args.datasets_root) / "QA"
    if root.exists():
        families = sorted({p.name for group in root.iterdir() if group.is_dir()
                           for p in group.iterdir() if p.is_dir()})
        print(f"\nfamilies: {', '.join(families) or '(无)'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    load_keys()
    config_path = Path(args.config)
    runs = args.runs or list(tasks.ALL_RUNS)
    results_root = Path(args.results_dir)

    failures = 0
    for family in args.families:
        for run in runs:
            qa_path = tasks.qa_path(args.datasets_root, family, run)
            if not qa_path.exists():
                print(f"[skip] {family}/{run}: QA 文件缺失 {qa_path}")
                continue

            print(f"\n=== {args.provider} × {family} × {run} ===")
            runtime = runtime_config(
                config_path=config_path,
                provider_name=args.provider,
                default_model="",
                cli_model=args.model,
                cli_temperature=args.temperature,
                cli_timeout=args.timeout,
                cli_max_retries=args.max_retries,
            )
            # 按题型覆盖抽帧数（见 providers.json 的 _frames_by_run_note）。
            # time 用全长视频，8 帧会让 31% 的题看不到被问的动作。
            by_run = _providers_cfg().get("frames_by_run", {}).get(run)
            if by_run:
                runtime["frames"] = dict(by_run)
            print(f"  provider={runtime['provider']} model={runtime['model']} "
                  f"frames={runtime['frames']['mode']}:{runtime['frames']['value']}"
                  + ("  ← 按题型覆盖" if by_run else ""))

            task = tasks.build(run,
                               strip_reasoning=args.strip_reasoning,
                               null_text_fix=args.null_text_fix,
                               zero_unscored=not getattr(args, "no_zero_unscored", False))

            # 解析媒体路径。**此前这里没有这一步** —— 因为 v1 的 QA 文件里
            # 写的是绝对路径（BC-08 就地改写留下的），拿来就能用。
            # 新数据把媒体路径写成相对 QA 文件的形式（可移植），必须解析。
            # 缺失会显式打印，不静默把原样路径传给下游。
            stats = ResolveStats()
            items = tasks.load_for_run(args.datasets_root, family, run, stats)
            if stats.unresolved or stats.ambiguous:
                print(f"  [media] 解析 {stats.resolved}/{stats.total}"
                      f"　未解析 {len(stats.unresolved)}　同名歧义 {len(stats.ambiguous)}")
                for x in (stats.unresolved + stats.ambiguous)[:3]:
                    print(f"          {x}")
            out_dir = results_root / args.provider / family
            store = ResultStore(
                out_dir / f"{run}.jsonl",
                meta={
                    "provider": runtime["provider"],
                    "model": runtime["model"],
                    "api_url": runtime["api_url"],
                    "family": family,
                    "run": run,
                    "input": str(qa_path),
                    "frames": runtime["frames"],
                    "generation": {
                        "temperature": runtime["temperature"],
                        "thinking": runtime["thinking"],
                        "max_new_tokens": runtime["max_new_tokens"],
                    },
                    "flags": {
                        "strip_reasoning": args.strip_reasoning,
                        "null_text_fix": args.null_text_fix,
                    },
                },
            )

            try:
                summary = engine.run(
                    task, items, runtime, store,
                    limit_items=args.limit_items,
                    limit_groups=args.limit_groups,
                    overwrite=args.overwrite,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  RUN FAILED: {type(exc).__name__}: {exc}")
                failures += 1
                continue

            (out_dir / f"{run}.summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            metric = tasks.PRIMARY_METRIC[run]
            print(f"  {metric} = {summary.get(metric)}  "
                  f"(answered {summary.get('answered')}/{summary.get('total')}, "
                  f"errors {summary.get('errors')})")
            breach = tasks.floor_breach(run, summary)      # 与 matrix 同口径
            if breach:
                print(f"    ⚠ {breach}")
    return 1 if failures else 0


def _parse_shard(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    try:
        index, total = (int(x) for x in text.split("/", 1))
    except ValueError as exc:
        raise SystemExit(f"--shard must look like i/N, got {text!r}") from exc
    if not 1 <= index <= total:
        raise SystemExit(f"--shard index out of range: {text}")
    return index, total


def _expand(args: argparse.Namespace):
    plan = matrix.load_plan(Path(args.plan))
    return plan, matrix.expand(
        plan,
        Path(args.datasets_root),
        shard=_parse_shard(getattr(args, "shard", None)),
        only_kind=getattr(args, "only", None),
        only_models=getattr(args, "models", None),
        only_runs=getattr(args, "runs", None),
    )


def cmd_plan(args: argparse.Namespace) -> int:
    plan, (specs, skipped) = _expand(args)
    print(f"{'#':>4}  {'model':<38} {'family':<14} {'run'}")
    print("-" * 78)
    for index, spec in enumerate(specs, 1):
        print(f"{index:>4}  {spec.model.name:<38} {spec.family:<14} {spec.run}")
    print("-" * 78)
    print(f"共 {len(specs)} 个 run，跳过 {len(skipped)} 个")
    for key, reason in skipped[:20]:
        print(f"  [skip] {key}: {reason}")
    if len(skipped) > 20:
        print(f"  ... 另有 {len(skipped) - 20} 个")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    plan, (specs, skipped) = _expand(args)
    if not specs:
        print("矩阵为空，无可估算内容")
        return 1
    by_model = report.estimate_matrix(specs, Path(args.datasets_root))
    print(report.format_estimate(by_model, plan))
    if skipped:
        print(f"\n跳过 {len(skipped)} 个组合（QA 缺失或稀疏规则）")
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    """按矩阵跑：model-major 调度，本地模型多卡并行。"""
    load_keys()
    plan, (specs, skipped) = _expand(args)
    if not specs:
        print("矩阵为空")
        return 1
    gpus = pool.visible_gpus(args.gpus)
    print(f"矩阵 {len(specs)} 个 run，跳过 {len(skipped)} 个，可用 GPU {gpus or '无'}")
    failures = matrix_run.run_matrix(
        plan, specs,
        config_path=Path(args.config),
        datasets_root=Path(args.datasets_root),
        results_root=Path(args.results_dir),
        gpus=gpus,
        flags={"strip_reasoning": args.strip_reasoning,
               "null_text_fix": args.null_text_fix,
               "zero_unscored": not args.no_zero_unscored},
        limit_items=args.limit_items,
        limit_groups=args.limit_groups,
        overwrite=args.overwrite,
        api_concurrency=args.api_concurrency,
        api_rate_limit=args.api_rate_limit,
    )
    return 1 if failures else 0


def cmd_dispatch(args: argparse.Namespace) -> int:
    """按模型所属环境分派子进程。见 robochrono/dispatch.py 的说明。"""
    plan = matrix.load_plan(Path(args.plan))
    env_config = dispatch.load_environments(Path(args.environments))

    passthrough = list(args.matrix_args)
    # argparse 的 REMAINDER 会把分隔用的 -- 留在列表里
    while passthrough and passthrough[0] == "--":
        passthrough.pop(0)
    if not passthrough or passthrough[0] != "matrix":
        passthrough = ["matrix", *passthrough]
    if "--plan" not in passthrough:
        passthrough += ["--plan", str(Path(args.plan).resolve())]

    # 全局选项必须显式转交给子进程，且要放在子命令**之前**。
    # 漏掉它们不会报错 —— 子进程用自己的默认值把结果写到别处，
    # 看起来一切正常。多机跑时 --results-dir 常指向共享挂载点，
    # 这种静默写错位置比直接崩溃难查得多。
    globals_ = [
        "--config", str(Path(args.config).resolve()),
        "--datasets-root", str(Path(args.datasets_root).resolve()),
        "--results-dir", str(Path(args.results_dir).resolve()),
    ]
    passthrough = [*globals_, *passthrough]

    return dispatch.run(
        env_config,
        [m.name for m in plan.models],
        passthrough,
        eval_root=EVAL_ROOT,
        dry_run=args.dry_run,
    )


def cmd_preflight(args: argparse.Namespace) -> int:
    load_keys()
    plan, (specs, skipped) = _expand(args)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    checks, fails = preflight.run_all(plan, specs, Path(args.datasets_root), config, skipped)
    print(preflight.format_checks(checks, skipped))
    return 1 if fails else 0


def cmd_pack(args: argparse.Namespace) -> int:
    info = report.pack(Path(args.results_dir), Path(args.output), full=args.full)
    print(f"打包 {info['files']} 个文件 -> {info['output']}  ({info['bytes']/1e6:.1f} MB)")
    print("回传：scp " + info["output"] + " <目标主机>:<路径>")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for directory in args.results_dirs or [Path(args.results_dir)]:
        rows.extend(report.collect(Path(directory)))
    if not rows:
        print("没有找到任何 *.summary.json")
        return 1
    print(report.to_markdown(rows))
    if args.csv:
        report.to_csv(rows, Path(args.csv))
        print(f"\nCSV 已写入 {args.csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robochrono", description="RoboChrono 评测框架")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--datasets-root", default=str(DEFAULT_DATASETS))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出 run、provider 与任务族").set_defaults(func=cmd_list)

    run_parser = sub.add_parser("run", help="跑一个或多个 (任务族 × 任务)")
    run_parser.add_argument("--provider", required=True)
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument("--families", nargs="+", default=["stack_cubes"])
    run_parser.add_argument("--runs", nargs="+", default=None,
                            help=f"默认全部：{' '.join(tasks.ALL_RUNS)}")
    # BC-05：拆分后的两个显式参数
    run_parser.add_argument("--limit-items", type=int, default=None, help="限制题数")
    run_parser.add_argument("--limit-groups", type=int, default=None,
                            help="限制调用组数（time 即视频数）")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument("--timeout", type=int, default=None)
    run_parser.add_argument("--max-retries", type=int, default=None)
    # 会改变分数的开关，默认关闭
    run_parser.add_argument("--strip-reasoning", action="store_true",
                            help="BC-02：剥离思考块并启用 JSON 兜底")
    run_parser.add_argument("--null-text-fix", action="store_true",
                            help="BC-10：text 为 null 的选项不参与文本匹配")
    run_parser.set_defaults(func=cmd_run)

    for name, help_text, func in (
        ("plan", "展开 (模型 × 任务族 × 任务) 矩阵", cmd_plan),
        ("estimate", "估算调用量与媒体体积（不调模型）", cmd_estimate),
        ("preflight", "开跑前自检：环境 / GPU / 权重 / 密钥 / 数据", cmd_preflight),
    ):
        sub_parser = sub.add_parser(name, help=help_text)
        sub_parser.add_argument("--plan", default=str(DEFAULT_PLAN))
        sub_parser.add_argument("--shard", default=None, help="形如 1/4，多机分工用")
        sub_parser.add_argument("--only", choices=["local", "api"], default=None)
        sub_parser.add_argument("--runs", nargs="+", default=None,
                                help=f"只看这些题型，默认全部：{' '.join(tasks.ALL_RUNS)}")
        sub_parser.set_defaults(func=func)

    matrix_parser = sub.add_parser("matrix", help="按 plan.json 跑整个矩阵（本地模型多卡并行）")
    matrix_parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    matrix_parser.add_argument("--shard", default=None, help="形如 1/4，多机分工用")
    matrix_parser.add_argument("--only", choices=["local", "api"], default=None)
    matrix_parser.add_argument("--gpus", type=int, default=None, help="使用前 N 张卡，默认全部")
    matrix_parser.add_argument("--models", nargs="+", default=None,
                               help="只跑这些模型（名字取自 plan.json），按环境分派时用")
    # **只重跑一个题型要靠它。** 没有它的时候，唯一看起来可行的写法是
    # `--overwrite` 起全矩阵再中途停手 —— 而那会先清掉全部 42 个 run（D-62）。
    matrix_parser.add_argument("--runs", nargs="+", default=None,
                               help=f"只跑这些题型，默认全部：{' '.join(tasks.ALL_RUNS)}")
    matrix_parser.add_argument("--api-concurrency", type=int, default=None,
                               help="API 模型的并发请求数，覆盖 providers.json；本地模型不受影响")
    matrix_parser.add_argument("--api-rate-limit", type=float, default=None,
                               help="API 每秒请求上限，0 为不限")
    matrix_parser.add_argument("--limit-items", type=int, default=None)
    matrix_parser.add_argument("--limit-groups", type=int, default=None)
    matrix_parser.add_argument("--overwrite", action="store_true")
    matrix_parser.add_argument("--strip-reasoning", action="store_true")
    matrix_parser.add_argument("--null-text-fix", action="store_true")
    matrix_parser.add_argument("--no-zero-unscored", action="store_true",
                               help="关闭 BC-15：轨迹 format 不合规时不计 0 分，"
                                    "回到冻结版口径（仅用于与旧数据对比）")
    matrix_parser.set_defaults(func=cmd_matrix)

    dispatch_parser = sub.add_parser(
        "dispatch",
        help="按 environments.json 把各模型分派到对应的 python 环境后跑矩阵")
    dispatch_parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    dispatch_parser.add_argument("--environments", default=str(DEFAULT_ENVIRONMENTS))
    dispatch_parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    dispatch_parser.add_argument("matrix_args", nargs=argparse.REMAINDER,
                                 help="-- 之后的参数原样透传给 matrix，例如 -- --gpus 8 --only local")
    dispatch_parser.set_defaults(func=cmd_dispatch)

    report_parser = sub.add_parser("report", help="汇总结果成对比表")
    report_parser.add_argument("results_dirs", nargs="*", type=Path,
                               help="结果目录，可给多个（多机合并）")
    report_parser.add_argument("--csv", default=None)
    report_parser.set_defaults(func=cmd_report)

    pack_parser = sub.add_parser("pack", help="打包结果供 scp 回传")
    pack_parser.add_argument("-o", "--output", default="robochrono-results.tar.gz")
    pack_parser.add_argument("--full", action="store_true", help="连媒体缓存一起带上")
    pack_parser.set_defaults(func=cmd_pack)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
