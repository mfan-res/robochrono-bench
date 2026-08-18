#!/usr/bin/env python3
# coding: utf-8
"""跑齐六套回归。

    python3 src/eval/tests/run_all.py

**「跳过」不算「通过」。** 需要完整 v1 数据（61 GB，留在旧仓库）的两套会跳过，
输出里单独一栏列出来 —— 混进通过数里就等于悄悄少测了。
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = sorted(p for p in HERE.glob("test_*.py"))

ok = skipped = failed = 0
print(f"{'回归':<38}{'结果'}")
print("-" * 52)
for t in TESTS:
    r = subprocess.run([sys.executable, str(t)], capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        state, failed = "✗ 未通过", failed + 1
    elif "跳过" in out:
        state, skipped = "○ 跳过（缺完整 v1 数据）", skipped + 1
    else:
        state, ok = "✓", ok + 1
    print(f"{t.stem:<38}{state}")
    if r.returncode != 0:
        for line in out.strip().splitlines()[-3:]:
            print(f"      {line}")
print("-" * 52)
print(f"通过 {ok}　跳过 {skipped}　未通过 {failed}")
raise SystemExit(1 if failed else 0)
