#!/usr/bin/env python3
# coding: utf-8
"""跑齐 ③ 标注这一段的回归。

    python3 src/label/tests/run_all.py

**为什么需要这个入口。** `src/eval/tests/run_all.py` 只扫 `src/eval/tests/`，
于是 `src/label/tests/` 下的测试**不在任何套件里** —— `test_core_replay.py`
因此长期是红的而没人知道（P-05 那次有意的语义改动没补进「已声明差异」表）。

一个没人跑的测试比没有测试更糟：它给人「有回归守着」的错觉。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = sorted(p for p in HERE.glob("test_*.py"))

ok = failed = 0
print(f"{'回归':<34}{'结果'}")
print("-" * 52)
for test in TESTS:
    result = subprocess.run([sys.executable, "-B", str(test)],
                            capture_output=True, text=True)
    if result.returncode == 0:
        state, ok = "✓", ok + 1
    else:
        state, failed = "✗ 未通过", failed + 1
    print(f"{test.stem:<34}{state}")
    if result.returncode != 0:
        for line in (result.stdout + result.stderr).strip().splitlines()[-4:]:
            print(f"      {line}")
print("-" * 52)
print(f"通过 {ok}　未通过 {failed}")
raise SystemExit(1 if failed else 0)
