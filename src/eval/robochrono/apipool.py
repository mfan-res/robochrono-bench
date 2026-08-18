#!/usr/bin/env python3
# coding: utf-8
"""API 调用的限流。

本地模型受 GPU 数量约束，API 模型不受 —— 瓶颈是网络往返与服务端排队。
串行跑一个任务族要 2,450 次调用，按每次 20 秒算约 13 小时；
真实矩阵里 API 侧有 245,000 次调用，串行是几周，不可行。

并发执行本身放在 ``engine.run`` 里，与串行路径共用同一份 unit 执行、落盘、
续跑、熔断逻辑 —— 不另写一份并发版，否则两条路径迟早会分叉。
这里只提供跨线程共享的限流器。
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """令牌桶。``rate`` 为每秒允许的请求数，``0`` 表示不限。

    桶容量默认等于 rate（即允许一秒的突发）。空闲一段时间后攒下的令牌
    让下一批请求可以立刻发出，而不必被均匀摊开。
    """

    def __init__(self, rate: float, burst: float | None = None) -> None:
        self.rate = float(rate)
        self.capacity = float(burst if burst is not None else max(1.0, rate))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rate <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            # 在锁外 sleep，否则其他线程连补充令牌都被挡住
            time.sleep(min(wait, 1.0))
