#!/usr/bin/env python3
# coding: utf-8
"""结果存储（BC-04）。

冻结版每答完一题就把整个结果文件重写一遍，且每行复制整条原始 item。
trajectory 的 3D 输入每题约 50 KB，一个 300 题任务累计写入约 2.2 GB，
整个 15×20×9 矩阵约 0.8 TB 无谓 IO。

这里改成 JSONL 追加：
  - 每题一行，写完即落盘，进程被杀也不丢已完成的部分
  - 断点恢复只需扫一遍已有行的 id，不必反序列化整个文件
  - 导出时再组装成与冻结版**完全同构**的 JSON，现有分析脚本零改动
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class ResultStore:
    """一个 (模型, 任务族, 任务) 的结果存储。"""

    def __init__(self, path: Path, meta: dict[str, Any] | None = None) -> None:
        self.path = path
        self.meta_path = path.with_suffix(".meta.json")
        self.meta = meta or {}

    # -- 写 ----------------------------------------------------------------

    def displace(self) -> int:
        """把已有结果挪到 ``<name>.jsonl.bak``，返回挪走了多少行。

        ``--overwrite`` 的语义是「重跑」，不是「销毁」——
        这些行是 GPU 一题一题跑出来的，**不可再生**。

        实际发生过（D-62）：想只重跑 time，而 ``matrix`` 没有 ``--runs``，
        于是用 ``--overwrite`` 起了全矩阵；跑到第 4 个 run 停手时，
        上一遍 42 个 run 的逐题数据已经全没了，约两小时机时。
        改名的代价是几百 KB 磁盘，删掉的代价是重跑。

        **只保留一代备份** —— 再多就成了没人清理的垃圾。
        后缀是 ``.jsonl.bak`` 而不是 ``.bak.jsonl``：``report.pack`` 按
        ``*.jsonl`` 收集文件，备份不该被打包回传。
        """
        if not self.path.exists():
            return 0
        rows = sum(1 for line in self.path.read_text(encoding="utf-8").splitlines()
                   if line.strip())
        backup = self.path.with_name(self.path.name + ".bak")
        backup.unlink(missing_ok=True)
        self.path.rename(backup)
        return rows

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.meta:
            self.meta_path.write_text(
                json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def append(self, rows: list[dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- 读 ----------------------------------------------------------------

    def rows(self) -> Iterator[dict[str, Any]]:
        """逐行读结果。**容忍尾部残行** —— 进程被 kill、OOM 或断电时，
        最后一行可能只写了一半。直接 ``json.loads`` 会抛异常，导致断点续跑
        和导出全部失败，等于因为半行数据丢掉整个 run 的成果。

        只在文件末尾容忍，中间出现坏行说明是别的问题（比如两个进程同时
        往一个文件追加），那种情况必须报出来而不是静默跳过。
        """
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
        for index, line in enumerate(lines):
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    print(f"  {self.path.name}: 丢弃尾部残行（上次运行被中断）", flush=True)
                    return
                raise

    def completed_ids(self) -> set[str]:
        """已完成的题目 id。与冻结版 ``is_finished`` 口径一致：
        有 model_output 且没有 error 才算完成，失败的会被重跑。"""
        done: set[str] = set()
        for row in self.rows():
            if row.get("model_output") and not row.get("error"):
                done.add(str(row.get("id")))
        return done

    def final_rows(self) -> list[dict[str, Any]]:
        """每个 id 只保留一行 —— 汇总与导出都必须走这里，不能直接用 ``rows()``。

        JSONL 是追加日志，同一个 id 可能出现多次：某次跑失败写了 error 行，
        续跑时 ``completed_ids`` 认为它没完成又跑了一遍并追加成功行。
        直接把两行都喂给 summarize，那道题就被计了两次，分母和分子都是错的。

        取舍规则与 ``completed_ids`` 保持一致，否则「算完成」和「算进分数」
        会是两套口径：
          - 有成功行就取**最后一条成功行**（重跑覆盖旧结果）
          - 全是失败行才取最后一条失败行（如实反映这题没做出来）

        注意不能简单「后来者覆盖」：进程被错误配置打断时，可能在已经成功的
        id 后面又追加了 error 行，那时最后一条恰恰是错的。
        """
        best: dict[str, dict[str, Any]] = {}
        for row in self.rows():
            key = str(row.get("id"))
            ok = bool(row.get("model_output")) and not row.get("error")
            previous = best.get(key)
            if previous is None:
                best[key] = row
                continue
            previous_ok = bool(previous.get("model_output")) and not previous.get("error")
            if ok or not previous_ok:
                best[key] = row
        return list(best.values())

    # -- 导出 --------------------------------------------------------------

    def export(self, output_path: Path, summary: dict[str, Any], items_by_id: dict[str, dict[str, Any]] | None = None) -> None:
        """导出成与冻结版同构的 JSON。

        ``items_by_id`` 给出时，把原始 item 字段合并回每行 —— 这样导出结果与
        冻结版逐字段一致；不给出时导出精简版。
        """
        results = []
        for row in self.final_rows():
            if items_by_id:
                item = items_by_id.get(str(row.get("id")))
                if item:
                    row = {**item, **row}
            results.append(row)

        payload = {**self.meta, "results": results, "summary": summary}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
