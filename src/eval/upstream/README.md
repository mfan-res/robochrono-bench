# src/eval/upstream —— 冻结的参考实现，**只读**

八个原始评测脚本，从旧仓库 `test/` 原样搬来，**一个字符都没改**。

它们不参与生产 —— 生产走 `robochrono/`。留着是为了让回归能比对：
`tests/test_parsing_equivalence.py` 与 `test_request_equivalence.py`
逐字调用这里的函数，确认重构后的解析与请求组装**与原实现完全一致**。

> 判据是「输出逐字节相同」，不是「看起来差不多」。
> 改这里等于改判据本身 —— **要改先问人**。
