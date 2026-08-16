# src

全部代码。三个阶段各一个目录，加一个共用层。

| 目录 | 产出 |
| --- | --- |
| `label/` | `data/label/` |
| `vqa/` | `data/vqa/` |
| `eval/` | `runs/` |
| `common/` | — （schema、manifest、路径、校验器基类） |

`common/` 存在的理由是四个校验器要共用 schema 和 manifest 逻辑，
各写一份必然慢慢分叉。

## 出处

`label/` 与 `vqa/` 的代码接管自数据方（`yyyyywv/egocentric` 的 `label/`、
原仓库的 `data/`）。**不再跟随上游更新，我们自己维护。**
但每个文件要注明出处 —— 不是为了合并，是为了将来发现某个行为很怪时，
能判断「这是原作者的设计，还是我们改出来的」。见 `docs/provenance.md`。
