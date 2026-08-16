# runs

每次评测的产物，按 run_id 一次一个目录。

```
<run_id>/
├── rows.jsonl      逐题结果
├── summary.json    汇总指标
├── meta.json       模型、配置、代码 commit、data/vqa 版本、耗时
└── report.html
```

`meta.json` 必须能回答「这份结果是用什么跑出来的」——
模型、provider、抽帧档位、`data/vqa` 的版本与 manifest 指纹、代码 commit。
少了任何一项，这份结果就不可复现，也就没有比较价值。
