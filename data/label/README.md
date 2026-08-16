# data/label

人类标注，**只读**。由 `src/label/` 的工具产出。

```
<family>/
├── segments/file-XXX.json     动作分段
└── categories.txt             该任务的候选动作标签集
```

分段结构（`src/common/schemas/segments.json`）：

```jsonc
{"segments": [{"id": "file-000-1", "start": 6.52, "end": 10.6,
               "start_frame": 163, "end_frame": 264,
               "objects": ["brush"], "main_verbs": ["pick"],
               "narration": " Pick the brush."}]}
```

七个评测任务的真值**全部**源自这里。
