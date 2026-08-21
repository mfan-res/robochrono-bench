# data/label

人类标注，**只读**。由 `src/label/` 的工具产出。

```
<family>/
├── subtasks.json                    该族的动作定义：id -> text（**文字只存在这里**）
├── segments/file-XXX_segments.json  动作分段，段里只存 subtask 的 id
├── segments.before_*/               每次数据修改前的备份，见 DEVLOG
└── categories.deprecated.txt        v1 的自由文本标签集，已废弃，只作留档
```

分段结构（权威契约是 `src/common/schemas/segments.json`）：

```jsonc
{"source": {"video": "tea/file-000/main.mp4", "fps": 25, "total_frames": 2994,
            "tool_version": "...", "subtasks_sha256": "...", "episode_bounds": null},
 "segments": [{"id": "file-000@f000070",
               "start": 2.8, "end": 9.24,
               "start_time": "00:00:02.800", "end_time": "00:00:09.240",
               "start_frame": 70, "end_frame": 230,
               "subtask": "open_teapot_lid"}]}
```

**帧号是权威，秒与时间串由 fps 派生**，校验器重算比对而不是相信文件里写的值。
`id` 由起始帧派生（`<episode>@f<6位帧号>`），不用序号 —— 序号会在中间插段时
让后面所有 id 平移，破坏下游引用。

> **这份文档曾经写的是 v1 的结构**（`segments/file-XXX.json`、`categories.txt`、
> 字段 `objects` / `main_verbs` / `narration`）。D-04 之后文字只存在
> `subtasks.json` 一处、段里存 id，上面才是现在的样子。

七个评测任务的真值**全部**源自这里。
