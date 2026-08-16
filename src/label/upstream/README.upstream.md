# Video Timestamp Labeler

`video_labeler_timestamp.py` 用于给视频打时间戳段标注。每个标注段包含起止帧、起止时间和类别描述，结果保存为和 egodata 示例一致的 `*_segments.json`。

## 标注文件
只需要标注每一个任务的一个视角即可（ego视角--第一视角）

## 需要准备

1. Python 3。
2. 依赖包：

```bash
pip install opencv-python numpy
```

3. 视频文件或视频目录。支持的视频后缀包括：

```text
.mp4 .avi .mov .mkv .flv
```

4. 类别 txt 文件。每一行是一个类别，空行会被忽略。

示例：

```text
N1:pick red cube
N2:place red cube
M1:pick yellow cube
M2:place yellow cube
```

类别前缀可以是 `C1:`、`M1:`、`N1:`、`label1 -` 等形式。保存 JSON 时会自动去掉前缀。

## 调用方式

建议始终用 `--txt` 指定类别文件：

```bash
python video_labeler_timestamp.py <视频文件或目录> --txt <类别txt路径>
```

标注单个视频：

```bash
python video_labeler_timestamp.py /path/to/file-000.mp4 --txt /path/to/classes.txt
```

标注一个目录下的所有视频：

```bash
python video_labeler_timestamp.py /path/to/video_dir --txt /path/to/classes.txt
```

指定输出路径：

```bash
python video_labeler_timestamp.py /path/to/file-000.mp4 --txt /path/to/classes.txt --output /path/to/file-000_segments.json
```

目录模式下 `--output` 需要指定输出目录，不能指定单个 json 文件：

```bash
python video_labeler_timestamp.py /path/to/video_dir --txt /path/to/classes.txt --output /path/to/json_dir
```

从目录中的某个视频开始标注：

```bash
python video_labeler_timestamp.py /path/to/video_dir --txt /path/to/classes.txt --start 23
python video_labeler_timestamp.py /path/to/video_dir --txt /path/to/classes.txt --start file-023.mp4
```

目录模式会按对应的 `*_segments.json` 判断是否已标注。只有对应 json 已存在时，才会跳过该视频。

## 操作方法

打开窗口后默认暂停。

- 拖动进度条或用方向键移动到起始帧。
- 按 `k` 标记当前段起始帧。
- 移动到结束帧，再按 `k` 标记结束帧。
- 按类别快捷键选择类别：`1-9`、`q-p`，最多支持 20 个类别。
- 按 `f` 将当前段结束设置为视频结尾。
- 按 `z` 或 Backspace 撤销/取消当前段。
- 按 Space 播放或暂停。
- 按 `s` 保存并进入下一个视频。
- 按 `q` 或 Esc 退出。

第一个类别的起始帧不需要是第 0 帧；中间也可以存在未标注帧。未标注帧不会写进输出 JSON。

## 标注段落选择

每个动作段落按以下规则选择起止帧：

- 起始帧：从夹爪在画面中出现并开始参与当前动作时开始。
- 结束帧：到当前动作完成后再保留约 1-2 秒。
- 如果两个动作之间存在明显等待、回位或无关运动，可以不标注这些空白帧。
- 每个标注段只覆盖一个明确动作，例如 `pick red cube`、`place red cube`；不要把多个动作合并成一个段。

示例数据可参考：

```text
./file-005_annotated.mp4
```

## 输出格式

默认输出文件名：

```text
<视频stem>_segments.json
```

如果视频名以 `_h264` 结尾，输出名会去掉这个后缀。例如：

```text
file-000_h264.mp4 -> file-000_segments.json
```

输出 JSON 示例：

```json
{
  "segments": [
    {
      "id": "file-000-1",
      "start": 0.0,
      "end": 25.7,
      "start_time": "00:00:00.000",
      "end_time": "00:00:25.700",
      "start_frame": 0,
      "end_frame": 513,
      "objects": [
        "red cube"
      ],
      "main_verbs": [
        "pick"
      ],
      "narration": " Pick the red cube."
    }
  ]
}
```

字段说明：

- `id`: 段 ID，格式为 `<视频stem>-<段序号>`。
- `start` / `end`: 起止时间，单位是秒。
- `start_time` / `end_time`: `HH:MM:SS.mmm` 格式时间戳。
- `start_frame` / `end_frame`: 起止帧号。
- `objects`: 从类别文本中解析出的对象；无法解析时为空数组。
- `main_verbs`: 从类别文本中解析出的动作；无法解析时为空数组。
- `narration`: 类别文本生成的描述句。

## 类别解析规则

类别文本会先去掉开头编号前缀：

```text
N1:pick red cube -> pick red cube
M1：pick red cube -> pick red cube
C10:place the yellow cube -> place the yellow cube
label1 - box free and closed -> box free and closed
```

如果类别以已知动作开头，例如 `pick red cube`，会解析为：

```json
{
  "objects": ["red cube"],
  "main_verbs": ["pick"],
  "narration": " Pick the red cube."
}
```

如果类别不是动作开头，例如 `box free and closed`，会保存为：

```json
{
  "objects": [],
  "main_verbs": [],
  "narration": " Box free and closed."
}
```

动作拆解依赖 `video_labeler_timestamp.py` 里的 `ACTION_VERBS`。只有类别文本的第一个词在 `ACTION_VERBS` 中，工具才会把它拆成 `main_verbs` 和 `objects`。

例如：

```text
pick red cube
```

如果 `pick` 在 `ACTION_VERBS` 中，会保存为：

```json
{
  "objects": ["red cube"],
  "main_verbs": ["pick"],
  "narration": " Pick the red cube."
}
```

如果需要支持新的动作词，例如 `grab red cube`，需要先在 `video_labeler_timestamp.py` 的 `ACTION_VERBS` 中加入 `grab`：

```python
ACTION_VERBS = {
    "grab",
    "pick",
    "place",
    ...
}
```

添加后，类别 txt 中以 `grab` 开头的类别就可以被拆解为：

```json
{
  "objects": ["red cube"],
  "main_verbs": ["grab"],
  "narration": " Grab the red cube."
}
```
