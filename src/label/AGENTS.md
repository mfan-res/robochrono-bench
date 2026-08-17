# AGENTS.md · src/label

给在这个目录里干活的编码 agent。先读同目录 `README.md` 了解这一段是干什么的，
再读这份。根目录 `DEVLOG.md` 有全部决策与踩坑记录。

---

## 每次改动后必须跑

```bash
cd <repo-root>
python3 src/label/tests/test_core_replay.py   # 必须全绿
python3 src/label/validate.py                 # 必须**恰好 64 条**发现
```

**64 这个数字是基线。** 变多 = 改坏了；变少 = 要么真修好了（那就在 DEVLOG 说明），
要么把检查弄失效了（更常见）。**不要为了让它变少而放松判据。**

schema 校验：

```bash
python3 - <<'PY'
import json, jsonschema
from pathlib import Path
v = jsonschema.Draft202012Validator(json.loads(
    Path("src/common/schemas/segments.json").read_text(encoding="utf-8")))
bad = [p for p in Path("data/label").rglob("segments/*_segments.json")
       if list(v.iter_errors(json.loads(p.read_text(encoding="utf-8"))))]
print(f"{320 - len(bad)}/320 通过", bad[:3])
PY
```

---

## 不可违反的约束

### 数据

- **`data/` 下的文件不要随手改。** `data/label` 与 `data/llm_cache` 是**不可再生层**
  （人工标注 / LLM 输出），没有源头可以重建。要改必须：先备份到新目录、
  写 `corrections.json` 记录映射与理由、跑回归确认没动到时间轴。
  先例见 `data/label/tea2/corrections.json`。
- **`data/label/**/.bak/` 不进 git**（`.gitignore` 已排除）。它是「改坏了还没提交」
  的即时兜底，**长期历史归 git**。不要把它加回版本控制。
- `src/label/upstream/` 三个文件是上游原样存档，**只读**。它们已不被使用，
  留着是为了在行为存疑时能比对。

### 语义

- **subtask ID 永不改变。** 改措辞只动 `subtasks.json` 的 `text`；`id` 一旦分配就冻结。
  这是整套设计的支点 —— 同一个动作在流水线各处曾有 11 种表示，
  表示越多越难保持一致，而不一致时不报错。
- **动作名一律精确相等匹配。** `pick_up_teapot` 是 `pick_up_teapot_lid` 的前缀，
  用 `startswith` / `in` / 子串会撞。
- **`start_frame` / `end_frame` 是权威**，`start` / `end` / `*_time` 是派生量，
  由 `core.Segment.as_record()` 按 fps 统一算。**任何地方都不要相信文件里写的秒**，
  要用就重算。
- **判断区间归属走帧号，不走秒。** `end = (end_frame + 1) / fps` 使相邻段的秒区间
  必然重叠一帧（全量 631 处）。按秒判「落在恰好一个段内」会在这些点上判到两个段。
- **段 id 由起始帧派生**（`file-000@f000277`），不用序号 —— 序号会在中间插段时
  让后面所有 id 平移。起始帧撞车时加 `-2` 后缀（`core._dedupe_ids`，
  `pen_inbox/file-037` 是唯一实例）。

### 契约

- **`schemas/segments.json` 拒绝 `metadata` / `window_type` / `original_*`。**
  那些是**出题产物**。曾经被回写进标注层，导致下一轮出题把上一轮的
  `pick_before_window=10.0` 当成人工真值。**不要为了让某份数据通过校验而放宽这条。**
- `source` 块必填。缺了就无法回答「这份标注用的哪版工具、哪版词表、对哪段视频」。
  `tool_version` 写 `unknown` 是**故意的**（上游工具不记版本，我们已在 wash 上撞过
  版本漂移）—— **不要编一个像样的版本号填进去**。

---

## 文件地图

| 文件 | 是什么 | 改动风险 |
| --- | --- | --- |
| `core.py` | 语义层：Segment、id 派生、subtask 定义、文档组装 | **高** —— 改了必跑回归 |
| `validate.py` | 六条检查。标注器与离线共用 | **高** —— 判据分叉过一次，代价很大 |
| `serve.py` | 后端：静态 + `/api/{families,episodes,episode,usage,save,subtasks}` + Range | 中 |
| `ui/index.html` | 单页前端，原生 JS 无构建 | 低 |
| `tests/test_core_replay.py` | 回归：拿 ID 化之前的 1,859 段当语料 | 改判据前先想清楚 |
| `migrate_to_subtask_ids.py` | 一次性迁移，已执行 | 不要再跑 |
| `restore_stackcubes.py` | 一次性还原，已执行 | 不要再跑 |
| `upstream/` | 上游原样存档 | **只读** |

---

## 回归怎么读

`test_core_replay.py` 的判据**不是「输出相同」，是「每处差异都被声明过」**：

```
逐字节相同      忠实保留了原信息
声明过的差异    DECLARED_B01 / DECLARED_B02 里列的（插 the、动词表硬编码）
未声明的差异    改坏了
```

语料是 `data/label/<族>/segments.before_subtask_id/`（迁移时自动留下）。
预期分布：express 50 处（B-02）、tea2 63 处（B-01）+ 21 处无对应（人工补 up 的那条）、
其余五族零差异、stack_cubes 跳过（还原后段数 300→200）。

**要修新 bug 就先往 `DECLARED_*` 加一条并写清理由**，不要直接让测试通过。

---

## 已经踩过的坑

- **`SimpleHTTPRequestHandler` 的 `directory` 设类属性无效** ——
  `__init__` 会用参数覆盖（默认 `os.getcwd()`）。必须 `partial(Handler, directory=...)`。
- **`Accept-Ranges: none` 会让浏览器整段下完才播。** 全帧内视频很大
  （tea2 单集 213 MB）。Range 支持在 `Handler._serve_media`，别删。
- **`pkill -f "<含命令行片段>"` 会匹配到执行它的那条 bash 命令本身，把 shell 杀掉。**
  停服务要按 PID：
  ```bash
  for p in $(ss -tlnp | grep ':8000' | grep -oP 'pid=\K[0-9]+' | sort -u); do kill $p; done
  ```
- **端口没释放时新进程会静默退出**（`Address already in use` 只在日志里）。
  重启后要确认新进程真的在跑，别看到旧版本还以为改动没生效。
- **`meta/episodes/*.parquet` 有多个 `*_file_index` 列** ——
  必须取 `videos/` 开头的那个（`serve.py:episode_bounds`、`validate.py:episode_bounds`
  都注明了）。取错会得出「一个视频装 40 轮」这种荒谬结论。
- **全局 `keydown` 必须给可输入元素让路。** `KEYS` 占了数字整排与 `q…p`，
  加上 `k/z/s/空格/方向键`，输入框里能打的字母只剩一半。
  症状是「能粘贴、打不出来」（粘贴不经过 keydown）。
  放行 `INPUT` / `TEXTAREA` / `contenteditable`，别只放行 `SELECT`。
- **`ffprobe -count_frames` 会解码整段视频**，320 个视频跑不完。
  用 `-show_entries stream=nb_frames`（读容器元数据，瞬时）。

---

## 没做的事（故意的）

- **`upstream/` 那 416 行 OpenCV GUI 没有重写。** 它无已知问题，
  且其产出（320 份标注）经全量核验质量很高；本机没有 cv2 与显示器，改了验证不了；
  而且**现在没有消费者** —— 网页版已经覆盖了标注场景。
- **`describe()` 不再用于产出标注**（段里只存 ID），保留它是给出题阶段
  按需推导动词/宾语用的，以及对迁移前语料做回归。
- **重复动作（wash）不禁止，只警告。** 洗两个盘子是任务本身如此。
  怎么处理属于能力定义层，见 `DEVLOG.md` 的 P-05，**待人决策，不要自行选定**。

---

## 遇到判断不清的地方

优先查 `DEVLOG.md` 的「被推翻的判断」一节 —— 里面有 19 条我们先错后纠的记录，
包括好几次「用一个自洽的观察给算术产物编故事」。
**如果你的解释听起来很合理但没有实测支撑，先怀疑它。**
（那 19 条里有好几条，都是画面看起来支持某个解释，但画面支持的是
「当时在发生什么」，不是「为什么那样处理」。）
