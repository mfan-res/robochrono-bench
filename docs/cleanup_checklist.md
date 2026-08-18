# 重构残留物清理清单

> 来源：2026-08-18 的第二轮独立复核（对第一轮 12 项发现逐条 double check，另出 9 项新发现）。
> 基线：`HEAD = 8e184f5`，工作区含 `src/vqa/plan.py` / `assets.py` 的未提交改动。
>
> **行号锚定在上述基线。** 由于有并发会话在改 `src/vqa/` 与 `src/eval/robochrono/tasks/`，
> 行号可能已经漂移 —— 所以每条都给了「搜索锚点」（一段可 grep 的原文），
> 找不到行号时按锚点定位。

## 怎么用这份清单

- 每条都是独立可做的，格式统一：**位置 → 现状 → 改成 → 怎么验**。
- `[ ]` 打勾即可。批次内的条目互不依赖，批次之间的依赖写在各批次开头。
- 标 **⚠决策** 的条目不要直接动手，先去 §0 拿到人的答复。
- 「怎么验」里的命令都是只读的，可以随便跑。

## 优先级速查

| 严重度 | 条目 |
| --- | --- |
| **高** | 1.1（抽帧覆盖只在 `run` 生效，影响 time 分数） |
| 中 | 4.1 4.2 4.3 4.4 4.5、2.1 2.4、3.1、5.2 5.3 |
| 低 | 批次 1 全部、3.2、5.1 5.4 5.5 |

---

## §0 · 动手之前要拿到的答复（⚠决策）

这些无法从仓库内部判断，答复会改变后面怎么做。

- [ ] **D-1 · 并发会话**：有另一个会话正在改 `src/vqa/plan.py` `assets.py` `compose.py` `pack.py`
      与 `src/eval/robochrono/tasks/choice.py`，并生成 `build/blind_v2/`（加 `left_right` /
      `image_in_video` 图选项题型）。**批次 1 的 1.4 1.8 1.9 与批次 4 的 4.5 会和它冲突**，
      先确认那边的进度。

- [ ] **D-2 · `frames_by_run` 与 `frame_variants` 谁优先？**（决定 1.1 怎么改）
      两者对 time 语义冲突：前者固定 32 帧，后者对 68 秒视频是 68/136 帧。
      是「time 固定 32、其余走 variant」，还是「time 也扫 variant」？

- [ ] **D-3 · 32 帧到底实测过没有？** 三方冲突：
      `providers.json` 的 `_note` 说「实测 32 帧 12.55G→6.31G，160 帧可跑」，
      同文件 `_frames_by_run_note` 说「⚠ 32 帧的显存未在 24 GB 卡上实测过」，
      commit `d88505a` 的 message 说「32 帧实测不 OOM」。
      如果前者是在别的卡上测的，要注明卡型。

- [ ] **D-4 · `robochrono` 有没有仓库外的 import 使用者？**
      影响 `ResultStore.export()`（5.3）、`extract.py` 全家（5.2）、`tasks.base.Task`（1.3）
      能不能按内部符号处理。

- [ ] **D-5 · `python -m robochrono export` 对外承诺过没有？**（决定 5.3 是接回还是删承诺）

- [ ] **D-6 · `extract.py` 是已放弃的方案，还是排期中的下一步？**（决定 5.2）
      它解决的问题（RynnBrain time 漏答 48%、trajectory 95% 用归一化坐标）
      在 `docs/disclosures.md` 里没有作为已知偏差被披露。

- [ ] **D-7 · 完整 v1 数据（61 GB）能不能进 CI？**（决定 3.4 走哪条路）

- [ ] **D-8 · `build/blind_*.json` 的 10 份变体，哪几份是论文/报告要引用的证据？**（决定 3.6）
      看名字 `blind_ship` / `blind_final2` / `blind_after_plates` 像是决策依据。**不可再生**。

- [ ] **D-9 · `REFACTOR_PLAN.md` 在哪？** 三处引用它，全仓库不存在：
      `src/eval/docs/dataset_structure.md:312`、`src/eval/robochrono/parsing.py:17`、
      `src/eval/configs/config_smoke.json:8`。留在旧仓库，还是已被 `docs/` 下某份取代？

- [ ] **D-10 · `src/vqa/recipes/` 是已废弃的设计，还是还没建？**
      `src/vqa/README.md` 与 `data/README.md` 都按「配方是 JSON 文件」描述，
      而 `plan.py` 把配方做成了代码常量 `RECIPE_VERSION = "v2.0"` + CLI 参数。

---

## §1 · 批次 1：零风险机械清理

**依赖**：无。除 1.1 外都可以立即做。删除类改错会立刻 `NameError`，不需要测试兜底。

### 1.1 ⚠决策 · `frames_by_run` 只在 `run` 生效，`matrix` 从不读它 —— **高危**

- [ ] **位置**：`src/eval/robochrono/cli.py:99-103`（唯一读取点）
      `src/eval/robochrono/matrix_run.py:140-153`（`_prepare`，不读）
      `src/eval/robochrono/matrix_run.py:32-49`（`_apply_frames`，只处理 `frame_variants`）
- **搜索锚点**：`by_run = _providers_cfg().get("frames_by_run"`
- **现状**：`frames_by_run`（time = uniform 32）全仓库只有 `cli.py` 一处读取。
  `matrix` 链路是 `cmd_matrix → run_matrix → _prepare → runtime_config + _apply_frames`，
  三者都不碰它。于是走 `matrix` 时：
  - API 模型 → `resolve_frames` 落 legacy 分支 → `num_segments = 16` → **10% 的 time 题看不到被问的动作**
  - 本地模型 → `frame_variants` 的 fps=1/2
  而 `run.sh` 第 3 步跑的正是 `matrix`。
  引入这条配置的 `d88505a` 把它列为「一处致命」，但改动只落在 `cli.py`。
- **改成**：把 `frames_by_run` 的解析下沉到 `vlm_api.runtime_config` 或 `matrix_run._prepare`，
  两条入口共用同一处覆盖逻辑；并按 **D-2** 的答复写清与 `frame_variants` 的优先级。
- **怎么验**：
  ```bash
  git grep -n frames_by_run           # 改前只有 cli.py + providers.json
  python -m robochrono plan --plan src/eval/configs/plan.json   # 只读，不调模型
  ```
  改完后建议加一条断言测试：给定 spec，`_prepare` 返回的 `runtime["frames"]` 等于期望值。
- **影响面**：所有经 `run.sh` / `matrix` 产出的 time 分数 —— **已产出的 time 结果需要重跑**。

### 1.2 · 删 13 条未使用 import

- [ ] 逐条删除（AST 扫描 + 逐条人工核对，全部可安全删）：
  ```
  src/eval/robochrono/cli.py:18                      import sys
  src/eval/robochrono/matrix_run.py:22               from .tasks.base import load_items
  src/eval/robochrono/normalize.py:45                from .tasks.base import Unit（保留 load_items）
  src/eval/robochrono/preflight.py:22                import subprocess
  src/eval/robochrono/preflight.py:29                from .tasks.base import load_items
  src/eval/robochrono/tasks/choice.py:16             from pathlib import Path
  src/eval/tests/test_normalized_equivalence.py:49   canonical_family
  src/eval/tests/test_request_equivalence.py:18      import json
  src/eval/tools/proposal_eval.py:22                 Any
  src/label/core.py:44                               field（保留 dataclass）
  src/migrate/fetch_raw.py:31                        import sys
  src/migrate/verify_bar_crop.py:43                  import sys
  src/migrate/verify_bar_crop.py:44                  Counter
  ```
- **注意**：`src/eval/robochrono/mediaindex.py` **是干净的**，第一轮审计说它有未使用 import 是误报，别动。
- **怎么验**：`python3 -c "import ast,sys; ast.parse(open('<file>').read())"`，或直接跑一次对应模块的 `--help`。

### 1.3 · 删 5 个零引用符号 + 让 `Task` 真正被用起来

- [ ] `src/eval/robochrono/pool.py:142-147` —— 删 `_CHOICE_RUNS` **连同它上面那两行注释**
      （注释写「现在只剩 preflight/report 在用」，`git grep _CHOICE_RUNS` 零外部命中，注释本身是假的）
- [ ] `src/vqa/compose.py:55` —— 删 `ASSETS = ROOT / "data" / "vqa" / "assets"`（全仓库仅此一行）
- [ ] `src/eval/robochrono/media_prep.py:251` —— 删 `ffmpeg_available()`
- [ ] `src/label/core.py:191` —— 删 `can_add()`（若在线重叠校验要用，改由 `serve.review()` 调用，见 4.3）
- [ ] `src/label/core.py:218` —— 删 `build_subtasks()`（**先确认 `subtasks.json` 的生成路径**，见 D-10 邻接）
- [ ] `src/eval/robochrono/tasks/base.py:43` —— `Task` Protocol **不要删**。
      改成把 `engine.py:54` `_run_unit(task: Any, ...)`、`engine.py:98` `run(task: Any, ...)`、
      `matrix_run.py` / `pool.py` 里的 `task: Any` 换成 `task: Task`。
      它描述的四钩子契约是这套架构的核心，删掉纯叙述抽象不如让它真的约束。
- **怎么验**：`git grep -n '<符号名>' -- . `，删前应只剩定义处。

### 1.4 · 删不可达代码

- [ ] `src/eval/tests/test_normalized_equivalence.py:157` —— `return 0` 之后的 `return 1`，不可达
- **搜索锚点**：`python3 tools/build_normalized.py   # 在旧仓库里跑`（紧接其后两行）

### 1.5 · 修两处互指的「不可能分叉」假声明

- [ ] `src/vqa/plan.py:237-238` —— 现写「**这是选项构造的唯一实现** —— `blind.py` 直接导入它，两边不可能分叉」
- [ ] `src/vqa/blind.py:65-67` —— 现写「需要模拟别的策略时也从那里导入，**两边不可能分叉**」
- **事实**：`blind.py` 从不 import `plan.py`（`blind.py:47-49` 只 import `distract` 与 `vocab`）。
  `options_as_built` 是读 `plan.json` 的产物（这条确实不会分叉），
  但 `options_cross` / `options_four` 是本地重写，**且已经分叉**：
  - 轮转盐不同：`plan.py:275-278` 用 `md5(f"{item_id}|in")` / `|out`；`blind.py:132,137` 用 `md5(id)` / `md5(id+"|x")`
  - 可借池不同：`plan.py:335-338` 是去重排序后的集合；`blind.py:135` 是含重复、未排序的列表
- **改成**：先只改注释，说清「`as_built` 读产物，其余五条是本地重写的历史策略」。
  真正收敛见 4.5。
- **不影响**：已发布的盲测主结论（走 `as_built`）。

### 1.6 · 修 6 条 Markdown 坏链接

- [ ] `docs/vqa/field_ownership.md:3` — `[v2 设计草案](schema_v2_proposal.md)` → `item_schema_design.md`
      （同目录，内容与 `src/eval/docs/schema_v2_proposal.md` 逐字节相同）
- [ ] `docs/vqa/item_schema_design.md:4` — `[数据结构现状](dataset_structure.md)` → `../../src/eval/docs/dataset_structure.md`
- [ ] `docs/vqa/item_schema_design.md:4` — `[重构方案](dataset_refactor_plan.md)` → `../../src/eval/docs/dataset_refactor_plan.md`
- [ ] `docs/vqa/item_schema_design.md:128` — `[清单](questions_for_data_team.md)` → `../questions_for_data_team.md`
- [ ] `docs/vqa/item_schema_design.md:344` — `[给数据方的清单](questions_for_data_team.md)` → `../questions_for_data_team.md`
- [ ] `src/eval/docs/dataset_structure.md:312` — `[...BC-16](../../REFACTOR_PLAN.md)` → 见 **D-9**
- **怎么验**（只读链接检查，可以直接跑）：
  ```bash
  python3 - <<'PY'
  import re, subprocess, pathlib
  root = pathlib.Path('.')
  mds = subprocess.run(['git','ls-files','*.md'],capture_output=True,text=True).stdout.split()
  pat = re.compile(r'\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
  bad = []
  for rel in mds:
      p = root/rel
      for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
          for m in pat.finditer(line):
              t = m.group(2)
              if t.startswith(('http://','https://','mailto:','#')): continue
              t = t.split('#')[0]
              if t and not (p.parent/t).resolve().exists():
                  bad.append(f'{rel}:{i}: {m.group(0)}')
  print('\n'.join(bad) or '无坏链接')
  print(f'共 {len(bad)} 条')
  PY
  ```
  基线是 **6 条**，改完应为 0（或只剩 D-9 待定的那条）。
- **建议顺带**：把这段检查加进 `src/eval/tests/run_all.py`，防止复发。

### 1.7 · 修 4 处非链接式失效引用

- [ ] `src/vqa/README.md:3` — `读 data/{raw,label,llm_cache} + recipes/` → 目录不存在
- [ ] `src/vqa/README.md:5` — `recipes/<family>.json 跟代码走 git` → 同上（见 **D-10**）
- [ ] `data/README.md:8` — `由上面三样 + src/vqa/recipes 确定性地产出` → 同上
- [ ] `src/eval/configs/plan.json:91` — `要跑 v1 时用 configs/plan_v1.json` → 该文件不存在
      （`ls src/eval/configs/` 只有 `config_smoke.json` `environments.json` `plan.json` `providers.json`）
- **另**：`src/vqa/README.md:10`「唯一的非确定性是 LLM 干扰项，已冻结在 `data/llm_cache/`」
  —— 干扰项自 D-38 起改为只用真实标签（`plan.py:595`），这句已失效，一并改。

### 1.8 · 修根 README 的过期状态与数字

- [ ] `README.md:37` — 「④ 出题 ⬜ **下一步**」→ 已完成（`src/vqa/` 七步齐全）
- [ ] `README.md:39` — 「⑥ 评测 …… 在旧仓库…… 待迁入」→ 已迁入（commit `eb565f6` `ef78c3c` `4502063` `b1ec6f8`）
- [ ] `README.md:86-87` — 「vqa（待建）」「eval（待从旧仓库迁入）」→ 同上
- [ ] `README.md:67` — 「7 个活跃族 …… tea2」→ **6 个**。
      `data/families.json` 记 tea2 为 `removed`，`ls data/label/` 只有 6 个族目录，
      `docs/disclosures.md` §8 也说 tea2 已移出
- [ ] `README.md:68` — 「277 集 · 1,759 个标注段 · 43 个 subtask」→ 实测 **249 份 segments 文件 · 1,390 段 · 40 个 subtask**
- [ ] `src/eval/README.md:26` — 「24 个文件 7,291 行」→ 实测 24 个文件 **7,372** 行（低价值，可选）
- **怎么验**：
  ```bash
  python3 - <<'PY'
  import json, glob
  segs = eps = subs = 0
  for f in sorted(glob.glob('data/label/*/segments/*_segments.json')):
      segs += len(json.load(open(f))['segments']); eps += 1
  for f in sorted(glob.glob('data/label/*/subtasks.json')):
      subs += len(json.load(open(f))['subtasks'])
  print('segments 文件', eps, '段', segs, 'subtask', subs)
  PY
  python3 -c "import json;d=json.load(open('data/families.json'))['families'];print([k for k,v in d.items() if v['status']=='active'])"
  ```
- **建议**：这些数字改成由 `index.py` 输出，或加一条「改数据后重跑 index.py 更新此处」的注记。

### 1.9 · 修 time 指标与抽帧状态的文档矛盾

以代码为准。**实际计算的指标**（`src/eval/robochrono/tasks/time_eqa.py:338-357`）：
`mean_tIoU`、`tIoU@0.3/@0.5/@0.7`、`center_inside_acc`、`pointing_acc`、
`mean_overlap_recall`、`mean_abs_start_error`、`mean_abs_end_error`、`parse_failure_rate`。
**主指标**由 `src/eval/robochrono/tasks/__init__.py:78` 单点决定，是 **`tIoU@0.5`**。

- [ ] `src/eval/RUNBOOK.md:19` — 表格里 time 的主指标写 `mean_tIoU` → 改 `tIoU@0.5`
- [ ] `src/eval/RUNBOOK.md:192-196` — 「抽帧策略尚未定案…… 只是让链路能跑，不是评测配置」
      → 已有 `frames_by_run: {time: uniform 32}` 与实测覆盖率表（8/16/24/32 → 31%/10%/3%/1%）。
      **注意**：这条要等 1.1 做完再改，否则文档会比代码超前
- [ ] `src/vqa/blind.py:32-33` — 「time 是开放作答，而『多少算答对』的容差目前没有定义」
      → `docs/disclosures.md` §10 已定 `tIoU@0.5`，换算约 ±2 秒
- [ ] ⚠决策 `src/eval/configs/providers.json` — 同一文件内两句关于 32 帧的话直接冲突，见 **D-3**
- **不用改**：`docs/disclosures.md` §10 是全仓库唯一与代码一致的那份，其余向它对齐。

### 1.10 · 修 `data/label/README.md` 展示的旧 schema

- [ ] `data/label/README.md:5-15` 现在展示的是迁移前的结构：
  | 文档写的 | 实际 |
  | --- | --- |
  | `segments/file-XXX.json` | `segments/file-XXX_segments.json` |
  | `categories.txt` | `categories.deprecated.txt` + `subtasks.json` |
  | 字段 `objects` / `main_verbs` / `narration` | 字段 `subtask`（ID 引用） |
- **权威来源**：`src/common/schemas/segments.json`
- **怎么验**：`head -c 800 data/label/tea/segments/file-000_segments.json`

### 1.11 · 修 `data/llm_cache/README.md` 的自相矛盾

- [ ] `data/llm_cache/README.md:22-25` — 目录块写「`v2/<族>.json`  本轮生成，**出题实际使用**」，
      与自己顶部横幅（`:3-12`「三代全部退场，出题已不读这里」）直接冲突；**且完全没有列 `v3/`**
      （`ls data/llm_cache/v3/` 有 7 份）
- [ ] `data/llm_cache/README.md:12` — 「唯一还在读它们的是 `vocab.py` 的 v1 命中率汇总」
      → 还有 `src/vqa/blind.py:209` 无条件读 `v3/`（见 1.13）

### 1.12 · `--policy both` 默认展开全部 6 套策略

- [ ] `src/vqa/blind.py:230` — `which = arg("--policy", "both")`，**`both` 是默认值**
- [ ] `src/vqa/blind.py:236` — `policies = list(POLICIES) if which == "both" else [which]`，
      而 `POLICIES`（`:144-145`）有 **6** 个键
- [ ] `src/vqa/blind.py:6` — 用法注释写「`--policy both`   # 现状与策略③ 两套对照」，与实际行为不符
- **后果**：裸跑 `python3 src/vqa/blind.py --n 20` 会发 6 倍的 API 请求。
- **改成**：默认值改 `as_built`，并把 `both` 改名为 `all`（或删掉这个聚合值）；同步改 `:6` 的注释。
- **背景**：`:271-276` 的 label 表显示 6 套里 4 套（`allreal` / `pool` / `three` / `cross`）
  是已被否决的历史方案；当前方案是四选一（`plan.py:126` `DISTRACTORS_PER_QUESTION = 3`）,
  对应 `as_built` 与 `four`。

### 1.13 · `blind.py` 无条件加载已退场的 `llm_cache/v3`

- [ ] `src/vqa/blind.py:209-210` — `pools = {...v3/{f}.json...}` 在 `main()` 顶部执行，**不看 `--policy`**
- **现状**：只有 `options_pool`（策略④，已否决）用得上 `pools`。
  当前 `v3/` 覆盖 7 个族、`vocab` 有 6 个族，所以能跑；
  但只要新增一个族或清理 `v3/`，`blind.py --policy as_built` 就会因一个已退场的资产直接崩。
- **改成**：惰性加载 —— 只在 `"pool" in policies` 时读。

### 1.14 · 统一「检查有几条」的说法

- [ ] 四处各说各的：
  ```
  src/label/validate.py:3     「六条检查」
  src/label/validate.py:58-59  SEVERITY 实际有 8 个键：污染/覆盖/歧义/重叠/派生/引用/序列/可疑
  src/label/validate.py:246   「六条检查全部通过」
  README.md:140               「六项核验」
  README.md:145               「七条检查全部通过」
  src/label/README.md:21      「六条」   src/label/README.md:107 「七条」
  src/label/AGENTS.md:13      「六条」
  ```
- **来由**：`README.md:152` 说「新增第七条『序列』检查」—— 「七」是加了序列之后的数，
  「六」是加之前的、`validate.py` 的 docstring 没跟着改；「可疑」是第八类但从未被计数。
- **改成**：统一为「八类检查（六条硬错 ✗ + 两条待人判断 ⚠）」，或按 `SEVERITY` 的 ✗/⚠ 分组表述。
- **怎么验**：`sed -n '56,60p' src/label/validate.py` 看 `SEVERITY` 的实际键数。

### 1.15 · 给已知失效的脚本加运行守卫

- [ ] `src/migrate/split_packed_episodes.py` — docstring 第一行就写「**当前不要运行**」，
      第 4 行写「⚠ **本脚本的核心假设是错的**」（假设三视角共享同一套文件结构，实测 tea2 不成立，D-42）。
      但它仍可以直接 `python3` 跑起来并写数据。
- **改成**：`main()` 开头加 `raise SystemExit("本脚本的核心假设已被证伪，见 docstring 与 D-42")`，
      或移入 `src/migrate/archive/`。
- **不要删** —— 它是 D-42 那个反直觉发现的来源记录。

### 1.16 · 标记已失效的 TODO

- [ ] `src/migrate/README.md:9-27` — 待办「反建缓存的键格式未验证（2026-08-17 暂缓）」，
      正文说「要用之前必须先确认这一点」。
      而 `src/vqa/plan.py:595` 与 `data/llm_cache/README.md:3-5` 都已明确三代缓存
      **永不再用于出题**（D-37 / D-38），此待办的前提已消失。
- **改成**：改写为「已失效：干扰项自 D-38 起不再使用任何生成物，此待办无需再跟进」，
      **保留原文** —— 符合本项目「判断被推翻时不删原文」的惯例（`README.md:178-179`）。

---

## §2 · 批次 2：先补测试，再清理

**依赖**：2.2 需要 1.4 先做完（删掉不可达的 `return 1`）。
批次 3 与批次 4 依赖本批次。

### 2.1 · 把 `smoke_v2.py` 纳入测试套件

- [ ] `src/eval/tests/run_all.py:15` — `TESTS = sorted(p for p in HERE.glob("test_*.py"))`
- **现状**：`src/eval/tests/smoke_v2.py` **不匹配这个 glob，从不执行**。
  `git grep smoke_v2` 在全仓库（含 `*.md` `*.sh`）零命中。
  而它的 docstring 写的是：「端到端冒烟：④ 出的题能不能被 ⑥ 完整处理……
  `pack.py` 自己的出厂检查只验字段形状；『评测端能不能真的处理』得由评测端自己回答。
  两边各写一套判据的话，迟早会对不上 —— **这个项目已经栽过三次**」。
  **这是全仓库唯一一条把 v2 出题产物与 v2 评测端连起来的自动检查。**
- **改成**：重命名为 `test_smoke_v2.py`，或把 glob 扩成 `("test_*.py", "smoke_*.py")`。
- **先做**：改名前本机跑一次确认它当前是通过的（它依赖 `data/vqa/eval/`，可能同样需要跳过分支）：
  ```bash
  python3 src/eval/tests/smoke_v2.py; echo "exit=$?"
  ```

### 2.2 · 补 `pool.run_pool` 的失败传播测试

- [ ] 新建 `src/eval/tests/test_pool_failure.py`
- **为什么**：`git grep -l pool -- src/eval/tests` 当前**为空** —— 多 GPU 路径零测试覆盖。
- **测什么**（用 fake queue，不需要 GPU）：
  1. worker 启动失败（`pool.py:86-88` 那条路径）→ `run_pool` 返回的 `errors >= 1`
  2. 某个 unit 抛异常 → `errors` 计数正确，且 `rows` 里有 `error_rows` 占位行
  3. 所有 worker 提前退出 → `run_pool` 不死锁，打印「所有 worker 已退出」并返回
- **这条是 4.1 和 4.2 的前置**。

### 2.3 · 补出题核心函数的单元测试

- [ ] `src/vqa/plan.py:235` `build_options` — 当前**无任何测试**
- [ ] `src/vqa/plan.py:165` `cooccurrence`
- [ ] `src/vqa/plan.py:155` `clip_for`
- [ ] `src/vqa/plan.py:147` `assert_no_leak`
- **为什么**：`src/vqa/tests/test_verifier.py` 只测 `distract.py` 的语义验证器
  （已退场的旧生成器），且需要 API key。当前出题链的核心函数一条测试都没有。
- **注意**：`plan.py:600-609` 的两次构建指纹自检只证明**确定性**，不证明**正确性**。

### 2.4 ⚠决策 · 落地测试分层（依赖 D-7）

- [ ] **现状**：`src/eval/tests/` 六套里有两套在普通 clone 上以退出码 0 跳过：
  - `test_normalized_equivalence.py:148-156` — `normalized/` 未构建（需完整 v1 QA）
  - `test_request_equivalence.py:43-46` — 需真实媒体（`paths.py:41-49` 的 `qa_root(need_media=True)`）
- **注意（第一轮审计此处偏严）**：`run_all.py:25-26,34` **确实**把跳过单列一栏、
  没有混进通过数，docstring `:7-8` 就是为这件事写的。真正的问题是
  `run_all.py:35` `raise SystemExit(1 if failed else 0)` —— **全部跳过时整体仍退出 0，
  用作 CI 门禁无法区分「验过了」与「没验」**。
- **注意 2**：`src/eval/README.md:15` 的「通过 5　跳过 1」在**有 v1 数据的机器上成立**
  （`ROBOCHRONO_V1_ROOT` 默认指向 `/mnt/.../benchmark/eval`，该路径存在）；
  **普通 clone 是 4 通过 2 跳过**。README 缺的是环境前提说明，不是数字写错。
- **改成**（四选一或组合）：
  1. **外部数据测试套件**（推荐，不依赖 D-7）：两套移进 `tests/external/` + `run_external.sh`，
     只在有 `ROBOCHRONO_V1_ROOT` 的机器上跑。`run_all.py` 保持只跑自足的四套并**全部要求通过**，
     不再有跳过分支
  2. **CI 分层**：PR 门禁跑 `run_all.py`（要求 0 跳过 0 失败）；nightly 在有 v1 数据的机器上跑 external
  3. **最小媒体 fixture**：为 `stack_cubes` 的 12 个抽样 unit 生成极小占位媒体（总计 < 2 MB）
     放进 `fixtures/media/`，让 `test_request_equivalence` 在普通 clone 也能跑起路径解析与请求组装。
     ⚠ 这会改变「比对的是真实媒体」的语义，应作为**第三档**而非替换
  4. **发布前门禁**：`RUNBOOK` 加一条前置项「跑 A4 新旧对照之前必须先过 normalized 等价性」，
     并让 `run_all.py` 在有跳过时打印「⚠ 本次运行未覆盖 N 项，不能作为发布依据」
- [ ] **顺带修**：`run_all.py:25` 用子串匹配 `"跳过" in out` 判定跳过 —— 任何测试只要输出里
      出现这两个字就会被记为跳过。当前只有那两套会打，但这是脆弱耦合。

---

## §3 · 批次 3：删除已确认的旧实现

**依赖**：3.1 需要 2.x 的移植验证；其余依赖批次 1 完成。

### 3.1 · `check_labels.py`：先移植，再删

- [ ] **位置**：`src/migrate/check_labels.py`
- **现状（比第一轮判断更严重）**：`:130`
  `narrations = [str(s.get("narration") or "")... for s in segments]`。
  实际段结构（见 `src/common/schemas/segments.json`，`additionalProperties: false`）
  只有 `id/start/end/start_time/end_time/start_frame/end_frame/subtask`，**没有 `narration`**。
  后果：`:131` 的 `dup` **恒为 `len(segments) - 1`**（所有 narration 都是空串，`set` 恒为 1 个元素）
  —— 不是静默归零，是**「重复动作」一栏对每一集报满**，即产生看起来像发现的假发现。
  `:182,195-197` 的「各族 narration 词表」恒为 `{'': N}`。
  调用方：零代码调用、零测试。
- **⚠ 但它不是被完全取代**（第一轮此处判断过头）。它有 `validate.py` 没有的能力：
  `:45 duration()` 与 `:54 frame_count()` **用 ffprobe 探真实视频**，据此做：
  1. 隐含 fps 与实际 fps 的一致性（`:117-121`）
  2. 帧号越界 `end_frame > frames + 2`（`:121`）
  3. 覆盖率 = 标注跨度 / 视频**真实**时长（`:139`）
  4. `gap_max` / `shortest` / `inverted`（`:141-144`）

  而 `src/label/validate.py` **完全不探视频**（import 里无 `subprocess`），
  fps 来自 `data/raw/<族>/meta.json`（`:103`），覆盖检查只比对 `episode_bounds`（`:157`）。
  「元数据声称的 fps 与视频实际 fps 是否一致」这条**目前没有任何替代者**，
  而它正是当年发现 tea2 问题的那类检查。
- **改成（按顺序）**：
  - [ ] ① 把 ffprobe 类检查移植进 `validate.py`，作为可选的 `--probe-video` 模式
        （默认关，因为要跑 249 次 ffprobe）
  - [ ] ② 移植完成并跑通后，再删 `check_labels.py`
  - [ ] ③ 若暂不移植：至少在文件头加一行
        `⚠ 读 narration，当前数据已改为 subtask，输出不可信，勿用`
- **怎么验**：`python3 src/label/validate.py` 应仍为**零条**（`README.md:154` 记录的基线）。

### 3.2 · 归档已退场的干扰项生成器

- [ ] `src/vqa/distract.py`（431 行）→ `archive/` 或 `experiments/`
- [ ] `src/vqa/pool.py`（273 行）→ 同上
- [ ] `src/vqa/tests/test_verifier.py`（只测 `distract.verify_prompt`，需 API key）→ 同上
- **前置**：`src/vqa/blind.py:48` `from distract import KEYS, MODEL, call_api` —— 移动前
  要么把这三个 API 工具提取到共用模块，要么让 `blind.py` 自带。
- **背景**：`src/vqa/plan.py:595` 已明写三代缓存全部退场，
  `build_options`（`plan.py:235`）只用真实标签。这两个生成器不再是出题链的一环。
- **数据不要删**：`data/llm_cache/` 的 v1-vendor / v2 / v3 是不可再生的 LLM 输出，
  也是 D-37/D-38 那个核心设计决定的全部实证依据。见 §6。

### 3.3 ⚠决策 · 归档 10 份盲测输出（依赖 D-8）

- [ ] `build/` 下除 `blind.json` 外的 10 份变体，`git grep` **全部零引用**（含 DEVLOG）：
  ```
  blind_after_plates.json  blind_cross.json  blind_final.json  blind_final2.json
  blind_four.json  blind_four_text.json  blind_n57.json  blind_ship.json
  blind_three.json  blind_v4.json
  ```
- **改成**：移入 `build/archive/blind/` 并加一份 `INDEX.md`，
  说明每份对应 `DEVLOG.md` 的哪条决策。
- **⚠ 不要只靠 Git 历史保存** —— 它们当前是被跟踪的工作区文件，删了要靠 commit hash 才能找回，
  而没有索引就等于找不回。它们是**不可再生的 LLM 输出**（DeepSeek，`temperature=0`
  但服务端不保证长期一致）。

---

## §4 · 批次 4：收敛调用链

**依赖**：4.1 4.2 需要 2.2 先做完（pool 的失败传播测试）。

### 4.1 · `_run_local_pool` 恒返回 0，与串行路径口径不一致

- [ ] **位置**：`src/eval/robochrono/matrix_run.py:228`（`return 0`）、`:194-196`（`_prepare` 失败只 `continue`）
- **对照**：`src/eval/robochrono/matrix_run.py:177-179` —— **同一个 `_prepare`、同一类异常**，
  在串行路径里是 `failures += 1`。这是同一份代码在两条路径上被计入 / 不计入，
  属实证的口径漂移，不是设计选择。
- **现状**：`stats`（`:218`）拿到了 `{"done", "errors"}`，只在 `:222` 打印，从不影响返回值；
  且 `stats` 定义在 `else:` 块内，函数末尾也拿不到。
  于是多卡路径下 QA 缺失、normalized 过期（`StaleNormalized`）、权重加载失败，
  `matrix` 一律经 `cli.py:245` 退出 **0**，并在 `:224-227` 照常写出 summary。
- **语义三分（判断依据）**：
  | 情况 | 当前行为 | 应当 |
  | --- | --- | --- |
  | 单题模型答错 | 不计失败 | ✅ 正确 |
  | worker 执行异常 | `pool.py:132-134` 记 error，计入 `stats["errors"]`，被丢弃 | ❌ 应传播 |
  | run 根本没跑起来（`_prepare` 抛错） | 串行路径认，多卡路径不认 | ❌ 应统一 |
- **改成**：`_run_local_pool` 累计 `_prepare` 失败数 + `stats["errors"]` 后返回。
- [ ] **顺带决策**：`engine.py:178` 的熔断只写 `summary["aborted"]`，
      `_run_serial` 也不计失败 —— 所以「连续 20 次失败」在**两条路径上**都不会让 `matrix` 退出非 0。
      这一层可能是有意的（`RUNBOOK.md:220` 有对应排查条目），但**没写进设计文档**。
      决定它是否应影响退出码，然后写进 RUNBOOK。
- **怎么验**：`git grep -n 'return 0' src/eval/robochrono/matrix_run.py`

### 4.2 · `pool._worker` 重复了 `engine._run_unit`，且已丢掉 `replay_key`

- [ ] **位置**：`src/eval/robochrono/pool.py:115-139` vs `src/eval/robochrono/engine.py:54-95`
- **现状**：两处逐段对应（`call_vlm` → `retry_parts` 循环 → `CallContext` → `task.rows`
  → `except` 走 `task.error_rows` → `row["timing"]`）。**已经分叉的三处**：
  1. `engine.py:65` 写 `runtime["replay_key"] = unit.key`，`pool.py` **没有**。
     `vlm_api.py:1271` 的 replay provider 读的正是这个键
  2. `engine.py:75` 打印 retry 提示，pool 静默
  3. `pool.py:138` 在 `timing` 里多一个 `worker` 字段
- **分叉 1 目前不可达**：`use_pool` 要求 `model.is_local`（`matrix_run.py:123`），
  而 replay provider 的 `kind` 不是 `local`（`matrix.py:38`）。
  所以这是**潜伏缺陷**，不是现行 bug —— 但它正说明「复制一份就会漂」。
- **改成**：`pool._worker` 里改调 `engine._run_unit(task, unit, runtime)`，再自行补 `worker` 字段。
  **不改任何可观察行为**（因为分叉 1 当前不可达）。

### 4.3 · `serve.py:review()` 是第二份判据，只覆盖 8 类检查里的 3 类

- [ ] **位置**：`src/label/serve.py:259-296` vs `src/label/validate.py:104-222`
- **现状**：`validate.py:58-59` 的 `SEVERITY` 声明 **8 类**：
  污染 / 覆盖 / 歧义 / 重叠 / 派生 / 引用 / 序列 / 可疑。
  `serve.review()` 只实现 **3 类**：帧重叠（`:270-274`）、多集漏标（`:277-283`）、
  同集重复 subtask（`:285-295`）。

  **在线保存时不会被拦下的 5 类**：
  | 检查 | 抓什么 | 为什么重要 |
  | --- | --- | --- |
  | 污染 | 出题产物回写进标注 | **就是 P-03 本身** |
  | 引用 | 未定义的 subtask ID | ID 化之后的新风险 |
  | 派生 | `start/end` 与帧号不自洽 | 上游 `end=(f+1)/fps` 的隐含语义 |
  | 序列 | 动作序列讲不通 | 抓出过 wash 两处真错误 |
  | 可疑 | 零长度段 | pen_inbox 的零长度段正是「标注连按两次 K」造出来的（`README.md:151`） |

  而**四处文档**都声称这是同一份代码：
  ```
  src/label/serve.py:22-23   「保存前强制跑 validate.py。标注工具与离线校验共用同一份判据」
  src/label/validate.py:5-9  「一套代码，两处使用 …… 必须是同一份代码」
  src/label/README.md:92
  src/label/AGENTS.md:88     并标「风险：高 —— 判据分叉过一次，代价很大」
  ```
  `serve.py:48` 的 import 只取 `core.Segment` 与 `core.build_document`，**没有 import `validate`**。
  连排序键都已不同：`validate.py:135` 按 `(start_frame, end_frame)`，`serve.py:269` 只按 `start_frame`。
- **改成（分两步）**：
  - [ ] ① **先补两条硬错到在线版**：`污染`（多余字段，判据见 `validate.py:53-55` 的 `ALLOWED_SEGMENT_KEYS`）
        与 `引用`（subtask 必须在 `subtasks.json` 里）。这两条判据简单、无需重构，**先做**
  - [ ] ② `serve.review()` 改为 import `validate` 的检查函数。
        主要成本是把 `validate.check_family` 从「按族扫目录」重构成「按单份文档检查」
- **不影响**：已有的 320 份标注（离线 `validate.py` 覆盖完整，当前为零条）。
- **怎么验**：`python3 src/label/validate.py` 仍为零条；在线保存一份带 `metadata` 字段的文档应被拦下。

### 4.4 · `schemas/segments.json` 无任何代码校验

- [ ] **位置**：`src/common/schemas/segments.json`（全文）
- **现状**：`git grep jsonschema -- '*.py'` 只命中 `src/vqa/compose.py:157`，加载的是 **`item.json`**。
  `segments.json` 只在两处 docstring 被提到（`src/label/core.py:238`、`data/label/README.md:11`），
  **无任何代码加载它**。
  它的规则被**手写复制**到了别处：`additionalProperties: false` 的白名单 ≈ `validate.py:53-55`
  的 `ALLOWED_SEGMENT_KEYS`；`not/anyOf` 拒绝 `metadata/window_type/original_*` ≈ `validate.py` 的「污染」检查。
  **同一条判据存了两份，一份没人执行。**
  而 schema 本身还比 `validate.py` 严，以下**目前完全没有被检查**：
  - `id` 的 pattern `^.+@f\d{6}(-\d+)?$`
  - `subtask` 的 pattern `^[a-z0-9]+(_[a-z0-9]+)*$`
  - `source` 的必填字段（`video` `fps` `total_frames` `tool_version`）
- **与 README 的冲突**：`README.md:124-126` 明写「每个层间边界都有 schema + 校验器
  （`src/common/schemas/`）。**校验器一套代码，产出时与离线核验共用**」
  —— 对 `item.json` 成立，对 `segments.json` 不成立。
- **改成**：在 `validate.py` 里加一步 `jsonschema` 校验，把手写白名单换成从 schema 读取。
  现成模式可抄 `src/vqa/compose.py:157-165`（「没装 jsonschema 就跳过并打印」）。

### 4.5 · `blind.options_four` 改用 `plan.build_options`

- [ ] **位置**：`src/vqa/blind.py:113-116`（`options_four`）、`:118-142`（`options_cross`）
- **现状**：`options_four` 意在复现出题实际用的借用逻辑，但轮转盐与可借池构造都与
  `plan.build_options` 不同（详见 1.5）。后果：`--policy four` 名义上是
  「统一 4 选项（借得最少）」即当前出题方案，**但它测的不是当前出题方案实际生成的那批选项**。
- **改成**：`from plan import build_options` 后直接调用；
  或把 `options_cross` / `options_four` 明确标注为「历史策略，非当前实现」。
- **不影响**：`as_built` 走的是 `plan.json` 的产物，已发布的盲测主结论仍成立。
- **前置**：需要 2.3 的 `build_options` 单元测试先到位。

---

## §5 · 批次 5：涉及架构或公开 API 的决策

**依赖**：全部依赖 §0 的对应答复。

### 5.1 ⚠决策 · `requirements.txt` 不是软链接（依赖无，但要拍板怎么改）

- [ ] **位置**：`src/eval/requirements.txt`、`src/eval/envs/groupA.txt`、`src/eval/docs/environments.md:42`
- **现状（第一轮此处判断被推翻）**：
  ```
  $ git ls-files -s src/eval/requirements.txt src/eval/envs/groupA.txt
  100644 efa2c663... 0  src/eval/envs/groupA.txt
  100644 efa2c663... 0  src/eval/requirements.txt      ← 100644 = 普通文件，软链接会是 120000
  ```
  两者 blob 相同、当前逐字节一致，但**都是普通文件**。
  而 `environments.md:42` 写「`requirements.txt` 是指向 `envs/groupA.txt` 的软链接」—— **文档是错的**。
  谁在读：`setup_env.sh:40` 读 `envs/groupA.txt`，`tools/setup_envs.sh:41` 读 `envs/${env_name}.txt`。
  **没有任何脚本读 `requirements.txt`**（`git grep requirements.txt` 只命中 `environments.md:42`）。
- **为什么值得修**：问题是**潜伏的** —— 任何一方被单独编辑，两份就静默分叉，
  而文档会让人以为不可能分叉。「以为是链接其实是副本」是最容易漂的形态。
- **改成（二选一）**：
  1. 真的换成符号链接（`git update-index` 后 mode 变 120000；注意 Windows 检出行为）
  2. **（推荐）** 删掉 `requirements.txt`，改 `environments.md:42` 为
     「依赖清单只有 `envs/*.txt` 一处」。没有任何脚本依赖它，
     而 pip 的默认约定不适用于这个多环境项目
- **怎么验**：`git ls-files -s src/eval/requirements.txt`

### 5.2 ⚠决策 · `extract.py` / `extract_llm.py` 686 行完全未接线（依赖 D-4 D-6）

- [ ] **位置**：`src/eval/robochrono/extract.py`（416 行）、`extract_llm.py`（270 行）
- **现状**：全仓库对 `extract_trajectory` / `extract_intervals` / `LlmExtractor` 的引用
  只有定义处与 `extract_llm.py:39` 对 `extract` 的内部 import。
  **无 CLI 注册、无配置项、无字符串动态引用、无 shell、无文档、无测试。**
  生产解析在别处：`time_eqa.py:110` `parse_multi_interval_text`、`:54` `parse_interval_text`；
  `trajectory.py:472` `parse_model_answer`、`:444` `coerce_point_list`；选择题走 `parsing.py`。
  `extract.py:22-25` 宣称的 L1/L2/L3 分层与「都失败 → BC-15 计 0 分」在 `engine.py` 里没有对应钩子。
- **Git 历史**：两文件都在 `b1ec6f8`（「从旧仓库迁入 ⑥」）**同一次提交**随迁入进来，
  与任务实现同批。所以不是「本仓库写了一半」，而是**旧仓库里就已经是未接线状态**。
- **改成（三选一，需拍板）**：
  1. **接入 engine** —— `extract.py` 的设计文档质量很高、判据明确，
     且它解决的是实测存在的问题（RynnBrain time 漏答 48%、SenseNova 轨迹 2D 95% 用归一化坐标）
  2. **移到 `experiments/` 或独立分支**
  3. **删代码，但把 docstring 里的实验数据搬进 `docs/`** —— 约束解码三轮实验的数字、
     「假阳性比漏掉更糟」「只抠不验的正则看起来救回 70%，加校验后真实可用只有 20%」这些教训
- **⚠ 不要直接删** —— 误删会丢失有实证支撑的设计结论。

### 5.3 ⚠决策 · `export` 子命令：docstring 宣传了，parser 没有（依赖 D-4 D-5）

- [ ] **位置**：`src/eval/robochrono/cli.py:7`、`src/eval/robochrono/store.py:108-124`
- **现状**：`cli.py:7` 的模块 docstring 第三行写 `python -m robochrono export --results-dir <dir>`，
  而 `main()`（`:312-393`）注册的子命令是
  list / run / plan / estimate / preflight / matrix / dispatch / report / pack
  —— **没有 export**。照 docstring 敲会得到 argparse 的 `invalid choice`。
  `ResultStore.export()`（`store.py:108`）零内部调用。
- **既不是「已被替代」也不是「未完成」**：`export()` 的功能（合并原始 item 字段、
  产出 v1 同构 JSON）在 `report` / `pack` 里**没有等价物**；
  而 `tasks/base.py:100-108` 的 BC-04 说明明确把 `export()` 设计成那个「导出兼容格式时再合并回去」的出口。
  **它是已实现但未接线的功能，且 A4 新旧对照很可能需要它。**
- **⚠ 不能判「安全删除」**：`robochrono` 是分发给同事在自有算力上跑的包，`ResultStore` 是公开类。
- **改成（二选一）**：
  1. **（推荐）** 接回 CLI —— `sub.add_parser("export")` + 一个 `cmd_export`，约 30 行。
     docstring 已经承诺了它，删承诺比兑现承诺更亏
  2. 删掉 `cli.py:7` 那一行，并在 `store.export()` 上加 docstring
     「库 API，无 CLI 入口，供 A4 对照使用」

### 5.4 ⚠决策 · `providers.json` 的 32 帧显存矛盾（依赖 D-3）

- [ ] 见 1.9 最后一条与 §0 的 D-3。这条决定要不要在正式跑之前先做 `--limit-items` 显存试探。

### 5.5 · 补一份 `src/eval/tools/README.md`

- [ ] **现状**：`src/eval/tools/` 下 12 个脚本，其中 5 个只被 `docs/` 引用、5 个零引用。
  但它们**不是残留** —— `build_normalized.py` 被 `tasks/__init__.py:114,169` 的错误提示明确指名
  （「重建：`python tools/build_normalized.py`」），
  `bc10_impact.py` / `version_compare.py` / `proposal_eval.py` / `frame_alignment_probe.py` 等
  是产出 `src/eval/docs/` 里那些实测数据的探针。
- **为什么要补**：这类「一次性但结论进了文档」的工具，删掉会让文档里的数字失去来源。
  但现在没有任何地方记录「哪个脚本对应哪份文档」。
- **改成**：加 `src/eval/tools/README.md`，一行一个脚本，标注它产出了哪份文档的哪一节。

### 5.6 · 修 `tools/smoke_all.sh` 的三处失效路径

- [ ] **位置**：`src/eval/tools/smoke_all.sh:12,14,19,20,53`
- **现状**：`:12` `REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"`
  在本仓库解析为 **`src/`**。于是：
  | 行 | 解析成 | 实际 |
  | --- | --- | --- |
  | `:14` `${REPO}/eval/models/…` | `src/eval/models/` | ❌ 权重在仓库根的 `models/` |
  | `:19-20` `${REPO}/eval/datasets/QA/…` | `src/eval/datasets/` | ❌ 不存在 |
  | `:53` `cd "${REPO}"` 后 `"test/${script}"` | `src/test/…` | ❌ 冻结脚本在 `src/eval/upstream/` |
  只有 `:15` 的 `configs/config_smoke.json` 与 `:17` 的 `results/smoke` 恰好对。
- **改成**：三处路径改对，或明确标注为「旧仓库遗物，不再维护」。
  `:6` 说它是「阶段 0 的门禁」—— 若门禁已完成使命，**标注比修更省**。
- **怎么验**：`ls -d src/test src/eval/models src/eval/datasets`（三个都应报不存在）

---

## §6 · 明确不要删的历史资产

清理时**不要碰**以下内容。每一条都有具体理由。

| 资产 | 位置 | 为什么留 |
| --- | --- | --- |
| **冻结上游实现（评测）** | `src/eval/upstream/` 11 文件 ≈ 4,900 行 | `README.upstream` 明写「一个字符都没改」；`test_request_equivalence.py:31` 与 `test_parsing_equivalence.py` 逐字调用其函数作为判据。**改它等于改判据**。含零引用的 `stitch_understanding_multiview_clips.py` —— 它属于「原样搬来的那一批」，单独删会破坏「整批未改」这个性质 |
| **冻结上游实现（标注）** | `src/label/upstream/` 3 文件 ≈ 1,220 行 | 同上；`video_labeler_timestamp.py` 是 `serve.py` 的前身，`tool_version: "upstream-video_labeler_timestamp/unknown"` 还写在每一份 segments 的 `source` 里 |
| **replay fixture** | `src/eval/fixtures/` 共 3.8 MB | 录下来的真实模型输出，**不可再生**（模型权重会变、服务端会变）。四套自足回归全靠它 |
| **迁移前标注语料** | `data/label/*/segments.before_*/`、`stack_cubes/segments.polluted/` | 每一份对应一次有据可查的数据修改（D-04 ID 化、P-05 加相对位置、序列检查修错、P-03 污染还原）。`segments.polluted` 尤其 —— 它是 P-03 的原始证据，`schemas/segments.json` 的 `not/anyOf` 规则就是照着它写的 |
| **corrections / provenance** | `data/label/wash/corrections.json`、各 segments 的 `source._restored` | 记录「哪一段被谁按什么依据改过」。`README.md:157-158` 那套纪律的可追溯性全在这里 |
| **不可再生的 LLM 输出** | `data/llm_cache/v1-vendor/`（16 份，含 8 份反建）、`v2/`（7 份）、`v3/`（7 份） | `llm_cache/README.md:9`「没有 v2 / v3 这两次失败，就没有那个结论」—— D-37/D-38「只用真实标签」这个核心设计决定的**全部实证依据**。重跑 LLM 会得到不同结果 |
| **盲测原始输出** | `build/blind*.json` 共 6.3 MB | 不可再生。至少 `blind.json` / `blind_final2.json` / `blind_ship.json` 极可能是最终方案的直接证据。见 3.3 —— **归档 + 写索引，不要只留 Git 历史** |
| **一次性迁移脚本** | `src/migrate/` 6 个脚本 | `README.md:6-7` 明确保留理由：`data/label/` 与 `data/llm_cache/` 不可再生，这些脚本是它们的**来源记录**。`fetch_raw.py` / `normalize_source.py` 还被根 `README.md:75-76` 列为「拿到数据」的正式步骤，**是活的入口** |
| **`docs/disclosures.md`** | 全文 | 全仓库唯一与代码一致的 time 指标表述来源，且明写「只增不删」的纪律。1.9 的修法应当是让别处向它对齐 |

---

## §7 · 第一轮审计中被推翻 / 降级的结论

清理时**不要按第一轮的这几条动手**：

1. **【推翻】`src/eval/requirements.txt` 是软链接** —— mode 是 `100644`，普通文件。见 5.1。
2. **【推翻】`mediaindex.py` 有未使用 import** —— AST 扫描零命中，逐条核对其 7 条 import 全部有使用。误报。
3. **【推翻】未使用 binding 约 17 个** —— 实测 **13** 个，清单见 1.2。
4. **【降级】「`run_all.py` 会让跳过混进通过数」** —— 不成立。`run_all.py:25-26,34` 明确单列跳过。
   真实问题降级为「整体退出码不区分『验过』与『没验』」，见 2.4。
5. **【降级】「文档写 5 通过 1 跳过是错的」** —— 在有 v1 数据的机器上**成立**。
   真实问题是 README 未声明环境前提。
6. **【重新归因】F-001 的「pool 吞失败」** —— `pool.run_pool` 尽职返回了 `{"done","errors"}`。
   吞掉它的是编排层 `_run_local_pool`。且「熔断不影响退出码」是**两条路径共有**的全局设计，
   不是多卡特有。见 4.1。
7. **【不是残留】** `src/eval/upstream/` `src/label/upstream/` `src/migrate/` `src/eval/tools/`
   —— 理由见 §6 与 5.5。
8. **【已排除动态引用误判】** `cli.py` 的 9 个 `cmd_*`（经 `set_defaults(func=...)` +
   `args.func(args)` 分派）、`serve.py` 的 `do_GET`/`do_POST`/`log_message`
   （`SimpleHTTPRequestHandler` 框架回调）、`tasks.build` 按字符串分派到
   `choice.SPECS`/`time_eqa`/`trajectory`。**这些不是死代码。**
