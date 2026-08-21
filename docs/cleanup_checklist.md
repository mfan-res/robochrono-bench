# 重构残留物清理清单

> **第三版**，2026-08-21 复核。基线 `HEAD = 8611e28`（上一版 `d6198a8`）。
> 本版把**首轮全量评测（SenseNova-2B，42 run / 9,037 unit）暴露的问题**
> 与静态审计的发现合并了 —— 有几条互相印证，见 §A。
>
> 行号按当前代码更新。仍建议用每条给的**搜索锚点**定位。

## 本次复核发生了什么

上一版之后有一个提交（`8611e28`），修了三处 time 相关缺陷并跑完首轮全量。

| 变化 | 条目 |
| --- | --- |
| 🟡 **修了一半** | **1.1** —— `matrix_run._prepare` 现在读 `frames_by_run` 了 ✅，但**多卡 pool 路径拿不到它** ❌。而且 meta 现在会记 32 帧、实际跑 8 帧 —— **从「没接线」变成「元数据与事实不符」，更难发现** |
| ✅ 已修 | time 提示词补视频时长（`time_eqa.py:_video_seconds` + `pack.py` 写 `input.video_seconds`，实测 200/200 已带上）；多题答案按序号回落匹配（gift_inhand 90/90 全废已解） |
| 🔺 **实证升级** | **5.2**（`extract.py` 未接线）、**4.2**（pool 重建 runtime）—— 全量跑的两处失败正好命中这两条，见 §A |
| 🆕 新增 | **A-1 … A-5** 五条，全部来自这次全量跑 |
| 🟢 无变化 | 1.2–1.18、2.x、3.x、4.1 4.3–4.7、5.1 5.3–5.7 仍然成立 |

## 一句话回答「模型矩阵要起吗」

**可以起了。** A-1（time 多卡路径现在也走 32 帧）与 A-2（`--runs` + 备份而非删除）
都已于 2026-08-21 修完，见 §A。

A-3（退化基线闸门）也已修完 —— 这一轮的报表会自己标出
「低于随机 / 低于最蠢的策略」的格子，不必再靠人工诊断。

## 怎么用这份清单

- 每条统一四段：**位置 → 现状 → 改成 → 怎么验**。`[ ]` 打勾即可。
- 批次内互不依赖，批次之间的依赖写在各批次开头。
- 标 **⚠决策** 的先去 §0 拿答复，别直接动手。
- 「怎么验」的命令都是只读的。

## 优先级速查

| 严重度 | 条目 |
| --- | --- |
| **P0 · 起下一轮之前必须做** | ~~A-1 A-2~~ ✅ 已修，run2 已跑完（三个模型齐了） |
| **P1 · 下一轮之前值得做** | ~~A-3b A-7~~ ✅ 已修 —— **time 的三份旧结果作废，需重跑**、**5.2**（`extract.py` 接入，实证已支持） |
| 中 | 4.1 4.2 4.3 4.4 4.5 4.6 4.7、2.1 2.4、3.1、5.3、1.17、~~A-3 A-4~~ ✅、**A-5 A-6 A-8** |
| 低 | 批次 1 其余、3.2 3.3、5.1 5.4 5.5 5.6 5.7 |

---

## §0 · 动手之前要拿到的答复（⚠决策）

- [x] ~~**D-1 · 并发会话**~~ —— 已落地为 8 个提交，工作区现在是干净的。本条关闭。

- [x] ~~**D-2 · `frames_by_run` 与 `frame_variants` 谁优先？**~~ —— `8611e28` 已定：
      `_prepare` 先套 `frames_by_run`，再让 `_apply_frames`（`frame_variants`）覆盖它。
      **判断是对的，但只落到了串行路径 → 见 A-1。**

- [ ] **D-3 · 32 帧到底实测过没有？** 三方冲突：
      `providers.json` `_note` 说「实测 32 帧 12.55G→6.31G，160 帧可跑」，
      同文件 `_frames_by_run_note` 说「⚠ 32 帧的显存未在 24 GB 卡上实测过」，
      commit `d88505a` 说「32 帧实测不 OOM」。

- [ ] **D-4 · `robochrono` 有没有仓库外的 import 使用者？**
      影响 `ResultStore.export()`（5.3）、`extract.py` 全家（5.2）、`tasks.base.Task`（1.3）、
      **以及本轮新增的 `tasks.load_run_items`（4.6）**。

- [ ] **D-5 · `python -m robochrono export` 对外承诺过没有？**（决定 5.3）

- [ ] **D-6 · `extract.py` 是已放弃的方案，还是排期中的下一步？**（决定 5.2）
      🔺 **首轮全量给了答案的一半**：它 docstring 里点名的失败模式（模型输出归一化坐标）
      这次真的发生了，且花了两小时人工诊断。问题已从「要不要」变成
      「**整套接入，还是先只接量级校验那一小块**」。

- [ ] **D-7 · 完整 v1 数据（61 GB）能不能进 CI？**（决定 2.4）

- [ ] **D-8 · `build/blind_*.json` 的 10 份变体，哪几份是论文/报告要引用的证据？**（决定 3.3）

- [ ] **D-9 · `REFACTOR_PLAN.md` 在哪？** 三处引用它，全仓库不存在：
      `src/eval/docs/dataset_structure.md:312`、`src/eval/robochrono/parsing.py:17`、
      `src/eval/configs/config_smoke.json:8`。

- [ ] **D-10 · `src/vqa/recipes/` 是已废弃的设计，还是还没建？**（决定 1.7）

- [x] ~~**D-15 · `time` 的分批策略选哪个？**~~ —— 已选**方案②（全局一题一问）**，
      2026-08-21 落地。依据：口径统一优先于省 13% 机时；
      按模型降批量会让强弱模型做的不是同一道题。本条关闭。

- [ ] 🆕 **D-16 · airpods 的 time 怎么处理？**（决定 A-8）
      1.8 秒的段在 `tIoU@0.5` 下无解。单独标注 / 移出 / 换指标，三选一。

- [x] ~~**D-13 · A-4 的三个选项选哪个？**~~ —— 已选①（只写进披露，不改数据），
      2026-08-21 落地为 `docs/disclosures.md` 第 1c 条。本条关闭。

- [ ] 🆕 **D-14 · Cosmos3-Edge-2B 怎么办？** 缺推理库。
      装（groupB 环境已有位置，`configs/environments.json` 里映射好了），
      还是先从 `plan.json` 的 models 里移出、在 `_why_models` 里记一句？
      **移出比留着 FAIL 好** —— 现在它是 preflight 里唯一的红色，会掩盖真的新问题。

- [ ] 🆕 **D-11 · `data/vqa/` 层的可再生中间件，哪些该进 git？**（决定 1.17）
      本轮 `.gitignore` 把 `build/frames.json`（5.6 MB）排除了，理由是「可再生」；
      但**同一条命令产出的 `build/frames_desc.npy`（21 MB 二进制）却进了 git**。
      判据是什么 —— 体积？格式？可 review 性？

- [ ] 🆕 **D-12 · `_run_local_pool` 遇到部分 spec 准备失败，应该 fail-fast 还是跑完能跑的？**（决定 4.1）
      本轮改成了 fail-fast（一个失败则 42 个全不跑），而 `_run_serial` 是逐个继续。
      两条路径行为相反，且都没写进文档。

---

## §A · 首轮全量评测暴露的问题（2026-08-21 新增）

首轮全量：SenseNova-SI-1.1-InternVL3-2B，42 run / 9,037 unit，8 卡，errors=0。
七个题型只有 image_in_video（44.7%，+16.2σ）与 understanding（40.9%，+13.7σ）测出信号，
time 归零。**这几条是从那次跑里落回代码的，不是模型结论。**

### A-1 ✅ **已修**（2026-08-21）· 多卡 pool 路径拿不到 `frames_by_run`

- [x] `pool.py:47` —— `frames_fps: float | None` 换成 `frames: dict | None` + `align_fps: bool`
- [x] `pool.py:113-123` —— 改为整份档位一起设，顺手消掉「改了不还原、漏给下一个 unit」
- [x] `matrix_run.py:213` —— `_, _, store, ...` 改为 `_, runtime, store, ...`，接住 runtime
- [x] `matrix_run.py:232-238` —— `WorkItem` 带上 `frames=runtime["frames"]` 与 `align_fps`

**验证**（不起 GPU，复刻 `vlm_api.py:1141-1153` 的取值逻辑逐 run 核对）：

```
run             WorkItem.frames                    worker 实际会用的帧数
image_in_video  uniform:8  (+派生键)               8 → 8
left_right      uniform:8                          8 → 8
planning        uniform:8                          8 → 8
planning_2      uniform:8                          8 → 8
step_order      uniform:8                          8 → 8
time            {'mode':'uniform','value':32}      8 → 32   ← 修好了
understanding   uniform:8                          8 → 8
```

片段类题维持 8 帧（本就够，D-52），只有 time 提到 32 —— 与 `_frames_by_run_note` 的意图一致。

**遗留**：`_meta` 记录的 `frames` 与 worker 实际用的现在一致了，
但两者的**字典形状**仍不统一 —— 见 A-6。

### A-2 ✅ **已修**（2026-08-21）· `matrix` 缺 `--runs`，`--overwrite` 无差别删除

- [x] **`--runs` 过滤** —— `matrix.expand()` 加 `only_runs` 参数；`cli._expand` 透传；
      `matrix` 与 `plan` / `estimate` / `preflight` 四个子命令都加上
      （后三个共用同一个注册循环，一处改四处受益）。
      拼错报错而不是筛出空集 —— `--runs tiem` 若静默跑空，表现是「矩阵为空」，
      看起来像数据没到位而不是打字错了。
- [x] **`--overwrite` 改为挪走而不是删掉** —— 新增 `ResultStore.displace()`，
      把 `<run>.jsonl` 改名为 `<run>.jsonl.bak` 并返回挪走的行数，只保留一代。
      `matrix_run._prepare` 与 `engine.run` **两处 unlink 都换掉了**（口径统一）。
      后缀选 `.jsonl.bak` 而不是 `.bak.jsonl`：`report.pack` 按 `*.jsonl` 收集文件，
      备份不该被打包回传。

**验证**：

```
不加 --runs                 : 210 个 spec
--runs time                 :  30 个 spec（6 族 x 5 模型）
--runs time understanding   :  60 个 spec
--runs time --models Sense  :   6 个 spec        <- 与 --models 正交
--runs tiem                 : ValueError，并列出可选题型

displace(): 7 行 -> time.jsonl.bak，原文件消失、备份 7 行俱在
再来一次    : 1 行 -> 覆盖同一个 .bak（只保留一代）
空 store    : 返回 0，不报错
rglob("*.jsonl") 不命中 .bak   <- pack 不会带上备份
```

**以后只重跑一个题型**：
```bash
python -m robochrono --datasets-root ../../data/vqa/eval --results-dir <dir> \
    matrix --only local --models <模型> --runs time --gpus 8 --overwrite
```
`--overwrite` 现在只影响 `--runs` 选中的那些 run，而且是挪走不是删。

**遗留**：`plan` / `estimate` / `preflight` 仍然没有 `--models`（只有 `matrix` 有）——
**这是本次之前就有的不对称**，没有一并改。要的话是同一处注册循环加一行。

### A-3 ✅ **已修**（2026-08-21）· 退化基线闸门

- [x] `tasks/__init__.py` —— 新增 `RANDOM_BASELINE` / `DEGENERATE_FLOOR` / `floor_breach()`。
      判据直接取自同文件里 `PRIMARY_METRIC` 的那张退化基线表
      （**数字上一轮就量过了，只是从没进过代码**）：
      | run | 指标 | 下限 | 下限是谁拿到的 |
      | --- | --- | --- | --- |
      | 六个选择题 | `accuracy` | 0.25 | 随机猜（④ 固定四选一） |
      | `time` | `mean_tIoU` | 0.13 | 「整段视频都报」 |
      | 任意 run | 主指标 | `== 0` | 与三个退化策略无区分度 |
      选择题的下限从 `choice.SPECS` 自动生成 —— 以后加题型不用改这里。
      用**名义**基线 25% 而非等效值（gift_inhand / pen_inbox 的 understanding 等效 33%）：
      等效值只会让门槛更高，名义值不会误报。
      trajectory 不设下限（退化基线没量过，且已搁置）。
- [x] `report.collect` 带上 `floor` 字段；`to_markdown` 给格子加 `⚠`，并在表下**逐条列出**
      —— 只留一个符号不够，「为什么低」和「低了」一样重要。
- [x] `matrix_run._write_summary` 与 `cli.cmd_run` **跑的时候就打印**，
      不等到出报表：一轮矩阵几小时，早两小时知道就能早两小时去查。

**验证**（拿首轮全量的真实分数造一份结果跑 `report`）：

```
| model                         | time | understanding | left_right | image_in_video | planning | planning_2 | step_order |
| SenseNova-SI-1.1-InternVL3-2B |  0⚠  |     0.409     |   0.262    |     0.447      | 0.241⚠   |    0.28    |   0.262*   |

⚠ 2 个格子低于退化基线：
| … | wash | planning | 低于随机猜（accuracy 0.241 < 0.25） |
| … | wash | time     | 低于「整段视频都报」（mean_tIoU 0 < 0.13） |
```

**这一轮人工挖了两小时才发现的两件事，现在是报表上的两行。**
另确认 `answered == 0` 时不报（全没答上是另一类问题，由 `answered` /
`parse_failure_rate` 各自反映，混进来会让这个记号失去意义）。

**回归**：`run_all.py` 通过 5 跳过 1 未通过 0，与改动前一致。


### A-3b ✅ **已修**（2026-08-21）· 100% 报错的 run 在报表上没有任何记号

- [x] `tasks/__init__.py` 新增 `execution_fault(summary)` —— **与 `floor_breach` 分开，不合并**：
      | 函数 | 说的是 | 该去查 |
      | --- | --- | --- |
      | `execution_fault` | 这个 run 根本没跑成 | **框架**（提示词 / 批量 / 解析器 / 媒体） |
      | `floor_breach` | 分数低到不看视频也能拿到 | **模型或题目** |
      三条判据：`answered == 0` / `errors ≥ 10%` / `parse_failure ≥ 50%`。
      **阈值不是拍脑袋**：两次全量各 9,037 次调用，Qwen3-VL 与 SenseNova 的
      errors 都是 **0**，RynnBrain 的 234 个错误全部集中在 time ——
      正常的 run 一个错都没有，10% 远在噪声之外。
      第三条（调用成功但解析不出来）单独一条，因为它指向的是提示词与解析器，
      不是请求与环境。
- [x] `report.to_markdown` 用 **`✗`** 标（与「低于基线」的 `⚠` 区分），
      并把执行故障表**排在退化基线表之前** —— 它的意思是「下面那个数不是分数」，
      读表的人得先知道哪些格子根本不该拿来比。
- [x] `matrix_run` 与 `cli.cmd_run` 跑的时候就打印 `✗`。

**验证**（拿 run2 的真实结果重新出报表）：

```
| RynnBrain-2B | 0✗ | 0.35 | ...        ← 与 Qwen 的 0.005⚠ 一眼分得开

✗ 6 个 run 没有正常执行完：
| RynnBrain-2B | airpods     | time | 0/200   | 200 | 一题都没答上          |
| RynnBrain-2B | gift_inhand | time | 0/90    |  90 | 一题都没答上          |
| RynnBrain-2B | pen_inbox   | time | 0/149   | 149 | 一题都没答上          |
| RynnBrain-2B | stack_cubes | time | 41/200  | 159 | 80% 的调用失败        |
| RynnBrain-2B | tea         | time | 0/234   | 234 | 一题都没答上          |
| RynnBrain-2B | wash        | time | 9/517   | 508 | 98% 的调用失败        |
```

**RynnBrain 那 1,340/1,390 的失败，现在是报表上的 6 行**；
另外 78 个 run 一个都没被误报。**回归 5 通过 1 跳过**，与改动前一致。

### A-4 ✅ **已修**（2026-08-21）· `planning` 的选项里必然包含「当前动作」

> **上一版判断有误，已更正**：曾写「会改 `build/plan.json` 与 `data/vqa/`，
> 必须在两轮评测之间做」—— **错了**。`plan.json` 里 `provenance.subtask` 就是
> 当前动作、`next_subtask` 是答案，**从现有产物直接算得出，不改数据、不挡评测**。

- [x] `docs/disclosures.md` 新增 **第 1c 条**，两张表都写进去了：
      各族「当前动作在选项中」的比例（由 `build/plan.json` 算得），
      以及首轮全量里模型的实际反应。
- [x] 顺带查了 **`planning_2` 结构完全相同**（合计 54.9%，与 planning 的 56.0% 相当），
      但首轮拿到 28.0%（+2.4σ）—— **同一个坑，这个模型在那道题上没踩下去**。
      原因未查清，已如实写进披露而不是略过。
- [x] 第 1 条末尾加了指向 1c 的一句：首轮数据从反方向回答了它的担忧
      （模型没用捷径，而是栽在更前面的一步）。
- [x] 页脚「怎么读这份清单」跟着改了 —— 原文说「前五条 / 后四条」，
      而实际已是 1、1b、1c、2–5 / 6–10 / 11。顺手把「第 10 条」的历史称呼
      也标注清楚（现为第 11 条，原文按「只增不删」留着）。

**结论没有改代码**：三个候选修法都更糟（三动作族凑不满 / 「当前动作缺席」
本身成新捷径 / 破坏统一四选一），所以这一条是**记录**，不是修复。

**留给 ⑦ 解读**：跨族比较 planning 时按这一列分层，比直接比总分有意义。
真要做成脚本的话是纯读 `build/plan.json` + 结果 jsonl，不动数据 —— 但不是必需。

### A-5 🟢 · 进 git 的运行日志有 94.5% 是同一行 transformers 噪声

- [ ] **位置**：`results/run1_sensenova_2b.log`（9,562 行 / 667 KB）、
      本地 InternVL adapter 的 `generate` 调用（`src/eval/robochrono/vlm_api.py`）
> **run2 更正（2026-08-21）**：这句噪声是 **InternVL 特有的**，Qwen3-VL / RynnBrain 都不发 —— run2 两个模型的日志一共只有 1,080 行、零噪声。
> 所以这条只影响 SenseNova 系，范围比原先写的小；但 SenseNova 还要重跑 time（见下），仍值得修。

- **现状**：
  ```
  总行 9562   其中 "Setting `pad_token_id` to `eos_token_id`:151645" 9037 行 = 94.5%
  ```
  正好一行一个 unit。真正有信息的只有 525 行。
  而 `.gitignore` **对 `results/` 没有任何规则** —— 下一轮三个模型进来会再加几倍。
- **改成**（二选一）：
  - [ ] ① 在本地 adapter 里显式传 `pad_token_id=tokenizer.eos_token_id`（或设 `generation_config.pad_token_id`），
        警告消失，日志回到几百行 —— **推荐**，这是警告本来就在提示的做法
  - [ ] ② 定 `results/` 的进 git 策略：日志过滤后再进，或只进 `*.summary.json` + 逐题 tar.gz
- **顺带**：`results/README.md` 已经写了这批结果是什么，很好 —— 补一句「日志是过滤过的/未过滤的」即可。

### A-6 🆕 · `frames_by_run` 覆盖出来的 frames 字典缺两个派生键，与 runtime 顶层说法不一致

- [ ] **位置**：`src/eval/robochrono/matrix_run.py:160-163`、`src/eval/robochrono/cli.py:101-103`
      （两处**同样的写法**）；对照 `src/eval/robochrono/vlm_api.py:150-157` 的 `resolve_frames`
- **搜索锚点**：`runtime["frames"] = dict(by_run)`
- **现状**：`resolve_frames` 产出的 frames 有四个键 ——
      `mode` / `value` / `video_sample_fps` / `num_segments`，后两个是**派生的**，
      注释写着「下游 adapter 仍读这两个字段」。
      而 `frames_by_run` 的覆盖是 `runtime["frames"] = dict(by_run)`，
      `by_run` 是配置里的原始片段，**只有 `mode` 和 `value`**：
      ```
      其它 run   {'mode':'uniform','value':8,'video_sample_fps':0.0,'num_segments':8}
      time       {'mode':'uniform','value':32}                    ← 少两个键
      ```
      同时 `runtime` 顶层的 `num_segments` / `video_sample_fps`（`vlm_api.py:265-266`
      在 `runtime_config` 时写入）**没有跟着更新，仍是 8**。
      于是一份 runtime 里有两个说法：`frames["value"]=32` 与 `runtime["num_segments"]=8`。
- **今天不影响帧数**：`vlm_api.py:1151` 的 uniform 分支是
      `int(frames.get("value") or runtime["num_segments"])` —— **先读 `frames["value"]`**，
      所以拿到的是 32。已在 A-1 的验证里逐 run 确认过。
- **但它是个上了膛的坑**：
      1. 谁把那行改成读 `runtime["num_segments"]`（看起来更"正规"），time 就**静默回到 8 帧**
      2. 写进结果 meta 的 frames 字典，time 与其它六个 run 形状不同，事后比对会绊一下
      3. `cli.py` 与 `matrix_run.py` 各写了一遍同样的覆盖 —— 又是「两处各写各的」
- **改成**：把覆盖收进 `vlm_api.py`，紧挨着 `resolve_frames` 放一个
      `apply_frames_override(runtime, spec) -> dict`，它负责补齐派生键**并同步顶层的
      `num_segments` / `video_sample_fps`**；`cli.cmd_run` 与 `matrix_run._prepare` 都调它。
      约 12 行，净减少一处重复。
- **怎么验**：
  ```bash
  git grep -n 'runtime\["frames"\] = dict(by_run)' -- src/    # 改后应为 0
  ```
  再跑 A-1 那段核对脚本，七个 run 的 frames 字典应当形状一致。
- **风险**：低。**置信度**：高（已实测两处字典形状不同）。**预估**：1 小时。

### A-7 ✅ **已修**（2026-08-21，方案②）· `time` 整组提问，RynnBrain-2B 因此一题都没被测到

- [x] `time_eqa.TimeEqaTask.units()` 改为 `one_item_per_unit(items)`；
      新增 `group_by_video: bool = False` 保留 v1 分组，**只给回放用**
      —— `fixtures/baseline/time.json` 是按视频录的，replay 表的键得对得上。
- [x] `build_prompt` **一个字没改**。它在 1 道题时本来就正常
      （补测里 `build_prompt(items[:1])` 输出 172 字符），
      不动它就少一处需要声明的差异。

**为什么选方案②而不是①**（按模型降批量）：分组让模型知道
「这段视频恰好有 N 个动作」、可以用排除法（披露第 5 条）。
强模型一次十三题拿到这个便利、弱模型一题一问拿不到，**两者做的不是同一道题**
—— 而 time 恰恰是刚开始出信号的那个题型。代价实测只有 +13%。

**真模型验证**（RynnBrain-2B，每族 8 题）：

```
                改之前              改之后
airpods         0/200 答上来        8/8 答上来   tIoU@0.5 0.125
stack_cubes    41/200              8/8          tIoU@0.5 0.375
pen_inbox       0/149              5/8          tIoU@0.5 0.25
tea             0/234              8/8          tIoU@0.5 0.0
wash            9/517              8/8          tIoU@0.5 0.125
gift_inhand     0/90               5/8
合计答上来率     3.6%               87.5%
```

**还剩 12% 空输出**（gift_inhand 3/8、pen_inbox 3/8，都是同一个
`Cannot parse multi-answer model output: ''`）。一题一问把 96% 降到 12%，
**但没有归零** —— 说明「产出几个答案」是主因，不是唯一因。剩下的那点
留给 5.2（`extract.py` 的量级/结构校验），或作为该模型自身的稳定性记录。
现在这 12% 会被 A-3b 的 `✗` 标出来，不会再伪装成分数。

**回归**：5 通过 1 跳过，与改动前一致 —— 包括两个 replay 测试
（它们照常按 video 分组回放 v1 录音）与 `test_request_equivalence`。

⚠ **`test_request_equivalence` 覆盖不到这次改动**：它把我们的
`unit.items` 原样喂给冻结实现再比 prompt，所以「怎么分组」这个决定
对它是隐形的（time 那行仍报 0 不一致）。**不是它坏了，是它不管这件事。**
补一条「units 切分不变」的断言值得做，已并入 2.3。

**遗留**：`--limit-groups` 对 time 的语义变了 —— 原本是「限制视频数」，
现在等同于 `--limit-items`。BC-05 当初拆这两个参数就是因为 v1 的
`--limit` 在 time 上限的是视频数。文档要跟着改（`cli.py` 的 help 文本）。

### A-8 🆕 · `time` 的分数由标注段长支配，跨族不可比

- [ ] **位置**：`docs/disclosures.md`（该补一条）；判据在 `tasks/__init__.py` 的
      `PRIMARY_METRIC` 注释里已有（`tIoU@0.5` 要求纯平移 `s ≤ D/3`）
- **run2 实测**（Qwen3-VL-8B，唯一真正做出定位的模型）：
  ```
  族             真值段 D   允许平移 D/3   实测平移中位   够不够   tIoU@0.5
  airpods          1.8s        0.6s         6.5s      ✗       0.5%
  wash             4.8s        1.6s         3.2s      ✗      31.5%
  pen_inbox        6.8s        2.2s         1.2s      ✓      73.8%
  gift_inhand      9.7s        3.2s         4.5s      ✗      11.1%
  stack_cubes     12.1s        4.0s         4.6s      ✗      21.5%
  tea             12.4s        4.1s         4.3s      ✗      46.2%
  ```
- **结论**：**模型的时间精度大致恒定（3–6.5 秒），而允许的误差随段长从
      0.6 秒变到 4.1 秒。** 中心命中率 40.6%–87.9%（airpods 除外 9.0%）——
      模型找对了大致位置，输在边界。
      **airpods 在当前指标下实质是无解题**：1.8 秒的段要求 0.6 秒精度，
      比标注本身的精度还高（`disclosures.md` 第 11 条：段边界是分段的交接点，
      不是动作的精确起始）。
- **与已有披露的关系**：第 4 条已经说过「片段时长与动作类型相关」，
      但那说的是**它泄答案**；这一条是它的另一面 —— **它还支配 time 的分数**。
- **改成（三选一，需拍板）**：① 只在披露里说明并按族分层读；
      ② airpods 的 time 单独标注或移出；③ 换成对段长不敏感的指标（如 center_inside）。
      **不建议动数据。**

---

## §1 · 批次 1：零风险机械清理

**依赖**：无。除 1.1 外都可立即做。

### 1.1 ✅🟡 **已被 A-1 取代** · `matrix` 读 `frames_by_run`

- [x] ~~串行 / API 路径~~ —— `8611e28` 已修：`matrix_run.py:159-163` 读 `frames_by_run`，
      `_run_serial:182` 用的就是这份 runtime。
- [ ] **多卡 pool 路径仍未修，且元数据开始说谎 → 见 §A 的 A-1。**
      本条不再单独跟踪，A-1 是它的当前形态。

### 1.2 · 删 18 条未使用 import（上一版是 13 条）

- [ ] 逐条删除（AST 扫描 + 人工核对）：
  ```
  src/eval/robochrono/cli.py:18                      import sys
  src/eval/robochrono/cli.py:24                      index_for_qa          🆕 本轮新增
  src/eval/robochrono/cli.py:24                      resolve_items         🆕 本轮新增
  src/eval/robochrono/cli.py:25                      load_items            🆕 本轮新增
  src/eval/robochrono/matrix_run.py:22               load_items
  src/eval/robochrono/normalize.py:45                Unit（保留 load_items）
  src/eval/robochrono/preflight.py:22                import subprocess
  src/eval/robochrono/preflight.py:29                load_items
  src/eval/robochrono/report.py:19                   load_items            🆕 本轮新增
  src/eval/robochrono/tasks/choice.py:16             from pathlib import Path
  src/eval/tests/test_normalized_equivalence.py:49   canonical_family
  src/eval/tests/test_request_equivalence.py:18      import json
  src/eval/tools/proposal_eval.py:22                 Any
  src/label/core.py:44                               field（保留 dataclass）
  src/migrate/fetch_raw.py:31                        import sys
  src/migrate/verify_bar_crop.py:43                  import sys
  src/migrate/verify_bar_crop.py:44                  Counter
  src/vqa/frames.py:53                               import sys           🆕 本轮新增
  src/vqa/tests/test_step_order_bound.py:34          import sys           🆕 本轮新增
  ```
- **为什么变多**：`3d2848a` 把 `cli` / `preflight` / `report` 的数据加载统一到
  `tasks.load_for_run`，三处旧的加载 import 就此悬空 —— **这是好改动留下的正常尾巴**，
  不是新引入的问题。
- **注意**：`src/eval/robochrono/mediaindex.py` **是干净的**，第一轮审计说它有未使用 import 是误报。
- **怎么验**：删完重跑本清单末尾 §8 的扫描脚本，应为 0 条。

### 1.3 · 删 5 个零引用符号 + 让 `Task` 真正被用起来

**本轮复核：五个符号全部仍是零引用，`Task` 仍未被用作类型标注。**

- [ ] `src/eval/robochrono/pool.py:142-147` —— 删 `_CHOICE_RUNS` **连同上面那两行注释**
      （注释写「现在只剩 preflight/report 在用」，实际零外部命中，注释本身是假的）
- [ ] `src/vqa/compose.py:55` —— 删 `ASSETS`
- [ ] `src/eval/robochrono/media_prep.py:251` —— 删 `ffmpeg_available()`
- [ ] `src/label/core.py:191` —— 删 `can_add()`
- [ ] `src/label/core.py:218` —— 删 `build_subtasks()`（先确认 `subtasks.json` 的生成路径）
- [ ] `src/eval/robochrono/tasks/base.py:43` —— `Task` Protocol **不要删**。
      把 `engine.py:54` `_run_unit(task: Any, ...)`、`engine.py:98` `run(task: Any, ...)`、
      `matrix_run.py` / `pool.py` 里的 `task: Any` 换成 `task: Task`。
- **怎么验**：`git grep -c '\b<符号名>\b' -- '*.py'`，删前应只有定义文件那 1 处。

### 1.4 · 删不可达代码

- [ ] `src/eval/tests/test_normalized_equivalence.py:157` —— `return 0` 之后的 `return 1`
- **本轮复核**：仍在。**搜索锚点**：`python3 tools/build_normalized.py   # 在旧仓库里跑`（其后两行）

### 1.5 · 修两处互指的「不可能分叉」假声明

**行号已更新**（`plan.py` 本轮长了 483 行）。

- [ ] `src/vqa/plan.py:311-312`（上一版 237-238）—— 「**这是选项构造的唯一实现** ——
      `blind.py` 直接导入它，两边不可能分叉」
- [ ] `src/vqa/blind.py:65-67` —— 「需要模拟别的策略时也从那里导入，**两边不可能分叉**」
- **事实（本轮复核 `blind.py` 一行未动，结论不变）**：`blind.py` 从不 import `plan.py`。
  `options_as_built` 读 `plan.json` 的产物（这条确实不会分叉），
  但 `options_cross` / `options_four` 是本地重写且**已经分叉**：
  - 轮转盐不同：`plan.py` 用 `md5(f"{item_id}|in")` / `|out`；`blind.py:132,137` 用 `md5(id)` / `md5(id+"|x")`
  - 可借池不同：`plan.py` 是去重排序后的集合；`blind.py:135` 是含重复、未排序的列表
- **改成**：先只改注释。真正收敛见 4.5。

### 1.6 ✅ **已修**（2026-08-21）· Markdown 坏链接 6 → 1

**本轮复核：这 6 条一条没修。** 另见 1.18 —— 本清单自己制造了 4 条假阳性。

- [ ] `docs/vqa/field_ownership.md:3` — `[v2 设计草案] (schema_v2_proposal.md)` → `item_schema_design.md`
- [ ] `docs/vqa/item_schema_design.md:4` — `[数据结构现状] (dataset_structure.md)` → `../../src/eval/docs/dataset_structure.md`
- [ ] `docs/vqa/item_schema_design.md:4` — `[重构方案] (dataset_refactor_plan.md)` → `../../src/eval/docs/dataset_refactor_plan.md`
- [ ] `docs/vqa/item_schema_design.md:128` — `[清单] (questions_for_data_team.md)` → `../questions_for_data_team.md`
- [ ] `docs/vqa/item_schema_design.md:344` — 同上 → `../questions_for_data_team.md`
- [ ] `src/eval/docs/dataset_structure.md:312` — `[...BC-16] (../../REFACTOR_PLAN.md)` → 见 **D-9**
- **结果**：6 条 → **1 条**。只剩 `src/eval/docs/dataset_structure.md:312` 的
  `REFACTOR_PLAN.md`，因为**目标文件全仓库不存在**，需先答 **D-9**。
  `docs/vqa/` 那四条统一指向 `src/eval/docs/`（canonical，其内部链接完整）。
- **建议顺带**：把 §8 的检查加进 `src/eval/tests/run_all.py`。

### 1.7 ✅ **已修**（2026-08-21）· 非链接式失效引用

**本轮复核：`src/vqa/README.md` 加了 14 行讲新步骤，但 `recipes/` 那两句原样留着。**

- [ ] `src/vqa/README.md:3` — `读 data/{raw,label,llm_cache} + recipes/` → 目录不存在
- [ ] `src/vqa/README.md:5` — `recipes/<family>.json 跟代码走 git` → 同上（见 **D-10**）
- [ ] `src/vqa/README.md:10` — 「唯一的非确定性是 LLM 干扰项，已冻结在 `data/llm_cache/`」
      → 干扰项自 D-38 起只用真实标签（`plan.py` 的 `main()` 里有注释说明），已失效
- [ ] `data/README.md:10` — `由上面三样 + src/vqa/recipes 确定性地产出` → 同上
- [ ] `src/eval/configs/plan.json` `_v2_note` — 「要跑 v1 时用 `configs/plan_v1.json`」→ 该文件不存在。
      **本轮该 `_v2_note` 已过期两处**：还写着「未建的任务：left_right / image_in_video / step_order」，
      而三者都已建成并进了 `runs`（同一个文件里 `_why_runs` 写的是对的）

### 1.8 ✅ **已修**（2026-08-21）· 根 README 的过期状态与数字

**本轮复核：`README.md` 一行未动，而代码走得更远了 —— 它比上一版更陈旧。**

- [ ] `README.md:37` — 「④ 出题 ⬜ **下一步**」→ **七个题型全部建成，10,178 道**
- [ ] `README.md:39` — 「⑥ 评测 …… 在旧仓库…… 待迁入」→ 已迁入且已跑通全 42 个组合
- [ ] `README.md:86` — 「vqa/  ④ 出题（待建）」→ 同上
- [ ] `README.md:206` — 「`src/vqa` ⬜ 待建，先定 A1 / A3 / A5」🆕 上一版漏列
- [ ] `README.md:207` — 「`src/eval` 🔶 在旧仓库，待迁入」🆕 上一版漏列
- [ ] `README.md:67` — 「7 个活跃族 …… tea2」→ **6 个**（`data/families.json` 记 tea2 为 `removed`）
- [ ] `README.md:68` — 「277 集 · 1,759 个标注段 · 43 个 subtask」
      → 实测 **249 份 segments 文件 · 1,390 段 · 40 个 subtask**（本轮重新量过，未变）
- **可以顺带补的真实数字**（本轮实测）：
  ```
  题量 10,178：left_right 2,640  understanding 1,390  time 1,390
               image_in_video 1,264  step_order 1,212  planning 1,141  planning_2 1,141
  ```
- **怎么验**：见 §8。

### 1.9 · 修 time 指标与抽帧状态的文档矛盾

**本轮复核：四处全部仍在。**（`docs/disclosures.md` 依旧是唯一与代码一致的那份。）

- [ ] `src/eval/RUNBOOK.md:19` — time 主指标写 `mean_tIoU` → 实际是 `tIoU@0.5`
      （`src/eval/robochrono/tasks/__init__.py` 的 `PRIMARY_METRIC` 单点决定）
- [ ] `src/eval/RUNBOOK.md:192-196` — 「抽帧策略尚未定案…… 不是评测配置」
      → **等 1.1 做完再改**，否则文档会比代码超前
- [ ] `src/vqa/blind.py:32-33` — 「time …… 容差目前没有定义」→ `disclosures.md` §10 已定 `tIoU@0.5`
- [ ] ⚠决策 `src/eval/configs/providers.json` — 同一文件内两句关于 32 帧的话冲突，见 **D-3**

### 1.10 ✅ **已修**（2026-08-21）· `data/label/README.md` 的旧 schema

- [ ] `data/label/README.md:5-15` 仍是迁移前的结构：
  | 文档写的 | 实际 |
  | --- | --- |
  | `segments/file-XXX.json` | `segments/file-XXX_segments.json` |
  | `categories.txt` | `categories.deprecated.txt` + `subtasks.json` |
  | `objects` / `main_verbs` / `narration` | `subtask`（ID 引用） |
- **权威来源**：`src/common/schemas/segments.json`

### 1.11 ✅ **已修**（2026-08-21）· `data/llm_cache/README.md` 的自相矛盾

- [ ] 目录块写「`v2/<族>.json`  本轮生成，**出题实际使用**」，与顶部横幅
      「三代全部退场，出题已不读这里」冲突；**且没有列 `v3/`**
- [ ] 「唯一还在读它们的是 `vocab.py`」→ 还有 `src/vqa/blind.py:209` 读 `v3/`（见 1.13）

### 1.12 · `--policy both` 默认展开全部 6 套策略

**本轮复核：`blind.py` 一行未动，全部仍在。**

- [ ] `src/vqa/blind.py:230` — `which = arg("--policy", "both")`，**`both` 是默认值**
- [ ] `src/vqa/blind.py:236` — `list(POLICIES) if which == "both"`，而 `POLICIES` 有 **6** 个键
- [ ] `src/vqa/blind.py:6` — 注释写「现状与策略③ 两套对照」，与实际行为不符
- **后果**：裸跑 `python3 src/vqa/blind.py --n 20` 发 6 倍 API 请求。
- **改成**：默认值改 `as_built`，`both` 改名 `all`（或删掉这个聚合值），同步改注释。

### 1.13 · `blind.py` 无条件加载已退场的 `llm_cache/v3`

- [ ] `src/vqa/blind.py:209-210` — 在 `main()` 顶部执行，**不看 `--policy`**。
      只有 `options_pool`（已否决）用得上。新增族或清理 `v3/` 就会让 `--policy as_built` 直接崩。
- **改成**：只在 `"pool" in policies` 时读。

### 1.14 ✅ **已修**（2026-08-21）· 统一「检查有几条」的说法

- [ ] 四处各说各的：`validate.py` docstring「六条」/ `SEVERITY` 实际 **8 个键** /
      `README.md:140`「六项核验」/ `README.md:145`「七条检查」/
      `src/label/README.md:21`「六条」与 `:107`「七条」/ `AGENTS.md:13`「六条」
- **已改成**「八类检查（六条硬错 ✗ + 两条待人判断 ⚠）」，改了五处。
- ⚠ **`serve.py:260` 特殊处理，没跟着改口径**：它写的是「六条检查的在线版」，
  而它实际只实现了三类（4.3）。跟着改成「八类」等于把假声明写得更死。
  改成了如实说明「只覆盖三类、拦不下哪五类、收敛计划见 4.3」——
  **这条文档修改反而把 4.3 那个代码缺口摆到了台面上。**

### 1.15 · 给已知失效的脚本加运行守卫

- [ ] `src/migrate/split_packed_episodes.py` — docstring 写「**当前不要运行**」「核心假设是错的」，
      但仍可直接跑起来并写数据。
- **改成**：`main()` 开头加 `raise SystemExit(...)`，或移入 `src/migrate/archive/`。**不要删**。

### 1.16 ✅ **已修**（2026-08-21）· 标记已失效的 TODO

- [ ] `src/migrate/README.md:9-27` — 「反建缓存的键格式未验证（2026-08-17 暂缓）」，
      前提已随 llm_cache 三代退场消失。
- **改成**：改写为「已失效」，**保留原文**（符合「判断被推翻时不删原文」的惯例）。

### 1.17 🆕 · `build/frames_desc.npy`（21 MB 二进制）进了 git，而同批产物 `frames.json` 被 ignore

- [ ] **位置**：`.gitignore:46-50`、`build/frames_desc.npy`、`src/vqa/frames.py`（`--write` 同时产出两者）
- **搜索锚点**：`# frames.json 是 19 MB 的像素描述子`
- **现状**：本轮 `.gitignore` 新增
  ```
  # frames.json 是 19 MB 的像素描述子，完全可由 data/source + index.json 重算。
  build/frames.json
  ```
  理由完全成立。但 `src/vqa/frames.py` 的 `--write` **同时**写 `build/frames.json`
  **和** `build/frames_desc.npy`，而后者**没有被 ignore，已经进了 git**：
  ```
  build/frames.json        5.6 MB   ← ignore 了
  build/frames_desc.npy   21.4 MB   ← 进了 git      ⟵ 同一条命令的产物，更大、更不可 review
  ```
  注释里的「19 MB」既不是 `frames.json` 的 5.6 MB，也不是 `.npy` 的 21.4 MB —— **两者很可能弄反了**。
- **改成**：按 **D-11** 的答复统一判据。若判据是「可再生的中间件不进 git」，
  则 `frames_desc.npy` 也该 ignore（`build/frames_floors.json` 1.4 KB 保留 —— 它是要被 review 的判据，
  `.gitignore` 的注释已经说清了这一点）。
  **注意**：从 git 移除大文件只是 `git rm --cached`，历史里那 21 MB 还在。
- **怎么验**：
  ```bash
  git ls-files -s build/ | awk '{print $4}'
  git cat-file -s $(git rev-parse HEAD:build/frames_desc.npy)
  grep -n 'frames' .gitignore
  ```

### 1.18 🆕 · 本清单自己让链接检查产生 4 条假阳性（我造成的）

- [ ] **位置**：本文件 §1.6 里那几条引用坏链接原文的行
- **现状**：上一版把 `[v2 设计草案] (schema_v2_proposal.md)` 这类**裸写**在清单里，
  于是链接检查器会去解析它们，报出 4 条根本不存在的坏链接（相对 `docs/` 解析）。
  基线因此从 6 条变成 10 条，没法用来判断 1.6 有没有做完。
- **改成**：本版已在 1.6 的链接示例里把 `]` 和 `(` 分开写成 `] (`。
  **只包反引号是没用的** —— 检查器读的是原始文本，不认 Markdown 的行内代码。
  §8 的 B 脚本也已改成先剥掉行内代码与围栏代码块，两道保险。
  **修完 1.6 后基线应为 0。**
- **怎么验**：跑 §8 的链接检查，输出里不应再出现 `docs/cleanup_checklist.md`。

---

## §2 · 批次 2：先补测试，再清理

**依赖**：2.2 需要 1.4 先做完。批次 3、4 依赖本批次。

### 2.1 · 把 `smoke_v2.py` 纳入测试套件

**本轮复核：仍未纳入。** `ls src/eval/tests/` 里 `smoke_v2.py` 在，但 `run_all.py` 的
glob 是 `test_*.py`，匹配不到；`git grep smoke_v2` 全仓库仍零命中。

- [ ] `src/eval/tests/run_all.py:15` — `TESTS = sorted(p for p in HERE.glob("test_*.py"))`
- **为什么要紧**：它的 docstring 写着「④ 出的题能不能被 ⑥ 完整处理…… 两边各写一套判据的话，
  迟早会对不上 —— **这个项目已经栽过三次**」。
  **本轮又新建了三个题型**（left_right / image_in_video / step_order），
  ④↔⑥ 的契约面比上一版大了近一倍，而这条唯一的自动检查仍然不跑。
- **改成**：重命名为 `test_smoke_v2.py`，或把 glob 扩成 `("test_*.py", "smoke_*.py")`。
- **先做**：`python3 src/eval/tests/smoke_v2.py; echo "exit=$?"` 确认它当前能过
  （新题型可能需要它跟着更新）。

### 2.2 · 补 `pool.run_pool` 的失败传播测试

- [ ] 新建 `src/eval/tests/test_pool_failure.py`
- **本轮复核**：`git grep -l pool -- src/eval/tests` 仍为空 —— 多 GPU 路径零测试覆盖。
- **测什么**（fake queue，不需要 GPU）：
  1. worker 启动失败 → `run_pool` 返回的 `errors >= 1`
  2. 某个 unit 抛异常 → `errors` 计数正确，`rows` 里有 `error_rows` 占位行
  3. 所有 worker 提前退出 → 不死锁，返回
  4. 🆕 部分 spec 准备失败 → `_run_local_pool` 的返回值符合 **D-12** 定下的语义
- **这条是 4.1 4.2 的前置。**
- 🆕 **首轮全量补的两个用例**（这两个 bug 都是靠人读结果才发现的，测试全程沉默）：
  5. **meta 与实际一致**：给一个 run 设 `frames_by_run`，断言 worker 实际用的
     `frames_used` 与 store meta 里的 `frames` 对得上（A-1 若不修，这条会红）
  6. **多题答案按序号回落**：喂一段 `{"id": "1", ...}` 形式的模型输出，
     断言能匹配回 `question_ids[0]`（`8611e28` 修的就是这个，gift_inhand 90/90 全废）

### 2.3 · 补出题核心函数的单元测试

- [ ] `src/vqa/plan.py` 的 `build_options` / `cooccurrence` / `clip_for` / `assert_no_leak`
- [ ] 🆕 `src/vqa/frames.py` 的画面距离判据（`frames_floors.json` 里那几个数）
      —— 本轮新增 238 行，是 `left_right` / `image_in_video` 干扰项质量的**唯一**兜底，
      目前只有 `src/vqa/tests/test_step_order_bound.py` 一个测试，覆盖的是 step_order 的盲基线上界
- **本轮进展**：新增了 `test_step_order_bound.py`（88 行）—— 好事，但它测的是
  「盲基线算得出来」这个论断，不是出题函数本身。
- **注意**：`plan.py` 的两次构建指纹自检只证明**确定性**，不证明**正确性**。

### 2.4 ⚠决策 · 落地测试分层（依赖 D-7）

**本轮复核：结构未变，仍是 5 通过 1 跳过（本机）/ 4 通过 2 跳过（普通 clone）。**

- [ ] `test_normalized_equivalence.py` 与 `test_request_equivalence.py` 在缺数据时以退出码 0 跳过
- [ ] `run_all.py:35` `SystemExit(1 if failed else 0)` —— 全部跳过时整体仍退出 0
- [ ] `run_all.py:25` 用子串 `"跳过" in out` 判定跳过，脆弱耦合
- **注意**：`run_all.py` **确实**把跳过单列一栏、没有混进通过数。真正的问题只是退出码。
- 🆕 **本轮新增了一个需要一起考虑的机制**：`test_request_equivalence.py` 加了 `DECLARED` 豁免表
  （当前豁免 `step_order/prompt`，理由是 v2 不拼宫格、改逐张发带标号的图）。
  判据从「输出相同」变成「**每处差异都被声明过**」。**这个设计是对的**（未声明的差异 = 改坏了），
  但它意味着 step_order 的 prompt 不再逐字节比 —— 分层方案里要把
  「DECLARED 表有几条」也当成一个要被 review 的数字。
- **改成**（四选一或组合）：
  1. **外部数据测试套件**（推荐，不依赖 D-7）：两套移进 `tests/external/` + `run_external.sh`；
     `run_all.py` 只跑自足的四套并**全部要求通过**，不再有跳过分支
  2. **CI 分层**：PR 跑 `run_all.py`（0 跳过 0 失败）；nightly 跑 external
  3. **最小媒体 fixture**：为抽样 unit 生成极小占位媒体（< 2 MB）。
     ⚠ 会改变「比对真实媒体」的语义，应作第三档
  4. **发布前门禁**：`run_all.py` 在有跳过时打印「⚠ 本次未覆盖 N 项，不能作为发布依据」

---

## §3 · 批次 3：删除已确认的旧实现

**依赖**：3.1 需要移植验证；其余依赖批次 1。

### 3.1 · `check_labels.py`：先移植，再删

**本轮复核：未动，全部仍成立。**

- [ ] **位置**：`src/migrate/check_labels.py:130`
- **现状**：读 `narration`，而当前数据只有 `subtask`。后果不是归零而是
  **`dup` 恒为 `len(segments) - 1`** —— 「重复动作」一栏对每一集报满，产生假发现。
- **⚠ 但它不是被完全取代**。它有 `validate.py` 没有的能力（`:45 duration()` / `:54 frame_count()`
  用 **ffprobe 探真实视频**）：
  1. 隐含 fps 与实际 fps 一致性　2. 帧号越界　3. 覆盖率 = 跨度 / 真实时长　4. gap / shortest / inverted

  `src/label/validate.py` **完全不探视频**，fps 来自 `data/raw/<族>/meta.json`。
  「元数据声称的 fps 与视频实际 fps 是否一致」**目前没有替代者**。
- **改成（按顺序）**：
  - [ ] ① ffprobe 类检查移植进 `validate.py --probe-video`（默认关，要跑 249 次 ffprobe）
  - [ ] ② 跑通后再删 `check_labels.py`
  - [ ] ③ 若暂不移植：文件头加 `⚠ 读 narration，当前数据已改为 subtask，输出不可信，勿用`
- **怎么验**：`python3 src/label/validate.py` 应仍为**零条**。

### 3.2 · 归档已退场的干扰项生成器

- [ ] `src/vqa/distract.py`（431 行）、`src/vqa/pool.py`（273 行）、
      `src/vqa/tests/test_verifier.py` → `archive/` 或 `experiments/`
- **前置**：`src/vqa/blind.py:48` `from distract import KEYS, MODEL, call_api` —— 移动前
      把这三个 API 工具提取到共用模块，或让 `blind.py` 自带。
- **数据不要删**：`data/llm_cache/` 三代是不可再生的 LLM 输出，见 §6。

### 3.3 ⚠决策 · 归档 10 份盲测输出（依赖 D-8）

- [ ] `build/` 下除 `blind.json` 外的 10 份变体，`git grep` 全部零引用（含 DEVLOG）：
  ```
  blind_after_plates  blind_cross  blind_final  blind_final2  blind_four
  blind_four_text  blind_n57  blind_ship  blind_three  blind_v4
  ```
- **改成**：移入 `build/archive/blind/` 并加 `INDEX.md` 说明每份对应 DEVLOG 哪条决策。
- **⚠ 不要只靠 Git 历史** —— 没有索引等于找不回。它们是不可再生的 LLM 输出。
- 🆕 **本轮相关**：`build/blind_v2/` 已被 `.gitignore` 排除，并由
  `src/vqa/blind_image.py` 机械导出（脚本才是可复现的那一份）—— **这是正确的做法**。
  旧的 10 份 `blind_*.json` 没有对应脚本，所以只能归档，不能靠重跑。

---

## §4 · 批次 4：收敛调用链

**依赖**：4.1 4.2 需要 2.2 先做完。

### 4.1 🟡 **部分修复** · `_run_local_pool` 的失败传播

- **✅ 本轮已修的部分**：`matrix_run.py:211-220` 新增 `broken` 列表，
  `_prepare` 失败现在会打印并 `return len(broken)`。
  上一版「42 个 spec 全抛异常 → work 为空 → 打印『全部已完成』→ 退出 0」的路径已堵上。
- [ ] **❌ 仍未修**：函数末尾（`matrix_run.py:243`）仍是 `return 0`，
      **`stats["errors"]`（worker 执行异常数，`matrix_run.py:236` 拿到）依旧被丢弃**。
      单题级的模型调用失败仍然不影响 `matrix` 的退出码。
- [ ] 🆕 **新引入的问题**：`if broken: ... return len(broken)` 是**提前返回**。
      只要 42 个 spec 里有 1 个准备失败，**其余 41 个准备好的 run 一个都不跑**，
      而且它们的 store 已经 `store.open()` 过、文件已建。
      而 `_run_serial`（`matrix_run.py:177-179`）是逐 spec 独立、失败一个继续下一个。
      **两条路径行为相反，都没写进文档。** 见 **D-12**。
- [ ] **仍待决**：`engine.py:178` 的熔断只写 `summary["aborted"]`，
      两条路径都不计失败 —— 「连续 20 次失败」不会让 `matrix` 退出非 0。
      决定它是否应影响退出码，然后写进 RUNBOOK。
- **怎么验**：
  ```bash
  grep -n 'return 0\|return len(broken)\|stats\[' src/eval/robochrono/matrix_run.py
  ```

### 4.2 🔺 **实证升级** · `pool._worker` 重复了 `engine._run_unit`，且自建 runtime

**这条不再是「潜伏缺陷」了 —— A-1 就是它造成的。**
`pool._worker` 不只复制了执行循环，还**从 config 重新构造了一份 runtime**
（`pool.py:78-83`），于是主进程在 `_prepare` 里对 runtime 做的任何事（BC-09 的
`frames_by_run`、未来任何 per-run 覆盖）都到不了 worker。
`8611e28` 修 `frames_by_run` 时正是在这里漏掉的。**修 4.2 等于顺手修掉 A-1，反之不然。**

**本轮复核：两处均未动，三处分叉仍在，另加 runtime 重建这第四处。**

- [ ] `src/eval/robochrono/pool.py:115-139` vs `src/eval/robochrono/engine.py:54-95`
- **已分叉的三处**：
  1. `engine.py:65` 写 `runtime["replay_key"] = unit.key`，`pool.py` **没有**。
     `vlm_api.py:1271` 的 replay provider 读的正是这个键
  2. `engine.py:75` 打印 retry 提示，pool 静默
  3. `pool.py:138` 在 `timing` 里多一个 `worker` 字段
- **分叉 1 目前不可达**（`use_pool` 要求 `model.is_local`，replay 的 `kind` 不是 local），
  是潜伏缺陷 —— 但它正说明「复制一份就会漂」。
  4. 🆕 **runtime 来源不同**：主进程 `_prepare` 算出的 runtime 被 `_run_local_pool:213` 丢弃，
     worker 用 `pool.py:78-83` 自建的那份 —— 两份的 `frames` 已经不一样（A-1）
- **改成**：`_run_local_pool` 把 `_prepare` 的 runtime 随 `WorkItem` 下发，
  worker 不再自建（只保留 `device_map` 覆盖）；`pool._worker` 改调
  `engine._run_unit(task, unit, runtime)`，再补 `worker` 字段。
  前半解掉 A-1，后半消掉执行循环的重复。

### 4.3 · `serve.py:review()` 是第二份判据，只覆盖 8 类检查里的 3 类

**本轮复核：`src/label/` 整个目录未动，全部仍成立。**

- [ ] `src/label/serve.py:259-296` vs `src/label/validate.py:104-222`
- **在线保存时不会被拦下的 5 类**：
  | 检查 | 抓什么 | 为什么重要 |
  | --- | --- | --- |
  | 污染 | 出题产物回写进标注 | **就是 P-03 本身** |
  | 引用 | 未定义的 subtask ID | ID 化之后的新风险 |
  | 派生 | `start/end` 与帧号不自洽 | 上游 `end=(f+1)/fps` 的隐含语义 |
  | 序列 | 动作序列讲不通 | 抓出过 wash 两处真错误 |
  | 可疑 | 零长度段 | pen_inbox 的零长度段就是「连按两次 K」造出来的 |
- **而四处文档都声称是同一份代码**：`serve.py:22-23`、`validate.py:5-9`、
  `src/label/README.md:92`、`src/label/AGENTS.md:88`（后者还标「风险：高 —— 判据分叉过一次」）。
  `serve.py:48` 的 import 只取 `core.Segment` / `core.build_document`，**没有 import `validate`**。
- **改成（分两步）**：
  - [ ] ① **先补两条硬错到在线版**：`污染`（判据见 `validate.py:53-55` 的 `ALLOWED_SEGMENT_KEYS`）
        与 `引用`。判据简单、无需重构
  - [ ] ② `serve.review()` 改为 import `validate` 的检查函数
        （主要成本：把 `check_family` 从「按族扫目录」重构成「按单份文档检查」）

### 4.4 · `schemas/segments.json` 无任何代码校验

**本轮复核：`compose.py` 改了 112 行，但校验的仍然只有 `item.json`。**

- [ ] `src/common/schemas/segments.json` —— 全仓库无任何代码加载它
      （`git grep jsonschema -- '*.py'` 只命中 `src/vqa/compose.py:161`，加载的是 `item.json`）
- **规则被手写复制到了别处**：`additionalProperties: false` 的白名单 ≈ `validate.py:53-55`；
  `not/anyOf` 拒绝 `metadata/window_type/original_*` ≈ `validate.py` 的「污染」检查。
  **同一条判据两份，一份没人执行。**
- **schema 还比 `validate.py` 严，以下目前完全没被检查**：
  `id` 的 pattern `^.+@f\d{6}(-\d+)?$`、`subtask` 的 pattern `^[a-z0-9]+(_[a-z0-9]+)*$`、
  `source` 的必填字段。
- **与 README 冲突**：`README.md:124-126` 明写「每个层间边界都有 schema + 校验器」
  —— 对 `item.json` 成立，对 `segments.json` 不成立。
- **改成**：`validate.py` 加一步 `jsonschema` 校验，手写白名单换成从 schema 读。
  现成模式可抄 `src/vqa/compose.py:161-169`。

### 4.5 · `blind.options_four` 改用 `plan.build_options`

- [ ] `src/vqa/blind.py:113-116`（`options_four`）、`:118-142`（`options_cross`）
- **现状**：意在复现出题实际用的借用逻辑，但轮转盐与可借池构造都与 `plan.build_options` 不同（见 1.5）。
  于是 `--policy four` 测的不是当前出题方案实际生成的那批选项。
- **不影响** `as_built`，已发布的盲测主结论仍成立。
- **前置**：2.3 的 `build_options` 单元测试。

### 4.6 🆕 · `tasks.load_run_items` 现在零调用，normalized 加载路径成了孤儿

- [ ] **位置**：`src/eval/robochrono/tasks/__init__.py:143-198`（`load_run_items`，约 55 行含长 docstring）
- **搜索锚点**：`def load_run_items(datasets_root: Any, family: str, run: str,`
- **现状**：`3d2848a` + `d6198a8` 把 `cli.run` / `preflight` / `estimate` / `matrix_run`
  四个调用点**全部**改到了新的 `tasks.load_for_run`。`git grep load_run_items` 现在只剩：
  ```
  定义处 1 处 + 自己的 docstring + DEVLOG + src/eval/docs/dataset_refactor_plan.md
  ```
  **没有任何代码调用它。**
- **连带**：`load_run_items(source="raw")` 与新的 `load_for_run` 功能重复
  （都是 `load_items` + `resolve_items(base=)`）；
  `load_run_items(source="normalized")` 是唯一读 `datasets/normalized/` 的路径，
  它一没人调用，`_require_fresh` / `StaleNormalized` 的 run 侧触发点也就没了。
  `check_freshness` 本身仍有用户（`preflight.py:108,131`、`tools/build_normalized.py`），
  **所以不要连 `normalize.py` 一起删**。
- **同时注意**：`preflight.check_normalized` 本轮新增了 v2 短路
  （根 `manifest.json` 有 `fingerprint` 就直接判 OK），所以 v2 数据上
  `check_freshness` 也不再被调到 —— normalized 这一层在 v2 上已经**整体不参与**了。
  而 `test_normalized_equivalence.py`（228 行）守的正是这条路。
- **改成**（需要和 D-4 一起定）：
  1. 若确认无外部使用者：删 `load_run_items`，把它 docstring 里
     「为什么不再『优先规范化、缺了自动回退』」那段实测结论
     （藏起 `stack_cubes/planning_2.jsonl` → 照跑 300 题、每题图从 3 张变回 1 张、BC-16 静默失效）
     搬进 `src/eval/docs/dataset_refactor_plan.md`
  2. 若 v1 对照还要用：保留，但在 docstring 顶部标明「**仅供 v1 对照，v2 走 `load_for_run`**」
- **怎么验**：`git grep -n 'load_run_items' -- '*.py'`

### 4.7 🆕 · 媒体缺失的显式报错被放宽成静默省略，preflight 看不见

- [ ] **位置**：`src/eval/robochrono/tasks/choice.py:214-238`
      （`media_head_and_options` / `media_clip_and_options`）
- **搜索锚点**：`没有主视角就【不放】主视角，而不是拿 None 去做图片`
- **现状**：本轮为了让 ⑤ 盲基线走**完全相同的代码路径**（这个动机是对的，见 `blind_image.py`
  的说明「不另写 harness，否则测的是那个脚本」），把两处改成了「缺媒体就不放这部分」。
  其中 `media_clip_and_options` **删掉了原来的**
  `raise ValueError(f"item {item.get('id')} has no input.clip_path")`。
- **风险**：现在「盲测故意挖掉」与「数据坏了」在代码里**没有区别**。
  一道本该带视频的 `image_in_video` 题若丢了 `clip_path`，会被静默地当成纯图题发给模型、
  拿到一个偏低的分数，而不是报错。
  这与本项目自己的原则冲突 —— `README.md:128-131`「**缺失要显式，不要静默跳过**」。
- **preflight 挡不住**：`preflight.check_data`（`preflight.py:167-171`）只检查
  `task.parts(unit)` **产出的** part 路径存不存在。媒体被省略后根本没有那个 part，
  于是抽样检查看到 0 个缺失，报 OK。
- **改成**（二选一）：
  1. **（推荐）** 显式化意图：题目里带一个 `blind: true` 标记（由 `blind_image.py` 写入），
     `choice.py` 只在该标记下允许省略，否则照旧 `raise`
  2. 保持现状，但在 `preflight.check_data` 里加一条：
     按题型断言 parts 里应有的媒体种类数（`image_in_video` 必须有 1 个 video part 等）
- **怎么验**：
  ```bash
  git log -1 --format=%h -S 'has no input.clip_path' -- src/eval/robochrono/tasks/choice.py
  grep -n 'if clip\|if head' src/eval/robochrono/tasks/choice.py
  ```

---

## §5 · 批次 5：涉及架构或公开 API 的决策

**依赖**：全部依赖 §0 的对应答复。

### 5.1 ⚠决策 · `requirements.txt` 不是软链接

- [ ] `src/eval/requirements.txt`、`src/eval/envs/groupA.txt`、`src/eval/docs/environments.md:42`
- **现状**：`git ls-files -s` 两者都是 **`100644`**（普通文件，软链接会是 `120000`），blob 相同。
  而 `environments.md:42` 写「`requirements.txt` 是指向 `envs/groupA.txt` 的软链接」—— **文档是错的**。
  **没有任何脚本读 `requirements.txt`**（`setup_env.sh:40` 与 `tools/setup_envs.sh:41` 都读 `envs/*.txt`）。
- **为什么值得修**：问题是潜伏的 —— 任一方被单独编辑就静默分叉，而文档让人以为不可能分叉。
- **改成（二选一）**：① 真换成符号链接；② **（推荐）** 删掉 `requirements.txt`，改文档为
  「依赖清单只有 `envs/*.txt` 一处」。
- **怎么验**：`git ls-files -s src/eval/requirements.txt`

### 5.2 🔺 **实证升级（原 ⚠决策）** · `extract.py` / `extract_llm.py` 686 行完全未接线

**这次全量跑独立撞上了 `extract.py` docstring 里写着的那个失败模式。**
它的开头就列着「SenseNova 轨迹 2D **95% 用归一化坐标而非像素**」，
而这次 time 归零的根因诊断是：**1,004 条预测无一超过 1.0，模型输出的是 [0,1] 归一化值**。
同一个病，换了个题型。

`extract.py` 的设计要点正是「**每一层的候选都必须过合理性校验，不通过就降级**」
（量级、画布范围可验）—— 一个「所有预测都 < 1.0 而视频长 89.6 秒」的量级校验
本可以把这件事变成结果里的一条 `format/unit` 标记，
而不是**两小时的人工诊断 + 一次白跑的全量**。

- [ ] `src/eval/robochrono/extract.py`（416 行）、`extract_llm.py`（270 行）
- **本轮复核**：仍零 import。生产解析仍在 `time_eqa.py` / `trajectory.py` / `parsing.py` 各自实现。
- 🆕 **建议**：不必整套接入。**先只接 L1 的「量级/范围合理性校验」这一小块**，
  用在 `time_eqa.parse_interval_row` 与 `trajectory.coerce_point_list` 上，
  产出一个 `unit_suspect` 标记进结果行。这是 A-3 的天然搭档：
  A-3 报「分数低于退化基线」，这条报「为什么低」。
- **Git 历史**：两者随 `b1ec6f8`「从旧仓库迁入 ⑥」一起进来，与任务实现同批
  —— 不是「本仓库写了一半」，而是**旧仓库里就已经是未接线状态**。
- **改成（三选一）**：① 接入 engine；② 移到 `experiments/`；
  ③ 删代码但把 docstring 里的实验数据（约束解码三轮实验、「假阳性比漏掉更糟」、
  「只抠不验的正则看起来救回 70%，加校验后真实可用只有 20%」）搬进 `docs/`。
- **⚠ 不要直接删** —— 误删会丢失有实证支撑的设计结论。

### 5.3 ⚠决策 · `export` 子命令：docstring 宣传了，parser 没有（依赖 D-4 D-5）

- [ ] `src/eval/robochrono/cli.py:7`、`src/eval/robochrono/store.py:108-124`
- **现状**：docstring 写 `python -m robochrono export --results-dir <dir>`，
  而 `main()` 注册的子命令里没有 `export` —— 照着敲得到 argparse 的 `invalid choice`。
  `ResultStore.export()` 零内部调用。
- **既不是「已被替代」也不是「未完成」**：它的功能（合并原始 item 字段、产出 v1 同构 JSON）
  在 `report` / `pack` 里没有等价物；`tasks/base.py` 的 BC-04 说明明确把它设计成
  「导出兼容格式时再合并回去」的出口。**A4 新旧对照很可能需要它。**
- **改成（二选一）**：① **（推荐）** 接回 CLI（约 30 行）；
  ② 删 `cli.py:7` 那一行，并在 `store.export()` 上标注「库 API，无 CLI 入口」。

### 5.4 ⚠决策 · `providers.json` 的 32 帧显存矛盾（依赖 D-3）

- [ ] 见 1.9 最后一条与 §0 的 D-3。决定要不要在正式跑之前先做 `--limit-items` 显存试探。

### 5.5 · 补一份 `src/eval/tools/README.md`

- [ ] `src/eval/tools/` 下 12 个脚本，5 个只被 `docs/` 引用、5 个零引用 —— 但**不是残留**：
  `build_normalized.py` 被 `tasks/__init__.py` 的错误提示指名，
  `bc10_impact.py` / `version_compare.py` / `proposal_eval.py` / `frame_alignment_probe.py`
  是产出 `src/eval/docs/` 那些实测数据的探针。
- **为什么补**：删掉会让文档里的数字失去来源，而现在没有地方记录「哪个脚本对应哪份文档」。

### 5.6 · 修 `tools/smoke_all.sh` 的三处失效路径

- [ ] `src/eval/tools/smoke_all.sh` —— `REPO` 解析为 `src/`，于是：
  | 行 | 解析成 | 实际 |
  | --- | --- | --- |
  | `${REPO}/eval/models/…` | `src/eval/models/` | ❌ 权重在仓库根的 `models/` |
  | `${REPO}/eval/datasets/QA/…` | `src/eval/datasets/` | ❌ 不存在 |
  | `cd "${REPO}"` 后 `test/${script}` | `src/test/…` | ❌ 冻结脚本在 `src/eval/upstream/` |
- **改成**：三处路径改对，或标注为「旧仓库遗物，不再维护」。
  它自称是「阶段 0 的门禁」—— 若已完成使命，**标注比修更省**。
- **怎么验**：`ls -d src/test src/eval/models src/eval/datasets`（三个都应报不存在）

### 5.7 🆕 · `media_step_order` 的 v1/v2 双形状分支，A4 之后应删

- [ ] **位置**：`src/eval/robochrono/tasks/choice.py:256-290` 附近
- **搜索锚点**：`v1 的题（`initial_image` + 一张宫格）仍按原样处理，A4 新旧对照要用`
- **现状**：本轮 step_order 改为「逐张发带标号的图」（v2），同时保留了 v1 的
  「initial 图 + 宫格图」分支，按 `initial_image` 是否存在二选一。
  **这是有正当理由的兼容 shim**（A4 新旧对照要用，且
  `test_request_equivalence.py` 的 `DECLARED` 里已把 `step_order/prompt` 声明为已知差异）。
- **为什么现在就记一笔**：这类分支的典型下场是「A4 跑完了，但没人记得回来删」。
  上一轮审计里 `extract.py`（5.2）和 `export`（5.3）就是这么留下来的。
- **改成**：不用现在动。在分支上方加一行
  `# A4 新旧对照跑完后删除此分支（负责人 / 预计时间）`，并在 A4 的验收清单里列一条。

---

## §6 · 明确不要删的历史资产

清理时**不要碰**以下内容。

| 资产 | 位置 | 为什么留 |
| --- | --- | --- |
| **冻结上游实现（评测）** | `src/eval/upstream/` 11 文件 ≈ 4,900 行 | `README.upstream` 明写「一个字符都没改」；`test_request_equivalence.py` 与 `test_parsing_equivalence.py` 逐字调用其函数作为判据。**改它等于改判据**。含零引用的 `stitch_understanding_multiview_clips.py` —— 单独删会破坏「整批未改」这个性质 |
| **冻结上游实现（标注）** | `src/label/upstream/` 3 文件 ≈ 1,220 行 | 同上；`tool_version: "upstream-video_labeler_timestamp/unknown"` 还写在每份 segments 的 `source` 里 |
| **replay fixture** | `src/eval/fixtures/` 共 3.8 MB | 录下来的真实模型输出，**不可再生**。四套自足回归全靠它 |
| **迁移前标注语料** | `data/label/*/segments.before_*/`、`stack_cubes/segments.polluted/` | 每份对应一次有据可查的数据修改。`segments.polluted` 是 P-03 的原始证据，`schemas/segments.json` 的 `not/anyOf` 就是照着它写的 |
| **corrections / provenance** | `data/label/wash/corrections.json`、`source._restored` | 「哪一段被谁按什么依据改过」的可追溯性全在这里 |
| **不可再生的 LLM 输出** | `data/llm_cache/v1-vendor/` `v2/` `v3/` | 「没有 v2 / v3 这两次失败，就没有那个结论」—— D-37/D-38 的全部实证依据 |
| **盲测原始输出** | `build/blind*.json` 共 6.3 MB | 不可再生，且没有对应脚本可重跑（对比：`build/blind_v2/` 有 `blind_image.py`，所以可以 ignore）。见 3.3 —— 归档 + 写索引 |
| 🆕 **`build/frames_floors.json`** | 1.4 KB | `.gitignore` 特意把它排除在 ignore 之外 —— **要被 review 的是那几个数字**。这是正确的取舍，别跟着 `frames.json` 一起清掉（见 1.17） |
| **一次性迁移脚本** | `src/migrate/` 6 个 | 是 `data/label/` 与 `data/llm_cache/` 的**来源记录**；`fetch_raw.py` / `normalize_source.py` 还是 `README.md` 里「拿到数据」的正式步骤 |
| **`docs/disclosures.md`** | 全文 | 全仓库唯一与代码一致的 time 指标表述来源，且**本轮跟着新题型更新到位了**（§6 重写、新增 1b / D-57 / D-58）。1.9 的修法是让别处向它对齐 |

---

## §7 · 不要按第一轮审计的这几条动手

1. **【推翻】`src/eval/requirements.txt` 是软链接** —— mode 是 `100644`。见 5.1。
2. **【推翻】`mediaindex.py` 有未使用 import** —— 逐条核对其 import 全部有使用。误报。
3. **【推翻】未使用 binding 约 17 个** —— 上一轮实测 13 个，本轮 18 个（原因见 1.2），都不是 17。
4. **【降级】「`run_all.py` 会让跳过混进通过数」** —— 不成立，它明确单列跳过。真实问题只是退出码。见 2.4。
5. **【降级】「文档写 5 通过 1 跳过是错的」** —— 在有 v1 数据的机器上**成立**。缺的是环境前提说明。
6. **【重新归因】「pool 吞失败」** —— `pool.run_pool` 尽职返回了 `{"done","errors"}`，
   吞掉它的是编排层 `_run_local_pool`。且熔断不影响退出码是**两条路径共有**的。见 4.1。
7. **【不是残留】** `src/eval/upstream/` `src/label/upstream/` `src/migrate/` `src/eval/tools/`。见 §6 与 5.5。
8. **【已排除动态引用误判】** `cli.py` 的 9 个 `cmd_*`（`set_defaults(func=...)` + `args.func(args)`）、
   `serve.py` 的 `do_GET`/`do_POST`/`log_message`（框架回调）、
   `tasks.build` 按字符串分派到 `choice.SPECS`/`time_eqa`/`trajectory`。**不是死代码。**

---

## §8 · 验证脚本

三段只读脚本，随时可跑。建议做完一个批次跑一次。

**A · 未使用 import**（基线 18 条，做完 1.2 应为 0）
```bash
python3 - <<'PY'
import ast, subprocess, pathlib
root = pathlib.Path('.')
for rel in subprocess.run(['git','ls-files','*.py'],capture_output=True,text=True).stdout.split():
    src = (root/rel).read_text(encoding='utf-8')
    try: tree = ast.parse(src)
    except Exception as e: print('PARSE-FAIL', rel, e); continue
    binds = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: binds[a.asname or a.name.split('.')[0]] = n.lineno
        elif isinstance(n, ast.ImportFrom) and n.module != '__future__':
            for a in n.names:
                if a.name != '*': binds[a.asname or a.name] = n.lineno
    used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name): used.add(n.id)
        elif isinstance(n, ast.Attribute):
            m = n
            while isinstance(m, ast.Attribute): m = m.value
            if isinstance(m, ast.Name): used.add(m.id)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            for b in binds:
                if b in n.value: used.add(b)
    for name, ln in sorted(binds.items(), key=lambda x: x[1]):
        if name not in used: print(f'{rel}:{ln}: {name}')
PY
```

**B · Markdown 坏链接**（当前基线 6 条，全在 1.6 名单里；做完 1.6 应为 0）
```bash
python3 - <<'PY'
import re, subprocess, pathlib
root = pathlib.Path('.')
pat = re.compile(r'\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
code = re.compile(r'`[^`]*`')
bad = []
for rel in subprocess.run(['git','ls-files','*.md'],capture_output=True,text=True).stdout.split():
    p = root/rel
    fenced = False
    for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
        if line.lstrip().startswith('```'):
            fenced = not fenced; continue
        if fenced: continue          # 围栏代码块里的不是链接
        line = code.sub('', line)    # 行内代码里的也不是
        for m in pat.finditer(line):
            t = m.group(2)
            if t.startswith(('http://','https://','mailto:','#')): continue
            t = t.split('#')[0]
            if t and not (p.parent/t).resolve().exists():
                bad.append(f'{rel}:{i}: {m.group(0)}')
print('\n'.join(bad) or '无坏链接'); print(f'共 {len(bad)} 条')
PY
```

**C · README 数字与抽帧档位**（1.1 / 1.8 用）
```bash
python3 - <<'PY'
import json, glob, sys
segs = eps = subs = 0
for f in sorted(glob.glob('data/label/*/segments/*_segments.json')):
    segs += len(json.load(open(f))['segments']); eps += 1
for f in sorted(glob.glob('data/label/*/subtasks.json')):
    subs += len(json.load(open(f))['subtasks'])
print(f'标注：{eps} 份 segments 文件 / {segs} 段 / {subs} 个 subtask')
fam = json.load(open('data/families.json'))['families']
print('活跃族：', [k for k, v in fam.items() if v['status'] == 'active'])
import collections
p = json.load(open('build/plan.json'))
print(f'题量 {len(p["items"])}：', dict(collections.Counter(i['task'] for i in p['items'])))
sys.path.insert(0, 'src/eval')
from robochrono.vlm_api import resolve_frames
cfg = json.load(open('src/eval/configs/providers.json'))
print('\n各 provider 实际抽帧档位（matrix 走的就是这个）：')
for name, pr in cfg['providers'].items():
    print(f'  {name:<40}', resolve_frames(pr, cfg['defaults']))
print('  frames_by_run（只有 cli.run 读）=', cfg.get('frames_by_run'))
PY
```
