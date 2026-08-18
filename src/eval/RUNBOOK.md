# RoboChrono 评测操作手册

面向在自有算力上执行评测的同事。照着走即可，不需要读框架代码。

---

## 0. 这套东西在做什么

对 **15 个模型 × 20 个任务族 × 9 个任务** 做视觉语言模型评测，
产出一张「模型 × 任务」的对比表。

九个任务对应八个任务类型 —— `trajectory` 的 2D 与 3D 是两份独立输入、
两次独立运行、两组独立分数，所以按 9 个 run 计。

| run | 主指标 |
| --- | --- |
| understanding / left_right / planning / planning_2 / step_order / image_in_video | `accuracy` |
| trajectory_2D / trajectory_3D | `mean_score` |
| time | `mean_tIoU` |

---

## 1. 准备环境

**需要两套环境，这是硬约束不是偏好。** InternVL 系只能在 transformers 4.x 上跑，
Cosmos3-Edge 只能在 5.x 上跑，两者互斥 —— 一套环境跑不完 15 个模型。

推荐用 uv（迁到别的集群时只要一条命令，不用打包传 6 GB 的 conda 环境）：

```bash
cd eval
bash tools/setup_envs.sh
# 国内集群走镜像：
UV_INDEX=https://mirrors.ustc.edu.cn/pypi/simple bash tools/setup_envs.sh
```

也保留了 conda 版（只建 groupA）：

```bash
bash setup_env.sh && conda activate robochrono
```

两条路读的是**同一份依赖清单** `envs/groupA.txt`，不会各建各的。

| | transformers | 覆盖 |
| --- | --- | --- |
| groupA | 4.57.6 | InternVL ×3、Qwen3-VL ×3、RynnBrain-2B、Cosmos-Reason2 |
| groupB | 5.15.0 | Cosmos3-Edge、RynnBrain1.1-2B |

版本是实测选的，不是抄官方推荐：`4.51.x` 因代码里的 `dtype=` 报
`unexpected keyword argument`；`5.x` 上 InternVL 加载失败（5.2.0 是 meta tensor
崩溃，5.15.0 是缺 `all_tied_weights_keys`）。详见 `docs/environments.md`。

⚠️ **换 transformers 版本会改变模型输出** —— 这是实测结论，不是理论担忧。
所以同一个模型的结果必须在同一个版本下产出，不要中途升级。详见报告第六节。

---

## 2. 准备数据与权重

```
eval/
├── datasets/
│   ├── QA/planning/<family>/…        生成流水线的产物
│   ├── QA/understanding/<family>/…
│   └── json/<family>/…               原始标注
└── models/<模型目录>/                 本地权重
```

两个已知的坑：

**权重目录名不要带点号。** transformers 4.x 按 HuggingFace repo id 建动态模块目录，
repo id 里的 `1.1` 会被当成包分隔符导致导入失败。
下载后把目录名里的点改成下划线，例如
`SenseNova-SI-1.1-InternVL3-2B` → `SenseNova-SI-1_1-InternVL3-2B`。

**`auto_map` 里的跨仓库引用要去掉。** 部分模型的 `config.json` 写成
`repo_id--module.Class`，同样会触发上面那个问题。跑一次：

```bash
python tools/patch_local_model.py models/<模型目录> --apply
```

`preflight` 会检测这两种情况并提示。

---

## 3. 配置 API key

**key 不要写进任何配置文件** —— 那些文件会进 git。写到仓库之外：

```bash
mkdir -p ~/.config/robochrono
cat > ~/.config/robochrono/keys.env <<'EOF'
DASHSCOPE_API_KEY=你的key
ZHIPUAI_API_KEY=
ARK_API_KEY=
GEMINI_API_KEY=
EOF
chmod 600 ~/.config/robochrono/keys.env
```

`providers.json` 里的 `api_key_env` 填的是**环境变量的名字**（如 `DASHSCOPE_API_KEY`），
不是 key 本身。填错了 `preflight` 会报出来。

---

## 4. 声明要跑什么

编辑 `configs/plan.json`：

```json
{
  "models":   [{"name": "...", "provider": "...", "kind": "local|api"}],
  "families": ["stack_cubes", "tea", ...],
  "runs":     ["time", "understanding", ...],
  "family_attrs": {"stack_cubes": {"two_handed": true}},
  "skip_rules":   [{"run": "left_right", "unless": "two_handed"}]
}
```

加模型、加任务族都只改这个文件，不用动代码。

---

## 5. 跑

如果配了多套环境，用 `dispatch` —— 它会把每个模型送到它该去的环境：

```bash
python -m robochrono dispatch -- --gpus 8
python -m robochrono dispatch --dry-run -- --gpus 8    # 先看会执行什么
```

单一环境下直接用 `run.sh`：

```bash
./run.sh                          # 自检 → 估算 → 执行 → 汇总 → 打包
./run.sh --only api               # 只跑 API 模型，不需要 GPU
./run.sh --only local --gpus 8    # 只跑本地模型，用 8 张卡
./run.sh --limit-items 4          # 冒烟：每个任务只跑 4 题
```

多机分工用稳定哈希切分，机器之间不重不漏：

```bash
./run.sh --shard 1/4      # A 机
./run.sh --shard 2/4      # B 机
```

也可以只跑单步：

```bash
python -m robochrono preflight        # 自检
python -m robochrono plan             # 看矩阵会展开成什么
python -m robochrono estimate         # 估算调用量与媒体体积
python -m robochrono matrix --gpus 8  # 执行
python -m robochrono matrix --only api --api-concurrency 24   # API 并发
python -m robochrono report results/ --csv results/summary.csv
python -m robochrono pack -o out.tar.gz
```

**中断了直接重跑同一条命令**，已完成的题会自动跳过。

---

## 6. 回传结果

```bash
python -m robochrono pack -o robochrono-results.tar.gz
scp robochrono-results.tar.gz <目标主机>:<路径>
```

默认只带汇总与逐题精简记录（全矩阵约几百 MB）。
`--full` 会连媒体缓存一起带，通常不需要。

---

## 7. 已知问题与注意事项

**Time EQA 的视频很大。** base64 后中位 10.47 MB，而 API 端的请求体上限约 10 MiB
（实测 10.32 MB 通过、10.99 MB 返回 413）。`providers.json` 里给 API provider 设了
`max_request_bytes`，超预算的视频会自动降分辨率重编码，变换记录写在结果的
`media_transforms` 字段里。已验证这个压缩**不改变模型实际获得的视觉预算**
（服务端报告的 `video_tokens` 压缩前后相同）。

**多卡不解决 Time EQA 的 OOM。** `device_map="auto"` 聚合的是权重容量，
而 OOM 失败的是单个连续张量（LLM 注意力矩阵，fp32 物化）。
实测把权重摊到 4 张卡后，仍然在同样的帧数上 OOM、申请量一字不差。
真正的解法是启用 FlashAttention，尚未验证。

**抽帧策略尚未定案。** `configs/providers.json` 里 `frames` 的取值目前只是
让链路能跑，不是评测配置。这个参数会显著影响 Time EQA 的成绩
（同一份数据上，2,048 视觉 token 得 tIoU 0.0，30,000 量级得 0.716）。
定案前 Time EQA 的横向对比不成立。详见
[`docs/frame_sampling_investigation.md`](docs/frame_sampling_investigation.md)。

**部分模型的官方推荐与我们的配置有出入**，逐条记录在
[`docs/official_config_survey.md`](docs/official_config_survey.md)，共 9 处分歧。

**`Cosmos3-Edge` 的定位存疑。** 官方描述是 omnimodal world model（生成视频/图像/动作），
README 里没有任何问答式推理示例。我们用 `AutoModelForImageTextToText` 加载它属于
非官方用法，能否稳定产出可解析的选择题答案未经验证。建议先单独试一个族。

---

## 8. 出问题时

```bash
python -m robochrono preflight      # 先跑这个，多数问题它会直接指出来
```

| 现象 | 原因 |
| --- | --- |
| `unexpected keyword argument 'dtype'` | transformers < 4.56 |
| `Tensor.item() cannot be called on meta tensors` | transformers 5.x + InternVL |
| `No module named 'transformers_modules.…'` | 权重目录名带点号，或 `auto_map` 跨仓库引用 |
| `413 RequestTooLarge` | 该 provider 没设 `max_request_bytes` |
| `Missing API key` | `keys.env` 没配，或 `api_key_env` 填成了 key 本身 |
| 某个 run 标 `aborted` | 连续 20 次失败触发熔断，看该 run 的 `.jsonl` 里的 error 字段 |

结果文件的组织：

```
results/<模型>/<任务族>/<run>.jsonl          逐题记录（含原始模型输出）
results/<模型>/<任务族>/<run>.summary.json   该 run 的全部指标
results/<模型>/<任务族>/<run>.meta.json      跑这一轮时的配置快照
```

`.jsonl` 里保留了**原始模型输出**，所以指标口径改了可以离线重算，
不必重新占卡或花钱。
