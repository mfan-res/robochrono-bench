# 多环境方案

## 为什么必须多套环境

不是偏好问题，是硬约束。实测（全部在本机跑过，不是查文档得来的）：

| 模型 | transformers 4.57.6 | transformers 5.15.0 |
|---|---|---|
| SenseNova-SI-1.1-InternVL3-2B | ✅ | ❌ `'InternVLChatModel' object has no attribute 'all_tied_weights_keys'` |
| RynnBrain-2B | ✅ | ✅ 2.13B |
| Qwen3-VL-8B-Instruct | ✅ | ✅ 8.77B |
| Cosmos3-Edge-2B | ❌ 架构 `cosmos3_edge` 未注册 | ✅ 2.44B |

**InternVL 只能在 4.x，Cosmos3-Edge 只能在 5.x，两者互斥。** 一套环境跑不完 15 个模型。

分界线在于模型是否依赖 `trust_remote_code` 的自定义代码：

- **原生架构**（Qwen3-VL、RynnBrain —— 都是 `Qwen3VLForConditionalGeneration`）
  由 transformers 自己维护，跨版本兼容
- **自定义代码**（InternVL）由模型仓库维护，跟不上 5.x 基类 API 的变化。
  5.2.0 上是 meta tensor 崩溃，5.15.0 上那个修好了，换成缺 `all_tied_weights_keys`

所以「随 transformers 升级而失效」是自定义代码模型的常态，不是偶发。

## 分组

| | transformers | 覆盖 |
|---|---|---|
| **groupA** | 4.57.6 | InternVL ×3、Qwen3-VL ×3、RynnBrain-2B、Cosmos-Reason2 —— 8 个 |
| **groupB** | 5.15.0 | Cosmos3-Edge、RynnBrain1.1-2B —— 2 个 |

只有两套，不是「一个模型一套」。官方推荐版本各不相同，但**推荐版本不等于唯一可用版本** ——
真正的判据是能不能加载、输出对不对，这个只能实测。按实测归并成两组，比按推荐值开
十套环境省得多，也少了十份要维护的依赖清单。

`huggingface_hub` 也被迫分家：A 组是 0.36.x，B 组是 1.27.x（transformers 5.x 要求）。
这是两套环境必须物理隔离、不能靠一套环境切 transformers 版本的另一个原因。

## 怎么建

依赖清单的唯一来源是 `envs/groupA.txt` / `envs/groupB.txt`。
`requirements.txt` 是指向 `envs/groupA.txt` 的软链接，`setup_env.sh`（conda 版）
和 `tools/setup_envs.sh`（uv 版）都从同一份清单读 —— 不存在两处版本号各写各的。

### uv（推荐，用于迁移到别的集群）

```bash
bash tools/setup_envs.sh                 # 建 groupA + groupB
bash tools/setup_envs.sh groupB          # 只建一套
UV_INDEX=https://mirrors.ustc.edu.cn/pypi/simple bash tools/setup_envs.sh   # 走镜像
```

脚本会把解释器路径写回 `configs/environments.json`（**相对 `eval/` 的路径**，
这样提交到 git 后在别人机器上依然有效）。

实测数据（本机，走 USTC 镜像）：

| | 大小 | 说明 |
|---|---|---|
| groupA | 5.6 GB | 其中约 4 GB 是 torch 捆绑的 nvidia CUDA 库 |
| groupB | 5.6 GB | 同上 |
| 合计 | 12 GB | 本机文件系统上两套没有硬链接去重 |

对比 conda 环境 6.0 GB。**uv 不省磁盘**，它的优势在别处：

- 环境完全由一个文本文件描述，`git clone` + 一条命令就能重建，不用打包传输 6 GB
- 解析安装是分钟级；conda solver 经常几分钟起步，多套环境时差距被放大
- 全部依赖都在 PyPI 上（已核实，没有 conda-only 的包），所以没有非用 conda 不可的理由
- 下载缓存跨环境共享，第二套环境的 torch 不用重新下

### conda（保留）

```bash
bash setup_env.sh          # 建 groupA 对应的 conda 环境
```

两条路都读同一份 `envs/groupA.txt`。

### 集群离线时

`envs/*.txt` 里全是精确版本号，可以在有网的机器上先 `uv pip download -r envs/groupA.txt -d wheels/`，
把 `wheels/` 拷过去后 `uv pip install --no-index --find-links wheels/ -r envs/groupA.txt`。

## 怎么跑

```bash
python -m robochrono dispatch -- --gpus 8              # 按模型自动分派到对应环境
python -m robochrono dispatch --dry-run -- --gpus 8    # 只看会执行什么
```

`dispatch` 读 `configs/environments.json`，把模型按环境分组，
每组用对应解释器起一个子进程跑 `matrix --models ...`。

分派粒度是**模型**而非进程内：矩阵本来就是 model-major 调度（一个模型跑完再换下一个），
按模型切一刀不打乱原有编排，还顺带隔离了崩溃 —— 一套环境装坏了不影响另一套。

刻意不用 `multiprocessing.set_executable`：跨解释器 spawn 依赖两边 python 版本与
pickle 协议一致，很脆且报错难查。

结果都写进同一个 `results/` 目录，各模型子目录互不重叠。

## 一个必须避开的坑

**不要对 venv 的 `bin/python` 调用 `Path.resolve()`。**

它是指向基础解释器的符号链接（uv 的情况下指向 `~/.local/share/uv/python/cpython-*/bin/python3.11`）。
解析之后 `sys.prefix` 会指向基础环境，**venv 的 site-packages 完全不生效** ——
B 组会静默用上 A 组的 transformers，而且不报任何错。

代码里已经改成词法运算（`os.path.abspath` / `os.path.relpath`），
`robochrono/dispatch.py` 与 `tools/setup_envs.sh` 两处都有注释标注。

验证方法：

```bash
for e in groupA groupB; do
  .venvs/$e/bin/python -c "import sys,transformers; print(e, transformers.__version__, sys.prefix)"
done
```

两边必须报各自的版本号，且 `sys.prefix` 落在各自的 venv 里。

## 版本会改变输出 —— 已实测确认

用 `tools/version_compare.py` 跑 RynnBrain-2B（两个版本都能加载），
同一批题、**同一张卡**、相同抽帧与生成参数，各跑一遍。

### 对照实验先排除随机性

| 实验 | 结果 |
|---|---|
| 同版本跑两遍（4.57.6 vs 4.57.6） | **13/13 run 完全一致，每一行都相同** |
| 跨版本（4.57.6 vs 5.15.0） | **13/13 run 全部不同** |

推理在同卡同版本下是确定性的，所以差异**可以归因到版本本身**，不是 GPU 随机性。

### 根因：视频 prompt 的包装格式变了

逐层拆解，同一段视频：

| | 4.57.6 | 5.15.0 | |
|---|---|---|---|
| chat template | md5 `afef5bd7` | md5 `afef5bd7` | 相同 |
| 视频像素张量 | `(30600,1536)` md5 `4e056216` | 同左 | **逐字节相同** |
| 抽帧数 | 4 / 10 / 72 / 146（按档位） | 同左 | 相同 |
| **input_ids** | **7802** | **7804** | **差 2 个 token** |
| 其中 `<\|video_pad\|>` | 7650 | 7650 | 相同 |

差的正是外面多套的一层：

```
4.57.6:  user\n <0.3 seconds><|vision_start|>[450 patches]<|vision_end|> <1.x seconds>...
5.15.0:  user\n <|vision_start|> <0.3 seconds><|vision_start|>[450]<|vision_end|> ... <|vision_end|>
                ^^^^^^^^^^^^^^^^ 多这一个                                          ^^^^^^^^^^^^^^ 和这一个
```

**5.15.0 在整段视频外面多包了一层 `<|vision_start|> … <|vision_end|>`。**
画面内容一模一样，包装不一样 —— 于是模型的输出跟着变。

这不是数值误差、不是 kernel 差异、不是我们传参传错了
（专门验过：`processor_kwargs` 那条警告是误导性的，
现有的 `**video_kwargs` 写法在 5.x 上是生效的，改成它建议的形式反而会被忽略）。
是 transformers 内部 Qwen3-VL 视频 prompt 模板本身的改动。

### 影响面（RynnBrain-2B，stack_cubes，每 run 25 个 unit，共 575 行）

| | 数量 | 占比 |
|---|---:|---:|
| 输出文本不同 | 361 | **63%** |
| 判定（对错 / tIoU）不同 | 45 | **8%** |

但**聚合指标的差值没有方向性**：13 个 run 的指标差基本都是 ±0.040，
也就是「一题」——有正有负，不存在哪个版本系统性更高。

| run | A | B | Δ |
|---|---:|---:|---:|
| left_right | 0.280 | 0.240 | −0.040 |
| planning_2 | 0.680 | 0.640 | −0.040 |
| step_order | 0.520 | 0.560 | +0.040 |
| fps1/image_in_video | 0.440 | 0.520 | +0.080 |
| fps1/time (tIoU) | 0.070 | 0.074 | +0.004 |
| fps2/time (tIoU) | 0.055 | 0.067 | +0.012 |
| fps1/understanding | 0.400 | 0.400 | 0 |

先前用 n=5 的样本量看到过 0.600→0.400 这种大幅波动，
**那其实也只是一题，被小样本放大成了 20 个百分点** —— 不要据此判断版本优劣。

按 8% 的翻转率、方向随机估算，一个 300 题的 run 聚合指标大约会有
**±1~2 个百分点**的不确定性。不致命，但足以影响接近的名次。

**哪一种包装格式是对的，取决于模型训练时用的是哪种** —— 用错会掉分，
而且不报任何错。这需要向模型作者确认，不是我们能从代码里判断的。

工程上的硬性要求：

1. **同一个模型的所有结果必须在同一个版本下产出。** 中途升级 transformers
   等于换了一次实验条件，前后数据不可合并。
2. **每份结果都记录 transformers 版本** —— 结果目录的 `.meta.json` 里有 `environment`
   字段（python / transformers / torch / qwen_vl_utils）。这是发现上述问题后补的：
   之前没记，事后根本无从判断两批数据能不能放一起比。
3. 跨模型对比时，如果两个模型跑在不同版本上，**这是一个真实的混淆变量**，
   报告里必须写明，不能当作模型能力差异。
