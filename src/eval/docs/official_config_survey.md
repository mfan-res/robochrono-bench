# 官方推荐配置调研

> 目的：为「capability 协议」确定每个模型的输入预处理配置，来源必须可追溯到官方。
> 状态：**只记录，不下结论。** 所有分歧原样保留，待讨论。
> 调研日期：2026-08-13

## 来源等级

| 等级 | 含义 | 实例 |
| --- | --- | --- |
| L1 | 官方明示推荐（专门的推荐段落） | Qwen3-VL-8B 的 Generation Hyperparameters |
| L2 | 官方示例调用值 | InternVL `load_video(num_segments=8)` |
| L3 | 官方函数签名默认值 | InternVL `def load_video(..., num_segments=32)` |
| L4 | 官方配套库默认值 | `qwen_vl_utils` 的 `FPS=2.0`（对 Qwen 系成立） |
| L5 | 同架构模型继承 | SenseNova 沿用 InternVL 惯例 |
| L6 | 我们按规则推导 | `min(上下文, 显存, 原生帧数)` |

L5 / L6 不是官方意见，报告中必须显式标注。

---

## 一、开源模型（9 个）

### InternVL3.5-8B / InternVL3.5-30B-A3B

来源：HF README 推理示例段（两个模型的 README 内容一致）

| 项 | 官方值 | 等级 | 我们当前值 | 差异 |
| --- | --- | --- | --- | --- |
| 视频抽帧 | `num_segments=32`（签名）<br>`num_segments=8`（示例调用） | L3 / L2 | `video_sample_fps=1.0`（≈53 帧） | **偏离**：官方无 fps 采样概念 |
| 视频切块 | `max_num=1` | L2 | `max_video_tiles=1` | 一致 |
| 图片切块 | `max_num=12` | L2 | `max_image_tiles=4` | **偏离** |
| 输入尺寸 | `input_size=448` | L2 | 448 | 一致 |
| 帧提示格式 | `Frame{i+1}: <image>\n` | L2 | `Video{n} Frame{i+1}: <image>\n` | 轻微偏离 |
| Thinking 模式采样 | `do_sample=True`, `temperature=0.6` | L1 | `thinking=disabled` | 条件不适用，但需记录 |

**分歧 #1：官方自己给了两个抽帧数。** 同一份 README 里函数签名默认 `num_segments=32`，而示例调用写 `num_segments=8`。需要定规则：L2（示例）优先还是 L3（签名）优先。

**分歧 #2：官方没有 fps 采样这条路径。** `video_sample_fps` 是本 repo 自创的参数，且它会静默架空官方的 `num_segments` 分支。我们代码里 `num_segments` 分支的帧索引公式与官方 `get_index` 逐字相同，说明当初照抄自官方，后来被自加的 fps 参数覆盖。

**附注：** README 中说明官方评测使用 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)。若追求与官方发布数字对齐，VLMEvalKit 的视频采样配置是更权威的参照，值得单独查一次。

---

### SenseNova-SI-1.1-InternVL3-2B

来源：HF README

| 项 | 官方值 | 等级 |
| --- | --- | --- |
| 全部推理配置 | **README 中没有任何推理示例** | 无 |

只能按 L5 继承 InternVL3 的惯例。

**分歧 #3：模型自身两个上下文数字矛盾。** `config.json` 的 `max_position_embeddings=32768`，而 `tokenizer_config.json` 的 `model_max_length=16384`。按 256 token/帧折算，前者对应 128 帧、后者对应 64 帧，差一倍。实测超过 16384 时只打警告仍返回结果，哪个是真实上限**需要实测**。

**附注（非配置问题，但影响部署）：** `config.json` 的 `auto_map` 使用跨仓库引用形式
`sensenova/SenseNova-SI-1.1-InternVL3-2B--modeling_internvl_chat.InternVLChatModel`。
transformers 按 repo id 建动态模块目录，而 id 中的 `1.1` 含点号会截断模块名导致导入失败。
已做成修复工具 `eval/tools/patch_local_model.py`。

---

### Qwen3-VL-8B-Instruct

来源：HF README

| 项 | 官方值 | 等级 | 我们当前值 | 差异 |
| --- | --- | --- | --- | --- |
| 生成参数 | `greedy=false`<br>`top_p=0.8`, `top_k=20`<br>**`temperature=0.7`**<br>`repetition_penalty=1.0`<br>`presence_penalty=1.5`<br>`out_seq_length=16384` | L1 | `temperature=0`<br>`max_new_tokens=1024` | **冲突** |
| 视频抽帧 | 未明示，指向 `qwen_vl_utils` | L4 | qwen_vl_utils 默认 | 一致 |
| 注意力实现 | 推荐 `flash_attention_2` | L1 | 未启用 | 偏离 |
| 视觉参数 | `patch_size=16`, `merge_size=2`, `temporal_patch_size=2` | L1（preprocessor_config） | 库默认 | 一致 |

**分歧 #4：官方推荐非贪心采样 temperature=0.7，与基准评测所需的确定性直接冲突。** 若严格遵循官方，同一道题跑两次结果不同，断点续跑与复现全部失效。

---

### Qwen3-VL-30B-A3B-Instruct / Qwen3-VL-235B-A22B-Instruct

来源：HF README

| 项 | 官方值 | 等级 |
| --- | --- | --- |
| 生成参数 | **README 中没有 Generation Hyperparameters 段** | 无 |
| 其余 | 同 8B（同系列） | L5 |

**分歧 #5：同一系列内文档完整度不一致。** 8B 有明确的生成参数推荐，30B 和 235B 没有。是否把 8B 的推荐外推到同系列其他尺寸，需要定规则。

---

### RynnBrain-2B

来源：HF README + GitHub `alibaba-damo-academy/RynnBrain`

| 项 | 官方值 | 等级 | 差异 |
| --- | --- | --- | --- |
| transformers 版本 | **`transformers==4.57.1`** | L1 | 我们用 4.56.2 |
| 推理示例位置 | GitHub cookbooks（notebook） | L2 | 未查完 |
| 采样参数（GitHub） | `temperature=0.8`, `top_p=0.95`（vLLM 用法） | L2 | 与 temperature=0 冲突 |
| 视频抽帧 | 未明示 | — | 我们经 `qwen_vl_utils`（L5，非其配套库） |

---

### RynnBrain1.1-2B

来源：HF README

| 项 | 官方值 | 等级 | 差异 |
| --- | --- | --- | --- |
| transformers 版本 | **`transformers==5.2.0`** | L1 | 我们用 4.56.2 |
| 调用方式 | `processor.apply_chat_template(..., enable_thinking=False)`<br>**不使用 `qwen_vl_utils`** | L2 | **偏离**：我们的 adapter 走 `qwen_vl_utils.process_vision_info` |
| dtype | `torch.bfloat16` | L2 | 一致 |
| 视频抽帧 | 未明示 | — | 我们经 `qwen_vl_utils`（L5） |

**分歧 #6：两个 RynnBrain 要求互不兼容的 transformers 版本。**

```
RynnBrain-2B      transformers==4.57.1
RynnBrain1.1-2B   transformers==5.2.0
InternVL 系       4.56.2 实测可用
冻结代码下限       ≥4.56（因为用了 dtype= 而非 torch_dtype=）
```

### 实测结论（2026-08-14）：4.57.6 覆盖除 RynnBrain1.1-2B 外的全部

在克隆环境里逐项实测（9 项检查：Auto 类导入、InternVL 自定义代码加载、
tokenizer、`dtype=` 关键字、decord 抽帧、`model.chat()` 视频推理、
`qwen_vl_utils`、`AutoModelForImageTextToText`、eval 全链路 replay 回归）：

| 版本 | 结果 |
| --- | --- |
| 4.51.3 | ✗ 冻结代码的 `dtype=` 被透传给模型构造函数，报 unexpected keyword argument |
| 4.56.2 | ✓ 可用，但低于 RynnBrain-2B 官方要求的 4.57.1 |
| **4.57.6** | **✓ 9/9 全过，且满足 ≥4.57.1** |
| 5.2.0 | ✗ InternVL 自定义代码在 meta device 下构造即崩 |

**5.2.0 失败的确切原因：**

```
modeling_intern_vit.py:312
    dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
RuntimeError: Tensor.item() cannot be called on meta tensors
```

transformers 5.x 在 meta 设备下构造模型，而这行代码调了 `.item()`。
`low_cpu_mem_usage` 与 `device_map` 的各种组合都试过，全部失败；
把这一行改成纯 Python 计算之后**仍然报同样的错**，说明同类不兼容点不止一处 ——
打补丁是无底洞，不是一行能解决的。

顺带一提：5.x 会把模块目录名里的连字符替换掉
（`SenseNova_hyphen_SI_hyphen_1_1_hyphen_InternVL3_hyphen_2B`），
也就是说我们为 4.x 做的 `patch_local_model.py` 那个修复在 5.x 上原本是不需要的。

**部署结论：** 主环境锁 `transformers==4.57.6`，覆盖 InternVL ×3、Qwen ×3、
RynnBrain-2B、Cosmos-Reason2 等；**只有 RynnBrain1.1-2B 需要单独环境**。
不是「每个模型一套环境」，而是「一主一副」，同事的部署成本可以接受。

升级 4.56.2 → 4.57.6 后重跑全部回归与真实模型冒烟，**分数逐位不变**
（understanding 0.5、trajectory_2D 0.278795），确认该升级不影响评测结果。

---

### Cosmos3-Edge-2B

来源：HF README + model card 元数据

| 项 | 发现 |
| --- | --- |
| 模型定位 | **世界模型，不是常规 VLM**。官方描述为「Omnimodal world models」，输出文本、图像、视频、动作 |
| `library_name` | `cosmos`（不是 transformers） |
| tags | 含 `diffusers` |
| `pipeline_tag` | 无 |
| README 推理示例 | **全部是视频生成**（`export_to_video`、`result.video`），**没有任何问答式推理示例** |
| 视频抽帧 | 无官方指引 |
| 上下文 | `max_position_embeddings=131072`，而 `tokenizer_config.model_max_length=262144` |

**分歧 #7（已证实为阻塞）：无法用标准 transformers 加载。**

2026-08-14 下载权重实测：

```
ValueError: The checkpoint you are trying to load has model type `cosmos3_edge`
but Transformers does not recognize this architecture.
```

逐项排查：

| 检查项 | 结果 |
| --- | --- |
| `model_type` | `cosmos3_edge`，架构 `Cosmos3EdgeForConditionalGeneration` |
| transformers 4.57.6 是否注册 | **否**（主干分支也没有） |
| 目录内是否有 `.py` 自定义代码 | **没有** —— `trust_remote_code` 无从下手 |
| 是否有 `auto_map` | **没有** |
| PyPI 上的 `cosmos` 包 | 同名无关包（"Thin server application framework"） |
| 官方安装方式 | `pip install -e packages/cosmos-framework`，从 GitHub 源码装 |
| 模型卡里的 transformers 示例 | **没有** `AutoModel.from_pretrained` 示例 |

目录里还有 `model_index.json`、`modular_model_index.json`、`scheduler/` —— diffusers 的产物。

**也就是说 repo 里 `local_cosmos_transformers` 这条 adapter 对已发布的权重是不工作的。**
（git 历史显示作者曾从 cosmos-framework 改为 transformers，可能当时用的是别的权重版本。）

**需要团队决定：** 引入 NVIDIA 的 cosmos-framework 作为额外依赖，
还是把 Cosmos3-Edge 移出评测名单。preflight 现在会在开跑前拦下它，
不会跑到一半才整个模型失败。

**分歧 #8：上下文两个数字矛盾**（131072 vs 262144）。位置编码容量通常是硬上限，应以 131072 为准，但需确认。

---

### Cosmos-Reason2-2B

| 项 | 发现 |
| --- | --- |
| 状态 | **gated 模型**，需在 HF 接受 NVIDIA 许可并配 token 才能访问 |
| 调研结果 | 未取到配置 |

---

## 二、闭源模型（6 个）

| 模型 | Provider | 状态 |
| --- | --- | --- |
| Qwen3-VL-235B-A22B | `qwen` | 注：该模型开源，但当前配置走 API。若改本地则归入开源组 |
| qwen3.8-max | `qwen` | 待查 DashScope 文档 |
| Seed2.0-Lite | `seed` | 待查 Ark 文档 |
| GLM-5V-Turbo | `glm` | 待查智谱文档 |
| Gemini-3.1-Pro | `gemini` | 待查 |
| Gemini-3.5-Flash | `gemini` | 待查 |

闭源模型的抽帧在服务端，即便文档有说明也**未必与实际行为一致**，最终需要用
`usage.prompt_tokens` 差分反推验证：

```
1. 发纯文本            → 文本基线 token
2. 发 D₁ 秒视频        → tokens₁
3. 发 D₂ 秒视频        → tokens₂（同素材不同长度）
4. (tokens₂-tokens₁)/(D₂-D₁) = 每秒 token 消耗
```

**前置条件：** 当前代码 `raw, model_text = call_vlm(...)` 拿到原始响应后直接丢弃，
`usage` 从未写入结果文件。必须先补上这个记录。

**分歧 #9：闭源模型的抽帧策略会随服务端更新漂移，且不通知、无版本号。**
同一模型不同时间跑出的 Time EQA 成绩不具备可复现性。

---

## 三、分歧汇总

| # | 分歧 | 影响范围 |
| --- | --- | --- |
| 1 | InternVL 官方给了 32 和 8 两个抽帧数 | 3 个模型 |
| 2 | `video_sample_fps` 是自创参数，架空了官方路径 | 3 个模型 |
| 3 | SenseNova 上下文 32768 vs 16384 | 1 个模型 |
| 4 | Qwen 官方推荐 temperature=0.7，与确定性冲突 | 3 个模型 |
| 5 | Qwen 同系列文档完整度不一致 | 2 个模型 |
| 6 | **两个 RynnBrain 要求互不兼容的 transformers 版本** | 部署方案 |
| 7 | Cosmos3-Edge 是世界模型，我们的用法非官方 | 1 个模型 |
| 8 | Cosmos3-Edge 上下文 131072 vs 262144 | 1 个模型 |
| 9 | 闭源模型抽帧会随服务端漂移 | 6 个模型 |

此外还有两处我们**已知偏离官方**但尚未讨论的：

- InternVL 图片切块官方 `max_num=12`，我们用 `4`
- RynnBrain1.1-2B 官方直接用 `processor.apply_chat_template`，我们走 `qwen_vl_utils`

---

## 四、尚未完成

- Cosmos-Reason2-2B（gated，需 HF token）
- RynnBrain 系列 GitHub cookbooks 中 notebook 的具体视频参数
- VLMEvalKit 中 InternVL 的视频采样配置（官方评测参照）
- 6 个闭源模型的官方 API 文档
- 全部闭源模型的 `usage` 反推实测（需 API key）
