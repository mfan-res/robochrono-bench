# models —— 本地模型权重

**34 GB，不进 git**（`.gitignore` 里 `models/*` 加 `!models/README.md`）。
从旧仓库 `eval/models/` 整体搬入，同盘 `mv`，内容一字未动。

| 目录 | 体积 | 权重布局 |
| --- | ---: | --- |
| `Qwen3-VL-8B-Instruct` | 17 G | 顶层 4 个 safetensors 分片 |
| `Cosmos3-Edge` | 8.6 G | **多模块**：`vision_encoder/` `vae/` `transformer/` 各自带权重 |
| `RynnBrain-2B` | 4.6 G | 顶层单份 |
| `SenseNova-SI-1_1-InternVL3-2B` | 3.9 G | 顶层单份 + 随附的 `modeling_*.py` |

> Cosmos3-Edge 的布局与其余三个不同 —— 只在顶层找 `*.safetensors` 会数出 0 个
> 而误判成「权重缺失」。检查完整性要递归找。

## 怎么被引用

路径写在 `src/eval/configs/providers.json` 里，**相对 `src/eval/`**，
形如 `../../models/<名字>`。

搬入时改过一次：原来相对旧仓库的 `eval/` 写成 `models/<名字>`，
根变了就得跟着改 —— 这类相对路径不会报错，只会「找不到权重」。

## 不在这里的

推理需要的 Python 环境（旧仓库 `eval/.venvs/`，12 GB）**没有搬**。
各模型依赖的 transformers 版本不同，`src/eval/envs/` 与
`configs/environments.json` 记录了对应关系，用 `setup_env.sh` 重建。
