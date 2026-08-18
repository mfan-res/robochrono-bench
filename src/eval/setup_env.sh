#!/usr/bin/env bash
# 建立 RoboChrono 评测专用 conda 环境。
#
# 说明：
#  - torch 2.6.0 的 PyPI 默认 wheel 自带 CUDA 12.4，与本机驱动 550.127.08 匹配，
#    不需要额外的 download.pytorch.org 源。
#  - 走清华 PyPI 镜像：本机 NO_PROXY 已包含 .tuna.tsinghua.edu.cn，直连不过代理。
#    注意该镜像的 JSON 元数据滞后（查不到 transformers 5.x），但按精确版本号安装可用。
#  - 不改动任何已有 env。
#
# transformers 版本为什么锁 4.57.6（实测结论，2026-08-14）：
#
#   4.51.3  ✗ 冻结代码用 dtype=（4.56 才引入的 torch_dtype 新名），会被透传给
#             模型构造函数并报 unexpected keyword argument 'dtype'
#   4.56.2  ✓ 可用，但低于 RynnBrain-2B 官方要求的 4.57.1
#   4.57.6  ✓ 9/9 项检查全通过，且满足 RynnBrain-2B 的 ≥4.57.1
#   5.2.0   ✗ InternVL 自定义代码在 meta device 下构造崩溃
#   5.15.0  ✗ meta tensor 那个问题没了，但换成 InternVLChatModel 缺少
#             all_tied_weights_keys —— 5.x 基类 API 变了，自定义代码没跟上
#
# 不被 4.57.6 覆盖的模型走 Group B（transformers 5.15.0），
# 见 configs/environments.json 与 envs/groupB.txt。
#
# 依赖清单的唯一来源是 envs/groupA.txt —— 本脚本与 tools/setup_envs.sh（uv 版）
# 都从它读，避免两处版本号各写各的。
set -euo pipefail

ENV_NAME="${ENV_NAME:-robochrono}"
CONDA_BASE="$(conda info --base)"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "==> 创建 conda 环境: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.11

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> 安装依赖（来自 envs/groupA.txt）"
pip install -i "${PIP_INDEX}" -r "$(dirname "$0")/envs/groupA.txt"

echo
echo "==> 自检"
python - <<'PY'
import torch, transformers, cv2, decord, PIL, numpy, requests
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
print(f"  transformers {transformers.__version__}")
print(f"  opencv       {cv2.__version__}")
print(f"  decord       ok")
print(f"  numpy        {numpy.__version__}")
import huggingface_hub as hh; print(f"  hf_hub       {hh.__version__}")
PY

echo
echo "完成。使用: conda activate ${ENV_NAME}"
