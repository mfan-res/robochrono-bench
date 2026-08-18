#!/usr/bin/env bash
# 用 uv 建出 configs/environments.json 里声明的各套 python 环境。
#
# 为什么是 uv 而不是 conda：
#   - conda 那套环境 6.0 GB，迁到新集群要么重装要么打包传输
#   - uv 解析 + 安装是秒级到分钟级，conda solver 常常几分钟起步
#   - 全部 17 个依赖都在 PyPI 上（已核实，没有 conda-only 的包）
#   - uv 的下载缓存跨环境共享 —— 两套环境的 torch 是同一个版本，只下一次
#
# 前提：集群能访问 PyPI（或镜像，见下面的 UV_INDEX）。
# 如果集群完全离线，见 docs/environments.md 里的离线方案。
#
#   bash tools/setup_envs.sh            # 建所有环境
#   bash tools/setup_envs.sh groupB     # 只建一个

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "没找到 uv。安装：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# 国内集群可以指到镜像；留空则用默认 PyPI
UV_INDEX="${UV_INDEX:-}"
INDEX_ARGS=()
[ -n "$UV_INDEX" ] && INDEX_ARGS=(--index-url "$UV_INDEX")

PYTHON_VERSION=3.11
VENV_ROOT="${ROBOCHRONO_VENVS:-$PWD/.venvs}"
TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(groupA groupB)
fi

mkdir -p "$VENV_ROOT"

for env_name in "${TARGETS[@]}"; do
    req="envs/${env_name}.txt"
    if [ ! -f "$req" ]; then
        echo "没有 $req，跳过 $env_name" >&2
        continue
    fi
    venv="$VENV_ROOT/$env_name"
    echo "=== $env_name -> $venv ==="
    uv venv --python "$PYTHON_VERSION" "$venv"
    VIRTUAL_ENV="$venv" uv pip install "${INDEX_ARGS[@]}" -r "$req"
    echo "  $("$venv/bin/python" -c 'import transformers,torch;print(f"transformers {transformers.__version__}  torch {torch.__version__}")')"
done

echo
echo "把解释器路径写进 configs/environments.json："
python - "$VENV_ROOT" "${TARGETS[@]}" <<'PY'
import json, os, sys
from pathlib import Path

venv_root, *names = sys.argv[1:]
path = Path("configs/environments.json")
config = json.loads(path.read_text(encoding="utf-8"))
for name in names:
    interpreter = Path(venv_root) / name / "bin/python"
    if name not in config["envs"] or not interpreter.exists():
        continue
    # 尽量写成相对 eval/ 的路径 —— 绝对路径提交到 git 后在别人机器上必然失效。
    # venv 在仓库外时（ROBOCHRONO_VENVS 指到别处）才退回绝对路径。
    # 纯词法运算，不能 resolve：venv 的 bin/python 是符号链接，
    # 跟过去就变成基础解释器了，venv 的包全部失效。
    value = os.path.relpath(os.path.abspath(interpreter), os.getcwd())
    if value.startswith(".."):
        value = os.path.abspath(interpreter)
    config["envs"][name]["python"] = value
    print(f"  {name}: {value}")
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo
echo "验证：python -m robochrono preflight"
