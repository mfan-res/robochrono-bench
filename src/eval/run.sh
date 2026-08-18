#!/usr/bin/env bash
# RoboChrono 评测一键入口。
#
#   ./run.sh                      自检 → 估算 → 跑完整矩阵 → 汇总 → 打包
#   ./run.sh --shard 1/4          多机分工，第 1 台
#   ./run.sh --only api           只跑 API 模型（不需要 GPU）
#   ./run.sh --only local --gpus 8
#   ./run.sh --limit-items 4      冒烟：每个任务只跑 4 题
#
# 任何一步失败都会停下，不会带着错误继续跑几个小时。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PY:-python}"
RESULTS="${RESULTS:-results}"
PACK_OUT="${PACK_OUT:-robochrono-results.tar.gz}"
ARGS=("$@")

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

banner "1/5 自检"
if ! "${PY}" -m robochrono --results-dir "${RESULTS}" preflight "${ARGS[@]}"; then
  echo
  echo "自检未通过，已中止。修复上面标 FAIL 的项目后重跑。"
  exit 1
fi

banner "2/5 估算调用量"
"${PY}" -m robochrono --results-dir "${RESULTS}" estimate "${ARGS[@]}"
echo
read -r -p "继续执行？[y/N] " reply
[[ "${reply}" =~ ^[Yy]$ ]] || { echo "已取消。"; exit 0; }

banner "3/5 执行矩阵"
"${PY}" -m robochrono --results-dir "${RESULTS}" matrix "${ARGS[@]}"

banner "4/5 汇总"
"${PY}" -m robochrono --results-dir "${RESULTS}" report "${RESULTS}" --csv "${RESULTS}/summary.csv"

banner "5/5 打包"
"${PY}" -m robochrono --results-dir "${RESULTS}" pack -o "${PACK_OUT}"

echo
echo "完成。把 ${PACK_OUT} 回传即可。"
