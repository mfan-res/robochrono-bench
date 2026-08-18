#!/usr/bin/env bash
# 用冻结的 test/ 脚本 + 真实本地模型，把九个任务各跑 N 道题。
#
# 这是阶段 0 的门禁：证明「规范化后的数据 + 本机环境 + 真实权重」能端到端跑通。
# 刻意用 test/ 下的原脚本而不是新框架 —— 先确立基线，再谈重构。
#
# 用法: eval/tools/smoke_all.sh [题数，默认 1]
set -uo pipefail

LIMIT="${1:-1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/mnt/public/users/wbcd/anaconda3/envs/robochrono/bin/python}"
MODEL="${MODEL:-${REPO}/eval/models/SenseNova-SI-1_1-InternVL3-2B}"
CONFIG="${CONFIG:-${REPO}/eval/configs/config_smoke.json}"
PROVIDER="${PROVIDER:-local_sensenova_si_1_1_internvl3_2b}"
OUT="${OUT:-${REPO}/eval/results/smoke}"

UND="${REPO}/eval/datasets/QA/understanding/stack_cubes"
PLN="${REPO}/eval/datasets/QA/planning/stack_cubes"

mkdir -p "${OUT}"
cd "${REPO}"

# task_name | script | input
TASKS=(
  "time|time_eqa_glm_test_multi.py|${UND}/time_vqa.json"
  "understanding|understanding_glm_test.py|${UND}/understanding_vqa.json"
  "left_right|left_right_glm_test.py|${UND}/left_right_vqa.json"
  "image_in_video|image_in_video_glm_test.py|${UND}/image_in_video_vqa.json"
  "planning|planning_glm_test.py|${PLN}/planning_vqa.json"
  "planning_2|planning_2_glm_test.py|${PLN}/planning_2_vqa.json"
  "step_order|step_order_glm_test.py|${PLN}/step_order_vqa.json"
  "trajectory_2D|trajectory_glm_test.py|${PLN}/trajectory_qa_2d.json"
  "trajectory_3D|trajectory_glm_test.py|${PLN}/trajectory_qa_3d.json"
)

printf '%-16s %-8s %-9s %-8s %s\n' task status answered errors note
printf -- '---------------------------------------------------------------\n'

rc=0
for row in "${TASKS[@]}"; do
  IFS='|' read -r name script input <<< "${row}"
  log="${OUT}/${name}.log"
  result="${OUT}/${name}.json"

  CUDA_VISIBLE_DEVICES=0 "${PY}" "test/${script}" \
      --config "${CONFIG}" \
      --provider "${PROVIDER}" \
      --model "${MODEL}" \
      --input "${input}" \
      --output "${result}" \
      --limit "${LIMIT}" \
      --overwrite > "${log}" 2>&1
  exit_code=$?

  if [ ! -f "${result}" ]; then
    printf '%-16s %-8s %-9s %-8s %s\n' "${name}" "FAIL" "-" "-" "$(tail -1 "${log}" | cut -c1-60)"
    rc=1
    continue
  fi

  read -r answered errors note < <(
    "${PY}" - "${result}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get("summary", {})
rows = d.get("results", [])
err = next((r.get("error") for r in rows if r.get("error")), "")
print(s.get("answered", "?"), s.get("errors", "?"), (err or "-")[:60])
PY
  )

  status="OK"
  if [ "${exit_code}" -ne 0 ] || [ "${errors}" != "0" ]; then status="FAIL"; rc=1; fi
  printf '%-16s %-8s %-9s %-8s %s\n' "${name}" "${status}" "${answered}" "${errors}" "${note}"
done

printf -- '---------------------------------------------------------------\n'
[ "${rc}" -eq 0 ] && echo "九个任务全部跑通" || echo "存在失败，详见 ${OUT}/*.log"
exit "${rc}"
