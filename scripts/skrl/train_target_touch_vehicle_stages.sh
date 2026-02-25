#!/usr/bin/env bash
set -euo pipefail

# Vehicle 系列多階段訓練：
# 1) Drone-Direct-Target-Touch-Vehicle-v0
# 2) Drone-Direct-Target-Touch-Vehicle-Faster-v0
# 3) Drone-Direct-Target-Touch-Vehicle-VeryFast-v0
# 4) Drone-Direct-Target-Touch-Vehicle-UltraFast-v0
#
# 用法：
#   bash scripts/skrl/train_target_touch_vehicle_stages.sh --headless --device cuda:0
#   bash scripts/skrl/train_target_touch_vehicle_stages.sh --headless --num_envs 2048 --max_iterations 2000
#
# 備註：
# - 傳入的參數會套用到所有階段。
# - 接續下一階段時，使用 logs/skrl/<experiment_dir>/<latest_run>/checkpoints 下最新的 agent_*.pt。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

COMMON_ARGS=("$@")

# 預設使用 8192 環境；若使用者已傳入 --num_envs，則尊重使用者設定。
HAS_NUM_ENVS_FLAG=0
for arg in "${COMMON_ARGS[@]}"; do
  if [[ "${arg}" == "--num_envs" || "${arg}" == --num_envs=* ]]; then
    HAS_NUM_ENVS_FLAG=1
    break
  fi
done
if [[ ${HAS_NUM_ENVS_FLAG} -eq 0 ]]; then
  COMMON_ARGS+=(--num_envs 8192)
fi

# 預設 headless；若使用者已傳入 --headless，則尊重使用者設定。
HAS_HEADLESS_FLAG=0
for arg in "${COMMON_ARGS[@]}"; do
  if [[ "${arg}" == "--headless" || "${arg}" == --headless=* ]]; then
    HAS_HEADLESS_FLAG=1
    break
  fi
done
if [[ ${HAS_HEADLESS_FLAG} -eq 0 ]]; then
  COMMON_ARGS+=(--headless)
fi

STAGE_NAMES=(
  "Stage-1 Vehicle Touch"
  "Stage-2 Vehicle Faster"
  "Stage-3 Vehicle VeryFast"
  "Stage-4 Vehicle UltraFast"
)

STAGE_TASKS=(
  "Drone-Direct-Target-Touch-Vehicle-v0"
  "Drone-Direct-Target-Touch-Vehicle-Faster-v0"
  "Drone-Direct-Target-Touch-Vehicle-VeryFast-v0"
  "Drone-Direct-Target-Touch-Vehicle-UltraFast-v0"
)

STAGE_EXP_DIRS=(
  "drone_target_touch_vehicle"
  "drone_target_touch_vehicle_faster"
  "drone_target_touch_vehicle_veryfast"
  "drone_target_touch_vehicle_ultrafast"
)

find_latest_checkpoint() {
  local exp_dir="$1"
  local log_base="${ROOT_DIR}/logs/skrl/${exp_dir}"

  if [[ ! -d "${log_base}" ]]; then
    echo "[ERROR] log directory not found: ${log_base}" >&2
    return 1
  fi

  local latest_run
  latest_run="$(ls -1dt "${log_base}"/* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_run}" ]]; then
    echo "[ERROR] no run directory found under: ${log_base}" >&2
    return 1
  fi

  local latest_ckpt
  latest_ckpt="$(ls -1t "${latest_run}"/checkpoints/agent_*.pt 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_ckpt}" ]]; then
    echo "${latest_ckpt}"
    return 0
  fi

  echo "[ERROR] no agent_*.pt checkpoint found in: ${latest_run}/checkpoints" >&2
  return 1
}

run_stage() {
  local stage_name="$1"
  local task_name="$2"
  local resume_ckpt="${3:-}"

  echo
  echo "=============================="
  echo "[INFO] Start ${stage_name}"
  echo "[INFO] Task: ${task_name}"
  if [[ -n "${resume_ckpt}" ]]; then
    echo "[INFO] Resume checkpoint: ${resume_ckpt}"
  fi
  echo "=============================="

  local cmd=(python scripts/skrl/train.py --task "${task_name}" --algorithm PPO)
  if [[ -n "${resume_ckpt}" ]]; then
    cmd+=(--checkpoint "${resume_ckpt}")
  fi
  cmd+=("${COMMON_ARGS[@]}")

  "${cmd[@]}"
}

prev_ckpt=""
final_ckpt=""
for i in "${!STAGE_TASKS[@]}"; do
  stage_name="${STAGE_NAMES[$i]}"
  task_name="${STAGE_TASKS[$i]}"
  exp_dir="${STAGE_EXP_DIRS[$i]}"

  if [[ -n "${prev_ckpt}" ]]; then
    run_stage "${stage_name}" "${task_name}" "${prev_ckpt}"
  else
    run_stage "${stage_name}" "${task_name}"
  fi

  stage_ckpt="$(find_latest_checkpoint "${exp_dir}")"
  echo "[INFO] ${stage_name} latest checkpoint: ${stage_ckpt}"

  prev_ckpt="${stage_ckpt}"
  final_ckpt="${stage_ckpt}"

  if [[ $((i + 1)) -lt ${#STAGE_TASKS[@]} ]]; then
    next_stage="${STAGE_NAMES[$((i + 1))]}"
    echo "[INFO] ${stage_name} complete -> next: ${next_stage}"
  fi
done

echo
echo "[DONE] Vehicle stages completed."
echo "[DONE] Final checkpoint: ${final_ckpt}"
