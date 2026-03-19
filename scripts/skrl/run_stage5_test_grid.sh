#!/usr/bin/env bash
set -euo pipefail

# Stage5-Test grid evaluation helper.
#
# Default grid:
# - distances: 20 30 40 50 60 70 80
# - speeds:    3 4 5 6 7 8 9 10 11 12
#
# Usage:
#   bash scripts/skrl/run_stage5_test_grid.sh \
#     --checkpoint /abs/path/to/best_agent.pt
#
# Optional:
#   --episodes 50
#   --num_envs 1
#   --distances "20 30 40 50 60 70 80"
#   --speeds "3 4 5 6 7 8 9 10 11 12"
#   --output_dir logs/stage5_grid_tests
#   --headless
#   --video
#   --video_length 2000
#   --device cuda:0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

TASK="Drone-Direct-Target-Touch-Vehicle-Moving-Stage5-Test-v0"
ALGORITHM="PPO"
CHECKPOINT=""
EPISODES=50
NUM_ENVS=1
DISTANCES_STR="20 30 40 50 60 70 80"
SPEEDS_STR="3 4 5 6 7 8 9 10 11 12"
OUTPUT_DIR="logs/stage5_grid_tests"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --num_envs|--env_nums)
      NUM_ENVS="$2"
      shift 2
      ;;
    --distances)
      DISTANCES_STR="$2"
      shift 2
      ;;
    --speeds)
      SPEEDS_STR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] --checkpoint is required"
  exit 1
fi

read -r -a DISTANCES <<< "${DISTANCES_STR}"
read -r -a SPEEDS <<< "${SPEEDS_STR}"

mkdir -p "${OUTPUT_DIR}"

if [[ ${#DISTANCES[@]} -eq 0 || ${#SPEEDS[@]} -eq 0 ]]; then
  echo "[ERROR] distances and speeds must not be empty"
  exit 1
fi

echo "[INFO] Task: ${TASK}"
echo "[INFO] Checkpoint: ${CHECKPOINT}"
echo "[INFO] Episodes per combo: ${EPISODES}"
echo "[INFO] Num envs: ${NUM_ENVS}"
echo "[INFO] Distances: ${DISTANCES[*]}"
echo "[INFO] Speeds: ${SPEEDS[*]}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"

for distance in "${DISTANCES[@]}"; do
  for speed in "${SPEEDS[@]}"; do
    log_file="${OUTPUT_DIR}/stage5_test_d${distance}_v${speed}.txt"
    echo
    echo "========================================"
    echo "[INFO] Stage5 grid test: distance=${distance}m speed=${speed}m/s"
    echo "[INFO] Log file: ${log_file}"
    echo "========================================"
    {
      echo "[INFO] Started at: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "[INFO] distance=${distance} speed=${speed} episodes=${EPISODES}"
      python scripts/skrl/play.py \
        --task "${TASK}" \
        --algorithm "${ALGORITHM}" \
        --num_envs "${NUM_ENVS}" \
        --checkpoint "${CHECKPOINT}" \
        --test_fixed_distance "${distance}" \
        --test_fixed_speed "${speed}" \
        --test_num_episodes "${EPISODES}" \
        "${EXTRA_ARGS[@]}"
    } 2>&1 | tee "${log_file}"
  done
done

echo
echo "[DONE] Stage5 grid test completed."
