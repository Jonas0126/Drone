#!/usr/bin/env bash
set -euo pipefail

TARGET_PANE="${1:-0.0}"
shift || true

NEXT_CMD="${*:-}"
POLL_INTERVAL="${TMUX_WATCH_POLL_SEC:-5}"
STABLE_CHECKS="${TMUX_WATCH_STABLE_CHECKS:-2}"

if [[ -z "${NEXT_CMD}" ]]; then
  echo "Usage: $0 <target-pane> <next-command>"
  echo "Example: $0 0.0 \"python scripts/skrl/train.py --task ...\""
  exit 1
fi

if ! tmux display-message -p -t "${TARGET_PANE}" "#{pane_id}" >/dev/null 2>&1; then
  echo "[ERROR] tmux pane '${TARGET_PANE}' not found"
  exit 1
fi

echo "[INFO] Watching tmux pane ${TARGET_PANE} ..."
echo "[INFO] Will run after current python exits: ${NEXT_CMD}"

idle_hits=0
while true; do
  current_cmd="$(tmux display-message -p -t "${TARGET_PANE}" "#{pane_current_command}")"

  if [[ "${current_cmd}" == python* ]]; then
    idle_hits=0
  else
    ((idle_hits+=1))
    if (( idle_hits >= STABLE_CHECKS )); then
      break
    fi
  fi

  sleep "${POLL_INTERVAL}"
done

ts="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] ${ts} detected pane idle (command=${current_cmd}). Sending next command..."
tmux send-keys -t "${TARGET_PANE}" "${NEXT_CMD}" C-m

echo "[INFO] Next command dispatched to ${TARGET_PANE}."
