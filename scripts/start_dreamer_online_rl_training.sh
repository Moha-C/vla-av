#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${DREAMER_ONLINE_RL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$ROOT_DIR/logs/dreamer_online_rl/$RUN_ID"
LATEST_FILE="$ROOT_DIR/logs/dreamer_online_rl/latest_training.txt"
PID_FILE="$RUN_DIR/training.pid"
LOG_FILE="$RUN_DIR/training_stdout.log"

mkdir -p "$RUN_DIR" "$(dirname "$LATEST_FILE")"
echo "$RUN_DIR" > "$LATEST_FILE"

if [[ -s "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && ps -p "$OLD_PID" >/dev/null 2>&1; then
    echo "[dreamer-online-rl] already running pid=$OLD_PID"
    echo "[dreamer-online-rl] run_dir=$RUN_DIR"
    exit 0
  fi
fi

nohup bash -c 'cd "$1" && exec python3 "$1/scripts/run_dreamer_online_rl_training.py" --run-id "$2"' \
  _ "$ROOT_DIR" "$RUN_ID" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "[dreamer-online-rl] launched"
echo "[dreamer-online-rl] pid=$(cat "$PID_FILE")"
echo "[dreamer-online-rl] run_dir=$RUN_DIR"
echo "[dreamer-online-rl] log=$LOG_FILE"
echo "[dreamer-online-rl] watch: bash scripts/watch_dreamer_online_rl_training.sh"
