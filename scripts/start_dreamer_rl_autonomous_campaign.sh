#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${DREAMER_RL_CAMPAIGN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$ROOT_DIR/logs/dreamer_rl_campaign/$RUN_ID"
LATEST_FILE="$ROOT_DIR/logs/dreamer_rl_campaign/latest_campaign.txt"
PID_FILE="$RUN_DIR/campaign.pid"
LOG_FILE="$RUN_DIR/campaign_stdout.log"

mkdir -p "$RUN_DIR" "$(dirname "$LATEST_FILE")"
echo "$RUN_DIR" > "$LATEST_FILE"

if [[ -s "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && ps -p "$OLD_PID" >/dev/null 2>&1; then
    echo "[dreamer-complement-campaign] already running pid=$OLD_PID"
    echo "[dreamer-complement-campaign] run_dir=$RUN_DIR"
    exit 0
  fi
fi

nohup setsid python3 "$ROOT_DIR/scripts/run_dreamer_rl_autonomous_campaign.py" \
  --run-id "$RUN_ID" \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "[dreamer-complement-campaign] launched"
echo "[dreamer-complement-campaign] pid=$(cat "$PID_FILE")"
echo "[dreamer-complement-campaign] run_dir=$RUN_DIR"
echo "[dreamer-complement-campaign] watch: bash scripts/watch_dreamer_rl_autonomous_campaign.sh"
