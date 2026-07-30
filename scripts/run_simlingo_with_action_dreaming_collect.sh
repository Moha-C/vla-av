#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECT_DIR="${ACTION_DREAMING_OUT_DIR:-$ROOT_DIR/logs/action_dreaming_collect}"
RUN_ID="${ACTION_DREAMING_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STATUS_PATH="${SIMLINGO_DREAMER_STATUS_PATH:-$ROOT_DIR/logs/simlingo_eval/dreamer_guard_status.json}"
SAMPLE_INTERVAL="${ACTION_DREAMING_SAMPLE_INTERVAL:-0.25}"
TRACE_PATH="${ACTION_DREAMING_TRACE_PATH:-$COLLECT_DIR/action_dreaming_${RUN_ID}.jsonl}"

mkdir -p "$COLLECT_DIR"
export SIMLINGO_DREAMER_STATUS_PATH="$STATUS_PATH"
export ACTION_DREAMING_TRACE_PATH="$TRACE_PATH"

cleanup() {
  if [[ -n "${COLLECT_PID:-}" ]] && kill -0 "$COLLECT_PID" 2>/dev/null; then
    kill "$COLLECT_PID" 2>/dev/null || true
    wait "$COLLECT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[action-dreaming] passive decision trace: $TRACE_PATH"
python -u "$ROOT_DIR/scripts/action_dreaming_collect_normal.py" \
  --status-path "$STATUS_PATH" \
  --output "$TRACE_PATH" \
  --interval "$SAMPLE_INTERVAL" \
  --route-id "${ROUTE_ID:-}" \
  --route-file "${ROUTE_FILE:-}" \
  --town "${TOWN:-}" \
  --seed "${SEED:-}" &
COLLECT_PID=$!
echo "$COLLECT_PID" > "$COLLECT_DIR/latest_collector.pid"
echo "$TRACE_PATH" > "$COLLECT_DIR/latest_trace.txt"

bash "$ROOT_DIR/scripts/run_simlingo_with_pov.sh"
