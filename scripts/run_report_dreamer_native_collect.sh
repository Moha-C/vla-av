#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SIMLINGO_PYTHON:-$HOME/miniconda3/envs/simlingo/bin/python}"
ROUTE_ID="${ROUTE_ID:-57}"
SEED="${SEED:-$((100000 + RANDOM % 899999))}"
RUN_ID="${REPORT_NATIVE_RUN_ID:-$(date +%Y%m%d_%H%M%S)_route_${ROUTE_ID}_seed_${SEED}}"
RUN_DIR="${REPORT_NATIVE_RUN_DIR:-$ROOT_DIR/data/report_dreamer/native/runs/$RUN_ID}"
TRACE_PATH="$RUN_DIR/trace.jsonl"
COLLISION_PATH="$RUN_DIR/collision_events.jsonl"
START_MARKER="$RUN_DIR/started.marker"

mkdir -p "$RUN_DIR"
export ROUTE_ID SEED
export SIMLINGO_REPORT_NATIVE_TRACE="$TRACE_PATH"
export SIMLINGO_COLLISION_EVENT_PATH="$COLLISION_PATH"
export SIMLINGO_REPORT_DREAMER_MODE="off"
export SIMLINGO_DREAMER_GUARD="0"
export SIMLINGO_DREAMER_RUNTIME=""
export SIMLINGO_CARDREAMER_MODE="off"
export SIMLINGO_RECORD="${SIMLINGO_RECORD:-0}"
export SIMLINGO_PLAYBACK_AFTER="${SIMLINGO_PLAYBACK_AFTER:-0}"

if [[ -n "${ROUTE_FILE:-}" ]]; then
  ROUTE_LABEL="$(basename "$ROUTE_FILE" .xml)"
else
  ROUTE_LABEL="bench2drive_${ROUTE_ID}"
fi
RESULT_JSON="${SIMLINGO_OUT_DIR:-$ROOT_DIR/logs/simlingo_eval}/results_${ROUTE_LABEL}_seed_${SEED}.json"

echo "[report-native] native SimLingo controls only"
echo "[report-native] trace=$TRACE_PATH"
echo "[report-native] expected_result=$RESULT_JSON"
: > "$START_MARKER"
set +e
bash "$ROOT_DIR/scripts/run_simlingo_with_pov.sh"
RUN_EXIT=$?
set -e

if [[ -s "$RESULT_JSON" && "$RESULT_JSON" -nt "$START_MARKER" && -s "$TRACE_PATH" ]]; then
  "$PYTHON" "$ROOT_DIR/scripts/finalize_report_native_trace.py" \
    --trace "$TRACE_PATH" \
    --result "$RESULT_JSON"
else
  echo "[report-native] no fresh finalized Bench2Drive result; this trace is not training-eligible" >&2
fi
exit "$RUN_EXIT"
