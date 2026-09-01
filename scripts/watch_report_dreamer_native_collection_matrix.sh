#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX_ID="${REPORT_NATIVE_MATRIX_ID:-native_report12_v1}"
MATRIX_DIR="$ROOT_DIR/data/report_dreamer/native/matrices/$MATRIX_ID"
PID_PATH="$MATRIX_DIR/campaign.pid"
UNIT_PATH="$MATRIX_DIR/campaign.unit"
STATUS_PATH="$MATRIX_DIR/status.env"
SUMMARY_PATH="$MATRIX_DIR/summary.tsv"
LOG_PATH="$MATRIX_DIR/campaign.nohup.log"

echo "=== Report Dreamer native collection: $MATRIX_ID ==="
if [[ -s "$UNIT_PATH" ]]; then
  unit_name="$(cat "$UNIT_PATH")"
  if systemctl --user is-active --quiet "$unit_name" 2>/dev/null; then
    main_pid="$(systemctl --user show "$unit_name" -p MainPID --value)"
    echo "process: RUNNING unit=$unit_name pid=$main_pid"
  else
    unit_state="$(systemctl --user is-active "$unit_name" 2>/dev/null || true)"
    echo "process: STOPPED unit=$unit_name state=${unit_state:-unknown}"
  fi
elif [[ -s "$PID_PATH" ]]; then
  pid="$(cat "$PID_PATH")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "process: RUNNING pid=$pid"
  else
    echo "process: STOPPED (last pid=${pid:-unknown})"
  fi
else
  echo "process: not started through durable launcher"
fi

if [[ -s "$STATUS_PATH" ]]; then
  echo
  echo "--- status ---"
  sed 's/^/  /' "$STATUS_PATH"
fi

if [[ -s "$SUMMARY_PATH" ]]; then
  accepted="$(awk -F '\t' 'NR > 1 && $6 ~ /^ACCEPTED/ { count += 1 } END { print count + 0 }' "$SUMMARY_PATH")"
  total="$(sed -n 's/^total=//p' "$STATUS_PATH" 2>/dev/null | tail -1)"
  total="${total:-12}"
  echo
  echo "accepted: $accepted/$total"
  echo "--- summary ---"
  column -t -s $'\t' "$SUMMARY_PATH" 2>/dev/null || cat "$SUMMARY_PATH"
fi

if [[ -s "$LOG_PATH" ]]; then
  echo
  echo "--- latest output ---"
  tail -40 "$LOG_PATH"
fi
