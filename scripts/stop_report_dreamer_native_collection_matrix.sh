#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX_ID="${REPORT_NATIVE_MATRIX_ID:-native_report12_v1}"
MATRIX_DIR="$ROOT_DIR/data/report_dreamer/native/matrices/$MATRIX_ID"
PID_PATH="$MATRIX_DIR/campaign.pid"
UNIT_PATH="$MATRIX_DIR/campaign.unit"

if [[ -s "$UNIT_PATH" ]]; then
  unit_name="$(cat "$UNIT_PATH")"
  systemctl --user stop "$unit_name" >/dev/null 2>&1 || true
  echo "[native-matrix] Stopped systemd campaign: $unit_name"
elif [[ -s "$PID_PATH" ]]; then
  pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    echo "[native-matrix] Stopped campaign pid=$pid"
  fi
fi

bash "$ROOT_DIR/scripts/stop_simlingo_dashboard.sh" >/dev/null 2>&1 || true
