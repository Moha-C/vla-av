#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX_ID="${REPORT_NATIVE_MATRIX_ID:-native_report12_v1}"
MATRIX_DIR="$ROOT_DIR/data/report_dreamer/native/matrices/$MATRIX_ID"
PID_PATH="$MATRIX_DIR/campaign.pid"
UNIT_PATH="$MATRIX_DIR/campaign.unit"
LOG_PATH="$MATRIX_DIR/campaign.nohup.log"
UNIT_SAFE="$(printf '%s' "$MATRIX_ID" | tr -c '[:alnum:]_.-' '-')"
UNIT_NAME="vla-av-report-native-${UNIT_SAFE}.service"

mkdir -p "$MATRIX_DIR"
if command -v systemd-run >/dev/null 2>&1 \
  && systemctl --user show-environment >/dev/null 2>&1; then
  if systemctl --user is-active --quiet "$UNIT_NAME"; then
    main_pid="$(systemctl --user show "$UNIT_NAME" -p MainPID --value)"
    echo "[native-matrix] Campaign already running: unit=$UNIT_NAME pid=$main_pid"
    echo "[native-matrix] Watch: bash scripts/watch_report_dreamer_native_collection_matrix.sh"
    exit 0
  fi

  systemctl --user reset-failed "$UNIT_NAME" >/dev/null 2>&1 || true
  run_args=(
    systemd-run --user
    --unit="$UNIT_NAME"
    --collect
    --property="WorkingDirectory=$ROOT_DIR"
    --property="StandardOutput=append:$LOG_PATH"
    --property="StandardError=append:$LOG_PATH"
    --setenv="REPORT_NATIVE_MATRIX_ID=$MATRIX_ID"
    --setenv="REPORT_NATIVE_MAX_ATTEMPTS=${REPORT_NATIVE_MAX_ATTEMPTS:-2}"
    --setenv="REPORT_NATIVE_INTER_RUN_DELAY=${REPORT_NATIVE_INTER_RUN_DELAY:-8}"
    --setenv="CARLA_QUALITY=${CARLA_QUALITY:-Low}"
    --setenv="SIMLINGO_VIEW_MODE=${SIMLINGO_VIEW_MODE:-chase}"
    --setenv="SIMLINGO_VIEW_WIDTH=${SIMLINGO_VIEW_WIDTH:-1280}"
    --setenv="SIMLINGO_VIEW_HEIGHT=${SIMLINGO_VIEW_HEIGHT:-720}"
    --setenv="SIMLINGO_RECORD=${SIMLINGO_RECORD:-0}"
    --setenv="SIMLINGO_PLAYBACK_AFTER=${SIMLINGO_PLAYBACK_AFTER:-0}"
    --setenv="PORT=${PORT:-2000}"
    --setenv="TM_PORT=${TM_PORT:-8000}"
    --setenv="DISPLAY=${DISPLAY:-:1}"
  )
  if [[ -n "${XAUTHORITY:-}" ]]; then
    run_args+=(--setenv="XAUTHORITY=$XAUTHORITY")
  fi
  if [[ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
    run_args+=(--setenv="DBUS_SESSION_BUS_ADDRESS=$DBUS_SESSION_BUS_ADDRESS")
  fi
  run_args+=(/bin/bash "$ROOT_DIR/scripts/run_report_dreamer_native_collection_matrix.sh")

  "${run_args[@]}" >/dev/null
  printf '%s\n' "$UNIT_NAME" > "$UNIT_PATH"
  sleep 1
  if ! systemctl --user is-active --quiet "$UNIT_NAME"; then
    echo "[native-matrix] systemd campaign failed to start. Log: $LOG_PATH" >&2
    tail -80 "$LOG_PATH" >&2 || true
    exit 1
  fi
  campaign_pid="$(systemctl --user show "$UNIT_NAME" -p MainPID --value)"
  printf '%s\n' "$campaign_pid" > "$PID_PATH"
  echo "[native-matrix] Started durable systemd campaign: unit=$UNIT_NAME pid=$campaign_pid"
  echo "[native-matrix] It survives terminal closure and resumes from accepted runs."
  echo "[native-matrix] Watch: bash scripts/watch_report_dreamer_native_collection_matrix.sh"
  echo "[native-matrix] Log: $LOG_PATH"
  exit 0
fi

if [[ -s "$PID_PATH" ]]; then
  old_pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[native-matrix] Campaign already running: pid=$old_pid matrix=$MATRIX_ID"
    echo "[native-matrix] Watch: bash scripts/watch_report_dreamer_native_collection_matrix.sh"
    exit 0
  fi
fi

nohup setsid --wait env \
  REPORT_NATIVE_MATRIX_ID="$MATRIX_ID" \
  REPORT_NATIVE_MAX_ATTEMPTS="${REPORT_NATIVE_MAX_ATTEMPTS:-2}" \
  REPORT_NATIVE_INTER_RUN_DELAY="${REPORT_NATIVE_INTER_RUN_DELAY:-8}" \
  bash "$ROOT_DIR/scripts/run_report_dreamer_native_collection_matrix.sh" \
  >> "$LOG_PATH" 2>&1 < /dev/null &
campaign_pid=$!
printf '%s\n' "$campaign_pid" > "$PID_PATH"

sleep 1
if ! kill -0 "$campaign_pid" 2>/dev/null; then
  echo "[native-matrix] Campaign failed to start. Log: $LOG_PATH" >&2
  tail -80 "$LOG_PATH" >&2 || true
  exit 1
fi

echo "[native-matrix] Started durable campaign: pid=$campaign_pid matrix=$MATRIX_ID"
echo "[native-matrix] It will survive terminal closure and resume from accepted runs."
echo "[native-matrix] Watch: bash scripts/watch_report_dreamer_native_collection_matrix.sh"
echo "[native-matrix] Log: $LOG_PATH"
