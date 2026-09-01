#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${DREAMER_CURRICULUM_CONDA_ENV:-simlingo}"
RUN_ID="${DREAMER_CURRICULUM_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$ROOT_DIR/logs/dreamer_curriculum/$RUN_ID"
PID_FILE="$ROOT_DIR/logs/dreamer_curriculum/latest_campaign.pid"
UNIT_FILE="$ROOT_DIR/logs/dreamer_curriculum/latest_campaign.unit"
LOG_FILE="$RUN_DIR/campaign.log"
UNIT="vla-av-dreamer-curriculum-${RUN_ID//[^A-Za-z0-9_.-]/-}.service"

mkdir -p "$RUN_DIR"
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && (( old_pid > 0 )) && kill -0 "$old_pid" 2>/dev/null; then
    echo "[dreamer-curriculum] already running: pid=$old_pid" >&2
    exit 1
  fi
fi

bash "$ROOT_DIR/scripts/stop_simlingo_dashboard.sh"
export DREAMER_CURRICULUM_RUN_ID="$RUN_ID"
export DREAMER_CURRICULUM_RUN_DIR="$RUN_DIR"
: >"$LOG_FILE"
systemd-run --user \
  --unit "$UNIT" \
  --collect \
  --quiet \
  --property "WorkingDirectory=$ROOT_DIR" \
  --setenv "DISPLAY=${DISPLAY:-:1}" \
  --setenv "XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}" \
  --setenv "DREAMER_CURRICULUM_RUN_ID=$RUN_ID" \
  --setenv "DREAMER_CURRICULUM_RUN_DIR=$RUN_DIR" \
  --setenv "DREAMER_CURRICULUM_CONDA_ENV=$CONDA_ENV" \
  --setenv "CONDA_SH=$CONDA_SH" \
  --setenv "DREAMER_CURRICULUM_DEVICE=${DREAMER_CURRICULUM_DEVICE:-auto}" \
  --setenv "DREAMER_CURRICULUM_CARLA_QUALITY=${DREAMER_CURRICULUM_CARLA_QUALITY:-Low}" \
  --setenv "DREAMER_CURRICULUM_ROUTE_RETRIES=${DREAMER_CURRICULUM_ROUTE_RETRIES:-1}" \
  "$ROOT_DIR/scripts/run_dreamer_curriculum_worker.sh" "$@"
pid="0"
for _ in $(seq 1 20); do
  pid="$(systemctl --user show "$UNIT" --property MainPID --value 2>/dev/null || echo 0)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 0 )); then
    break
  fi
  sleep 0.25
done
echo "$pid" > "$PID_FILE"
echo "$UNIT" > "$UNIT_FILE"
echo "$RUN_DIR" > "$ROOT_DIR/logs/dreamer_curriculum/latest_campaign.txt"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || (( pid <= 0 )); then
  echo "[dreamer-curriculum] service did not remain active; inspect $LOG_FILE" >&2
  tail -40 "$LOG_FILE" >&2 || true
  exit 1
fi
echo "[dreamer-curriculum] started pid=$pid"
echo "[dreamer-curriculum] unit=$UNIT"
echo "[dreamer-curriculum] run_dir=$RUN_DIR"
echo "[dreamer-curriculum] watch: bash scripts/watch_dreamer_curriculum_training.sh"
