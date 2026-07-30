#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/experiments/maram_dreamer_carla/full_carla_gpu_retry}"
LOG_DIR="${MARAM_LOCAL_LOG_DIR:-$ROOT_DIR/logs/maram_dreamer}"
PID_FILE="${MARAM_LOCAL_PID_FILE:-$LOG_DIR/full_carla_gpu_retry.pid}"
RUN_LOG="${MARAM_LOCAL_RUN_LOG:-$LOG_DIR/full_carla_gpu_retry.log}"
CARLA_LOG="${MARAM_LOCAL_CARLA_LOG:-$LOG_DIR/carla_server_2000.log}"
CARLA_PID_FILE="${MARAM_LOCAL_CARLA_PID_FILE:-$LOG_DIR/carla_server_2000.pid}"

CONDA_ENV="${MARAM_DREAMER_CONDA_ENV:-vla-av}"
CARLA_ROOT="${CARLA_ROOT:-$HOME/carla_simulator}"
CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2000}"
CARLA_QUALITY="${CARLA_QUALITY:-Low}"
CARLA_WAIT_SECONDS="${CARLA_WAIT_SECONDS:-240}"
CLEAN_OUT="${CLEAN_OUT:-1}"
RESTART_EXISTING="${RESTART_EXISTING:-1}"

EPISODES="${EPISODES:-30}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-600}"
WM_EPOCHS="${WM_EPOCHS:-30}"
RL_ITERS="${RL_ITERS:-1000}"
RUN_COLLECT="${RUN_COLLECT:-1}"
RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_RL="${RUN_RL:-1}"

mkdir -p "$LOG_DIR"

if [[ "$CLEAN_OUT" == "1" ]]; then
  echo "[maram-local-start] cleaning output: $OUT_DIR"
  rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    if [[ "$RESTART_EXISTING" == "1" ]]; then
      echo "[maram-local-start] stopping previous training: pid=$old_pid"
      kill "$old_pid" || true
      sleep 2
    else
      echo "[maram-local-start] previous training still running: pid=$old_pid"
      echo "[maram-local-start] stop it manually before launching a duplicate."
      exit 1
    fi
  fi
fi

carla_ready() {
  conda run -n "$CONDA_ENV" python -c "import carla; client = carla.Client('$CARLA_HOST', int('$CARLA_PORT')); client.set_timeout(5.0); client.get_world()" >/dev/null 2>&1
}

if carla_ready; then
  echo "[maram-local-start] CARLA already reachable on $CARLA_HOST:$CARLA_PORT"
else
  if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
    echo "[maram-local-start] CARLA executable not found: $CARLA_ROOT/CarlaUE4.sh" >&2
    exit 1
  fi
  echo "[maram-local-start] starting CARLA on port $CARLA_PORT"
  nohup setsid bash -c '
    set -euo pipefail
    cd "$1"
    exec ./CarlaUE4.sh \
      -quality-level="$2" \
      -nosound \
      -carla-rpc-port="$3" \
      -graphicsadapter=0 \
      -RenderOffScreen
  ' _ "$CARLA_ROOT" "$CARLA_QUALITY" "$CARLA_PORT" > "$CARLA_LOG" 2>&1 &
  echo $! > "$CARLA_PID_FILE"

  deadline=$((SECONDS + CARLA_WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if carla_ready; then
      echo "[maram-local-start] CARLA is ready"
      break
    fi
    sleep 5
  done
  if ! carla_ready; then
    echo "[maram-local-start] CARLA did not become ready within ${CARLA_WAIT_SECONDS}s" >&2
    echo "[maram-local-start] check: $CARLA_LOG" >&2
    exit 1
  fi
fi

echo "[maram-local-start] launching training"
nohup setsid bash -c '
  set -euo pipefail
  cd "$1"
  EPISODES="$2" \
  STEPS_PER_EPISODE="$3" \
  WM_EPOCHS="$4" \
  RL_ITERS="$5" \
  OUT_DIR="$6" \
  CARLA_HOST="$7" \
  CARLA_PORT="$8" \
  MARAM_DREAMER_CONDA_ENV="$9" \
  RUN_COLLECT="${10}" \
  RUN_PRETRAIN="${11}" \
  RUN_RL="${12}" \
    bash scripts/no_sleep_run.sh bash scripts/run_maram_dreamer_carla_training.sh
' _ \
  "$ROOT_DIR" \
  "$EPISODES" \
  "$STEPS_PER_EPISODE" \
  "$WM_EPOCHS" \
  "$RL_ITERS" \
  "$OUT_DIR" \
  "$CARLA_HOST" \
  "$CARLA_PORT" \
  "$CONDA_ENV" \
  "$RUN_COLLECT" \
  "$RUN_PRETRAIN" \
  "$RUN_RL" \
  > "$RUN_LOG" 2>&1 &

echo $! > "$PID_FILE"

echo
echo "[maram-local-start] launched"
echo "[maram-local-start] pid=$(cat "$PID_FILE")"
echo "[maram-local-start] out_dir=$OUT_DIR"
echo "[maram-local-start] run_log=$RUN_LOG"
echo "[maram-local-start] carla_log=$CARLA_LOG"
echo "[maram-local-start] carla_pid_file=$CARLA_PID_FILE"
