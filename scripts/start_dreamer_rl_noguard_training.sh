#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KIND="${DREAMER_RL_KIND:-ppo}"            # ppo | sdbs
RUN_ID="${DREAMER_RL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${DREAMER_RL_CONDA_ENV:-simlingo}"
DEVICE="${DREAMER_RL_DEVICE:-cuda}"
EPISODES="${DREAMER_RL_EPISODES:-100}"
TOWN="${DREAMER_RL_TOWN:-Town10HD}"
CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2000}"
CARLA_ROOT="${CARLA_ROOT:-$HOME/carla_simulator}"
CARLA_QUALITY="${CARLA_QUALITY:-Low}"
CARLA_WAIT_SECONDS="${CARLA_WAIT_SECONDS:-240}"
AUTO_START_CARLA="${AUTO_START_CARLA:-1}"
MAX_EPISODE_STEPS="${DREAMER_RL_MAX_EPISODE_STEPS:-700}"
ROLLOUT_SIZE="${DREAMER_RL_ROLLOUT_SIZE:-1024}"
EVAL_INTERVAL="${DREAMER_RL_EVAL_INTERVAL:-25}"
RESTART_EXISTING="${RESTART_EXISTING:-0}"
MOCK="${DREAMER_RL_MOCK:-0}"
INSTALL_LATEST="${DREAMER_RL_INSTALL_LATEST:-0}"
FOREGROUND="${DREAMER_RL_FOREGROUND:-0}"

case "$KIND" in
  ppo)
    REPO_DIR="$ROOT_DIR/experiments/dreamer_ppo_carla"
    CKPT_ROOT="$ROOT_DIR/external/simlingo/checkpoints/dreamer_ppo_rl_noguard"
    TRAIN_ARGS=()
    LOG_NAME="ppo_rl_noguard.csv"
    ;;
  sdbs)
    REPO_DIR="$ROOT_DIR/experiments/dreamer_ppo_carla_sdbs_fresh"
    CKPT_ROOT="$ROOT_DIR/external/simlingo/checkpoints/dreamer_sdbs_rl_noguard"
    TRAIN_ARGS=(--sdbs)
    LOG_NAME="sdbs_rl_noguard.csv"
    ;;
  *)
    echo "[dreamer-rl-start] DREAMER_RL_KIND must be 'ppo' or 'sdbs', got: $KIND" >&2
    exit 1
    ;;
esac

if [[ "$MOCK" == "1" ]]; then
  TRAIN_ARGS+=(--mock)
fi

INIT_WM="$CKPT_ROOT/init_guarded_world_model.pt"
RUN_DIR="$ROOT_DIR/logs/dreamer_rl_noguard/$KIND/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
RUN_CKPT_DIR="$CKPT_ROOT/runs/$RUN_ID"
PID_FILE="$RUN_DIR/training.pid"
DONE_FILE="$RUN_DIR/training.done"
FAILED_FILE="$RUN_DIR/training.failed"
RUN_LOG="$RUN_DIR/training_stdout.log"
RUN_SCRIPT="$RUN_DIR/run_training.sh"
CARLA_LOG="$RUN_DIR/carla.log"
CARLA_PID_FILE="$RUN_DIR/carla.pid"
LATEST_FILE="$ROOT_DIR/logs/dreamer_rl_noguard/latest_${KIND}_run.txt"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$RUN_CKPT_DIR" "$(dirname "$LATEST_FILE")"
rm -f "$DONE_FILE" "$FAILED_FILE"

if [[ ! -s "$INIT_WM" ]]; then
  echo "[dreamer-rl-start] init checkpoint missing: $INIT_WM" >&2
  echo "[dreamer-rl-start] run: bash scripts/prepare_dreamer_rl_noguard_checkpoints.sh" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    if [[ "$RESTART_EXISTING" == "1" ]]; then
      echo "[dreamer-rl-start] stopping previous training pid=$old_pid"
      kill "$old_pid" || true
      sleep 2
    else
      echo "[dreamer-rl-start] training already running pid=$old_pid" >&2
      exit 1
    fi
  fi
fi

carla_ready() {
  conda run -n "$CONDA_ENV" python -c "import carla; c=carla.Client('$CARLA_HOST', int('$CARLA_PORT')); c.set_timeout(5.0); c.get_world()" >/dev/null 2>&1
}

if [[ "$MOCK" == "1" ]]; then
  echo "[dreamer-rl-start] MOCK=1: skipping CARLA startup"
elif carla_ready; then
  echo "[dreamer-rl-start] CARLA already reachable on $CARLA_HOST:$CARLA_PORT"
elif [[ "$AUTO_START_CARLA" == "1" ]]; then
  if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
    echo "[dreamer-rl-start] CARLA executable not found: $CARLA_ROOT/CarlaUE4.sh" >&2
    exit 1
  fi
  echo "[dreamer-rl-start] starting CARLA on $CARLA_HOST:$CARLA_PORT"
  nohup setsid bash -c '
    set -euo pipefail
    cd "$1"
    exec ./CarlaUE4.sh -quality-level="$2" -nosound -carla-rpc-port="$3" -graphicsadapter=0 -RenderOffScreen
  ' _ "$CARLA_ROOT" "$CARLA_QUALITY" "$CARLA_PORT" > "$CARLA_LOG" 2>&1 &
  echo $! > "$CARLA_PID_FILE"

  deadline=$((SECONDS + CARLA_WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if carla_ready; then
      echo "[dreamer-rl-start] CARLA ready"
      break
    fi
    sleep 5
  done
  if ! carla_ready; then
    echo "[dreamer-rl-start] CARLA did not become ready within ${CARLA_WAIT_SECONDS}s" >&2
    echo "[dreamer-rl-start] check: $CARLA_LOG" >&2
    exit 1
  fi
else
  echo "[dreamer-rl-start] CARLA is not reachable and AUTO_START_CARLA=0" >&2
  exit 1
fi

{
  echo "kind=$KIND"
  echo "run_id=$RUN_ID"
  echo "repo=$REPO_DIR"
  echo "run_dir=$RUN_DIR"
  echo "checkpoint_dir=$RUN_CKPT_DIR"
  echo "init_world_model=$INIT_WM"
  echo "town=$TOWN"
  echo "episodes=$EPISODES"
  echo "max_episode_steps=$MAX_EPISODE_STEPS"
  echo "rollout_size=$ROLLOUT_SIZE"
  echo "device=$DEVICE"
  echo "mock=$MOCK"
  echo "install_latest=$INSTALL_LATEST"
  echo "foreground=$FOREGROUND"
  echo "started_at=$(date -Iseconds)"
} > "$RUN_DIR/run.env"

echo "$RUN_DIR" > "$LATEST_FILE"

echo "[dreamer-rl-start] launching $KIND RL no-guard training"
{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "$REPO_DIR"
  printf 'export PYTHONPATH=%q:${PYTHONPATH:-}\n' "$REPO_DIR"
  printf 'conda run -n %q python -m training.dreamer_ppo' "$CONDA_ENV"
  for arg in \
    "${TRAIN_ARGS[@]}" \
    --episodes "$EPISODES" \
    --device "$DEVICE" \
    --host "$CARLA_HOST" \
    --port "$CARLA_PORT" \
    --town "$TOWN" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --rollout-size "$ROLLOUT_SIZE" \
    --eval-interval "$EVAL_INTERVAL" \
    --log-dir "$LOG_DIR" \
    --ckpt-dir "$RUN_CKPT_DIR" \
    --log-name "$LOG_NAME" \
    --init-world-model "$INIT_WM"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  printf 'if [[ %q == "1" && -s %q ]]; then\n' "$INSTALL_LATEST" "$RUN_CKPT_DIR/best_model.pt"
  printf '  cp -a %q %q\n' "$RUN_CKPT_DIR/best_model.pt" "$CKPT_ROOT/latest_rl_model.pt"
  printf '  echo %q > %q\n' "$RUN_CKPT_DIR/best_model.pt" "$CKPT_ROOT/latest_rl_model_source.txt"
  printf 'fi\n'
} > "$RUN_SCRIPT"
chmod +x "$RUN_SCRIPT"

if [[ "$FOREGROUND" == "1" ]]; then
  echo "$$" > "$PID_FILE"
  echo "[dreamer-rl-start] foreground mode enabled"
  if bash "$RUN_SCRIPT" > "$RUN_LOG" 2>&1; then
    date -Iseconds > "$DONE_FILE"
  else
    status=$?
    echo "$status" > "$FAILED_FILE"
    exit "$status"
  fi
else
  nohup setsid bash -c '
    set +e
    run_script="$1"
    done_file="$2"
    failed_file="$3"
    bash "$run_script"
    status=$?
    if [[ "$status" == "0" ]]; then
      date -Iseconds > "$done_file"
    else
      echo "$status" > "$failed_file"
    fi
    exit "$status"
  ' _ "$RUN_SCRIPT" "$DONE_FILE" "$FAILED_FILE" > "$RUN_LOG" 2>&1 &
  echo $! > "$PID_FILE"
fi

echo "[dreamer-rl-start] launched"
echo "[dreamer-rl-start] pid=$(cat "$PID_FILE")"
echo "[dreamer-rl-start] run_dir=$RUN_DIR"
echo "[dreamer-rl-start] log=$RUN_LOG"
echo "[dreamer-rl-start] watch: DREAMER_RL_KIND=$KIND bash scripts/watch_dreamer_rl_noguard_training.sh"
