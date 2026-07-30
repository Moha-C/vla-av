#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${OUT_DIR:-$ROOT_DIR/experiments/maram_dreamer_carla/full_carla_gpu}}"
LOG_DIR="$OUT_DIR/logs"

DATA_FILE="$OUT_DIR/data/carla_normal.npz"
WM_CKPT="$OUT_DIR/checkpoints/wm_pretrained_carla.pt"
RL_CKPT="$OUT_DIR/checkpoints/sdbs_checkpoint.pt"

EPISODES="${EPISODES:-30}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-600}"
WM_EPOCHS="${WM_EPOCHS:-30}"
RL_ITERS="${RL_ITERS:-1000}"
TOTAL_TRANSITIONS=$((EPISODES * STEPS_PER_EPISODE))

exists_label() {
  local label="$1"
  local file="$2"
  if [[ -s "$file" ]]; then
    printf "  [OK]      %-32s %s bytes\n" "$label" "$(stat -c '%s' "$file")"
  else
    printf "  [MISSING] %-32s %s\n" "$label" "$file"
  fi
}

last_match() {
  local file="$1"
  local pattern="$2"
  if [[ -s "$file" ]]; then
    grep -aE "$pattern" "$file" | tail -n 1 || true
  fi
}

echo "[maram-dreamer-watch] out_dir=$OUT_DIR"
echo
echo "Required outputs:"
exists_label "normal driving dataset" "$DATA_FILE"
exists_label "world model checkpoint" "$WM_CKPT"
exists_label "Dreamer-PPO checkpoint" "$RL_CKPT"

echo
echo "Live processes:"
pgrep -af 'run_maram_dreamer_carla_training|collect_carla_driving_data|pretrain_world_model|scripts/run_training.py|CarlaUE4' || true

echo
if [[ -e "$LOG_DIR/01_collect.log" && ! -s "$DATA_FILE" ]]; then
  latest_collect="$(last_match "$LOG_DIR/01_collect.log" '\[carla\].*transitions')"
  latest_collect_error="$(last_match "$LOG_DIR/01_collect.log" 'Aborted|RuntimeError|Traceback')"
  echo "Phase: collection"
  echo "Expected transitions: $TOTAL_TRANSITIONS"
  if [[ -n "$latest_collect" ]]; then
    echo "Latest: $latest_collect"
  elif [[ -n "$latest_collect_error" ]]; then
    echo "Latest: $latest_collect_error"
  else
    echo "Latest: collection process started; waiting for first episode log"
  fi
elif [[ -s "$DATA_FILE" && ! -s "$WM_CKPT" ]]; then
  latest_wm="$(last_match "$LOG_DIR/02_pretrain_wm.log" 'epoch|Epoch|loss|ERROR|Traceback')"
  echo "Phase: world-model pretrain"
  echo "Expected epochs: $WM_EPOCHS"
  [[ -n "$latest_wm" ]] && echo "Latest: $latest_wm"
elif [[ -s "$WM_CKPT" && ! -s "$RL_CKPT" ]]; then
  latest_rl="$(last_match "$LOG_DIR/03_train_rl.log" 'iter|Iteration|reward|checkpoint|ERROR|Traceback')"
  echo "Phase: Dreamer-PPO RL training"
  echo "Expected iterations: $RL_ITERS"
  [[ -n "$latest_rl" ]] && echo "Latest: $latest_rl"
elif [[ -s "$DATA_FILE" && -s "$WM_CKPT" && -s "$RL_CKPT" ]]; then
  echo "Phase: complete"
else
  echo "Phase: not started or crashed before required outputs"
fi

echo
echo "Last logs:"
for log in "$LOG_DIR/00_preflight.log" "$LOG_DIR/01_collect.log" "$LOG_DIR/02_pretrain_wm.log" "$LOG_DIR/03_train_rl.log"; do
  if [[ -s "$log" ]]; then
    echo "--- $(basename "$log")"
    tail -n 5 "$log"
  fi
done
