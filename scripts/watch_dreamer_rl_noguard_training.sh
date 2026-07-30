#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${DREAMER_RL_KIND:-ppo}"          # ppo | sdbs
LATEST_FILE="$ROOT_DIR/logs/dreamer_rl_noguard/latest_${KIND}_run.txt"

if [[ -n "${DREAMER_RL_RUN_DIR:-}" ]]; then
  RUN_DIR="$DREAMER_RL_RUN_DIR"
elif [[ -s "$LATEST_FILE" ]]; then
  RUN_DIR="$(cat "$LATEST_FILE")"
else
  echo "[dreamer-rl-watch] no latest run for kind=$KIND"
  echo "[dreamer-rl-watch] start one with: DREAMER_RL_KIND=$KIND bash scripts/start_dreamer_rl_noguard_training.sh"
  exit 1
fi

PID_FILE="$RUN_DIR/training.pid"
RUN_ENV="$RUN_DIR/run.env"
RUN_LOG="$RUN_DIR/training_stdout.log"
CARLA_LOG="$RUN_DIR/carla.log"
DONE_FILE="$RUN_DIR/training.done"
FAILED_FILE="$RUN_DIR/training.failed"

if [[ -s "$RUN_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$RUN_ENV"
else
  episodes="${DREAMER_RL_EPISODES:-unknown}"
  checkpoint_dir=""
fi

CSV="$RUN_DIR/logs/${KIND}_rl_noguard.csv"
if [[ "$KIND" == "sdbs" ]]; then
  CSV="$RUN_DIR/logs/sdbs_rl_noguard.csv"
fi
if [[ "$KIND" == "ppo" ]]; then
  CSV="$RUN_DIR/logs/ppo_rl_noguard.csv"
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -s "$FAILED_FILE" ]]; then
  train_state="FAILED exit=$(cat "$FAILED_FILE")"
elif [[ -s "$DONE_FILE" ]]; then
  train_state="DONE at $(cat "$DONE_FILE")"
elif [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
  train_state="RUNNING pid=$pid"
else
  train_state="STOPPED"
fi

rows=0
last_row=""
if [[ -s "$CSV" ]]; then
  rows=$(( $(wc -l < "$CSV") - 1 ))
  if (( rows < 0 )); then rows=0; fi
  last_row="$(tail -n 1 "$CSV")"
fi

total="${episodes:-unknown}"
if [[ "$total" =~ ^[0-9]+$ ]] && (( total > 0 )); then
  pct="$(awk -v r="$rows" -v t="$total" 'BEGIN { printf "%.1f", (100.0*r/t) }')"
  progress="$rows/$total episodes ($pct%)"
else
  progress="$rows episodes"
fi

best_ckpt="${checkpoint_dir:-}/best_model.pt"
latest_runtime=""
case "$KIND" in
  ppo) latest_runtime="$ROOT_DIR/external/simlingo/checkpoints/dreamer_ppo_rl_noguard/latest_rl_model.pt" ;;
  sdbs) latest_runtime="$ROOT_DIR/external/simlingo/checkpoints/dreamer_sdbs_rl_noguard/latest_rl_model.pt" ;;
esac

echo "=== Dreamer RL no-guard training ==="
echo "kind:       $KIND"
echo "run_dir:    $RUN_DIR"
echo "training:   $train_state"
echo "progress:   $progress"
[[ -n "${town:-}" ]] && echo "town:       $town"
[[ -n "${device:-}" ]] && echo "device:     $device"
echo
echo "Outputs:"
if [[ -s "$best_ckpt" ]]; then
  echo "  [OK] best checkpoint: $best_ckpt ($(stat -c '%s' "$best_ckpt") bytes)"
else
  echo "  [..] best checkpoint: $best_ckpt"
fi
if [[ -s "$latest_runtime" ]]; then
  echo "  [OK] latest runtime copy: $latest_runtime ($(stat -c '%s' "$latest_runtime") bytes)"
else
  echo "  [..] latest runtime copy: $latest_runtime"
fi
if [[ -s "$CSV" ]]; then
  echo "  [OK] csv log: $CSV"
else
  echo "  [..] csv log: $CSV"
fi

echo
if [[ -n "$last_row" ]]; then
  echo "Last CSV row:"
  echo "$last_row"
fi

echo
echo "Live processes:"
ps -eo pid=,cmd= | awk '/python -m training[.]dreamer_ppo|CarlaUE4/ && !/awk/ { print }' || true

echo
echo "Training log tail:"
if [[ -s "$RUN_LOG" ]]; then
  tail -n 35 "$RUN_LOG"
else
  echo "no training log yet: $RUN_LOG"
fi

if [[ -s "$CARLA_LOG" ]]; then
  echo
  echo "CARLA log tail:"
  tail -n 10 "$CARLA_LOG"
fi
