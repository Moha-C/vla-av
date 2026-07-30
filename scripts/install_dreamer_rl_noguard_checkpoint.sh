#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${DREAMER_RL_KIND:-ppo}" # ppo | sdbs

case "$KIND" in
  ppo)
    CKPT_ROOT="$ROOT_DIR/external/simlingo/checkpoints/dreamer_ppo_rl_noguard"
    LATEST_FILE="$ROOT_DIR/logs/dreamer_rl_noguard/latest_ppo_run.txt"
    ;;
  sdbs)
    CKPT_ROOT="$ROOT_DIR/external/simlingo/checkpoints/dreamer_sdbs_rl_noguard"
    LATEST_FILE="$ROOT_DIR/logs/dreamer_rl_noguard/latest_sdbs_run.txt"
    ;;
  *)
    echo "[dreamer-rl-install] DREAMER_RL_KIND must be 'ppo' or 'sdbs', got: $KIND" >&2
    exit 1
    ;;
esac

if [[ -n "${DREAMER_RL_RUN_DIR:-}" ]]; then
  RUN_DIR="$DREAMER_RL_RUN_DIR"
elif [[ -s "$LATEST_FILE" ]]; then
  RUN_DIR="$(cat "$LATEST_FILE")"
else
  echo "[dreamer-rl-install] no latest run found for kind=$KIND" >&2
  exit 1
fi

RUN_ENV="$RUN_DIR/run.env"
if [[ -s "$RUN_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$RUN_ENV"
else
  checkpoint_dir="$CKPT_ROOT/runs/$(basename "$RUN_DIR")"
fi

BEST_CKPT="${DREAMER_RL_BEST_CKPT:-${checkpoint_dir:-}/best_model.pt}"
LATEST_CKPT="$CKPT_ROOT/latest_rl_model.pt"
SOURCE_FILE="$CKPT_ROOT/latest_rl_model_source.txt"

if [[ ! -s "$BEST_CKPT" ]]; then
  echo "[dreamer-rl-install] best checkpoint missing: $BEST_CKPT" >&2
  exit 1
fi

mkdir -p "$CKPT_ROOT"
cp -a "$BEST_CKPT" "$LATEST_CKPT"
{
  echo "$BEST_CKPT"
  echo "installed_at=$(date -Iseconds)"
  echo "kind=$KIND"
} > "$SOURCE_FILE"

echo "[dreamer-rl-install] installed $KIND no-guard checkpoint"
echo "[dreamer-rl-install] source=$BEST_CKPT"
echo "[dreamer-rl-install] target=$LATEST_CKPT"
