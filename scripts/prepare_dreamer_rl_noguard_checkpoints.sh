#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMLINGO_CKPT_DIR="$ROOT_DIR/external/simlingo/checkpoints"

PPO_SRC="$SIMLINGO_CKPT_DIR/dreamer_guard/best_world_model.pt"
SDBS_SRC="$SIMLINGO_CKPT_DIR/dreamer_sdbs_fresh/best_world_model.pt"
PPO_DIR="$SIMLINGO_CKPT_DIR/dreamer_ppo_rl_noguard"
SDBS_DIR="$SIMLINGO_CKPT_DIR/dreamer_sdbs_rl_noguard"

mkdir -p "$PPO_DIR/runs" "$SDBS_DIR/runs"

copy_once() {
  local src="$1"
  local dst="$2"
  if [[ ! -s "$src" ]]; then
    echo "[dreamer-rl-prepare] missing source checkpoint: $src" >&2
    exit 1
  fi
  if [[ -e "$dst" ]]; then
    echo "[dreamer-rl-prepare] keeping existing init checkpoint: $dst"
  else
    cp -a "$src" "$dst"
    echo "[dreamer-rl-prepare] created init checkpoint: $dst"
  fi
}

copy_once "$PPO_SRC" "$PPO_DIR/init_guarded_world_model.pt"
copy_once "$SDBS_SRC" "$SDBS_DIR/init_guarded_world_model.pt"

{
  echo "Dreamer RL no-guard initialization"
  echo "Created/checked: $(date -Iseconds)"
  echo
  echo "These files are only warm-start seeds for RL no-guard experiments."
  echo "They are copied from the current guarded checkpoints and must not replace"
  echo "the guarded runtime checkpoints unless explicitly validated later."
  echo
  echo "PPO init:  $PPO_DIR/init_guarded_world_model.pt"
  echo "SDBS init: $SDBS_DIR/init_guarded_world_model.pt"
} > "$SIMLINGO_CKPT_DIR/dreamer_rl_noguard_README.txt"

sha256sum \
  "$PPO_DIR/init_guarded_world_model.pt" \
  "$SDBS_DIR/init_guarded_world_model.pt" \
  > "$SIMLINGO_CKPT_DIR/dreamer_rl_noguard_SHA256SUMS.txt"

echo "[dreamer-rl-prepare] done"
echo "[dreamer-rl-prepare] ppo_dir=$PPO_DIR"
echo "[dreamer-rl-prepare] sdbs_dir=$SDBS_DIR"
