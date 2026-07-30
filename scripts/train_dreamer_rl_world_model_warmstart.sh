#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${DREAMER_RL_KIND:-ppo}" # ppo | sdbs
RUN_ID="${DREAMER_RL_WM_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${DREAMER_RL_CONDA_ENV:-simlingo}"
DEVICE_REQUEST="${DREAMER_RL_DEVICE:-auto}"
EPOCHS="${DREAMER_RL_WM_EPOCHS:-80}"
BATCH_SIZE="${DREAMER_RL_WM_BATCH_SIZE:-128}"
LR="${DREAMER_RL_WM_LR:-3e-4}"
SET_AS_INIT="${DREAMER_RL_SET_AS_INIT:-0}"

LATEST_DATASET_FILE="$ROOT_DIR/data/dreamer_rl/latest_dataset.txt"
DATASET="${DREAMER_RL_DATASET:-}"
if [[ -z "$DATASET" && -s "$LATEST_DATASET_FILE" ]]; then
  DATASET="$(cat "$LATEST_DATASET_FILE")"
fi
if [[ -z "$DATASET" || ! -s "$DATASET" ]]; then
  echo "[dreamer-rl-wm] missing dataset. Build one first:" >&2
  echo "  bash scripts/build_dreamer_rl_dataset.sh" >&2
  exit 1
fi

case "$KIND" in
  ppo)
    CKPT_ROOT="$ROOT_DIR/external/simlingo/checkpoints/dreamer_ppo_rl_noguard"
    ;;
  sdbs)
    CKPT_ROOT="$ROOT_DIR/external/simlingo/checkpoints/dreamer_sdbs_rl_noguard"
    ;;
  *)
    echo "[dreamer-rl-wm] DREAMER_RL_KIND must be 'ppo' or 'sdbs', got: $KIND" >&2
    exit 1
    ;;
esac

REPO_DIR="$ROOT_DIR/experiments/dreamer_ppo_carla"
OUT_DIR="$ROOT_DIR/logs/dreamer_rl_warmstart/$KIND/$RUN_ID"
LATEST_WARMSTART="$ROOT_DIR/logs/dreamer_rl_warmstart/latest_${KIND}_warmstart.txt"
RUNTIME_WARMSTART_DIR="$CKPT_ROOT/offline_warmstarts/$RUN_ID"

mkdir -p "$OUT_DIR" "$(dirname "$LATEST_WARMSTART")" "$RUNTIME_WARMSTART_DIR"

if [[ "$DEVICE_REQUEST" == "auto" ]]; then
  DEVICE="$(conda run -n "$CONDA_ENV" python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
else
  DEVICE="$DEVICE_REQUEST"
  if [[ "$DEVICE" == cuda* ]]; then
    CUDA_AVAILABLE="$(conda run -n "$CONDA_ENV" python -c 'import torch; print(1 if torch.cuda.is_available() else 0)')"
    if [[ "$CUDA_AVAILABLE" != "1" ]]; then
      if [[ "${DREAMER_RL_STRICT_DEVICE:-0}" == "1" ]]; then
        echo "[dreamer-rl-wm] requested $DEVICE but CUDA is unavailable" >&2
        exit 1
      fi
      echo "[dreamer-rl-wm] requested $DEVICE but CUDA is unavailable; falling back to cpu"
      DEVICE="cpu"
    fi
  fi
fi

{
  echo "kind=$KIND"
  echo "run_id=$RUN_ID"
  echo "dataset=$DATASET"
  echo "out_dir=$OUT_DIR"
  echo "runtime_warmstart_dir=$RUNTIME_WARMSTART_DIR"
  echo "device=$DEVICE"
  echo "epochs=$EPOCHS"
  echo "batch_size=$BATCH_SIZE"
  echo "lr=$LR"
  echo "started_at=$(date -Iseconds)"
} > "$OUT_DIR/run.env"

echo "$OUT_DIR" > "$LATEST_WARMSTART"

echo "[dreamer-rl-wm] training world-model warm-start"
echo "[dreamer-rl-wm] kind=$KIND dataset=$DATASET"
PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" conda run -n "$CONDA_ENV" python \
  "$REPO_DIR/tools/train_simlingo_world_model_offline.py" \
  --data "$DATASET" \
  --output-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE"

if [[ ! -s "$OUT_DIR/best_world_model.pt" ]]; then
  echo "[dreamer-rl-wm] training finished but checkpoint is missing: $OUT_DIR/best_world_model.pt" >&2
  exit 1
fi

cp -a "$OUT_DIR/best_world_model.pt" "$RUNTIME_WARMSTART_DIR/best_world_model.pt"
cp -a "$OUT_DIR/summary.json" "$RUNTIME_WARMSTART_DIR/summary.json" 2>/dev/null || true
cp -a "$OUT_DIR/history.json" "$RUNTIME_WARMSTART_DIR/history.json" 2>/dev/null || true

if [[ "$SET_AS_INIT" == "1" ]]; then
  cp -a "$OUT_DIR/best_world_model.pt" "$CKPT_ROOT/init_offline_world_model.pt"
  {
    echo "source=$OUT_DIR/best_world_model.pt"
    echo "dataset=$DATASET"
    echo "installed_at=$(date -Iseconds)"
    echo "kind=$KIND"
    echo "status=offline warm-start copied as init_offline_world_model.pt; pass DREAMER_RL_INIT_WORLD_MODEL to use it for live RL"
  } > "$CKPT_ROOT/init_offline_world_model_source.txt"
fi

echo "[dreamer-rl-wm] OK"
echo "[dreamer-rl-wm] checkpoint=$OUT_DIR/best_world_model.pt"
echo "[dreamer-rl-wm] runtime_copy=$RUNTIME_WARMSTART_DIR/best_world_model.pt"
echo "[dreamer-rl-wm] live RL example:"
echo "  DREAMER_RL_KIND=$KIND DREAMER_RL_INIT_WORLD_MODEL=$OUT_DIR/best_world_model.pt DREAMER_RL_EPISODES=10 bash scripts/start_dreamer_rl_noguard_training.sh"
