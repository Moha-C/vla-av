#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${DREAMER_COMPLEMENT_KIND:-both}" # ppo | sdbs | both
RUN_ID="${DREAMER_COMPLEMENT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${DREAMER_COMPLEMENT_CONDA_ENV:-simlingo}"
DEVICE_REQUEST="${DREAMER_COMPLEMENT_DEVICE:-auto}"
EPOCHS="${DREAMER_COMPLEMENT_EPOCHS:-120}"
BATCH_SIZE="${DREAMER_COMPLEMENT_BATCH_SIZE:-128}"
LR="${DREAMER_COMPLEMENT_LR:-3e-4}"
INSTALL="${DREAMER_COMPLEMENT_INSTALL:-1}"
DATASET="${DREAMER_COMPLEMENT_DATASET:-}"

latest_dataset_from_file() {
  local latest_file="$ROOT_DIR/data/dreamer_rl/latest_dataset.txt"
  if [[ -s "$latest_file" ]]; then
    local candidate
    candidate="$(cat "$latest_file")"
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  return 1
}

latest_campaign_dataset() {
  find "$ROOT_DIR/logs/dreamer_rl_campaign" \
    -path '*/dataset_all_traces/dreamer_rl_dataset.npz' \
    -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

if [[ -z "$DATASET" ]]; then
  DATASET="$(latest_dataset_from_file || true)"
fi
if [[ -z "$DATASET" ]]; then
  DATASET="$(latest_campaign_dataset || true)"
fi
if [[ -z "$DATASET" || ! -s "$DATASET" ]]; then
  echo "[dreamer-complement] missing dataset." >&2
  echo "[dreamer-complement] Expected a converted SimLingo trace dataset, for example:" >&2
  echo "  DREAMER_COMPLEMENT_DATASET=logs/dreamer_rl_campaign/<run>/dataset_all_traces/dreamer_rl_dataset.npz bash scripts/train_dreamer_complement_from_traces.sh" >&2
  exit 1
fi

case "$KIND" in
  ppo|sdbs|both) ;;
  *)
    echo "[dreamer-complement] DREAMER_COMPLEMENT_KIND must be ppo, sdbs, or both; got: $KIND" >&2
    exit 1
    ;;
esac

REPO_DIR="$ROOT_DIR/experiments/dreamer_ppo_carla"
TRAINER="$REPO_DIR/tools/train_simlingo_world_model_offline.py"
if [[ ! -s "$TRAINER" ]]; then
  echo "[dreamer-complement] trainer missing: $TRAINER" >&2
  exit 1
fi

if [[ "$DEVICE_REQUEST" == "auto" ]]; then
  DEVICE="$(conda run -n "$CONDA_ENV" python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
else
  DEVICE="$DEVICE_REQUEST"
  if [[ "$DEVICE" == cuda* ]]; then
    CUDA_AVAILABLE="$(conda run -n "$CONDA_ENV" python -c 'import torch; print(1 if torch.cuda.is_available() else 0)')"
    if [[ "$CUDA_AVAILABLE" != "1" ]]; then
      if [[ "${DREAMER_COMPLEMENT_STRICT_DEVICE:-0}" == "1" ]]; then
        echo "[dreamer-complement] requested $DEVICE but CUDA is unavailable" >&2
        exit 1
      fi
      echo "[dreamer-complement] requested $DEVICE but CUDA is unavailable; falling back to cpu"
      DEVICE="cpu"
    fi
  fi
fi

OUT_BASE="$ROOT_DIR/logs/dreamer_complement_training/$RUN_ID"
mkdir -p "$OUT_BASE" "$ROOT_DIR/data/dreamer_complement"
echo "$OUT_BASE" > "$ROOT_DIR/logs/dreamer_complement_training/latest_complement_training.txt"

copy_audit() {
  local out_dir="$1"
  local dataset_dir
  dataset_dir="$(dirname "$DATASET")"
  if [[ -s "$dataset_dir/audit.json" ]]; then
    cp -a "$dataset_dir/audit.json" "$out_dir/dataset_audit.json"
  fi
  if [[ -s "$dataset_dir/manifest.txt" ]]; then
    cp -a "$dataset_dir/manifest.txt" "$out_dir/dataset_manifest.txt"
  fi
}

train_one() {
  local kind="$1"
  local out_dir="$OUT_BASE/$kind"
  local ckpt_root
  case "$kind" in
    ppo) ckpt_root="$ROOT_DIR/external/simlingo/checkpoints/dreamer_ppo_complement" ;;
    sdbs) ckpt_root="$ROOT_DIR/external/simlingo/checkpoints/dreamer_sdbs_complement" ;;
  esac

  mkdir -p "$out_dir"
  {
    echo "kind=$kind"
    echo "run_id=$RUN_ID"
    echo "dataset=$DATASET"
    echo "out_dir=$out_dir"
    echo "checkpoint_root=$ckpt_root"
    echo "device=$DEVICE"
    echo "epochs=$EPOCHS"
    echo "batch_size=$BATCH_SIZE"
    echo "lr=$LR"
    echo "install=$INSTALL"
    echo "started_at=$(date -Iseconds)"
    echo "training_objective=SimLingo-relative world model scorer; SimLingo remains the base driver"
  } > "$out_dir/run.env"
  copy_audit "$out_dir"

  echo "[dreamer-complement] training $kind complement checkpoint"
  echo "[dreamer-complement] dataset=$DATASET"
  echo "[dreamer-complement] output=$out_dir"
  PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" conda run -n "$CONDA_ENV" python \
    "$TRAINER" \
    --data "$DATASET" \
    --output-dir "$out_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device "$DEVICE"

  if [[ ! -s "$out_dir/best_world_model.pt" ]]; then
    echo "[dreamer-complement] training finished but checkpoint is missing: $out_dir/best_world_model.pt" >&2
    exit 1
  fi

  if [[ "$INSTALL" == "1" ]]; then
    mkdir -p "$ckpt_root/runs/$RUN_ID"
    cp -a "$out_dir/best_world_model.pt" "$ckpt_root/latest_world_model.pt"
    cp -a "$out_dir/best_world_model.pt" "$ckpt_root/runs/$RUN_ID/best_world_model.pt"
    cp -a "$out_dir/summary.json" "$ckpt_root/summary.json" 2>/dev/null || true
    cp -a "$out_dir/history.json" "$ckpt_root/history.json" 2>/dev/null || true
    cp -a "$out_dir/summary.json" "$ckpt_root/runs/$RUN_ID/summary.json" 2>/dev/null || true
    cp -a "$out_dir/history.json" "$ckpt_root/runs/$RUN_ID/history.json" 2>/dev/null || true
    cp -a "$out_dir/dataset_audit.json" "$ckpt_root/runs/$RUN_ID/dataset_audit.json" 2>/dev/null || true
    {
      echo "source=$out_dir/best_world_model.pt"
      echo "dataset=$DATASET"
      echo "installed_at=$(date -Iseconds)"
      echo "kind=$kind"
      echo "run_id=$RUN_ID"
      echo "objective=Dreamer complement to SimLingo: score base/control candidates from SimLingo trace data, not standalone CarlaEnv policy"
    } > "$ckpt_root/latest_world_model_source.txt"
    echo "[dreamer-complement] installed=$ckpt_root/latest_world_model.pt"
  fi
}

case "$KIND" in
  ppo) train_one ppo ;;
  sdbs) train_one sdbs ;;
  both)
    train_one ppo
    train_one sdbs
    ;;
esac

cat > "$OUT_BASE/README.txt" <<EOF
Dreamer complement training run: $RUN_ID

This run trains the Dreamer world-model scorer from SimLingo closed-loop traces.
It does not train an autonomous CARLA policy. Runtime usage remains:

  SimLingo proposes the base control every tick.
  Dreamer scores SimLingo-relative candidate actions.
  The guard may override only when the selected mode allows it.

Dataset:
$DATASET

Installed checkpoints:
  PPO complement:  external/simlingo/checkpoints/dreamer_ppo_complement/latest_world_model.pt
  SDBS complement: external/simlingo/checkpoints/dreamer_sdbs_complement/latest_world_model.pt
EOF

echo "[dreamer-complement] OK"
echo "[dreamer-complement] run=$OUT_BASE"
