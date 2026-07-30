#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${SIMLINGO_MODEL_DIR:-$ROOT_DIR/models/simlingo_hf}"
REPO_ID="${SIMLINGO_REPO_ID:-RenzKa/simlingo}"
CONDA_PYTHON="${SIMLINGO_PYTHON:-$HOME/miniconda3/envs/${SIMLINGO_ENV_NAME:-simlingo}/bin/python}"

mkdir -p "$MODEL_DIR" "$ROOT_DIR/logs"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

echo "[simlingo-download] repo=$REPO_ID"
echo "[simlingo-download] dst=$MODEL_DIR"

if command -v hf >/dev/null 2>&1; then
  HF_BIN="$(command -v hf)"
elif [[ -x "$HOME/.local/bin/hf" ]]; then
  HF_BIN="$HOME/.local/bin/hf"
else
  HF_BIN=""
fi

if [[ -n "$HF_BIN" ]]; then
  "$HF_BIN" download "$REPO_ID" \
    --local-dir "$MODEL_DIR" \
    --include "simlingo/.hydra/*" \
    --include "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt" \
    --include "simlingo/checkpoints/epoch=013.ckpt/latest" \
    --include "simlingo/checkpoints/epoch=013.ckpt/zero_to_fp32.py" \
    --max-workers "${HF_MAX_WORKERS:-4}" \
    2>&1 | tee "$ROOT_DIR/logs/simlingo_download.log"
elif [[ -x "$CONDA_PYTHON" ]]; then
  "$CONDA_PYTHON" - "$REPO_ID" "$MODEL_DIR" "${HF_MAX_WORKERS:-4}" <<'PY' \
    2>&1 | tee "$ROOT_DIR/logs/simlingo_download.log"
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir, max_workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=[
        "simlingo/.hydra/*",
        "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
        "simlingo/checkpoints/epoch=013.ckpt/latest",
        "simlingo/checkpoints/epoch=013.ckpt/zero_to_fp32.py",
    ],
    max_workers=max_workers,
)
PY
else
  echo "[simlingo-download] Could not find 'hf' or $CONDA_PYTHON." >&2
  echo "[simlingo-download] Run first: bash scripts/setup_simlingo_env.sh" >&2
  exit 1
fi

test -f "$MODEL_DIR/simlingo/.hydra/config.yaml"
test -f "$MODEL_DIR/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"

du -sh "$MODEL_DIR" | tee "$MODEL_DIR/DOWNLOAD_OK.txt"
echo "[simlingo-download] OK"
