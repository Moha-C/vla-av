#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMLINGO_ROOT="${SIMLINGO_ROOT:-$ROOT_DIR/external/simlingo}"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${SIMLINGO_ENV_NAME:-simlingo}"

mkdir -p "$ROOT_DIR/logs"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "[simlingo-env] conda not found at $CONDA_BIN" >&2
  exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[simlingo-env] Reusing conda env: $ENV_NAME"
else
  echo "[simlingo-env] Creating conda env: $ENV_NAME"
  "$CONDA_BIN" env create -f "$SIMLINGO_ROOT/environment.yaml" \
    2>&1 | tee "$ROOT_DIR/logs/simlingo_env_create.log"
fi

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
PY

if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  else
    CUDA_NVCC="$(find /usr/local "$HOME/miniconda3/envs" -maxdepth 5 -type f -name nvcc 2>/dev/null | head -n 1 || true)"
    if [[ -n "$CUDA_NVCC" ]]; then
      export CUDA_HOME="$(cd "$(dirname "$CUDA_NVCC")/.." && pwd)"
    fi
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("flash_attn") else 1)
PY
then
  echo "[simlingo-env] flash_attn already installed"
else
  if command -v nvcc >/dev/null 2>&1; then
    echo "[simlingo-env] Installing flash-attn"
    if ! pip install flash-attn==2.7.0.post2 --no-build-isolation \
      2>&1 | tee "$ROOT_DIR/logs/simlingo_flash_attn_install.log"; then
      echo "[simlingo-env] WARNING: flash-attn install failed; continuing without it."
      echo "[simlingo-env] SimLingo/InternVL can usually run with the standard attention path, slower but usable."
    fi
  else
    echo "[simlingo-env] WARNING: nvcc/CUDA_HOME not found, skipping flash-attn."
    echo "[simlingo-env] This is not fatal for the first CARLA eval; it may only make inference slower."
  fi
fi

python - <<'PY' | tee "$ROOT_DIR/logs/simlingo_env_check.log"
import carla, cv2, hydra, torch, transformers
print("carla:", getattr(carla, "__version__", "ok"))
print("cv2:", cv2.__version__)
print("hydra: ok")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
PY

echo "[simlingo-env] OK"
