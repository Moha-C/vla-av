#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALPAMAYO_ROOT="${ALPAMAYO_ROOT:-$ROOT_DIR/external/alpamayo_official}"
VENV_ACTIVATE="$ALPAMAYO_ROOT/ar1_venv/bin/activate"
CUDA_TOOLKIT="${CUDA_TOOLKIT:-$HOME/miniforge3/envs/cuda-toolkit-128}"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Missing Alpamayo venv: $VENV_ACTIVATE" >&2
  exit 1
fi

set +u
source "$VENV_ACTIVATE"
set -u

export PYTHONPATH="$ALPAMAYO_ROOT:$ALPAMAYO_ROOT/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
if [[ -d "$CUDA_TOOLKIT" ]]; then
  export CUDA_HOME="$CUDA_TOOLKIT"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "Alpamayo official venv active"
echo "ALPAMAYO_ROOT=$ALPAMAYO_ROOT"
