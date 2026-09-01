#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SIMLINGO_PYTHON:-/home/mohm/miniconda3/envs/simlingo/bin/python}"
OUTPUT="${RESIDUAL_DREAMERV3_OUTPUT:-${ROOT}/checkpoints/residual_dreamerv3/candidate}"
PHASE="${RESIDUAL_DREAMERV3_PHASE:-all}"

mkdir -p "${OUTPUT}"
cd "${ROOT}"

exec "${PYTHON_BIN}" scripts/train_residual_dreamerv3.py "${PHASE}" \
  --config "${RESIDUAL_DREAMERV3_CONFIG:-${ROOT}/configs/residual_dreamerv3.yaml}" \
  --output "${OUTPUT}" \
  --device "${RESIDUAL_DREAMERV3_DEVICE:-cuda}" \
  --world-epochs "${RESIDUAL_DREAMERV3_WORLD_EPOCHS:-40}" \
  --actor-epochs "${RESIDUAL_DREAMERV3_ACTOR_EPOCHS:-40}" \
  --max-windows "${RESIDUAL_DREAMERV3_MAX_WINDOWS:-0}" \
  "$@"
