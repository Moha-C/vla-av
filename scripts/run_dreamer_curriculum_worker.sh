#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${DREAMER_CURRICULUM_CONDA_ENV:-simlingo}"
RUN_DIR="${DREAMER_CURRICULUM_RUN_DIR:?DREAMER_CURRICULUM_RUN_DIR is required}"

mkdir -p "$RUN_DIR"
exec >>"$RUN_DIR/campaign.log" 2>&1

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

cd "$ROOT_DIR"
exec python -u -m scripts.run_dreamer_curriculum_training "$@"
