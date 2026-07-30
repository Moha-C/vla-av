#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMLINGO_ROOT="${SIMLINGO_ROOT:-$ROOT_DIR/external/simlingo}"
CARLA_ROOT="${CARLA_ROOT:-$HOME/carla_simulator}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${SIMLINGO_ENV_NAME:-simlingo}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing conda init file: $CONDA_SH" >&2
  exit 1
fi

set +u
source "$CONDA_SH"
conda activate "$ENV_NAME"
set -u

export CARLA_ROOT
export WORK_DIR="$SIMLINGO_ROOT"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SCENARIO_RUNNER_ROOT="$SIMLINGO_ROOT/Bench2Drive/scenario_runner"
export LEADERBOARD_ROOT="$SIMLINGO_ROOT/Bench2Drive/leaderboard"
export PYTHONPATH="$SIMLINGO_ROOT:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"

echo "SimLingo env active: $CONDA_DEFAULT_ENV"
echo "SIMLINGO_ROOT=$SIMLINGO_ROOT"
echo "CARLA_ROOT=$CARLA_ROOT"
