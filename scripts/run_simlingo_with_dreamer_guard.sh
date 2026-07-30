#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# First-test default is shadow mode: the guard logs possible overrides but does
# not change SimLingo controls. Use SIMLINGO_DREAMER_GUARD=1 for applied mode.
export SIMLINGO_DREAMER_GUARD="${SIMLINGO_DREAMER_GUARD:-shadow}"
export SIMLINGO_DREAMER_GUARD_MODE="${SIMLINGO_DREAMER_GUARD_MODE:-apply}"
export SIMLINGO_DREAMER_RISK_MARGIN="${SIMLINGO_DREAMER_RISK_MARGIN:-0.05}"
export SIMLINGO_DREAMER_MAX_PROGRESS_DROP="${SIMLINGO_DREAMER_MAX_PROGRESS_DROP:-0.01}"
export SIMLINGO_DREAMER_MAX_STEER_DELTA="${SIMLINGO_DREAMER_MAX_STEER_DELTA:-0.12}"
export SIMLINGO_DREAMER_MAX_BRAKE_INCREASE="${SIMLINGO_DREAMER_MAX_BRAKE_INCREASE:-0.45}"
export SIMLINGO_DREAMER_LOG_EVERY="${SIMLINGO_DREAMER_LOG_EVERY:-40}"

if [[ -z "${SIMLINGO_DREAMER_CHECKPOINT:-}" ]]; then
  export SIMLINGO_DREAMER_CHECKPOINT="$ROOT_DIR/external/simlingo/checkpoints/dreamer_guard/best_world_model.pt"
fi

echo "[simlingo-dreamer] guard=${SIMLINGO_DREAMER_GUARD} mode=${SIMLINGO_DREAMER_GUARD_MODE}"
echo "[simlingo-dreamer] checkpoint=${SIMLINGO_DREAMER_CHECKPOINT}"

bash "$ROOT_DIR/scripts/run_simlingo_with_pov.sh"
