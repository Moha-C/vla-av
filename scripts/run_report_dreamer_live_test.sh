#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATION="${REPORT_DREAMER_ABLATION:-D}"
SHADOW="${REPORT_DREAMER_SHADOW:-1}"
ROUTE_ID="${ROUTE_ID:-57}"
SEED="${SEED:-20260818}"
CONFIG="${REPORT_DREAMER_CONFIG:-$ROOT_DIR/configs/dreamer_report_aligned.yaml}"

case "$ABLATION" in
  A)
    export SIMLINGO_REPORT_DREAMER_MODE=off
    export SIMLINGO_DREAMER_GUARD=0
    export SIMLINGO_DREAMER_RUNTIME=""
    export SIMLINGO_CARDREAMER_MODE=off
    ;;
  B)
    echo "Ablation B is the preserved legacy guard. Launch Dreamer PPO from the dashboard so its versioned legacy preset is used unchanged." >&2
    exit 2
    ;;
  C|D|E)
    if [[ "$ABLATION" == "E" ]]; then
      DEFAULT_CHECKPOINT="$ROOT_DIR/checkpoints/report_aligned_dreamer/production/report_dreamer_pairwise.pt"
    else
      DEFAULT_CHECKPOINT="$ROOT_DIR/checkpoints/report_aligned_dreamer/production/report_dreamer.pt"
    fi
    CHECKPOINT="${REPORT_DREAMER_CHECKPOINT:-$DEFAULT_CHECKPOINT}"
    if [[ ! -f "$CHECKPOINT" ]]; then
      echo "Missing validated report-aligned checkpoint: $CHECKPOINT" >&2
      echo "A smoke or unvalidated candidate is never selected automatically." >&2
      exit 2
    fi
    RUN_ID="$(date +%Y%m%d_%H%M%S)_ablation_${ABLATION}_route_${ROUTE_ID}_seed_${SEED}"
    RUN_DIR="${REPORT_DREAMER_RUN_DIR:-$ROOT_DIR/logs/report_dreamer_runtime/$RUN_ID}"
    mkdir -p "$RUN_DIR"
    export SIMLINGO_REPORT_DREAMER_MODE="$([[ "$SHADOW" == "1" ]] && echo shadow || echo apply)"
    export SIMLINGO_REPORT_DREAMER_SHADOW="$SHADOW"
    export SIMLINGO_REPORT_DREAMER_ABLATION="$ABLATION"
    export SIMLINGO_REPORT_DREAMER_CHECKPOINT="$CHECKPOINT"
    export SIMLINGO_REPORT_DREAMER_CONFIG="$CONFIG"
    export SIMLINGO_REPORT_DREAMER_DEVICE="${REPORT_DREAMER_DEVICE:-cpu}"
    export SIMLINGO_REPORT_DREAMER_TRACE="$RUN_DIR/trace.jsonl"
    export SIMLINGO_REPORT_DREAMER_STATUS_PATH="$ROOT_DIR/logs/simlingo_eval/dreamer_guard_status.json"
    export SIMLINGO_DREAMER_GUARD=0
    export SIMLINGO_DREAMER_RUNTIME=""
    export SIMLINGO_DREAMER_RECOVERY=0
    export SIMLINGO_DREAMER_COLLISION_SHIELD=0
    export SIMLINGO_CARDREAMER_MODE=off
    printf '%s\n' "$SIMLINGO_REPORT_DREAMER_TRACE" > "$ROOT_DIR/logs/simlingo_eval/latest_report_dreamer_trace.txt"
    ;;
  *)
    echo "REPORT_DREAMER_ABLATION must be A, B, C, D, or E." >&2
    exit 2
    ;;
esac

if [[ -z "${ROUTE_FILE:-}" ]]; then
  export ROUTE_FILE="$ROOT_DIR/external/simlingo/leaderboard/data/bench2drive_split/bench2drive_${ROUTE_ID}.xml"
fi
if [[ ! -f "$ROUTE_FILE" ]]; then
  echo "Route XML not found: $ROUTE_FILE" >&2
  exit 2
fi

export ROUTE_ID SEED
export CARLA_QUALITY="${CARLA_QUALITY:-Low}"
export SIMLINGO_VIEW_MODE="${SIMLINGO_VIEW_MODE:-chase}"
export SIMLINGO_VIEW_WIDTH="${SIMLINGO_VIEW_WIDTH:-1280}"
export SIMLINGO_VIEW_HEIGHT="${SIMLINGO_VIEW_HEIGHT:-720}"
export SIMLINGO_RECORD="${SIMLINGO_RECORD:-0}"
export SIMLINGO_PLAYBACK_AFTER="${SIMLINGO_PLAYBACK_AFTER:-0}"

echo "[report-dreamer-live] ablation=$ABLATION shadow=$SHADOW route=$ROUTE_FILE seed=$SEED"
exec bash "$ROOT_DIR/scripts/run_simlingo_with_pov.sh"
