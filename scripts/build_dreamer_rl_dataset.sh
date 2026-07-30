#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${DREAMER_RL_DATASET_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${DREAMER_RL_DATASET_OUT_DIR:-$ROOT_DIR/data/dreamer_rl/$RUN_ID}"
DATASET="$OUT_DIR/dreamer_rl_dataset.npz"
AUDIT_JSON="$OUT_DIR/audit.json"
LATEST_FILE="$ROOT_DIR/data/dreamer_rl/latest_dataset.txt"

STATE_DIM="${DREAMER_RL_STATE_DIM:-28}"
ACTION_KEY="${DREAMER_RL_ACTION_KEY:-chosen_action}"
MAX_STATIONARY="${DREAMER_RL_MAX_STATIONARY_PER_RUN:-60}"
RECOVERY_OVERSAMPLE="${DREAMER_RL_RECOVERY_OVERSAMPLE:-3}"
NEAR_HAZARD_OVERSAMPLE="${DREAMER_RL_NEAR_HAZARD_OVERSAMPLE:-2}"
MAX_RECOVERY_RISK="${DREAMER_RL_MAX_RECOVERY_RISK:-0.88}"
MAX_RECOVERY_RISK_INCREASE="${DREAMER_RL_MAX_RECOVERY_RISK_INCREASE:-0.03}"
MIN_RECOVERY_TTC="${DREAMER_RL_MIN_RECOVERY_TTC:-3.2}"
MAX_RECOVERY_PROGRESS_DROP="${DREAMER_RL_MAX_RECOVERY_PROGRESS_DROP:-0.01}"

MIN_TRANSITIONS="${DREAMER_RL_MIN_TRANSITIONS:-1000}"
MIN_RUNS="${DREAMER_RL_MIN_RUNS:-2}"
MIN_ROUTES="${DREAMER_RL_MIN_ROUTES:-2}"
MAX_STATIONARY_FRACTION="${DREAMER_RL_MAX_STATIONARY_FRACTION:-0.65}"
MIN_RECOVERY_FRACTION="${DREAMER_RL_MIN_RECOVERY_FRACTION:-0.02}"

mkdir -p "$OUT_DIR" "$(dirname "$LATEST_FILE")"

TRACE_FILES=()
if (( "$#" > 0 )); then
  for path in "$@"; do
    TRACE_FILES+=("$path")
  done
elif [[ -n "${DREAMER_RL_TRACE_FILES:-}" ]]; then
  # shellcheck disable=SC2206
  TRACE_FILES=(${DREAMER_RL_TRACE_FILES})
else
  while IFS= read -r path; do
    TRACE_FILES+=("$path")
  done < <(find "$ROOT_DIR/logs/action_dreaming_collect" -maxdepth 1 -type f -name '*.jsonl' 2>/dev/null | sort)
fi

if (( "${#TRACE_FILES[@]}" == 0 )); then
  echo "[dreamer-rl-dataset] no trace files found." >&2
  echo "[dreamer-rl-dataset] collect first with the dashboard mode: CARLA POV + Action Dreaming collect" >&2
  exit 1
fi

{
  echo "run_id=$RUN_ID"
  echo "created_at=$(date -Iseconds)"
  echo "dataset=$DATASET"
  echo "audit=$AUDIT_JSON"
  echo "state_dim=$STATE_DIM"
  echo "action_key=$ACTION_KEY"
  echo "traces:"
  for path in "${TRACE_FILES[@]}"; do
    echo "  $path"
  done
} > "$OUT_DIR/manifest.txt"

echo "[dreamer-rl-dataset] building dataset -> $DATASET"
python3 "$ROOT_DIR/scripts/dreamer_trace_jsonl_to_npz.py" \
  --input "${TRACE_FILES[@]}" \
  --output "$DATASET" \
  --state-dim "$STATE_DIM" \
  --action-key "$ACTION_KEY" \
  --max-stationary-per-run "$MAX_STATIONARY" \
  --recovery-oversample "$RECOVERY_OVERSAMPLE" \
  --near-hazard-oversample "$NEAR_HAZARD_OVERSAMPLE" \
  --max-recovery-risk "$MAX_RECOVERY_RISK" \
  --max-recovery-risk-increase "$MAX_RECOVERY_RISK_INCREASE" \
  --min-recovery-ttc "$MIN_RECOVERY_TTC" \
  --max-recovery-progress-drop "$MAX_RECOVERY_PROGRESS_DROP"

echo "[dreamer-rl-dataset] auditing dataset -> $AUDIT_JSON"
python3 "$ROOT_DIR/scripts/audit_dreamer_rl_dataset.py" "$DATASET" \
  --json-output "$AUDIT_JSON" \
  --expected-state-dim "$STATE_DIM" \
  --min-transitions "$MIN_TRANSITIONS" \
  --min-runs "$MIN_RUNS" \
  --min-routes "$MIN_ROUTES" \
  --max-stationary-fraction "$MAX_STATIONARY_FRACTION" \
  --min-recovery-fraction "$MIN_RECOVERY_FRACTION"

echo "$DATASET" > "$LATEST_FILE"
echo "[dreamer-rl-dataset] OK"
echo "[dreamer-rl-dataset] latest=$LATEST_FILE"
echo "[dreamer-rl-dataset] dataset=$DATASET"
