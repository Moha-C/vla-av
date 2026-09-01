#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPORT_DREAMER_PYTHON:-$HOME/miniconda3/envs/simlingo/bin/python}"
CANDIDATE="${REPORT_DREAMER_CHECKPOINT:-$ROOT_DIR/checkpoints/report_aligned_dreamer/candidate/report_dreamer_candidate.pt}"
MANIFEST="${REPORT_DREAMER_MANIFEST:-$ROOT_DIR/checkpoints/report_aligned_dreamer/candidate/dataset_manifest.json}"
PREDICTION_METRICS="${REPORT_DREAMER_PREDICTION_METRICS:-$ROOT_DIR/checkpoints/report_aligned_dreamer/candidate/test_prediction_metrics.json}"

SHADOW_ROUTE="${REPORT_DREAMER_SHADOW_ROUTE:-57}"
SHADOW_SEED="${REPORT_DREAMER_SHADOW_SEED:-20260818}"
CAMPAIGN_ROUTES="${REPORT_DREAMER_CAMPAIGN_ROUTES:-55,57}"
CAMPAIGN_SEEDS="${REPORT_DREAMER_CAMPAIGN_SEEDS:-20260818,20260819,20260820}"
CAMPAIGN_WEATHER="${REPORT_DREAMER_CAMPAIGN_WEATHER:-day}"
CAMPAIGN_TIMEOUT="${REPORT_DREAMER_CAMPAIGN_TIMEOUT:-1800}"
QUALITY="${CARLA_QUALITY:-Low}"
EVAL_DEVICE="${REPORT_DREAMER_EVAL_DEVICE:-cpu}"
RUNTIME_DEVICE="${REPORT_DREAMER_RUNTIME_DEVICE:-cpu}"

RUN_ID="${REPORT_DREAMER_VALIDATION_ID:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_DIR="$ROOT_DIR/logs/report_dreamer_validation/$RUN_ID"
SHADOW_RUN_DIR="$ROOT_DIR/logs/report_dreamer_runtime/${RUN_ID}_shadow_D_route_${SHADOW_ROUTE}_seed_${SHADOW_SEED}"
SHADOW_TRACE="$SHADOW_RUN_DIR/trace.jsonl"
SHADOW_VERIFICATION="$SHADOW_RUN_DIR/shadow_verification.json"
CAMPAIGN_DIR="$ROOT_DIR/logs/report_dreamer_campaigns/${RUN_ID}_paired_D"
CAMPAIGN_SUMMARY="$CAMPAIGN_DIR/closed_loop_ab_summary.json"
STATUS_FILE="$PIPELINE_DIR/status.env"
LOG_FILE="$PIPELINE_DIR/pipeline.log"

mkdir -p "$PIPELINE_DIR"
printf '%s\n' "$PIPELINE_DIR" > "$ROOT_DIR/logs/report_dreamer_validation/latest_run.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

phase="initialization"
write_status() {
  local state="$1"
  local detail="${2:-}"
  {
    printf 'STATE=%s\n' "$state"
    printf 'PHASE=%s\n' "$phase"
    printf 'DETAIL=%s\n' "$detail"
    printf 'RUN_ID=%s\n' "$RUN_ID"
    printf 'CANDIDATE=%s\n' "$CANDIDATE"
    printf 'PREDICTION_METRICS=%s\n' "$PREDICTION_METRICS"
    printf 'SHADOW_VERIFICATION=%s\n' "$SHADOW_VERIFICATION"
    printf 'CAMPAIGN_SUMMARY=%s\n' "$CAMPAIGN_SUMMARY"
    printf 'LOG_FILE=%s\n' "$LOG_FILE"
  } > "$STATUS_FILE"
}

fail() {
  local exit_code=$?
  write_status failed "command failed with exit code $exit_code"
  echo
  echo "[report-dreamer-pipeline] FAILED during: $phase"
  echo "[report-dreamer-pipeline] log: $LOG_FILE"
  exit "$exit_code"
}
trap fail ERR

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    return 1
  fi
}

require_file "$PYTHON"
require_file "$CANDIDATE"
require_file "$MANIFEST"

echo "=== Report-aligned Dreamer validation pipeline ==="
echo "candidate: $CANDIDATE"
echo "shadow:    route $SHADOW_ROUTE / seed $SHADOW_SEED"
echo "campaign:  routes $CAMPAIGN_ROUTES / seeds $CAMPAIGN_SEEDS"
echo "quality:   $QUALITY"
echo "devices:   evaluation=$EVAL_DEVICE / CARLA runtime=$RUNTIME_DEVICE"
echo "run:       $RUN_ID"
echo "log:       $LOG_FILE"
write_status running "pipeline started"

phase="frozen_evaluation"
write_status running "evaluating the frozen candidate on the held-out test split"
echo
echo "[1/4] Frozen test-split evaluation"
"$PYTHON" "$ROOT_DIR/scripts/evaluate_report_dreamer.py" \
  --checkpoint "$CANDIDATE" \
  --manifest "$MANIFEST" \
  --device "$EVAL_DEVICE" \
  --output "$PREDICTION_METRICS"
require_file "$PREDICTION_METRICS"

phase="carla_shadow"
write_status running "running CARLA shadow and proving native-control invariance"
echo
echo "[2/4] CARLA shadow integration test"
REPORT_DREAMER_ABLATION=D \
REPORT_DREAMER_SHADOW=1 \
REPORT_DREAMER_CHECKPOINT="$CANDIDATE" \
REPORT_DREAMER_RUN_DIR="$SHADOW_RUN_DIR" \
REPORT_DREAMER_DEVICE="$RUNTIME_DEVICE" \
ROUTE_ID="$SHADOW_ROUTE" \
SEED="$SHADOW_SEED" \
CARLA_QUALITY="$QUALITY" \
SIMLINGO_RECORD=0 \
SIMLINGO_PLAYBACK_AFTER=0 \
  bash "$ROOT_DIR/scripts/run_report_dreamer_live_test.sh"
require_file "$SHADOW_TRACE"
"$PYTHON" "$ROOT_DIR/scripts/verify_report_dreamer_shadow_trace.py" \
  --trace "$SHADOW_TRACE" \
  --output "$SHADOW_VERIFICATION"
require_file "$SHADOW_VERIFICATION"

phase="paired_ab_campaign"
write_status running "running paired native-versus-Dreamer CARLA evaluations"
echo
echo "[3/4] Paired A/D CARLA campaign"
REPORT_DREAMER_DEVICE="$RUNTIME_DEVICE" \
"$PYTHON" "$ROOT_DIR/scripts/run_report_dreamer_ab_campaign.py" \
  --checkpoint "$CANDIDATE" \
  --ablation D \
  --routes "$CAMPAIGN_ROUTES" \
  --seeds "$CAMPAIGN_SEEDS" \
  --weather "$CAMPAIGN_WEATHER" \
  --quality "$QUALITY" \
  --timeout "$CAMPAIGN_TIMEOUT" \
  --output "$CAMPAIGN_DIR"
require_file "$CAMPAIGN_SUMMARY"

phase="production_promotion"
write_status running "checking strict promotion gates"
echo
echo "[4/4] Conditional production promotion"
if "$PYTHON" "$ROOT_DIR/scripts/promote_report_dreamer_checkpoint.py" \
  --candidate "$CANDIDATE" \
  --prediction-metrics "$PREDICTION_METRICS" \
  --closed-loop-summary "$CAMPAIGN_SUMMARY"; then
  phase="complete"
  write_status complete "candidate promoted to production"
  echo
  echo "[report-dreamer-pipeline] COMPLETE: candidate promoted to production."
else
  promotion_exit=$?
  phase="promotion_rejected"
  write_status rejected "strict promotion criteria were not met; candidate preserved"
  echo
  echo "[report-dreamer-pipeline] VALIDATION FINISHED, PROMOTION REJECTED."
  echo "[report-dreamer-pipeline] The candidate was preserved and production was not modified."
  echo "[report-dreamer-pipeline] Review: $CAMPAIGN_SUMMARY"
  exit "$promotion_exit"
fi

echo "[report-dreamer-pipeline] status:  $STATUS_FILE"
echo "[report-dreamer-pipeline] summary: $CAMPAIGN_SUMMARY"
echo "[report-dreamer-pipeline] log:     $LOG_FILE"
