#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SIMLINGO_PYTHON:-$HOME/miniconda3/envs/simlingo/bin/python}"
OUT_DIR="${REPORT_DREAMER_SMOKE_DIR:-$(mktemp -d /tmp/report_dreamer_smoke.XXXXXX)}"

cd "$ROOT_DIR"
echo "[report-dreamer-test] output=$OUT_DIR"
"$PYTHON" -m py_compile src/world_model/*.py \
  external/simlingo/team_code/report_dreamer_adapter.py \
  external/simlingo/team_code/agent_simlingo.py \
  scripts/train_report_dreamer.py scripts/evaluate_report_dreamer.py \
  scripts/run_report_dreamer_ab_campaign.py \
  scripts/summarize_report_dreamer_run.py \
  scripts/verify_report_dreamer_shadow_trace.py \
  scripts/promote_report_dreamer_checkpoint.py \
  scripts/finalize_report_native_trace.py
"$PYTHON" -m unittest \
  tests.test_report_aligned_dreamer \
  tests.test_report_dashboard_integration -v
"$PYTHON" scripts/train_report_dreamer.py all \
  --sequence-length 8 \
  --batch-size 4 \
  --max-traces 30 \
  --max-windows 32 \
  --epochs 1 \
  --source-policy any \
  --allow-missing-event-ground-truth \
  --trace-glob 'logs/dreamer_curriculum/**/trace.jsonl' \
  --trace-glob 'logs/dreamer_online_rl/**/trace.jsonl' \
  --trace-glob 'logs/dreamer_rl_campaign/**/traces/*.jsonl' \
  --trace-glob 'logs/action_dreaming_collect/**/*.jsonl' \
  --device cpu \
  --output "$OUT_DIR"
"$PYTHON" scripts/evaluate_report_dreamer.py \
  --checkpoint "$OUT_DIR/report_dreamer_candidate.pt" \
  --manifest "$OUT_DIR/dataset_manifest.json" \
  --device cpu \
  --max-windows 16 \
  --output "$OUT_DIR/test_prediction_metrics.json"
echo "[report-dreamer-test] PASS"
echo "[report-dreamer-test] Smoke artifacts are diagnostic candidates only: $OUT_DIR"
