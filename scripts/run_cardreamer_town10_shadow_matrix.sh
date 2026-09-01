#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS_TEXT="${CARDREAMER_SHADOW_SEEDS:-14823 14837 14851}"
TRACE_TARGET="${CARDREAMER_SHADOW_TRACE_TARGET:-110}"
MAX_WALL_SECONDS="${CARDREAMER_SHADOW_MAX_WALL_SECONDS:-420}"
LATERAL_ADAPTER="${CARDREAMER_SHADOW_LATERAL_ADAPTER:-mirror}"
DISPLAY_VALUE="${DISPLAY:-:1}"
SUMMARY_DIR="$ROOT_DIR/artifacts/cardreamer_integration_20260817/shadow"

mkdir -p "$SUMMARY_DIR"
traces=()

for seed in $SEEDS_TEXT; do
  run_id="town10hd_${LATERAL_ADAPTER}_seed_${seed}"
  run_dir="$ROOT_DIR/logs/cardreamer_shadow/$run_id"
  trace="$run_dir/trace.jsonl"
  traces+=("$trace")
  mkdir -p "$run_dir"

  existing=0
  if [[ -f "$trace" ]]; then
    existing="$(wc -l < "$trace")"
  fi
  if (( existing >= TRACE_TARGET )); then
    echo "[cardreamer-matrix] seed=$seed already complete decisions=$existing"
    continue
  fi

  bash "$ROOT_DIR/scripts/stop_simlingo_dashboard.sh" >/dev/null 2>&1 || true
  echo "[cardreamer-matrix] seed=$seed adapter=$LATERAL_ADAPTER target=$TRACE_TARGET"
  setsid env \
    DISPLAY="$DISPLAY_VALUE" \
    ROUTE_ID=148 \
    TOWN=Town10HD \
    SEED="$seed" \
    PORT=2000 \
    TM_PORT=8000 \
    SIMLINGO_CARDREAMER_MODE=shadow \
    SIMLINGO_CARDREAMER_LATERAL_ADAPTER="$LATERAL_ADAPTER" \
    SIMLINGO_CARDREAMER_RUN_ID="$run_id" \
    SIMLINGO_DREAMER_GUARD=0 \
    SIMLINGO_VLM_COT=off \
    SIMLINGO_RECORD=0 \
    SIMLINGO_PLAYBACK_AFTER=0 \
    SIMLINGO_VIEW_MODE=chase \
    SIMLINGO_VIEW_WIDTH=1280 \
    SIMLINGO_VIEW_HEIGHT=720 \
    SIMLINGO_VISUAL_WEATHER=day \
    CARLA_QUALITY=Low \
    bash "$ROOT_DIR/scripts/run_simlingo_with_pov.sh" \
    >"$run_dir/launcher.log" 2>&1 &
  run_pid=$!
  started="$(date +%s)"

  while kill -0 "$run_pid" 2>/dev/null; do
    decisions=0
    if [[ -f "$trace" ]]; then
      decisions="$(wc -l < "$trace")"
    fi
    elapsed=$(( $(date +%s) - started ))
    echo "[cardreamer-matrix] seed=$seed decisions=$decisions elapsed=${elapsed}s"
    if (( decisions >= TRACE_TARGET || elapsed >= MAX_WALL_SECONDS )); then
      kill -TERM -- "-$run_pid" 2>/dev/null || true
      break
    fi
    sleep 10
  done
  wait "$run_pid" 2>/dev/null || true
  bash "$ROOT_DIR/scripts/stop_simlingo_dashboard.sh" >/dev/null 2>&1 || true
  final=0
  if [[ -f "$trace" ]]; then
    final="$(wc -l < "$trace")"
  fi
  echo "[cardreamer-matrix] seed=$seed finished decisions=$final"
done

summary="$SUMMARY_DIR/${LATERAL_ADAPTER}_matrix_summary.json"
summary_status=0
python3 "$ROOT_DIR/scripts/summarize_cardreamer_shadow.py" \
  "${traces[@]}" \
  --output "$summary" || summary_status=$?

echo "[cardreamer-matrix] summary=$summary gate_exit=$summary_status"
exit 0
