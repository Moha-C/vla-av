#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SIMLINGO_PYTHON:-$HOME/miniconda3/envs/simlingo/bin/python}"
MATRIX_ID="${REPORT_NATIVE_MATRIX_ID:-native_report12_v1}"
MATRIX_DIR="${REPORT_NATIVE_MATRIX_DIR:-$ROOT_DIR/data/report_dreamer/native/matrices/$MATRIX_ID}"
RUNS_DIR="${REPORT_NATIVE_RUNS_DIR:-$ROOT_DIR/data/report_dreamer/native/runs/$MATRIX_ID}"
SUMMARY_PATH="$MATRIX_DIR/summary.tsv"
EVENTS_PATH="$MATRIX_DIR/events.tsv"
STATUS_PATH="$MATRIX_DIR/status.env"
CAMPAIGN_LOG="$MATRIX_DIR/campaign.log"
DRY_RUN="${REPORT_NATIVE_MATRIX_DRY_RUN:-0}"
MAX_ATTEMPTS="${REPORT_NATIVE_MAX_ATTEMPTS:-2}"
INTER_RUN_DELAY="${REPORT_NATIVE_INTER_RUN_DELAY:-8}"
PORT="${PORT:-2000}"
RUNNER="${REPORT_NATIVE_RUNNER:-$ROOT_DIR/scripts/run_report_dreamer_native_collect.sh}"

# route_id|seed|town|scenario
RUN_MATRIX=(
  "148|2026082201|Town10HD|Accident"
  "148|2026082202|Town10HD|Accident"
  "32|2026082203|Town12|Accident"
  "32|2026082204|Town12|Accident"
  "06|2026082205|Town12|AccidentTwoWays"
  "06|2026082206|Town12|AccidentTwoWays"
  "57|2026082207|Town12|CrossingBicycleFlow"
  "57|2026082208|Town12|CrossingBicycleFlow"
  "113|2026082209|Town12|PedestrianCrossing"
  "113|2026082210|Town12|PedestrianCrossing"
  "70|2026082211|Town13|AccidentTwoWays"
  "70|2026082212|Town13|AccidentTwoWays"
)
if [[ "${REPORT_NATIVE_MATRIX_LIMIT:-}" =~ ^[0-9]+$ ]] \
  && (( REPORT_NATIVE_MATRIX_LIMIT > 0 )) \
  && (( REPORT_NATIVE_MATRIX_LIMIT < ${#RUN_MATRIX[@]} )); then
  RUN_MATRIX=("${RUN_MATRIX[@]:0:REPORT_NATIVE_MATRIX_LIMIT}")
fi

mkdir -p "$MATRIX_DIR" "$RUNS_DIR"
if [[ ! -s "$SUMMARY_PATH" ]]; then
  printf 'index\troute_id\tseed\ttown\tscenario\tstatus\tepisode\n' > "$SUMMARY_PATH"
fi
if [[ ! -s "$EVENTS_PATH" ]]; then
  printf 'timestamp\tevent\tindex\troute_id\tseed\tdetail\n' > "$EVENTS_PATH"
fi

CURRENT_INDEX=0
CURRENT_ROUTE="-"
CURRENT_SEED="-"
CAMPAIGN_STATE="starting"

write_status() {
  local completed
  completed="$(awk -F '\t' 'NR > 1 && $6 ~ /^ACCEPTED/ { count += 1 } END { print count + 0 }' "$SUMMARY_PATH")"
  {
    printf 'matrix_id=%q\n' "$MATRIX_ID"
    printf 'state=%q\n' "$CAMPAIGN_STATE"
    printf 'pid=%q\n' "$$"
    printf 'current_index=%q\n' "$CURRENT_INDEX"
    printf 'current_route=%q\n' "$CURRENT_ROUTE"
    printf 'current_seed=%q\n' "$CURRENT_SEED"
    printf 'accepted=%q\n' "$completed"
    printf 'total=%q\n' "${#RUN_MATRIX[@]}"
    printf 'updated_at=%q\n' "$(date --iso-8601=seconds)"
  } > "$STATUS_PATH.tmp"
  mv "$STATUS_PATH.tmp" "$STATUS_PATH"
}

record_event() {
  local event="$1"
  local detail="${2:-}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$event" "$CURRENT_INDEX" \
    "$CURRENT_ROUTE" "$CURRENT_SEED" "$detail" >> "$EVENTS_PATH"
  printf '[%s] %s index=%s route=%s seed=%s %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$event" "$CURRENT_INDEX" \
    "$CURRENT_ROUTE" "$CURRENT_SEED" "$detail" >> "$CAMPAIGN_LOG"
}

on_signal() {
  local signal_name="$1"
  CAMPAIGN_STATE="interrupted"
  record_event "CAMPAIGN_INTERRUPTED" "signal=$signal_name"
  write_status
  exit 130
}

trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
# A terminal hangup must not silently stop a multi-hour collection campaign.
trap 'record_event "HANGUP_IGNORED" "campaign continues"' HUP

completed_episode_for() {
  local run_base="$1"
  local candidate
  shopt -s nullglob
  local candidates=("${run_base}"_attempt_*/episode.json)
  shopt -u nullglob
  for candidate in "${candidates[@]}"; do
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

summary_contains_accepted_run() {
  local route_id="$1"
  local seed="$2"
  awk -F '\t' -v route_id="$route_id" -v seed="$seed" '
    $2 == route_id && $3 == seed && $6 ~ /^ACCEPTED/ { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "$SUMMARY_PATH"
}

next_attempt_for() {
  local run_base="$1"
  local attempt=1
  while [[ -e "${run_base}_attempt_${attempt}" ]]; do
    attempt=$((attempt + 1))
  done
  printf '%s\n' "$attempt"
}

validate_episode() {
  local episode_path="$1"
  local trace_path="$2"
  "$PYTHON" - "$episode_path" "$trace_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
trace_path = Path(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
with trace_path.open("r", encoding="utf-8") as handle:
    first_trace_row = json.loads(next(line for line in handle if line.strip()))
trace_status = first_trace_row.get("status", {})
policy_source = trace_status.get("policy_source")
if policy_source != "simlingo_native":
    raise SystemExit(
        f"unexpected policy_source in {trace_path}: {policy_source!r}"
    )
if not payload.get("bench2drive_ground_truth", False):
    raise SystemExit(f"missing Bench2Drive event ground truth in {path}")
if not payload.get("terminal_validation", {}).get("accepted", False):
    raise SystemExit(f"terminal validation was not accepted in {path}")
PY
}

wait_for_previous_run_shutdown() {
  local waited=0
  local max_wait="${REPORT_NATIVE_SHUTDOWN_TIMEOUT:-120}"
  local pattern
  local patterns=(
    "$ROOT_DIR/scripts/run_simlingo_with_pov.sh"
    "$ROOT_DIR/scripts/run_simlingo_local_eval.sh"
    "$ROOT_DIR/scripts/carla_ego_viewer.py"
    "leaderboard_evaluator.py"
  )

  while (( waited < max_wait )); do
    local alive=0
    for pattern in "${patterns[@]}"; do
      if pgrep -f "$pattern" >/dev/null 2>&1; then
        alive=1
        break
      fi
    done
    if (( alive == 0 )) && "$PYTHON" - "$PORT" <<'PY'
import socket
import sys

base = int(sys.argv[1])
sockets = []
try:
    for port in (base, base + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", port))
        sockets.append(sock)
except OSError:
    raise SystemExit(1)
finally:
    for sock in sockets:
        sock.close()
PY
    then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done

  record_event "STALE_PROCESS_CLEANUP" "waited=${waited}s port=$PORT"
  for pattern in "${patterns[@]}"; do
    pkill -TERM -f "$pattern" >/dev/null 2>&1 || true
  done
  pkill -TERM -f "CarlaUE4.*carla-rpc-port=$PORT" >/dev/null 2>&1 || true
  sleep 4
  for pattern in "${patterns[@]}"; do
    pkill -KILL -f "$pattern" >/dev/null 2>&1 || true
  done
  pkill -KILL -f "CarlaUE4.*carla-rpc-port=$PORT" >/dev/null 2>&1 || true

  # Do not start the next route until the OS also releases CARLA's RPC and
  # streaming sockets. A listening-process check misses sockets in teardown.
  while (( waited < max_wait + 120 )); do
    if "$PYTHON" - "$PORT" <<'PY'
import socket
import sys

base = int(sys.argv[1])
sockets = []
try:
    for port in (base, base + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", port))
        sockets.append(sock)
except OSError:
    raise SystemExit(1)
finally:
    for sock in sockets:
        sock.close()
PY
    then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done

  echo "[native-matrix] CARLA ports $PORT/$((PORT + 1)) remain unavailable after cleanup." >&2
  return 1
}

run_isolated() {
  local route_id="$1"
  local seed="$2"
  local run_dir="$3"
  local run_log="$run_dir/launcher.log"
  local run_status

  # CARLA and the evaluator have their own process-group cleanup. Isolating the
  # complete route prevents that cleanup from terminating this matrix runner.
  set +e
  ROUTE_ID="$route_id" \
  SEED="$seed" \
  REPORT_NATIVE_RUN_DIR="$run_dir" \
  CARLA_QUALITY="${CARLA_QUALITY:-Low}" \
  SIMLINGO_VIEW_MODE="${SIMLINGO_VIEW_MODE:-chase}" \
  SIMLINGO_RECORD="${SIMLINGO_RECORD:-0}" \
  SIMLINGO_PLAYBACK_AFTER="${SIMLINGO_PLAYBACK_AFTER:-0}" \
  SIMLINGO_VLM_COT="off" \
    setsid --wait bash "$RUNNER" 2>&1 | tee -a "$run_log"
  run_status=${PIPESTATUS[0]}
  set -e
  return "$run_status"
}

TOTAL="${#RUN_MATRIX[@]}"
write_status
echo "=== Native SimLingo report collection: $MATRIX_ID ==="
echo "runs: $TOTAL | sequential | native controls only"
echo "summary: $SUMMARY_PATH"
echo "campaign log: $CAMPAIGN_LOG"
echo
record_event "CAMPAIGN_STARTED" "total=$TOTAL max_attempts=$MAX_ATTEMPTS"

for offset in "${!RUN_MATRIX[@]}"; do
  index=$((offset + 1))
  IFS='|' read -r route_id seed town scenario <<< "${RUN_MATRIX[$offset]}"
  CURRENT_INDEX="$index"
  CURRENT_ROUTE="$route_id"
  CURRENT_SEED="$seed"
  CAMPAIGN_STATE="running"
  write_status
  run_base="$RUNS_DIR/run_$(printf '%02d' "$index")_route_${route_id}_seed_${seed}"

  if episode_path="$(completed_episode_for "$run_base")"; then
    existing_trace="$(dirname "$episode_path")/trace.jsonl"
    set +e
    validate_episode "$episode_path" "$existing_trace"
    validation_status=$?
    set -e
    if [[ "$validation_status" -eq 0 ]]; then
      echo "[$index/$TOTAL] SKIP route=$route_id seed=$seed $town/$scenario"
      echo "             accepted episode: $episode_path"
      if ! summary_contains_accepted_run "$route_id" "$seed"; then
        printf '%s\t%s\t%s\t%s\t%s\tACCEPTED_EXISTING\t%s\n' \
          "$index" "$route_id" "$seed" "$town" "$scenario" "$episode_path" >> "$SUMMARY_PATH"
      fi
      record_event "RUN_SKIPPED_ACCEPTED" "episode=$episode_path"
      write_status
      continue
    fi
    record_event "EXISTING_EPISODE_REJECTED" "episode=$episode_path validation_exit=$validation_status"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    attempt="$(next_attempt_for "$run_base")"
    echo "[$index/$TOTAL] DRY route=$route_id seed=$seed $town/$scenario"
    echo "             dry-run: ${run_base}_attempt_${attempt}"
    continue
  fi

  route_accepted=0
  for (( route_attempt=1; route_attempt<=MAX_ATTEMPTS; route_attempt++ )); do
    wait_for_previous_run_shutdown
    attempt="$(next_attempt_for "$run_base")"
    run_dir="${run_base}_attempt_${attempt}"
    mkdir -p "$run_dir"

    echo "[$index/$TOTAL] RUN route=$route_id seed=$seed $town/$scenario (attempt $attempt)"
    record_event "RUN_STARTED" "attempt=$attempt run_dir=$run_dir"
    if run_isolated "$route_id" "$seed" "$run_dir"; then
      run_status=0
    else
      run_status=$?
    fi
    printf '%s\n' "$run_status" > "$run_dir/launcher_exit_code.txt"

    episode_path="$run_dir/episode.json"
    trace_path="$run_dir/trace.jsonl"
    if [[ ! -s "$episode_path" || ! -s "$trace_path" ]]; then
      record_event "RUN_INCOMPLETE" "attempt=$attempt exit=$run_status episode=$episode_path"
      echo "[native-matrix] Incomplete run $index/$TOTAL (attempt $attempt, exit=$run_status); cleaning up and retrying." >&2
      wait_for_previous_run_shutdown
      continue
    fi

    set +e
    validate_episode "$episode_path" "$trace_path"
    validation_status=$?
    set -e
    if [[ "$validation_status" -ne 0 ]]; then
      printf '%s\t%s\t%s\t%s\t%s\tREJECTED_VALIDATION\t%s\n' \
        "$index" "$route_id" "$seed" "$town" "$scenario" "$episode_path" >> "$SUMMARY_PATH"
      record_event "RUN_REJECTED" "attempt=$attempt validation_exit=$validation_status episode=$episode_path"
      echo "[native-matrix] Run $index/$TOTAL was finalized but rejected by provenance validation; continuing." >&2
      break
    fi

    accepted_status="ACCEPTED"
    if [[ "$run_status" -ne 0 ]]; then
      accepted_status="ACCEPTED_NONZERO_EXIT"
      echo "[native-matrix] Evaluator exit=$run_status, but the authoritative Bench2Drive episode is complete and eligible; continuing."
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$index" "$route_id" "$seed" "$town" "$scenario" "$accepted_status" "$episode_path" >> "$SUMMARY_PATH"
    record_event "RUN_ACCEPTED" "attempt=$attempt status=$accepted_status episode=$episode_path"
    echo "[$index/$TOTAL] ACCEPTED: $episode_path"
    route_accepted=1
    write_status
    break
  done

  if [[ "$route_accepted" -ne 1 ]]; then
    if ! summary_contains_accepted_run "$route_id" "$seed"; then
      printf '%s\t%s\t%s\t%s\t%s\tFAILED_AFTER_RETRIES\t%s\n' \
        "$index" "$route_id" "$seed" "$town" "$scenario" "${episode_path:-}" >> "$SUMMARY_PATH"
    fi
    record_event "RUN_GAVE_UP" "attempts=$MAX_ATTEMPTS; continuing to next matrix row"
    echo "[native-matrix] Route $index/$TOTAL failed after $MAX_ATTEMPTS attempt(s); later routes will still run." >&2
  fi

  wait_for_previous_run_shutdown
  if (( index < TOTAL && INTER_RUN_DELAY > 0 )); then
    CAMPAIGN_STATE="inter_run_pause"
    write_status
    echo "[native-matrix] Next route starts in ${INTER_RUN_DELAY}s."
    sleep "$INTER_RUN_DELAY"
  fi
  echo
done

if [[ "$DRY_RUN" == "1" ]]; then
  CAMPAIGN_STATE="dry_run_complete"
  write_status
  echo "[native-matrix] Dry-run complete; no simulation was launched."
  exit 0
fi

accepted_total="$(awk -F '\t' 'NR > 1 && $6 ~ /^ACCEPTED/ { count += 1 } END { print count + 0 }' "$SUMMARY_PATH")"
CAMPAIGN_STATE="complete"
write_status
record_event "CAMPAIGN_COMPLETE" "accepted=$accepted_total total=$TOTAL"
echo "[native-matrix] Campaign complete: $accepted_total/$TOTAL native SimLingo runs accepted."
echo "[native-matrix] Dataset summary: $SUMMARY_PATH"
echo "[native-matrix] Next step: audit the dataset before any training."
