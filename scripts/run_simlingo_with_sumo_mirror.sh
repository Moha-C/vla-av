#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/mohm/miniconda3/envs/simlingo/bin/python}"
SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export SUMO_HOME

PORT="${PORT:-2000}"
ROUTE_ID="${ROUTE_ID:-114}"
ROUTE_FILE="${ROUTE_FILE:-}"
SUMO_MIRROR_LOG_DIR="${SUMO_MIRROR_LOG_DIR:-$ROOT_DIR/logs/sumo_mirror}"
SUMO_MIRROR_LOG="$SUMO_MIRROR_LOG_DIR/latest_mirror.log"
SUMO_MIRROR_SUMMARY="$SUMO_MIRROR_LOG_DIR/latest_summary.json"
SUMO_MIRROR_ATTACK_COMMANDS="${SUMO_MIRROR_ATTACK_COMMANDS:-$SUMO_MIRROR_LOG_DIR/attack_commands.jsonl}"
SUMO_MIRROR_LIVE_STATE="${SUMO_MIRROR_LIVE_STATE:-$SUMO_MIRROR_LOG_DIR/live_state.json}"
SUMO_MIRROR_GUI="${SUMO_MIRROR_GUI:-1}"
SUMO_MIRROR_START_DELAY="${SUMO_MIRROR_START_DELAY:-0}"
SUMO_MIRROR_SYNC_TLS="${SUMO_MIRROR_SYNC_TLS:-1}"
SUMO_MIRROR_POLL="${SUMO_MIRROR_POLL:-0.05}"
SUMO_MIRROR_NO_WARNINGS="${SUMO_MIRROR_NO_WARNINGS:-1}"
SUMO_MIRROR_WAIT_FOR_VEHICLES="${SUMO_MIRROR_WAIT_FOR_VEHICLES:-1}"
SUMO_MIRROR_WAIT_TIMEOUT="${SUMO_MIRROR_WAIT_TIMEOUT:-240}"
SUMO_MIRROR_ATTACK_RED_STOP_DISTANCE="${SUMO_MIRROR_ATTACK_RED_STOP_DISTANCE:-42}"
mkdir -p "$SUMO_MIRROR_LOG_DIR"

cleanup() {
  if [[ -n "${MIRROR_PID:-}" ]] && kill -0 "$MIRROR_PID" 2>/dev/null; then
    kill -INT "$MIRROR_PID" 2>/dev/null || true
    wait "$MIRROR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[simlingo+sumo] preparing SUMO net/route overlay"
PREPARE_ENV=(
  SUMO_PREPARE_ONLY=1
  ROUTE_ID="$ROUTE_ID"
  SUMO_VALIDATE_END=1
)
if [[ -n "$ROUTE_FILE" ]]; then
  PREPARE_ENV+=(ROUTE_FILE="$ROUTE_FILE")
fi
env "${PREPARE_ENV[@]}" bash "$ROOT_DIR/scripts/prepare_simlingo_route_sumo_view.sh"

if [[ -n "$ROUTE_FILE" ]]; then
  ROUTE_BASENAME="$(basename "$ROUTE_FILE" .xml)"
else
  if [[ "$ROUTE_ID" =~ ^[0-9]+$ ]]; then
    ROUTE_BASENAME="bench2drive_$(printf "%02d" "$((10#$ROUTE_ID))")"
    if [[ ! -d "$ROOT_DIR/generated_sumo_routes/$ROUTE_BASENAME" ]]; then
      ROUTE_BASENAME="bench2drive_$((10#$ROUTE_ID))"
    fi
  else
    ROUTE_BASENAME="bench2drive_$ROUTE_ID"
  fi
fi

SUMMARY_FILE="$ROOT_DIR/generated_sumo_routes/$ROUTE_BASENAME/$ROUTE_BASENAME.summary.json"
if [[ ! -f "$SUMMARY_FILE" ]]; then
  echo "[simlingo+sumo] missing SUMO route summary: $SUMMARY_FILE" >&2
  exit 1
fi

read_json_field() {
  "$PYTHON" - "$SUMMARY_FILE" "$1" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
value = data
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

NET_FILE="$(read_json_field net_file)"
ADDITIONAL_FILE="$(read_json_field additional_file)"
TOWN_FROM_ROUTE="$(read_json_field metadata.town)"
export TOWN="${TOWN:-$TOWN_FROM_ROUTE}"

echo "[simlingo+sumo] route=$ROUTE_BASENAME town=$TOWN"
echo "[simlingo+sumo] net=$NET_FILE"
echo "[simlingo+sumo] overlay=$ADDITIONAL_FILE"
echo "[simlingo+sumo] attack_commands=$SUMO_MIRROR_ATTACK_COMMANDS"
echo "[simlingo+sumo] live_state=$SUMO_MIRROR_LIVE_STATE"
echo "[simlingo+sumo] wait_for_vehicles=$SUMO_MIRROR_WAIT_FOR_VEHICLES timeout=${SUMO_MIRROR_WAIT_TIMEOUT}s"
echo "[simlingo+sumo] attack_red_stop_distance=${SUMO_MIRROR_ATTACK_RED_STOP_DISTANCE}m"
echo "[simlingo+sumo] starting SUMO mirror; log=$SUMO_MIRROR_LOG"
(
  sleep "$SUMO_MIRROR_START_DELAY"
  ARGS=(
    --host 127.0.0.1
    --port "$PORT"
    --tm-port "${TM_PORT:-8000}"
    --net-file "$NET_FILE"
    --additional-file "$ADDITIONAL_FILE"
    --output-dir "$SUMO_MIRROR_LOG_DIR/runtime"
    --poll-interval "$SUMO_MIRROR_POLL"
    --wait-for-vehicles "$SUMO_MIRROR_WAIT_FOR_VEHICLES"
    --wait-for-vehicles-timeout "$SUMO_MIRROR_WAIT_TIMEOUT"
    --attack-red-stop-distance "$SUMO_MIRROR_ATTACK_RED_STOP_DISTANCE"
    --summary "$SUMO_MIRROR_SUMMARY"
    --attack-command-file "$SUMO_MIRROR_ATTACK_COMMANDS"
    --live-state-file "$SUMO_MIRROR_LIVE_STATE"
  )
  if [[ "$SUMO_MIRROR_GUI" == "1" ]]; then
    ARGS+=(--sumo-gui)
  fi
  if [[ "$SUMO_MIRROR_SYNC_TLS" == "1" ]]; then
    ARGS+=(--sync-traffic-lights)
  fi
  if [[ "$SUMO_MIRROR_NO_WARNINGS" == "1" ]]; then
    ARGS+=(--no-warnings)
  fi
  "$PYTHON" "$ROOT_DIR/scripts/carla_sumo_mirror.py" "${ARGS[@]}"
) >"$SUMO_MIRROR_LOG" 2>&1 &
MIRROR_PID=$!
echo "$MIRROR_PID" > "$SUMO_MIRROR_LOG_DIR/latest_mirror.pid"

echo "[simlingo+sumo] starting SimLingo POV run"
ROUTE_ID="$ROUTE_ID" ROUTE_FILE="$ROUTE_FILE" PORT="$PORT" bash "$ROOT_DIR/scripts/run_simlingo_with_pov.sh"
STATUS=$?

cleanup
echo "[simlingo+sumo] SUMO mirror log tail:"
tail -40 "$SUMO_MIRROR_LOG" || true
exit "$STATUS"
