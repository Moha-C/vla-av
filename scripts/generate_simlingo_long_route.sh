#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-$HOME/carla_simulator}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${SIMLINGO_ENV_NAME:-simlingo}"
PORT="${PORT:-2000}"
TOWN="${TOWN:-Town12}"
SEED="${SEED:-$(date +%s)}"
SEGMENTS="${SEGMENTS:-8}"
MIN_LEG_DISTANCE="${MIN_LEG_DISTANCE:-250}"
KEYPOINT_SPACING="${KEYPOINT_SPACING:-25}"
SAMPLING_RESOLUTION="${SAMPLING_RESOLUTION:-2}"
MAX_KEYPOINTS="${MAX_KEYPOINTS:-420}"
QUALITY="${CARLA_QUALITY:-Low}"
ROUTE_OUTPUT="${ROUTE_OUTPUT:-$ROOT_DIR/generated_routes/bench2drive_long_${TOWN}_seed_${SEED}.xml}"
USE_EXISTING_CARLA="${SIMLINGO_USE_EXISTING_CARLA:-0}"
ROUTE_MODE="${SIMLINGO_ROUTE_GENERATION_MODE:-offline}"
LOCAL_CONNECTIONS="${SIMLINGO_LONG_ROUTE_LOCAL_CONNECTIONS:-1}"
CONNECT_TIMEOUT="${SIMLINGO_ROUTE_CONNECT_TIMEOUT:-180}"
LOAD_TIMEOUT="${SIMLINGO_ROUTE_LOAD_TIMEOUT:-300}"
RPC_TIMEOUT="${SIMLINGO_ROUTE_RPC_TIMEOUT:-180}"
CARLA_LOG="$ROOT_DIR/logs/simlingo_eval/route_generator_carla.log"
ROUTES_ROOT="${SIMLINGO_ROUTES_ROOT:-$ROOT_DIR/external/simlingo/leaderboard/data/bench2drive_split}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[simlingo-route] Missing conda init: $CONDA_SH" >&2
  exit 1
fi
if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" && "$USE_EXISTING_CARLA" != "1" ]]; then
  echo "[simlingo-route] Missing CARLA executable: $CARLA_ROOT/CarlaUE4.sh" >&2
  exit 1
fi

mkdir -p "$(dirname "$ROUTE_OUTPUT")" "$ROOT_DIR/logs/simlingo_eval"

if [[ "$ROUTE_MODE" == "offline" ]]; then
  echo "[simlingo-route] Generating route offline from installed Bench2Drive XMLs."
  LOCAL_ARGS=()
  if [[ "$LOCAL_CONNECTIONS" != "0" ]]; then
    LOCAL_ARGS+=(--prefer-local-connections)
  fi
  python3 "$ROOT_DIR/scripts/generate_bench2drive_offline_long_route.py" \
    --routes-root "$ROUTES_ROOT" \
    --town "$TOWN" \
    --seed "$SEED" \
    --segments "$SEGMENTS" \
    --max-keypoints "$MAX_KEYPOINTS" \
    --output "$ROUTE_OUTPUT" \
    "${LOCAL_ARGS[@]}"
  echo "[simlingo-route] ROUTE_FILE=$ROUTE_OUTPUT"
  exit 0
fi

cleanup() {
  if [[ -n "${CARLA_PID:-}" ]] && kill -0 "$CARLA_PID" 2>/dev/null; then
    kill "$CARLA_PID" 2>/dev/null || true
    wait "$CARLA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$USE_EXISTING_CARLA" != "1" ]]; then
  echo "[simlingo-route] Starting temporary CARLA for route generation on port $PORT..."
  "$CARLA_ROOT/CarlaUE4.sh" \
    -quality-level="$QUALITY" \
    -RenderOffScreen \
    -nosound \
    -carla-rpc-port="$PORT" \
    -graphicsadapter=0 \
    > "$CARLA_LOG" 2>&1 &
  CARLA_PID=$!
else
  echo "[simlingo-route] Using already running CARLA on port $PORT."
fi

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"

if ! python -u "$ROOT_DIR/scripts/generate_bench2drive_long_route.py" \
  --port "$PORT" \
  --town "$TOWN" \
  --seed "$SEED" \
  --segments "$SEGMENTS" \
  --min-leg-distance "$MIN_LEG_DISTANCE" \
  --keypoint-spacing "$KEYPOINT_SPACING" \
  --sampling-resolution "$SAMPLING_RESOLUTION" \
  --max-keypoints "$MAX_KEYPOINTS" \
  --connect-timeout "$CONNECT_TIMEOUT" \
  --load-timeout "$LOAD_TIMEOUT" \
  --rpc-timeout "$RPC_TIMEOUT" \
  --output "$ROUTE_OUTPUT"; then
  echo "[simlingo-route] Route generation failed. CARLA server log tail:" >&2
  tail -80 "$CARLA_LOG" >&2 || true
  exit 1
fi

echo "[simlingo-route] ROUTE_FILE=$ROUTE_OUTPUT"
