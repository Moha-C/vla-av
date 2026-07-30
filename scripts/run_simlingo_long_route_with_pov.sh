#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOWN="${TOWN:-Town12}"
SEED="${SEED:-$(date +%s)}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
SEGMENTS="${SEGMENTS:-12}"
MAX_KEYPOINTS="${MAX_KEYPOINTS:-420}"
ROUTE_OUTPUT="${ROUTE_OUTPUT:-$ROOT_DIR/generated_routes/bench2drive_long_${TOWN}_seed_${SEED}.xml}"

echo "[simlingo-long] Generating long custom route for $TOWN..."
TOWN="$TOWN" \
SEED="$SEED" \
PORT="$PORT" \
SEGMENTS="$SEGMENTS" \
MAX_KEYPOINTS="$MAX_KEYPOINTS" \
ROUTE_OUTPUT="$ROUTE_OUTPUT" \
bash "$ROOT_DIR/scripts/generate_simlingo_long_route.sh"

echo "[simlingo-long] Launching SimLingo on custom route:"
echo "[simlingo-long] $ROUTE_OUTPUT"
ROUTE_FILE="$ROUTE_OUTPUT" \
TOWN="$TOWN" \
SEED="$SEED" \
PORT="$PORT" \
TM_PORT="$TM_PORT" \
SIMLINGO_VIEW_MODE="${SIMLINGO_VIEW_MODE:-chase}" \
SIMLINGO_VISUAL_WEATHER="${SIMLINGO_VISUAL_WEATHER:-day}" \
bash "$ROOT_DIR/scripts/run_simlingo_with_pov.sh"
