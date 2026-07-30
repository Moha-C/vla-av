#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/mohm/miniconda3/envs/simlingo/bin/python}"
SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export SUMO_HOME

ROUTE_ID="${ROUTE_ID:-114}"
if [[ "$ROUTE_ID" =~ ^[0-9]+$ ]]; then
  ROUTE_PADDED="$(printf "%02d" "$((10#$ROUTE_ID))")"
else
  ROUTE_PADDED="$ROUTE_ID"
fi
ROUTE_FILE="${ROUTE_FILE:-$ROOT_DIR/external/simlingo/leaderboard/data/bench2drive_split/bench2drive_${ROUTE_ID}.xml}"
if [[ ! -f "$ROUTE_FILE" ]]; then
  ROUTE_FILE="$ROOT_DIR/external/simlingo/leaderboard/data/bench2drive_split/bench2drive_${ROUTE_PADDED}.xml"
fi
if [[ ! -f "$ROUTE_FILE" ]]; then
  echo "[simlingo-sumo-view] route XML not found for ROUTE_ID=$ROUTE_ID" >&2
  exit 1
fi

TOWN="${TOWN:-$("$PYTHON" - "$ROUTE_FILE" <<'PY'
import sys
import xml.etree.ElementTree as ET
route = ET.parse(sys.argv[1]).getroot().find("route")
print(route.attrib.get("town", "Town12"))
PY
)}"

NET_OUTPUT_DIR="${NET_OUTPUT_DIR:-$ROOT_DIR/generated_sumo_nets}"
ROUTE_OUTPUT_DIR="${ROUTE_OUTPUT_DIR:-$ROOT_DIR/generated_sumo_routes}"

echo "[simlingo-sumo-view] route=$ROUTE_FILE"
echo "[simlingo-sumo-view] town=$TOWN"

"$PYTHON" "$ROOT_DIR/scripts/generate_carla_sumo_net.py" \
  --town "$TOWN" \
  --output-dir "$NET_OUTPUT_DIR"

NET_FILE="$NET_OUTPUT_DIR/$TOWN/$TOWN.net.xml"
"$PYTHON" "$ROOT_DIR/scripts/bench2drive_route_to_sumo.py" \
  --route-xml "$ROUTE_FILE" \
  --net-file "$NET_FILE" \
  --output-dir "$ROUTE_OUTPUT_DIR"

SUMOCFG="$ROUTE_OUTPUT_DIR/$(basename "$ROUTE_FILE" .xml)/$(basename "$ROUTE_FILE" .xml).sumocfg"
echo "[simlingo-sumo-view] SUMOCFG=$SUMOCFG"

if [[ "${SUMO_PREPARE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${SUMO_GUI:-0}" == "1" ]]; then
  echo "[simlingo-sumo-view] opening SUMO GUI"
  sumo-gui -c "$SUMOCFG"
else
  echo "[simlingo-sumo-view] validating with headless SUMO"
  sumo -c "$SUMOCFG" --begin 0 --end "${SUMO_VALIDATE_END:-3}" --no-warnings true
fi
