#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_alpamayo_r1_local_test.sh

Useful environment variables:
  VLA_CONTROL=0|1        0 = CARLA autopilot drives while Alpamayo R1 predicts; 1 = Alpamayo R1 drives.
  LANE_ASSIST=0..1       Steering blend for safety when VLA_CONTROL=1. Use 0 for pure VLA.
  MODEL_PATH=...         Fine-tuned Alpamayo R1 checkpoint directory.
  CAMERA_WIDTH=640       Camera width for inference.
  CAMERA_HEIGHT=360      Camera height for inference.
  SPAWN_PRESET=straight  Spawn preset: straight, straight_turn, junction, traffic_law.
  SPAWN_INDEX=...        Optional deterministic CARLA spawn index.
  NAV_MANEUVER=follow_lane|auto|straight|left|right
  ROUTE_TARGET_NAV=1     Inject SimLingo-style local target point into Alpamayo prompt.
  ROUTE_TARGET_DISTANCE=18
  ROUTE_TARGET_SCAN_DISTANCE=55
  ROUTE_TARGET_STEER_BLEND=0.35
  DURATION=0             Seconds to run; 0 = until the Pygame window is closed.
EOF
  exit 0
fi

MODEL_PATH="${MODEL_PATH:-vm_backups/official_sft/intermediate/stage2/checkpoint-10528}"
REPO_PATH="${REPO_PATH:-external/alpamayo_official}"
PYTHON_PATH="${PYTHON_PATH:-external/alpamayo_official/ar1_venv/bin/python}"
VLA_CONTROL="${VLA_CONTROL:-0}"
LANE_ASSIST="${LANE_ASSIST:-0}"
CARLA_QUALITY="${CARLA_QUALITY:-Low}"
VLA_AV_CONDA_ENV="${VLA_AV_CONDA_ENV:-vla-av-step18}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "[alpamayo-r1-test] Missing python: $PYTHON_PATH" >&2
  echo "[alpamayo-r1-test] Run the local Alpamayo R1 setup first." >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "[alpamayo-r1-test] Missing checkpoint at $MODEL_PATH" >&2
  exit 1
fi

args=(
  --real
  --model alpamayo_r1
  --alpamayo-r1-model-path "$MODEL_PATH"
  --alpamayo-r1-repo "$REPO_PATH"
  --alpamayo-r1-python "$PYTHON_PATH"
  --alpamayo-r1-attn-implementation eager
  --alpamayo-dtype bfloat16
  --alpamayo-num-frames 4
  --alpamayo-history-steps 16
  --alpamayo-plan-horizon 8
  --alpamayo-lookahead-index 8
  --alpamayo-max-generation-length 256
  --alpamayo-temperature 0.6
  --alpamayo-top-p 0.98
  --camera-width "${CAMERA_WIDTH:-640}"
  --camera-height "${CAMERA_HEIGHT:-360}"
  --camera-fov "${CAMERA_FOV:-95}"
  --duration "${DURATION:-0}"
  --spawn-preset "${SPAWN_PRESET:-straight}"
  --demo-vehicles "${VEHICLES:-24}"
  --demo-two-wheelers "${TWO_WHEELERS:-12}"
  --demo-walkers "${WALKERS:-50}"
  --demo-pedestrian-cross-factor "${PEDESTRIAN_CROSS_FACTOR:-0.95}"
  --target-speed-kmh "${TARGET_SPEED_KMH:-12}"
  --alpamayo-target-speed-kmh "${ALPAMAYO_TARGET_SPEED_KMH:-12}"
  --max-vla-throttle "${MAX_VLA_THROTTLE:-0.28}"
  --alpamayo-max-throttle "${ALPAMAYO_MAX_THROTTLE:-0.28}"
  --alpamayo-max-brake "${ALPAMAYO_MAX_BRAKE:-0.75}"
  --lane-assist "$LANE_ASSIST"
  --safety-max-speed "${SAFETY_MAX_SPEED:-35}"
  --off-road-distance "${OFF_ROAD_DISTANCE:-3.0}"
  --autopilot-demo-label "CARLA autopilot"
  --nav-maneuver "${NAV_MANEUVER:-follow_lane}"
  --route-target-distance "${ROUTE_TARGET_DISTANCE:-18}"
  --route-target-scan-distance "${ROUTE_TARGET_SCAN_DISTANCE:-55}"
  --route-target-steer-blend "${ROUTE_TARGET_STEER_BLEND:-0.35}"
  --instruction "${INSTRUCTION:-You are controlling an autonomous urban vehicle in CARLA. At every frame, choose the safest immediate driving action. Stay inside the current drivable lane and follow the road geometry; if the lane bends or the road does not continue straight, steer with the lane instead of driving off road. Brake before red lights, yellow lights, stop lines, stop signs, blocked crosswalks, pedestrians, cyclists, scooters, motorcycles, and vehicles with priority. At a stop sign, make a complete stop, wait about three seconds, then proceed only when the path and right-of-way are clear. On green lights, proceed only if the lane, junction, and crosswalk are clear. Keep smooth steering, throttle, and braking. Never leave the drivable road surface.}"
)

if [[ "${ROUTE_TARGET_NAV:-1}" == "1" || "${ROUTE_TARGET_NAV:-1}" == "true" ]]; then
  args+=(--route-target-nav)
else
  args+=(--no-route-target-nav)
fi

if [[ -n "${SPAWN_INDEX:-}" ]]; then
  args+=(--spawn-index "$SPAWN_INDEX")
fi

if [[ "$VLA_CONTROL" == "1" || "$VLA_CONTROL" == "true" ]]; then
  args+=(--vla-control --warmup-autopilot)
else
  args+=(--compare)
fi

CARLA_QUALITY="$CARLA_QUALITY" VLA_AV_CONDA_ENV="$VLA_AV_CONDA_ENV" ./start.sh "${args[@]}"
