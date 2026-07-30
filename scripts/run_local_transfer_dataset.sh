#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  RUNS=1 FRAMES=49 TRANSFER_RESOLUTION=480 TRANSFER_STEPS=24 bash scripts/run_local_transfer_dataset.sh

Environment variables:
  RUNS, START_INDEX, RUN_PREFIX, DATASET_DIR
  FRAMES, FPS, WIDTH, HEIGHT, CAMERA_PRESET
  TRANSFER_RESOLUTION=480|720, TRANSFER_STEPS, TRANSFER_GUIDANCE
  VEHICLES, TWO_WHEELERS, WALKERS, SPAWN_PRESETS, WEATHER_PROMPTS
  CARLA_PATH, CARLA_QUALITY, CARLA_OFFSCREEN
  DRY_RUN=1 prints the planned runs without launching CARLA or Transfer.
EOF
  exit 0
fi

CONDA_BIN="${CONDA_BIN:-/home/mohm/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-vla-av-step18}"
DRY_RUN="${DRY_RUN:-0}"

RUNS="${RUNS:-2}"
START_INDEX="${START_INDEX:-1}"
RUN_PREFIX="${RUN_PREFIX:-transfer25_local_hq}"
DATASET_DIR="${DATASET_DIR:-data/alpamayo_transfer_dataset_local_hq}"

FRAMES="${FRAMES:-49}"
FPS="${FPS:-10}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-540}"
CAMERA_PRESET="${CAMERA_PRESET:-hood}"

VEHICLES="${VEHICLES:-30}"
TWO_WHEELERS="${TWO_WHEELERS:-16}"
WALKERS="${WALKERS:-70}"
PEDESTRIAN_CROSS_FACTOR="${PEDESTRIAN_CROSS_FACTOR:-0.95}"
TRAFFIC_SPEED_DIFFERENCE="${TRAFFIC_SPEED_DIFFERENCE:-20}"
EGO_SPEED_DIFFERENCE="${EGO_SPEED_DIFFERENCE:-10}"
SPAWN_TOP_K="${SPAWN_TOP_K:-80}"
SEED_BASE="${SEED_BASE:-10017}"

TRANSFER_RESOLUTION="${TRANSFER_RESOLUTION:-480}"
TRANSFER_STEPS="${TRANSFER_STEPS:-24}"
TRANSFER_GUIDANCE="${TRANSFER_GUIDANCE:-6.5}"
TRANSFER_MAX_FRAMES="${TRANSFER_MAX_FRAMES:-$FRAMES}"
SEG_WEIGHT="${SEG_WEIGHT:-1.0}"
DEPTH_WEIGHT="${DEPTH_WEIGHT:-0.0}"
VIS_WEIGHT="${VIS_WEIGHT:-0.0}"
NUM_GPUS="${NUM_GPUS:-1}"

CARLA_PATH="${CARLA_PATH:-/home/mohm/carla_simulator/CarlaUE4.sh}"
CARLA_QUALITY="${CARLA_QUALITY:-Epic}"
CARLA_OFFSCREEN="${CARLA_OFFSCREEN:-0}"

INSTRUCTION="${INSTRUCTION:-Drive like a safe autonomous vehicle in an urban environment. Follow the current lane and road markings, keep a smooth centered trajectory, respect speed limits, obey red lights, green lights, stop signs, lane arrows, crosswalks, priority rules, and right-of-way. Yield to pedestrians, cyclists, scooters, motorbikes, parked cars pulling out, and other vehicles. Stop when the path is blocked, wait until it is clear, then continue smoothly without leaving the drivable lane.}"

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-CGI, video game, cyberpunk, neon glow, handheld camera, phone in hand, dashcam holder, cartoon, anime, oversaturated colors, distorted buildings, warped lane markings, melted texture, flicker, motion smear, motion blur, blurry, out of focus, soft focus, depth of field blur, compression artifacts, black border, letterbox, pillarbox, low quality}"

SPAWN_PRESETS="${SPAWN_PRESETS:-traffic_law,straight_turn,junction,traffic_light,stop_or_light}"
WEATHER_PROMPTS="${WEATHER_PROMPTS:-sharp forward-facing hood-mounted automotive camera, crisp focus, high detail, stable camera, natural realistic colors, accurate lane markings, realistic buildings, realistic vehicles, clean road texture, no cinematic blur, clear daytime urban street, realistic exposure|cloudy morning realistic hood-mounted automotive camera, crisp focus, neutral colors, realistic buildings, accurate road markings, clear road texture|wet road after rain, realistic hood-mounted automotive camera, crisp focus, natural reflections on asphalt, accurate lane markings, realistic vehicles, no cinematic blur|golden hour urban automotive camera, crisp focus, realistic warm sunlight, accurate lane markings, realistic buildings and vehicles, stable camera|night urban hood-mounted automotive camera, crisp focus, realistic headlights and street lights, accurate lane markings, no neon cyberpunk}"

format_seconds() {
  local total="${1:-0}"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf "%02dh%02dm%02ds" "$h" "$m" "$s"
}

pick_csv() {
  local csv="$1"
  local idx="$2"
  IFS=',' read -r -a items <<< "$csv"
  local count="${#items[@]}"
  printf "%s" "${items[$(((idx - 1) % count))]}"
}

pick_pipe() {
  local value="$1"
  local idx="$2"
  IFS='|' read -r -a items <<< "$value"
  local count="${#items[@]}"
  printf "%s" "${items[$(((idx - 1) % count))]}"
}

echo "[local-dataset] Local CARLA -> Cosmos Transfer2.5 -> Alpamayo manifest"
echo "[local-dataset] runs=${RUNS} frames=${FRAMES} resolution=${TRANSFER_RESOLUTION} steps=${TRANSFER_STEPS} dataset=${DATASET_DIR}"
echo "[local-dataset] conda=${CONDA_BIN} env=${CONDA_ENV}"

"$CONDA_BIN" run -n "$CONDA_ENV" python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible from the conda environment.")
print(f"[local-dataset] CUDA OK: {torch.cuda.get_device_name(0)}")
PY

start_ts="$(date +%s)"
completed=0

for offset in $(seq 0 $((RUNS - 1))); do
  run_number=$((START_INDEX + offset))
  run_name="$(printf "%s_%04d" "$RUN_PREFIX" "$run_number")"
  run_dir="data/synthetic/transferred_real/${run_name}"
  expected_video="${run_dir}/transfer_output/${run_name}.mp4"
  spawn_preset="$(pick_csv "$SPAWN_PRESETS" "$run_number")"
  weather="$(pick_pipe "$WEATHER_PROMPTS" "$run_number")"
  scenario_seed=$((SEED_BASE + run_number * 17))
  percent=$((completed * 100 / RUNS))

  elapsed=$(( $(date +%s) - start_ts ))
  eta="unknown"
  if [[ "$completed" -gt 0 ]]; then
    avg=$((elapsed / completed))
    eta="$(format_seconds $(((RUNS - completed) * avg)))"
  fi

  echo
  echo "[local-dataset] progress=${percent}% done=${completed}/${RUNS} elapsed=$(format_seconds "$elapsed") eta=${eta}"
  echo "[local-dataset] run=${run_name} seed=${scenario_seed} spawn=${spawn_preset}"

  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "[local-dataset] dry-run weather=${weather}"
    completed=$((completed + 1))
    continue
  fi

  if [[ -f "$expected_video" ]]; then
    echo "[local-dataset] transfer output already exists, skipping generation: ${expected_video}"
    completed=$((completed + 1))
    continue
  fi

  reuse_args=()
  if [[ -d "$run_dir" && -f "${run_dir}/carla_rgb.mp4" && -f "${run_dir}/carla_seg.mp4" && -f "${run_dir}/carla_depth.mp4" ]]; then
    echo "[local-dataset] reusing existing CARLA controls for incomplete run ${run_name}"
    reuse_args=(--reuse-run-dir "$run_dir")
  elif [[ -d "$run_dir" ]]; then
    echo "[local-dataset] ERROR: ${run_dir} exists but does not contain complete CARLA videos." >&2
    echo "[local-dataset] Move it to backup or choose START_INDEX/RUN_PREFIX before retrying." >&2
    exit 1
  fi

  carla_offscreen_args=()
  if [[ "$CARLA_OFFSCREEN" == "1" || "$CARLA_OFFSCREEN" == "true" ]]; then
    carla_offscreen_args=(--carla-offscreen)
  fi

  CARLA_QUALITY="$CARLA_QUALITY" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$CONDA_BIN" run -n "$CONDA_ENV" python scripts/cosmos_transfer_real.py \
    "${reuse_args[@]}" \
    --run-name "$run_name" \
    --camera-preset "$CAMERA_PRESET" \
    --frames "$FRAMES" \
    --fps "$FPS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --spawn-preset "$spawn_preset" \
    --spawn-top-k "$SPAWN_TOP_K" \
    --vehicles "$VEHICLES" \
    --two-wheelers "$TWO_WHEELERS" \
    --walkers "$WALKERS" \
    --pedestrian-cross-factor "$PEDESTRIAN_CROSS_FACTOR" \
    --traffic-speed-difference "$TRAFFIC_SPEED_DIFFERENCE" \
    --ego-speed-difference "$EGO_SPEED_DIFFERENCE" \
    --scenario-seed "$scenario_seed" \
    --weather "$weather" \
    --instruction "$INSTRUCTION" \
    --negative-prompt "$NEGATIVE_PROMPT" \
    --guidance "$TRANSFER_GUIDANCE" \
    --seg-weight "$SEG_WEIGHT" \
    --depth-weight "$DEPTH_WEIGHT" \
    --vis-weight "$VIS_WEIGHT" \
    --transfer-resolution "$TRANSFER_RESOLUTION" \
    --transfer-max-frames "$TRANSFER_MAX_FRAMES" \
    --transfer-num-steps "$TRANSFER_STEPS" \
    --no-keep-input-resolution \
    --carla-path "$CARLA_PATH" \
    "${carla_offscreen_args[@]}" \
    --num-gpus "$NUM_GPUS" \
    --run-transfer

  completed=$((completed + 1))
done

elapsed=$(( $(date +%s) - start_ts ))
echo
echo "[local-dataset] progress=100% done=${completed}/${RUNS} elapsed=$(format_seconds "$elapsed") eta=00h00m00s"
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "[local-dataset] dry-run complete; manifest build skipped."
  exit 0
fi
echo "[local-dataset] Building manifest at ${DATASET_DIR}"

"$CONDA_BIN" run -n "$CONDA_ENV" python scripts/prepare_alpamayo_transfer_dataset.py \
  --runs-dir data/synthetic/transferred_real \
  --run-glob "${RUN_PREFIX}_*" \
  --output-dir "$DATASET_DIR" \
  --history-steps 16 \
  --future-steps 64 \
  --dt 0.1 \
  --camera-index 1 \
  --jpeg-quality 97

echo "[local-dataset] Done: ${DATASET_DIR}/manifest.jsonl"
