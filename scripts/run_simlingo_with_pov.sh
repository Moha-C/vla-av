#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${SIMLINGO_ENV_NAME:-simlingo}"
PORT="${PORT:-2000}"
VIEW_WIDTH="${SIMLINGO_VIEW_WIDTH:-1280}"
VIEW_HEIGHT="${SIMLINGO_VIEW_HEIGHT:-720}"
VIEW_FOV="${SIMLINGO_VIEW_FOV:-95}"
VIEW_FPS="${SIMLINGO_VIEW_FPS:-45}"
VIEW_MODE="${SIMLINGO_VIEW_MODE:-chase}"
VISUAL_WEATHER="${SIMLINGO_VISUAL_WEATHER:-day}"
export SIMLINGO_PROFILE_EVERY="${SIMLINGO_PROFILE_EVERY:-40}"
export SIMLINGO_DRAW_WAYPOINTS="${SIMLINGO_DRAW_WAYPOINTS:-1}"
RECORD_DIR="${SIMLINGO_RECORD_DIR:-$ROOT_DIR/logs/simlingo_eval/recordings}"
RECORD_PATH="${SIMLINGO_RECORD_PATH:-$RECORD_DIR/simlingo_$(date +%Y%m%d_%H%M%S).mp4}"
PLAYBACK_AFTER="${SIMLINGO_PLAYBACK_AFTER:-1}"
PLAYBACK_SPEED="${SIMLINGO_PLAYBACK_SPEED:-5}"
PLAYBACK_LOG="${SIMLINGO_PLAYBACK_LOG:-$ROOT_DIR/logs/simlingo_eval/latest_replay.log}"
VIEWER_LOG="${SIMLINGO_VIEWER_LOG:-$ROOT_DIR/logs/simlingo_eval/latest_pov_viewer.log}"
DREAMER_STATUS_PATH="${SIMLINGO_DREAMER_STATUS_PATH:-$ROOT_DIR/logs/simlingo_eval/dreamer_guard_status.json}"
export SIMLINGO_DREAMER_STATUS_PATH="$DREAMER_STATUS_PATH"
VLM_COT_MODE="${SIMLINGO_VLM_COT:-off}"
VLM_COT_STATUS_PATH="${SIMLINGO_VLM_COT_STATUS_PATH:-$ROOT_DIR/logs/simlingo_eval/vlm_cot_status.json}"
VLM_COT_FRAME_PATH="${SIMLINGO_VLM_COT_FRAME_PATH:-$ROOT_DIR/logs/simlingo_eval/vlm_cot_frame.jpg}"
VLM_COT_LOG_PATH="${SIMLINGO_VLM_COT_LOG_PATH:-$ROOT_DIR/logs/simlingo_eval/vlm_cot_reasoning.jsonl}"
export SIMLINGO_VLM_COT_STATUS_PATH="$VLM_COT_STATUS_PATH"
export SIMLINGO_VLM_COT_FRAME_PATH="$VLM_COT_FRAME_PATH"
export SIMLINGO_VLM_COT_LOG_PATH="$VLM_COT_LOG_PATH"
export SDL_VIDEO_X11_FORCE_EGL="${SDL_VIDEO_X11_FORCE_EGL:-1}"

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

mkdir -p "$ROOT_DIR/logs/simlingo_eval"
rm -f "$DREAMER_STATUS_PATH"
rm -f "$VLM_COT_STATUS_PATH" "$VLM_COT_FRAME_PATH" "$VLM_COT_LOG_PATH"
: > "$VIEWER_LOG"

export SIMLINGO_RENDER_MODE="${SIMLINGO_RENDER_MODE:-offscreen}"

cleanup() {
  if [[ -n "${VIEWER_PID:-}" ]] && kill -0 "$VIEWER_PID" 2>/dev/null; then
    kill "$VIEWER_PID" 2>/dev/null || true
    wait "$VIEWER_PID" 2>/dev/null || true
  fi
  if [[ -n "${EVAL_PID:-}" ]] && kill -0 "$EVAL_PID" 2>/dev/null; then
    kill "$EVAL_PID" 2>/dev/null || true
    wait "$EVAL_PID" 2>/dev/null || true
  fi
  if [[ -n "${COT_PID:-}" ]] && kill -0 "$COT_PID" 2>/dev/null; then
    kill "$COT_PID" 2>/dev/null || true
    wait "$COT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[simlingo-pov] Opening Pygame viewer first; it will wait for CARLA on port $PORT."
echo "[simlingo-pov] DISPLAY=${DISPLAY:-<unset>} | mode=$VIEW_MODE | size=${VIEW_WIDTH}x${VIEW_HEIGHT} | visual_weather=$VISUAL_WEATHER"
echo "[simlingo-pov] Viewer log: $VIEWER_LOG"
echo "[simlingo-pov] Native SimLingo mode: full model every tick, no fast cache, no added control post-process."
echo "[simlingo-pov] Waypoint overlay: red=predicted path, green=predicted speed, blue=target points."
if [[ "${SIMLINGO_DREAMER_GUARD:-0}" != "0" && "${SIMLINGO_DREAMER_GUARD:-0}" != "false" && "${SIMLINGO_DREAMER_GUARD:-0}" != "no" ]]; then
  echo "[simlingo-pov] Dreamer Guard enabled: variant=${SIMLINGO_DREAMER_VARIANT:-dreamer_guard_v1} mode=${SIMLINGO_DREAMER_GUARD_MODE:-apply} risk_margin=${SIMLINGO_DREAMER_RISK_MARGIN:-0.05} max_progress_drop=${SIMLINGO_DREAMER_MAX_PROGRESS_DROP:-0.01}"
  echo "[simlingo-pov] Dreamer overlay status: ${DREAMER_STATUS_PATH}"
  if [[ -n "${SIMLINGO_DREAMER_CHECKPOINT:-}" ]]; then
    echo "[simlingo-pov] Dreamer checkpoint: ${SIMLINGO_DREAMER_CHECKPOINT} source=${SIMLINGO_DREAMER_CHECKPOINT_SOURCE:-unknown}"
  else
    echo "[simlingo-pov] Dreamer checkpoint: default external/simlingo/checkpoints/dreamer_guard/best_world_model.pt"
  fi
fi
if [[ -n "${SIMLINGO_CUSTOM_PROMPT:-}" ]]; then
  echo "[simlingo-pov] Native instruction prompt enabled: ${SIMLINGO_CUSTOM_PROMPT}"
fi
if [[ "$VLM_COT_MODE" != "off" && "$VLM_COT_MODE" != "0" && "$VLM_COT_MODE" != "false" && "$VLM_COT_MODE" != "no" ]]; then
  echo "[simlingo-pov] External VLM-CoT enabled: mode=$VLM_COT_MODE model=${SIMLINGO_VLM_COT_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
  echo "[simlingo-pov] CoT frame: $VLM_COT_FRAME_PATH"
  echo "[simlingo-pov] CoT status: $VLM_COT_STATUS_PATH"
  python -u "$ROOT_DIR/scripts/vlm_cot_sidecar.py" \
    --mode "$VLM_COT_MODE" \
    --model "${SIMLINGO_VLM_COT_MODEL:-Qwen/Qwen2-VL-7B-Instruct}" \
    --frame-path "$VLM_COT_FRAME_PATH" \
    --status-path "$VLM_COT_STATUS_PATH" \
    --log-path "$VLM_COT_LOG_PATH" \
    --interval "${SIMLINGO_VLM_COT_INTERVAL:-2.0}" \
    --max-new-tokens "${SIMLINGO_VLM_COT_MAX_TOKENS:-180}" \
    --device "${SIMLINGO_VLM_COT_DEVICE:-auto}" \
    ${SIMLINGO_VLM_COT_LOCAL_ONLY:+--local-files-only} &
  COT_PID=$!
  echo "$COT_PID" > "$ROOT_DIR/logs/simlingo_eval/vlm_cot_sidecar.pid"
fi
echo "[simlingo-pov] Recording enabled: $RECORD_PATH | replay x${PLAYBACK_SPEED} after route."
echo "$RECORD_PATH" > "$ROOT_DIR/logs/simlingo_eval/latest_pygame_recording.txt"
VIEWER_ARGS=(
  --port "$PORT"
  --width "$VIEW_WIDTH"
  --height "$VIEW_HEIGHT"
  --fov "$VIEW_FOV"
  --mode "$VIEW_MODE"
  --visual-weather "$VISUAL_WEATHER"
  --brightness "${SIMLINGO_VIEW_BRIGHTNESS:-8}"
  --contrast "${SIMLINGO_VIEW_CONTRAST:-1.08}"
  --saturation "${SIMLINGO_VIEW_SATURATION:-1.10}"
  --min-valid-brightness "${SIMLINGO_VIEW_MIN_BRIGHTNESS:-12}"
  --min-valid-p95 "${SIMLINGO_VIEW_MIN_P95:-24}"
  --dark-drop-ratio "${SIMLINGO_VIEW_DARK_DROP_RATIO:-0.45}"
  --stale-frame-seconds "${SIMLINGO_VIEW_STALE_SECONDS:-60}"
  --max-fps "$VIEW_FPS"
  --record-path "$RECORD_PATH"
  --record-fps "${SIMLINGO_RECORD_FPS:-30}"
  --dreamer-status-path "$DREAMER_STATUS_PATH"
  --cot-status-path "$VLM_COT_STATUS_PATH"
  --cot-frame-path "$VLM_COT_FRAME_PATH"
  --cot-frame-interval "${SIMLINGO_VLM_COT_FRAME_INTERVAL:-1.0}"
  --cot-frame-width "${SIMLINGO_VLM_COT_FRAME_WIDTH:-1280}"
  --timeout 900
)
if [[ "${SIMLINGO_TRAFFIC_LIGHT_OVERLAY:-0}" != "0" && "${SIMLINGO_TRAFFIC_LIGHT_OVERLAY:-0}" != "false" && "${SIMLINGO_TRAFFIC_LIGHT_OVERLAY:-0}" != "no" ]]; then
  echo "[simlingo-pov] Traffic-light overlay enabled: CARLA light state badges."
  VIEWER_ARGS+=(
    --traffic-light-overlay
    --traffic-light-overlay-distance "${SIMLINGO_TRAFFIC_LIGHT_OVERLAY_DISTANCE:-160}"
    --traffic-light-overlay-max "${SIMLINGO_TRAFFIC_LIGHT_OVERLAY_MAX:-80}"
  )
fi
python -u "$ROOT_DIR/scripts/carla_ego_viewer.py" "${VIEWER_ARGS[@]}" >"$VIEWER_LOG" 2>&1 &
VIEWER_PID=$!
echo "$VIEWER_PID" > "$ROOT_DIR/logs/simlingo_eval/pov_viewer.pid"
sleep 2
if ! kill -0 "$VIEWER_PID" 2>/dev/null; then
  echo "[simlingo-pov] Pygame viewer exited immediately. Check: $VIEWER_LOG" >&2
  tail -80 "$VIEWER_LOG" >&2 || true
  exit 1
fi

echo "[simlingo-pov] Starting SimLingo closed-loop eval."
bash "$ROOT_DIR/scripts/run_simlingo_local_eval.sh" &
EVAL_PID=$!
echo "$EVAL_PID" > "$ROOT_DIR/logs/simlingo_eval/latest_eval.pid"

EVAL_STATUS=0
wait "$EVAL_PID" || EVAL_STATUS=$?

if [[ -n "${VIEWER_PID:-}" ]] && kill -0 "$VIEWER_PID" 2>/dev/null; then
  kill -INT "$VIEWER_PID" 2>/dev/null || true
  wait "$VIEWER_PID" 2>/dev/null || true
fi

if [[ "$PLAYBACK_AFTER" != "0" && -s "$RECORD_PATH" ]]; then
  echo "[simlingo-pov] Replaying recorded demo at x${PLAYBACK_SPEED}: $RECORD_PATH"
  echo "[simlingo-pov] Replay log: $PLAYBACK_LOG"
  (
    cd "$ROOT_DIR"
    nohup python -u "$ROOT_DIR/scripts/play_recorded_video.py" "$RECORD_PATH" \
      --speed "$PLAYBACK_SPEED" \
      --title "SimLingo replay" \
      >"$PLAYBACK_LOG" 2>&1 &
    echo $! > "$ROOT_DIR/logs/simlingo_eval/latest_replay.pid"
  )
elif [[ "$PLAYBACK_AFTER" != "0" ]]; then
  echo "[simlingo-pov] Replay skipped: recording missing or empty: $RECORD_PATH" >&2
fi

exit "$EVAL_STATUS"
