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
RECORD_ENABLED="${SIMLINGO_RECORD:-1}"
PLAYBACK_AFTER="${SIMLINGO_PLAYBACK_AFTER:-1}"
PLAYBACK_SPEED="${SIMLINGO_PLAYBACK_SPEED:-5}"
PLAYBACK_LOG="${SIMLINGO_PLAYBACK_LOG:-$ROOT_DIR/logs/simlingo_eval/latest_replay.log}"
VIEWER_LOG="${SIMLINGO_VIEWER_LOG:-$ROOT_DIR/logs/simlingo_eval/latest_pov_viewer.log}"
DREAMER_STATUS_PATH="${SIMLINGO_DREAMER_STATUS_PATH:-$ROOT_DIR/logs/simlingo_eval/dreamer_guard_status.json}"
export SIMLINGO_DREAMER_STATUS_PATH="$DREAMER_STATUS_PATH"
REPORT_DREAMER_MODE="${SIMLINGO_REPORT_DREAMER_MODE:-off}"
REPORT_DREAMER_CHECKPOINT="${SIMLINGO_REPORT_DREAMER_CHECKPOINT:-$ROOT_DIR/checkpoints/report_aligned_dreamer/production/report_dreamer.pt}"
REPORT_DREAMER_TRACE="${SIMLINGO_REPORT_DREAMER_TRACE:-}"
CARDREAMER_MODE="${SIMLINGO_CARDREAMER_MODE:-off}"
CARDREAMER_PYTHON="${SIMLINGO_CARDREAMER_PYTHON:-$HOME/miniconda3/envs/cardreamer/bin/python}"
CARDREAMER_UPSTREAM="${SIMLINGO_CARDREAMER_UPSTREAM:-$ROOT_DIR/external/cardreamer_upstream}"
CARDREAMER_CHECKPOINT="${SIMLINGO_CARDREAMER_CHECKPOINT:-$ROOT_DIR/external/cardreamer_checkpoints/CarDreamer_checkpoints/overtake.ckpt}"
CARDREAMER_STATUS_PATH="${SIMLINGO_CARDREAMER_STATUS_PATH:-$ROOT_DIR/logs/simlingo_eval/cardreamer_runtime_status.json}"
CARDREAMER_CONTROL_STATUS_PATH="${SIMLINGO_CARDREAMER_CONTROL_STATUS_PATH:-$ROOT_DIR/logs/simlingo_eval/cardreamer_residual_control.json}"
CARDREAMER_RUN_ID="${SIMLINGO_CARDREAMER_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CARDREAMER_TRACE_PATH="${SIMLINGO_CARDREAMER_TRACE_PATH:-$ROOT_DIR/logs/cardreamer_runtime/$CARDREAMER_RUN_ID/trace.jsonl}"
CARDREAMER_BEV_PATH="${SIMLINGO_CARDREAMER_BEV_PATH:-$ROOT_DIR/logs/cardreamer_runtime/$CARDREAMER_RUN_ID/latest_bev.png}"
CARDREAMER_LOG_PATH="${SIMLINGO_CARDREAMER_LOG_PATH:-$ROOT_DIR/logs/cardreamer_runtime/$CARDREAMER_RUN_ID/sidecar.log}"
CARDREAMER_LATERAL_ADAPTER="${SIMLINGO_CARDREAMER_LATERAL_ADAPTER:-native}"
export SIMLINGO_CARDREAMER_STATUS_PATH="$CARDREAMER_STATUS_PATH"
export SIMLINGO_CARDREAMER_CONTROL_STATUS_PATH="$CARDREAMER_CONTROL_STATUS_PATH"
export SIMLINGO_CARDREAMER_TRACE_PATH="$CARDREAMER_TRACE_PATH"
VLM_COT_MODE="${SIMLINGO_VLM_COT:-off}"
VLM_COT_STATUS_PATH="${SIMLINGO_VLM_COT_STATUS_PATH:-$ROOT_DIR/logs/simlingo_eval/vlm_cot_status.json}"
VLM_COT_FRAME_PATH="${SIMLINGO_VLM_COT_FRAME_PATH:-$ROOT_DIR/logs/simlingo_eval/vlm_cot_frame.jpg}"
VLM_COT_LOG_PATH="${SIMLINGO_VLM_COT_LOG_PATH:-$ROOT_DIR/logs/simlingo_eval/vlm_cot_reasoning.jsonl}"
export SIMLINGO_VLM_COT_STATUS_PATH="$VLM_COT_STATUS_PATH"
export SIMLINGO_VLM_COT_FRAME_PATH="$VLM_COT_FRAME_PATH"
export SIMLINGO_VLM_COT_LOG_PATH="$VLM_COT_LOG_PATH"
export SDL_VIDEO_X11_FORCE_EGL="${SDL_VIDEO_X11_FORCE_EGL:-1}"

wait_for_carla_ports() {
  local timeout="${SIMLINGO_CARLA_PORT_WAIT_SECONDS:-180}"
  local started_at=$SECONDS

  while ! python - "$PORT" <<'PY'
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
  do
    if (( SECONDS - started_at >= timeout )); then
      echo "[simlingo-pov] CARLA ports $PORT/$((PORT + 1)) did not become available within ${timeout}s." >&2
      return 1
    fi
    sleep 2
  done
}

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

mkdir -p "$ROOT_DIR/logs/simlingo_eval"
rm -f "$DREAMER_STATUS_PATH"
rm -f "$CARDREAMER_STATUS_PATH" "$CARDREAMER_CONTROL_STATUS_PATH"
rm -f "$VLM_COT_STATUS_PATH" "$VLM_COT_FRAME_PATH" "$VLM_COT_LOG_PATH"
: > "$VIEWER_LOG"

# Bench2Drive may otherwise silently increment a busy RPC port while the
# viewer keeps waiting on the requested one. Wait for CARLA's RPC/streaming
# pair so evaluator and Pygame always connect to the same server.
echo "[simlingo-pov] Waiting for CARLA ports $PORT/$((PORT + 1)) to be free."
wait_for_carla_ports

export SIMLINGO_RENDER_MODE="${SIMLINGO_RENDER_MODE:-offscreen}"

cleanup() {
  stop_viewer "${VIEWER_PID:-}"
  if [[ -n "${EVAL_PID:-}" ]] && kill -0 "$EVAL_PID" 2>/dev/null; then
    kill "$EVAL_PID" 2>/dev/null || true
    wait "$EVAL_PID" 2>/dev/null || true
  fi
  if [[ -n "${COT_PID:-}" ]] && kill -0 "$COT_PID" 2>/dev/null; then
    kill "$COT_PID" 2>/dev/null || true
    wait "$COT_PID" 2>/dev/null || true
  fi
  if [[ -n "${CARDREAMER_PID:-}" ]] && kill -0 "$CARDREAMER_PID" 2>/dev/null; then
    kill "$CARDREAMER_PID" 2>/dev/null || true
    wait "$CARDREAMER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

stop_viewer() {
  local pid="${1:-}"
  local grace_seconds="${SIMLINGO_VIEWER_SHUTDOWN_SECONDS:-10}"
  local watchdog_pid
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return
  fi
  kill -TERM "$pid" 2>/dev/null || true
  (
    sleep "$grace_seconds"
    kill -KILL "$pid" 2>/dev/null || true
  ) &
  watchdog_pid=$!
  wait "$pid" 2>/dev/null || true
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
}

echo "[simlingo-pov] Opening Pygame viewer first; it will wait for CARLA on port $PORT."
echo "[simlingo-pov] DISPLAY=${DISPLAY:-<unset>} | mode=$VIEW_MODE | size=${VIEW_WIDTH}x${VIEW_HEIGHT} | visual_weather=$VISUAL_WEATHER"
echo "[simlingo-pov] Viewer log: $VIEWER_LOG"
echo "[simlingo-pov] Native SimLingo inference: full model every tick, no fast cache."
echo "[simlingo-pov] Waypoint overlay: red=predicted path, green=predicted speed, blue=target points."
if [[ "$REPORT_DREAMER_MODE" != "off" && "$REPORT_DREAMER_MODE" != "0" && "$REPORT_DREAMER_MODE" != "false" && "$REPORT_DREAMER_MODE" != "no" ]]; then
  if [[ ! -f "$REPORT_DREAMER_CHECKPOINT" ]]; then
    echo "[simlingo-pov] Missing validated report-aligned Dreamer checkpoint: $REPORT_DREAMER_CHECKPOINT" >&2
    exit 1
  fi
  echo "[simlingo-pov] Report-aligned RSSM complement: mode=$REPORT_DREAMER_MODE ablation=${SIMLINGO_REPORT_DREAMER_ABLATION:-D} shadow=${SIMLINGO_REPORT_DREAMER_SHADOW:-0}."
  echo "[simlingo-pov] Report RSSM checkpoint: $REPORT_DREAMER_CHECKPOINT"
  echo "[simlingo-pov] Report RSSM trace: ${REPORT_DREAMER_TRACE:-disabled}"
  echo "[simlingo-pov] Candidate 0 is exact post-PID native SimLingo; final control uses continuous alpha blending."
fi
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
if [[ "$CARDREAMER_MODE" == "shadow" || "$CARDREAMER_MODE" == "residual" ]]; then
  if [[ ! -x "$CARDREAMER_PYTHON" ]]; then
    echo "[simlingo-pov] Missing CarDreamer Python 3.10 environment: $CARDREAMER_PYTHON" >&2
    exit 1
  fi
  if [[ ! -f "$CARDREAMER_CHECKPOINT" ]]; then
    echo "[simlingo-pov] Missing official CarDreamer checkpoint: $CARDREAMER_CHECKPOINT" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$CARDREAMER_TRACE_PATH")"
  CARDREAMER_ROUTE_FILE="${SIMLINGO_CARDREAMER_ROUTE_FILE:-${ROUTE_FILE:-}}"
  if [[ -z "$CARDREAMER_ROUTE_FILE" && "${ROUTE_ID:-}" =~ ^[0-9]+$ ]]; then
    CARDREAMER_ROUTE_FILE="$ROOT_DIR/external/simlingo/leaderboard/data/bench2drive_split/bench2drive_${ROUTE_ID}.xml"
  fi
  CARDREAMER_ARGS=(
    --host "127.0.0.1"
    --port "$PORT"
    --checkpoint "$CARDREAMER_CHECKPOINT"
    --upstream "$CARDREAMER_UPSTREAM"
    --status-path "$CARDREAMER_STATUS_PATH"
    --trace-path "$CARDREAMER_TRACE_PATH"
    --bev-path "$CARDREAMER_BEV_PATH"
    --seed "${SEED:-1}"
    --timeout "${SIMLINGO_CARDREAMER_TIMEOUT:-900}"
    --interval-game-seconds "${SIMLINGO_CARDREAMER_INTERVAL:-0.1}"
    --minimum-clearance "${SIMLINGO_CARDREAMER_MINIMUM_CLEARANCE:-5.0}"
    --minimum-oncoming-ttc "${SIMLINGO_CARDREAMER_MINIMUM_ONCOMING_TTC:-7.0}"
    --minimum-rear-ttc "${SIMLINGO_CARDREAMER_MINIMUM_REAR_TTC:-5.0}"
    --lateral-adapter "$CARDREAMER_LATERAL_ADAPTER"
    --runtime-mode "$CARDREAMER_MODE"
  )
  if [[ -n "$CARDREAMER_ROUTE_FILE" && -f "$CARDREAMER_ROUTE_FILE" ]]; then
    CARDREAMER_ARGS+=(--route-file "$CARDREAMER_ROUTE_FILE")
  fi
  if [[ "$CARDREAMER_MODE" == "residual" ]]; then
    echo "[simlingo-pov] CarDreamer official RSSM residual enabled: proposals alter SimLingo control."
    echo "[simlingo-pov] Task-scoped authority: RSSM only acts during an accepted blocked-lane overtake."
    echo "[simlingo-pov] Explicit traffic gate: clearance>=${SIMLINGO_CARDREAMER_MINIMUM_CLEARANCE:-5.0}m, rear TTC>=${SIMLINGO_CARDREAMER_MINIMUM_REAR_TTC:-5.0}s, oncoming TTC>=${SIMLINGO_CARDREAMER_MINIMUM_ONCOMING_TTC:-7.0}s."
    echo "[simlingo-pov] Residual blend alpha=${SIMLINGO_CARDREAMER_RESIDUAL_ALPHA:-0.35}."
    echo "[simlingo-pov] Signed longitudinal arbitration: engage=${SIMLINGO_CARDREAMER_ENGAGE_DECISIONS:-2} decisions, minimum hold=${SIMLINGO_CARDREAMER_MIN_ENGAGEMENT_DECISIONS:-20}, release=${SIMLINGO_CARDREAMER_RELEASE_DECISIONS:-6}."
  else
    echo "[simlingo-pov] CarDreamer official shadow enabled: READ ONLY, zero control authority."
  fi
  echo "[simlingo-pov] CarDreamer runs in Python 3.10; SimLingo remains in Python 3.8."
  echo "[simlingo-pov] CarDreamer uses privileged full-state BEV, not camera-only input."
  echo "[simlingo-pov] CarDreamer lateral adapter: $CARDREAMER_LATERAL_ADAPTER (weights unchanged)."
  echo "[simlingo-pov] CarDreamer trace: $CARDREAMER_TRACE_PATH"
  "$CARDREAMER_PYTHON" -u "$ROOT_DIR/scripts/cardreamer_shadow_sidecar.py" \
    "${CARDREAMER_ARGS[@]}" >"$CARDREAMER_LOG_PATH" 2>&1 &
  CARDREAMER_PID=$!
  echo "$CARDREAMER_PID" > "$ROOT_DIR/logs/simlingo_eval/cardreamer_runtime.pid"
  echo "$CARDREAMER_TRACE_PATH" > "$ROOT_DIR/logs/simlingo_eval/latest_cardreamer_trace.txt"
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
if [[ "$RECORD_ENABLED" != "0" && "$RECORD_ENABLED" != "false" && "$RECORD_ENABLED" != "no" ]]; then
  echo "[simlingo-pov] Recording enabled: $RECORD_PATH | replay x${PLAYBACK_SPEED} after route."
  echo "$RECORD_PATH" > "$ROOT_DIR/logs/simlingo_eval/latest_pygame_recording.txt"
else
  echo "[simlingo-pov] Recording disabled for this automated run."
fi
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
  --dreamer-status-path "$DREAMER_STATUS_PATH"
  --cardreamer-status-path "$CARDREAMER_STATUS_PATH"
  --cardreamer-control-status-path "$CARDREAMER_CONTROL_STATUS_PATH"
  --cot-status-path "$VLM_COT_STATUS_PATH"
  --cot-frame-path "$VLM_COT_FRAME_PATH"
  --cot-frame-interval "${SIMLINGO_VLM_COT_FRAME_INTERVAL:-1.0}"
  --cot-frame-width "${SIMLINGO_VLM_COT_FRAME_WIDTH:-1280}"
  --timeout 900
)
if [[ "$RECORD_ENABLED" != "0" && "$RECORD_ENABLED" != "false" && "$RECORD_ENABLED" != "no" ]]; then
  VIEWER_ARGS+=(
    --record-path "$RECORD_PATH"
    --record-fps "${SIMLINGO_RECORD_FPS:-30}"
  )
fi
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

stop_viewer "${VIEWER_PID:-}"

if [[ "$RECORD_ENABLED" != "0" && "$RECORD_ENABLED" != "false" && "$RECORD_ENABLED" != "no" && "$PLAYBACK_AFTER" != "0" && -s "$RECORD_PATH" ]]; then
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
elif [[ "$RECORD_ENABLED" != "0" && "$RECORD_ENABLED" != "false" && "$RECORD_ENABLED" != "no" && "$PLAYBACK_AFTER" != "0" ]]; then
  echo "[simlingo-pov] Replay skipped: recording missing or empty: $RECORD_PATH" >&2
fi

exit "$EVAL_STATUS"
