#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARDREAMER_ENV="${CARDREAMER_ENV:-/home/mohm/miniconda3/envs/cardreamer}"
CARLA_ROOT="${CARLA_ROOT:-/home/mohm/carla_simulator}"
CHECKPOINT="${CARDREAMER_CHECKPOINT:-$ROOT_DIR/external/cardreamer_checkpoints/CarDreamer_checkpoints/overtake.ckpt}"
PORT="${CARDREAMER_CARLA_PORT:-2100}"
SEEDS="${CARDREAMER_EVAL_SEEDS:-11,23,37,51,73}"
EPISODES="${CARDREAMER_EVAL_EPISODES:-3}"
QUALITY="${CARDREAMER_CARLA_QUALITY:-Low}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${CARDREAMER_EVAL_OUT_DIR:-$ROOT_DIR/artifacts/cardreamer_integration_20260817/official_eval/$RUN_ID}"

mkdir -p "$OUT_DIR"
printf '%s\n' "$OUT_DIR" > "$ROOT_DIR/artifacts/cardreamer_integration_20260817/official_eval/latest_run.txt"

SERVER_PID=""
SERVER_PGID=""
stop_server() {
  if [[ -n "$SERVER_PGID" ]]; then
    kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! pgrep -g "$SERVER_PGID" >/dev/null 2>&1; then
        break
      fi
      sleep 0.25
    done
    if pgrep -g "$SERVER_PGID" >/dev/null 2>&1; then
      kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
    fi
  fi
  [[ -z "$SERVER_PID" ]] || wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  SERVER_PGID=""
}
trap stop_server EXIT INT TERM

start_server() {
  local seed="$1"
  echo "[cardreamer-official] starting CARLA seed=$seed port=$PORT quality=$QUALITY"
  setsid "$CARLA_ROOT/CarlaUE4.sh" \
    -RenderOffScreen \
    -nosound \
    -quality-level="$QUALITY" \
    -carla-rpc-port="$PORT" \
    -benchmark \
    -fps=10 \
    >"$OUT_DIR/carla_seed_${seed}.log" 2>&1 &
  SERVER_PID=$!
  SERVER_PGID="$(ps -o pgid= -p "$SERVER_PID" | tr -d '[:space:]')"

  local ready=0
  for _ in $(seq 1 180); do
    if "$CARDREAMER_ENV/bin/python" -c "import carla; c=carla.Client('127.0.0.1',$PORT); c.set_timeout(2.0); print(c.get_world().get_map().name)" >/dev/null 2>&1; then
      sleep 3
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[cardreamer-official] CARLA exited before becoming ready" >&2
      return 1
    fi
    sleep 1
  done
  if [[ "$ready" != "1" ]]; then
    echo "[cardreamer-official] CARLA RPC did not become ready on port $PORT" >&2
    return 1
  fi
}

IFS=',' read -r -a seed_values <<< "$SEEDS"
for seed in "${seed_values[@]}"; do
  seed="${seed//[[:space:]]/}"
  start_server "$seed"
  echo "[cardreamer-official] seed=$seed episodes=$EPISODES"
  "$CARDREAMER_ENV/bin/python" -u "$ROOT_DIR/scripts/cardreamer_official_eval.py" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUT_DIR/seed_${seed}.json" \
    --logdir "$OUT_DIR/log_seed_${seed}" \
    --carla-port "$PORT" \
    --seed "$seed" \
    --episodes "$EPISODES" \
    2>&1 | tee "$OUT_DIR/seed_${seed}.log"
  stop_server
done

set +e
"$CARDREAMER_ENV/bin/python" "$ROOT_DIR/scripts/summarize_cardreamer_official_eval.py" \
  --input-dir "$OUT_DIR" \
  --output "$OUT_DIR/summary.json" \
  | tee "$OUT_DIR/summary.txt"
SUMMARY_STATUS=${PIPESTATUS[0]}
set -e

echo "[cardreamer-official] results=$OUT_DIR"
exit "$SUMMARY_STATUS"
