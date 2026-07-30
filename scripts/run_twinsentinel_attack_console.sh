#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TWINSENTINEL_ROOT="${TWINSENTINEL_ROOT:-$ROOT_DIR/experiments/TwinSentinel_Project}"
DASHBOARD_DIR="${TWINSENTINEL_DASHBOARD_DIR:-$TWINSENTINEL_ROOT/node_dashboard}"
LOG_DIR="$ROOT_DIR/logs/twinsentinel_live"
SUMO_MIRROR_LOG_DIR="${SUMO_MIRROR_LOG_DIR:-$ROOT_DIR/logs/sumo_mirror}"
NODE_BIN="${NODE_BIN:-$ROOT_DIR/.conda_node/bin/node}"
NPM_BIN="${NPM_BIN:-$ROOT_DIR/.conda_node/bin/npm}"
NPM_CONFIG_CACHE="${NPM_CONFIG_CACHE:-$ROOT_DIR/.npm-cache}"

PORT="${TWINSENTINEL_PORT:-3100}"
STATE_FILE="${TWINSENTINEL_STATE_FILE:-$SUMO_MIRROR_LOG_DIR/live_state.json}"
COMMAND_FILE="${TWINSENTINEL_COMMAND_FILE:-$SUMO_MIRROR_LOG_DIR/attack_commands.jsonl}"
LOG_FILE="$LOG_DIR/dashboard.log"
PID_FILE="$LOG_DIR/dashboard.pid"

mkdir -p "$LOG_DIR" "$SUMO_MIRROR_LOG_DIR"

if [[ ! -f "$DASHBOARD_DIR/server.js" ]]; then
  echo "[twinsentinel-live] missing TwinSentinel dashboard server: $DASHBOARD_DIR/server.js" >&2
  echo "[twinsentinel-live] expected repo at: $TWINSENTINEL_ROOT" >&2
  exit 1
fi

if [[ ! -d "$DASHBOARD_DIR/node_modules" ]]; then
  echo "[twinsentinel-live] missing node_modules in $DASHBOARD_DIR; installing from TwinSentinel package-lock..."
  if [[ ! -x "$NPM_BIN" ]]; then
    NPM_BIN="$(command -v npm || true)"
  fi
  if [[ -z "$NPM_BIN" || ! -x "$NPM_BIN" ]]; then
    echo "[twinsentinel-live] npm executable not found." >&2
    exit 1
  fi
  (
    cd "$DASHBOARD_DIR"
    NPM_CONFIG_CACHE="$NPM_CONFIG_CACHE" PATH="$ROOT_DIR/.conda_node/bin:$PATH" "$NPM_BIN" ci --omit=dev
  )
fi

if [[ ! -x "$NODE_BIN" ]]; then
  NODE_BIN="$(command -v node || true)"
fi
if [[ -z "$NODE_BIN" || ! -x "$NODE_BIN" ]]; then
  echo "[twinsentinel-live] node executable not found. Expected $ROOT_DIR/.conda_node/bin/node" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[twinsentinel-live] already running pid=$OLD_PID url=http://127.0.0.1:$PORT"
    exit 0
  fi
fi

echo "[twinsentinel-live] state=$STATE_FILE"
echo "[twinsentinel-live] commands=$COMMAND_FILE"
echo "[twinsentinel-live] starting dashboard on http://127.0.0.1:$PORT"

(
  cd "$DASHBOARD_DIR"
  PATH="$ROOT_DIR/.conda_node/bin:$PATH" \
  TWINSENTINEL_LIVE_MODE=1 \
  TWINSENTINEL_ROOT="$TWINSENTINEL_ROOT" \
  TWINSENTINEL_STATE_FILE="$STATE_FILE" \
  TWINSENTINEL_COMMAND_FILE="$COMMAND_FILE" \
  HOST=127.0.0.1 \
  PORT="$PORT" \
  "$NODE_BIN" server.js
) >"$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "[twinsentinel-live] pid=$PID log=$LOG_FILE"
