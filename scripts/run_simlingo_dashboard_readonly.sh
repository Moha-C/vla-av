#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export SIMLINGO_DASHBOARD_HOST="${SIMLINGO_DASHBOARD_HOST:-0.0.0.0}"
export SIMLINGO_DASHBOARD_PORT="${SIMLINGO_DASHBOARD_PORT:-8875}"
export SIMLINGO_DASHBOARD_READ_ONLY=1

echo "[simlingo-dashboard-share] Starting read-only presentation server."
echo "[simlingo-dashboard-share] Launch, stop, replay and attack APIs are disabled server-side."
echo "[simlingo-dashboard-share] Bind: ${SIMLINGO_DASHBOARD_HOST}:${SIMLINGO_DASHBOARD_PORT}"

exec bash "$ROOT_DIR/scripts/run_simlingo_dashboard.sh"
