#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_FILE="$ROOT_DIR/logs/dreamer_curriculum/latest_campaign.unit"
UNIT="$(cat "$UNIT_FILE" 2>/dev/null || true)"

if [[ -n "$UNIT" ]]; then
  systemctl --user stop "$UNIT" 2>/dev/null || true
fi
bash "$ROOT_DIR/scripts/stop_simlingo_dashboard.sh"
echo "[dreamer-curriculum] stopped${UNIT:+ unit=$UNIT}"
