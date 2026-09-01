#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST="$ROOT_DIR/logs/dreamer_curriculum/latest_campaign.txt"
PID_FILE="$ROOT_DIR/logs/dreamer_curriculum/latest_campaign.pid"
UNIT_FILE="$ROOT_DIR/logs/dreamer_curriculum/latest_campaign.unit"

if [[ ! -s "$LATEST" ]]; then
  echo "No Dreamer curriculum campaign has been started."
  exit 1
fi
RUN_DIR="$(cat "$LATEST")"
STATUS="$RUN_DIR/status.json"
PID="$(cat "$PID_FILE" 2>/dev/null || true)"
UNIT="$(cat "$UNIT_FILE" 2>/dev/null || true)"
if [[ -n "$UNIT" ]] && systemctl --user is-active --quiet "$UNIT" 2>/dev/null; then
  PID="$(systemctl --user show "$UNIT" --property MainPID --value 2>/dev/null || echo "$PID")"
  PROCESS_STATE="RUNNING pid=$PID unit=$UNIT"
elif [[ "$PID" =~ ^[0-9]+$ ]] && (( PID > 0 )) && kill -0 "$PID" 2>/dev/null; then
  PROCESS_STATE="RUNNING pid=$PID"
else
  UNIT_STATE="$(systemctl --user is-active "$UNIT" 2>/dev/null || true)"
  PROCESS_STATE="STOPPED${PID:+ pid=$PID}${UNIT:+ unit=$UNIT state=$UNIT_STATE}"
fi

echo "=== Dreamer PPO Curriculum ==="
echo "process: $PROCESS_STATE"
echo "run_dir:  $RUN_DIR"
if [[ -s "$STATUS" ]]; then
  python3 - "$STATUS" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
done = int(data.get("completed_simulations", 0))
total = int(data.get("total_simulations", 0))
percent = 100.0 * done / max(1, total)
eta = float(data.get("eta_seconds", 0.0) or 0.0)
hours, rem = divmod(int(eta), 3600)
minutes, seconds = divmod(rem, 60)
print(f"phase:    {data.get('phase', '-')}")
print(f"progress: {done}/{total} simulations ({percent:.1f}%)")
print(f"ETA:      {hours:02d}:{minutes:02d}:{seconds:02d}")
current = data.get("current") or {}
if current:
    print(
        "current:  route {route_id} / {town} / {scenario} / seed {seed}".format(
            **current
        )
    )
print(f"production: {str(data.get('production_sha256', '-'))[:16]}")
print(f"candidate:  {str(data.get('candidate_sha256', '-'))[:16]}")
for stage in data.get("stages") or []:
    gate = stage.get("gate") or {}
    print(
        f"stage {stage.get('name')}: {stage.get('status')} "
        f"gate={gate.get('approved', '-')} runs={len(stage.get('training_runs') or [])}"
    )
if data.get("error"):
    print(f"error: {data['error']}")
PY
else
  echo "status: waiting for initialization"
fi
echo
echo "--- campaign log tail ---"
tail -30 "$RUN_DIR/campaign.log" 2>/dev/null || true
