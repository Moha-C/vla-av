#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST_FILE="$ROOT_DIR/logs/dreamer_rl_campaign/latest_campaign.txt"

if [[ -n "${DREAMER_RL_CAMPAIGN_RUN_DIR:-}" ]]; then
  RUN_DIR="$DREAMER_RL_CAMPAIGN_RUN_DIR"
elif [[ -s "$LATEST_FILE" ]]; then
  RUN_DIR="$(cat "$LATEST_FILE")"
else
  echo "[dreamer-rl-campaign-watch] no campaign has been launched yet"
  exit 1
fi

STATUS="$RUN_DIR/status.json"
SUMMARY="$RUN_DIR/summary.md"
LOG="$RUN_DIR/campaign_stdout.log"
PID_FILE="$RUN_DIR/campaign.pid"
UNIT_NAME_FILE="$RUN_DIR/systemd_unit_name.txt"

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "$PID" ]] && ps -p "$PID" >/dev/null 2>&1; then
  STATE="RUNNING pid=$PID"
elif [[ -s "$UNIT_NAME_FILE" ]]; then
  UNIT="$(cat "$UNIT_NAME_FILE")"
  UNIT_STATE="$(systemctl --user is-active "$UNIT" 2>/dev/null || true)"
  STATE="systemd ${UNIT_STATE:-unknown} unit=$UNIT"
else
  STATE="STOPPED"
fi

echo "=== Dreamer RL Autonomous Campaign ==="
echo "run_dir:  $RUN_DIR"
echo "state:    $STATE"
echo

if [[ -s "$STATUS" ]]; then
  python3 - "$STATUS" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
print(f"phase:    {data.get('phase')}")
print(f"progress: {data.get('completed_runs', 0)}/{data.get('total_runs', 0)} route runs")
cur = data.get("current_run")
if cur:
    print(f"current:  route {cur.get('route_id')} {cur.get('town')} {cur.get('scenario_type')} seed={cur.get('seed')}")
if data.get("dataset"):
    print(f"dataset:  {data['dataset'].get('path')}")
if data.get("warmstarts"):
    for kind, info in data["warmstarts"].items():
        print(f"warm {kind}: {info.get('status')} {info.get('checkpoint', '')}")
if data.get("rl_runs"):
    for kind, info in data["rl_runs"].items():
        print(f"rl {kind}: {info.get('status')} {info.get('run_dir', '')}")
print()
for run in data.get("runs", [])[-5:]:
    m = run.get("metrics") or {}
    print(
        f"- {run.get('quality')} route={run.get('route_id')} "
        f"{run.get('town')} {run.get('scenario_type')} "
        f"score={m.get('driving_score')} route_score={m.get('route_score')} "
        f"coll={m.get('collisions')} offroad={m.get('offroad')}"
    )
PY
else
  echo "status: pending"
fi

echo
echo "--- summary tail ---"
if [[ -s "$SUMMARY" ]]; then
  tail -n 60 "$SUMMARY"
else
  echo "no summary yet: $SUMMARY"
fi

echo
echo "--- log tail ---"
if [[ -s "$LOG" ]]; then
  tail -n 40 "$LOG"
else
  echo "no log yet: $LOG"
fi
