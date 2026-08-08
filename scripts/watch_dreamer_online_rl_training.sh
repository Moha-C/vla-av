#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST_FILE="$ROOT_DIR/logs/dreamer_online_rl/latest_training.txt"

if [[ -n "${DREAMER_ONLINE_RL_RUN_DIR:-}" ]]; then
  RUN_DIR="$DREAMER_ONLINE_RL_RUN_DIR"
elif [[ -s "$LATEST_FILE" ]]; then
  RUN_DIR="$(cat "$LATEST_FILE")"
else
  echo "[dreamer-online-rl-watch] no online RL training has been launched yet"
  exit 1
fi

STATUS="$RUN_DIR/status.json"
SUMMARY="$RUN_DIR/summary.md"
LOG="$RUN_DIR/training_stdout.log"
PID_FILE="$RUN_DIR/training.pid"

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "$PID" ]] && ps -p "$PID" >/dev/null 2>&1; then
  STATE="RUNNING pid=$PID"
else
  STATE="STOPPED"
fi

echo "=== Dreamer Online RL No-Guard Training ==="
echo "run_dir:  $RUN_DIR"
echo "state:    $STATE"
echo

if [[ -s "$STATUS" ]]; then
  python3 - "$STATUS" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
done = int(data.get("completed_episodes", 0) or 0)
total = int(data.get("total_episodes", 0) or 0)
progress = (100.0 * done / total) if total else 0.0
print(f"phase:    {data.get('phase')}")
print(f"progress: {done}/{total} episodes ({progress:.1f}%)")
print(f"no_guard: {data.get('no_guard')} | complement_to_simlingo: {data.get('complement_to_simlingo')}")
cur = data.get("current_episode")
if cur:
    print(
        "current:  "
        f"{cur.get('kind')} route={cur.get('route_id')} town={cur.get('town')} "
        f"scenario={cur.get('scenario_type')} seed={cur.get('seed')}"
    )
    print(f"trace:    {cur.get('trace')}")
if data.get("checkpoints"):
    print("checkpoints:")
    for kind, info in data["checkpoints"].items():
        print(f"  - {kind}: {info.get('active')}")
        print(f"    backup: {info.get('backup_before_online')}")
episodes = data.get("episodes") or []
if episodes:
    print()
    print("last episodes:")
for ep in episodes[-6:]:
    metrics = ep.get("metrics") or {}
    update = ep.get("update") or {}
    print(
        f"  - {ep.get('kind')} route={ep.get('route_id')} "
        f"rows={ep.get('trace_rows')} rl_rows={ep.get('rl_trace_rows')} update={update.get('status')} "
        f"score={metrics.get('driving_score')} route={metrics.get('route_score')} "
        f"coll={metrics.get('collisions')} offroad={metrics.get('offroad')} "
        f"reward={update.get('reward_sum')}"
    )
PY
else
  echo "status: pending"
fi

echo
echo "--- summary tail ---"
if [[ -s "$SUMMARY" ]]; then
  tail -n 80 "$SUMMARY"
else
  echo "no summary yet: $SUMMARY"
fi

echo
echo "--- log tail ---"
if [[ -s "$LOG" ]]; then
  tail -n 60 "$LOG"
else
  echo "no log yet: $LOG"
fi
