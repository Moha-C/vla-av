#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${SIMLINGO_OUT_DIR:-$ROOT_DIR/logs/simlingo_eval}"

echo "==== PROCESS ===="
ps -eo pid,etime,cmd | grep -E "run_simlingo_local_eval|leaderboard_evaluator|CarlaUE4" | grep -v grep || true

echo
echo "==== GPU ===="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true

echo
echo "==== LOG ===="
latest_log="$(find "$OUT_DIR" -maxdepth 1 -type f -name 'run*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
if [[ -n "${latest_log:-}" ]]; then
  echo "$latest_log"
  tail -n 80 "$latest_log"
else
  echo "No SimLingo eval log yet."
fi

echo
echo "==== RESULT ===="
latest_result="$(find "$OUT_DIR" -maxdepth 1 -type f -name 'results*.json' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
if [[ -n "${latest_result:-}" ]]; then
  echo "$latest_result"
  python - "$latest_result" <<'PY' || true
import json, sys
p = sys.argv[1]
with open(p) as f:
    data = json.load(f)
cp = data.get("_checkpoint", {})
print("progress:", cp.get("progress"))
for r in cp.get("records", [])[-5:]:
    print(r.get("route_id"), r.get("status"), r.get("scores", {}))
PY
else
  echo "No result JSON yet."
fi
