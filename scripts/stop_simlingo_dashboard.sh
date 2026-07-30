#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
twinsentinel_pid_file="$ROOT_DIR/logs/twinsentinel_live/dashboard.pid"
if [[ -f "$twinsentinel_pid_file" ]]; then
  twinsentinel_pid="$(cat "$twinsentinel_pid_file" 2>/dev/null || true)"
  if [[ -n "$twinsentinel_pid" ]] && kill -0 "$twinsentinel_pid" 2>/dev/null; then
    kill -TERM "$twinsentinel_pid" || true
  fi
fi

patterns=(
  "scripts/simlingo_dashboard.py"
  "scripts/run_simlingo_with_pov.sh"
  "scripts/run_simlingo_with_sumo_mirror.sh"
  "scripts/run_simlingo_local_eval.sh"
  "scripts/carla_ego_viewer.py"
  "scripts/carla_sumo_mirror.py"
  "scripts/vlm_cot_sidecar.py"
  "scripts/run_twinsentinel_attack_console.sh"
  "experiments/TwinSentinel_Project/node_dashboard/server.js"
  "leaderboard_evaluator.py"
  "sumo-gui"
  "sumo -c"
  "CarlaUE4"
)

for pattern in "${patterns[@]}"; do
  pkill -TERM -f "$pattern" || true
done
sleep 4
for pattern in "${patterns[@]}"; do
  pkill -9 -f "$pattern" || true
done
echo "[simlingo-dashboard] stopped dashboard, SimLingo eval, Pygame viewer, and CARLA."
