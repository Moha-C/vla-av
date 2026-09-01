#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-$HOME/carla_simulator}"
SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
CONDA_ENV="${SIMLINGO_ENV_NAME:-simlingo}"
FAIL=0

ok() { printf '[OK] %s\n' "$1"; }
warn() { printf '[WARN] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; FAIL=1; }

has_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    ok "command '$1' found: $(command -v "$1")"
  else
    fail "command '$1' not found"
  fi
}

minimum_file_size() {
  local path="$1" label="$2" minimum="$3"
  if [[ ! -f "$path" ]]; then
    fail "$label missing: $path"
    return
  fi
  local size
  size="$(stat -c '%s' "$path" 2>/dev/null || printf '0')"
  if (( size < minimum )); then
    fail "$label is too small ($size bytes); it may be an unresolved Git LFS pointer: $path"
  else
    ok "$label: $path ($size bytes)"
  fi
}

echo "=== VLA-AV fresh install check ==="
echo "root=$ROOT_DIR"
echo "carla_root=$CARLA_ROOT"
echo "sumo_home=$SUMO_HOME"
echo "conda_env=$CONDA_ENV"
echo

for cmd in git git-lfs conda sumo sumo-gui ffmpeg node npm; do
  has_cmd "$cmd"
done

if command -v nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi found"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/[GPU] /'
else
  warn "nvidia-smi not found; CARLA/SimLingo GPU execution is not ready"
fi

[[ -d "$SUMO_HOME" ]] && ok "SUMO_HOME exists: $SUMO_HOME" || fail "SUMO_HOME missing: $SUMO_HOME"
[[ -x "$CARLA_ROOT/CarlaUE4.sh" ]] && ok "CARLA executable found" || fail "CARLA executable missing: $CARLA_ROOT/CarlaUE4.sh"

ROUTE_DIR="$ROOT_DIR/external/simlingo/leaderboard/data/bench2drive_split"
ROUTE_COUNT="$(find "$ROUTE_DIR" -maxdepth 1 -type f -name 'bench2drive_*.xml' 2>/dev/null | wc -l)"
if (( ROUTE_COUNT >= 200 )); then
  ok "Bench2Drive route catalog present: $ROUTE_COUNT XML routes"
else
  fail "Bench2Drive route catalog incomplete: $ROUTE_COUNT routes in $ROUTE_DIR"
fi

[[ -f "$ROOT_DIR/external/simlingo/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py" ]] \
  && ok "Bench2Drive evaluator source present" \
  || fail "Bench2Drive evaluator source missing"
[[ -f "$ROOT_DIR/experiments/TwinSentinel_Project/node_dashboard/server.js" ]] \
  && ok "TwinSentinel attack-console source present" \
  || fail "TwinSentinel attack-console source missing"

minimum_file_size \
  "$ROOT_DIR/models/simlingo_hf/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt" \
  "SimLingo model" 100000000
minimum_file_size \
  "$ROOT_DIR/external/simlingo/checkpoints/dreamer_guard/best_world_model.pt" \
  "Dreamer PPO runtime checkpoint" 100000
minimum_file_size \
  "$ROOT_DIR/checkpoints/report_aligned_dreamer/production/report_dreamer.pt" \
  "Promoted report RSSM checkpoint" 1000000

[[ -f "$ROOT_DIR/streamlit_share/dashboard_snapshot.html" ]] \
  && ok "read-only Streamlit HTML snapshot present" \
  || fail "read-only Streamlit HTML snapshot missing"
[[ -f "$ROOT_DIR/streamlit_share/kpi_snapshot.json" ]] \
  && ok "read-only KPI snapshot present" \
  || fail "read-only KPI snapshot missing"

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  ok "conda env exists: $CONDA_ENV"
  if conda run -n "$CONDA_ENV" python -c \
    'import carla, cv2, hydra, numpy, pygame, torch, transformers; print("imports OK")'
  then
    ok "core Python imports OK in '$CONDA_ENV'"
  else
    fail "core Python imports failed in '$CONDA_ENV'"
  fi
else
  fail "conda env missing: $CONDA_ENV"
fi

if python3 -m py_compile \
  "$ROOT_DIR/scripts/simlingo_dashboard.py" \
  "$ROOT_DIR/scripts/export_streamlit_dashboard.py" \
  "$ROOT_DIR/external/simlingo/team_code/agent_simlingo.py"
then
  ok "dashboard and agent Python syntax OK"
else
  fail "dashboard or agent Python syntax failed"
fi

echo
if (( FAIL == 0 )); then
  echo "[vla-av-check] READY: installation prerequisites look good."
else
  echo "[vla-av-check] NOT READY: fix the FAIL lines above."
fi
exit "$FAIL"
