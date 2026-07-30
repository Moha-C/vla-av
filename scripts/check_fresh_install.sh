#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-$HOME/carla_simulator}"
SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
CONDA_ENV="${SIMLINGO_ENV_NAME:-simlingo}"
FAIL=0

ok() {
  printf '[OK] %s\n' "$1"
}

warn() {
  printf '[WARN] %s\n' "$1"
}

fail() {
  printf '[FAIL] %s\n' "$1"
  FAIL=1
}

exists_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    ok "$label: $path"
  else
    fail "$label missing: $path"
  fi
}

exists_exec() {
  local path="$1"
  local label="$2"
  if [[ -x "$path" ]]; then
    ok "$label: $path"
  else
    fail "$label missing or not executable: $path"
  fi
}

has_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    ok "command '$cmd' found: $(command -v "$cmd")"
  else
    fail "command '$cmd' not found"
  fi
}

echo "=== VLA-AV fresh install check ==="
echo "root=$ROOT_DIR"
echo "carla_root=$CARLA_ROOT"
echo "sumo_home=$SUMO_HOME"
echo "conda_env=$CONDA_ENV"
echo

has_cmd git
has_cmd conda
has_cmd sumo
has_cmd sumo-gui
has_cmd ffmpeg

if command -v nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi found"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/[GPU] /'
else
  warn "nvidia-smi not found; CPU-only checks may pass, but CARLA/SimLingo GPU performance will suffer"
fi

if [[ -d "$SUMO_HOME" ]]; then
  ok "SUMO_HOME directory exists: $SUMO_HOME"
else
  fail "SUMO_HOME directory missing: $SUMO_HOME"
fi

exists_exec "$CARLA_ROOT/CarlaUE4.sh" "CARLA executable"

exists_file "$ROOT_DIR/models/simlingo_hf/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt" "SimLingo model"
exists_file "$ROOT_DIR/external/simlingo/checkpoints/dreamer_guard/best_world_model.pt" "Dreamer PPO guarded checkpoint"
exists_file "$ROOT_DIR/external/simlingo/checkpoints/dreamer_sdbs_fresh/best_world_model.pt" "Dreamer SDBS guarded checkpoint"
exists_file "$ROOT_DIR/external/simlingo/checkpoints/dreamer_ppo_rl_noguard/latest_rl_model.pt" "Dreamer PPO RL no-guard checkpoint"
exists_file "$ROOT_DIR/external/simlingo/checkpoints/dreamer_sdbs_rl_noguard/latest_rl_model.pt" "Dreamer SDBS RL no-guard checkpoint"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  ok "conda env exists: $CONDA_ENV"
  if conda run -n "$CONDA_ENV" python - <<'PY'
import importlib
mods = ["torch", "carla", "pygame", "transformers", "hydra", "numpy"]
for name in mods:
    importlib.import_module(name)
print("imports OK")
PY
  then
    ok "Python imports OK in env '$CONDA_ENV'"
  else
    fail "Python imports failed in env '$CONDA_ENV'"
  fi
else
  fail "conda env missing: $CONDA_ENV"
fi

if python3 -m py_compile "$ROOT_DIR/scripts/simlingo_dashboard.py" >/dev/null 2>&1; then
  ok "dashboard Python syntax OK"
else
  fail "dashboard Python syntax failed"
fi

echo
if [[ "$FAIL" -eq 0 ]]; then
  echo "[vla-av-check] READY: fresh install prerequisites look good."
else
  echo "[vla-av-check] NOT READY: fix the FAIL lines above."
fi
exit "$FAIL"
