#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMLINGO_ROOT="${SIMLINGO_ROOT:-$ROOT_DIR/external/simlingo}"
MODEL_DIR="${SIMLINGO_MODEL_DIR:-$ROOT_DIR/models/simlingo_hf}"
CARLA_ROOT="${CARLA_ROOT:-$HOME/carla_simulator}"
CONDA_ENV="${SIMLINGO_ENV_NAME:-simlingo}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
TOWN="${TOWN:-Town10HD}"
ROUTE_ID="${ROUTE_ID:-}"
ROUTE_FILE="${ROUTE_FILE:-}"
SEED="${SEED:-1}"
QUALITY="${CARLA_QUALITY:-Low}"
RENDER_MODE="${SIMLINGO_RENDER_MODE:-offscreen}"
OUT_DIR="${SIMLINGO_OUT_DIR:-$ROOT_DIR/logs/simlingo_eval}"

MODEL_PT="$MODEL_DIR/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
if [[ -n "$ROUTE_FILE" ]]; then
  ROUTE_XML="$ROUTE_FILE"
elif [[ -n "$ROUTE_ID" ]]; then
  if [[ "$ROUTE_ID" == "random" ]]; then
    ROUTE_XML="$(python3 - "$SIMLINGO_ROOT" "$CARLA_ROOT" "$TOWN" <<'PY'
import glob
import os
import random
import re
import sys

root, carla_root, wanted_town = sys.argv[1], sys.argv[2], sys.argv[3]
maps_dir = os.path.join(carla_root, "CarlaUE4", "Content", "Carla", "Maps")
installed = set()
for path in glob.glob(os.path.join(maps_dir, "**", "Town*.umap"), recursive=True):
    town = os.path.splitext(os.path.basename(path))[0]
    if "_Tile_" not in town:
        installed.add(town)
routes = []
for path in sorted(glob.glob(os.path.join(root, "leaderboard/data/bench2drive_split/*.xml"))):
    with open(path, errors="ignore") as f:
        text = f.read(4096)
    match = re.search(r'town="([^"]+)"', text)
    if not match:
        continue
    town = match.group(1)
    if town not in installed:
        continue
    if wanted_town not in ("any", "random", "*") and town != wanted_town:
        continue
    routes.append(path)
if not routes:
    raise SystemExit(f"No compatible route found for TOWN={wanted_town}; run scripts/list_simlingo_routes.py")
print(random.choice(routes))
PY
)"
  else
    ROUTE_XML="$SIMLINGO_ROOT/leaderboard/data/bench2drive_split/bench2drive_${ROUTE_ID}.xml"
  fi
else
  ROUTE_XML="$(python3 - "$SIMLINGO_ROOT" "$TOWN" <<'PY'
import glob
import os
import re
import sys

root, town = sys.argv[1], sys.argv[2]
for path in sorted(glob.glob(os.path.join(root, "leaderboard/data/bench2drive_split/*.xml"))):
    with open(path, errors="ignore") as f:
        text = f.read(4096)
    if f'town="{town}"' in text:
        print(path)
        break
PY
)"
fi
ROUTE_LABEL="$(basename "$ROUTE_XML" .xml)"
RESULT_JSON="$OUT_DIR/results_${ROUTE_LABEL}_seed_${SEED}.json"
RUN_LOG="$OUT_DIR/run_${ROUTE_LABEL}_seed_${SEED}.log"
CARLA_LOG="$OUT_DIR/carla_${ROUTE_LABEL}_seed_${SEED}.log"

mkdir -p "$OUT_DIR"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

if [[ ! -f "$MODEL_PT" ]]; then
  echo "[simlingo-eval] Missing model: $MODEL_PT" >&2
  echo "[simlingo-eval] Run: bash scripts/download_simlingo_model.sh" >&2
  exit 1
fi
if [[ ! -f "$ROUTE_XML" ]]; then
  echo "[simlingo-eval] Missing route: $ROUTE_XML" >&2
  echo "[simlingo-eval] Try: TOWN=Town10HD bash scripts/run_simlingo_local_eval.sh" >&2
  exit 1
fi
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[simlingo-eval] Missing conda init: $CONDA_SH" >&2
  exit 1
fi

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

export CARLA_ROOT
export CARLA_QUALITY="$QUALITY"
export SIMLINGO_RENDER_MODE="$RENDER_MODE"
export WORK_DIR="$SIMLINGO_ROOT"
export SCENARIO_RUNNER_ROOT="$SIMLINGO_ROOT/Bench2Drive/scenario_runner"
export LEADERBOARD_ROOT="$SIMLINGO_ROOT/Bench2Drive/leaderboard"
export SAVE_PATH="$OUT_DIR/viz/"
export ROUTES="$ROUTE_XML"
export SIMLINGO_CARLA_LOG="$CARLA_LOG"
# Use carla==0.9.15 from the simlingo conda env, plus CARLA's PythonAPI/carla
# directory only for the "agents" helper package. Do not add the py3.7 egg:
# it can require old system libs such as libtiff.so.5 on newer Ubuntu installs.
export PYTHONPATH="$SIMLINGO_ROOT:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"

echo "[simlingo-eval] route=$ROUTE_XML"
echo "[simlingo-eval] route_town=$(grep -o 'town="[^"]*"' "$ROUTE_XML" | head -n 1 | cut -d'"' -f2)"
echo "[simlingo-eval] scenario=$(grep -o '<scenario name=\"[^\"]*\" type=\"[^\"]*\"' "$ROUTE_XML" | head -n 1 | sed 's/^ *//')"
echo "[simlingo-eval] model=$MODEL_PT"
echo "[simlingo-eval] result=$RESULT_JSON"
echo "[simlingo-eval] log=$RUN_LOG"
echo "[simlingo-eval] carla_log=$CARLA_LOG"

python -u "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py" \
  --routes="$ROUTE_XML" \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint="$RESULT_JSON" \
  --timeout=600 \
  --agent="$SIMLINGO_ROOT/team_code/agent_simlingo.py" \
  --agent-config="$MODEL_PT" \
  --traffic-manager-seed="$SEED" \
  --port="$PORT" \
  --traffic-manager-port="$TM_PORT" \
  2>&1 | tee "$RUN_LOG"
