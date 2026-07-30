#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${MARAM_DREAMER_REPO:-$ROOT_DIR/external/maram_dreamer}"
CONDA_ENV="${MARAM_DREAMER_CONDA_ENV:-vla-av}"

HOST="${CARLA_HOST:-localhost}"
PORT="${CARLA_PORT:-2000}"
TOWN="${CARLA_TOWN:-}"
DEVICE="${DEVICE:-cuda}"

EPISODES="${EPISODES:-30}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-600}"
N_VEHICLES="${N_VEHICLES:-15}"
N_WALKERS="${N_WALKERS:-20}"

WM_EPOCHS="${WM_EPOCHS:-30}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-256}"
RL_ITERS="${RL_ITERS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-50}"
EVAL_EVERY="${EVAL_EVERY:-25}"
SEED="${SEED:-0}"

RUN_COLLECT="${RUN_COLLECT:-1}"
RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_RL="${RUN_RL:-1}"
TRAFFIC_PREDICTOR="${TRAFFIC_PREDICTOR:-1}"
DOMAIN_RANDOMIZATION="${DOMAIN_RANDOMIZATION:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/experiments/maram_dreamer_carla/$RUN_ID}"
DATA_FILE="$OUT_DIR/data/carla_normal.npz"
WM_CKPT="$OUT_DIR/checkpoints/wm_pretrained_carla.pt"
RL_CKPT="$OUT_DIR/checkpoints/sdbs_checkpoint.pt"
LOG_DIR="$OUT_DIR/logs"

usage() {
  cat <<EOF
Usage: bash scripts/run_maram_dreamer_carla_training.sh

Runs the maram-br/Dreamer README pipeline against a live CARLA server:
  1. preflight CARLA + torch/CUDA
  2. collect normal driving with Traffic Manager
  3. pretrain WorldModel offline
  4. train S-DBS Dreamer-PPO in CARLA

Common environment overrides:
  DEVICE=cuda|cpu                  default: $DEVICE
  CARLA_HOST=localhost             default: $HOST
  CARLA_PORT=2000                  default: $PORT
  CARLA_TOWN=Town03                default: current loaded CARLA map
  EPISODES=30                      default: $EPISODES
  STEPS_PER_EPISODE=600            default: $STEPS_PER_EPISODE
  WM_EPOCHS=30                     default: $WM_EPOCHS
  RL_ITERS=1000                    default: $RL_ITERS
  RUN_COLLECT=0|1                  default: $RUN_COLLECT
  RUN_PRETRAIN=0|1                 default: $RUN_PRETRAIN
  RUN_RL=0|1                       default: $RUN_RL
  TRAFFIC_PREDICTOR=0|1            default: $TRAFFIC_PREDICTOR
  OUT_DIR=/path/to/output          default: $OUT_DIR

Example smoke-sized CARLA run:
  EPISODES=2 STEPS_PER_EPISODE=100 WM_EPOCHS=2 RL_ITERS=10 \\
    bash scripts/run_maram_dreamer_carla_training.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

run_py() {
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

mkdir -p "$OUT_DIR/data" "$OUT_DIR/checkpoints" "$LOG_DIR"

echo "[maram-dreamer] repo=$REPO_DIR"
echo "[maram-dreamer] conda_env=$CONDA_ENV"
echo "[maram-dreamer] out_dir=$OUT_DIR"
echo "[maram-dreamer] carla=${HOST}:${PORT} town=$TOWN device=$DEVICE"

cd "$REPO_DIR"

echo "[maram-dreamer] preflight"
cat > "$OUT_DIR/preflight.py" <<PY
import sys
print("python:", sys.version)

import torch
print("torch:", torch.__version__)
print("torch.cuda.is_available:", torch.cuda.is_available())
if "$DEVICE" == "cuda":
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this environment. Fix local NVIDIA driver/device nodes "
            "or run with DEVICE=cpu for a non-GPU debug run."
        )
    print("cuda device:", torch.cuda.get_device_name(0))

import carla
print("carla module: OK")
client = carla.Client("$HOST", int("$PORT"))
client.set_timeout(10.0)
world = client.get_world()
print("server version:", client.get_server_version())
print("current map:", world.get_map().name)
PY
run_py "$OUT_DIR/preflight.py" 2>&1 | tee "$LOG_DIR/00_preflight.log"

if [[ "$RUN_COLLECT" == "1" ]]; then
  echo "[maram-dreamer] collecting CARLA normal driving -> $DATA_FILE"
  run_py scripts/collect_carla_driving_data.py \
    --host "$HOST" --port "$PORT" \
    --town "$TOWN" \
    --episodes "$EPISODES" \
    --steps_per_episode "$STEPS_PER_EPISODE" \
    --n_vehicles "$N_VEHICLES" \
    --n_walkers "$N_WALKERS" \
    --seed "$SEED" \
    --out "$DATA_FILE" \
    2>&1 | tee "$LOG_DIR/01_collect.log"
else
  echo "[maram-dreamer] skipping collection; using existing $DATA_FILE"
fi

if [[ "$RUN_PRETRAIN" == "1" ]]; then
  echo "[maram-dreamer] pretraining world model -> $WM_CKPT"
  run_py scripts/pretrain_world_model_offline.py \
    --data "$DATA_FILE" \
    --out "$WM_CKPT" \
    --epochs "$WM_EPOCHS" \
    --batch_size "$WM_BATCH_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/02_pretrain_wm.log"
else
  echo "[maram-dreamer] skipping pretrain; using existing $WM_CKPT"
fi

if [[ "$RUN_RL" == "1" ]]; then
  echo "[maram-dreamer] training S-DBS Dreamer-PPO -> $RL_CKPT"
  args=(
    scripts/run_training.py
    --mode carla
    --device "$DEVICE"
    --carla_host "$HOST"
    --carla_port "$PORT"
    --town "$TOWN"
    --wm_checkpoint "$WM_CKPT"
    --iters "$RL_ITERS"
    --eval_every "$EVAL_EVERY"
    --save_every "$SAVE_EVERY"
    --save_path "$RL_CKPT"
    --seed "$SEED"
  )
  if [[ "$TRAFFIC_PREDICTOR" == "1" ]]; then
    args+=(--traffic_predictor)
  fi
  if [[ "$DOMAIN_RANDOMIZATION" == "1" ]]; then
    args+=(--domain_randomization)
  fi
  run_py "${args[@]}" 2>&1 | tee "$LOG_DIR/03_train_rl.log"
else
  echo "[maram-dreamer] skipping RL training"
fi

cat > "$OUT_DIR/README_RESULTS.txt" <<EOF
maram-br/Dreamer CARLA training run

Repo: $REPO_DIR
Conda env: $CONDA_ENV
CARLA: $HOST:$PORT
Town: $TOWN
Device: $DEVICE

Outputs:
  data/carla_normal.npz
  checkpoints/wm_pretrained_carla.pt
  checkpoints/sdbs_checkpoint.pt
  logs/00_preflight.log
  logs/01_collect.log
  logs/02_pretrain_wm.log
  logs/03_train_rl.log
EOF

echo
echo "[maram-dreamer] done"
echo "[maram-dreamer] results=$OUT_DIR"
find "$OUT_DIR" -maxdepth 3 -type f -printf "  %p (%s bytes)\n" | sort
