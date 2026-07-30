#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_HOST="${AXIS1_SSH_HOST:-ucloud@ssh.cloud.sdu.dk}"
SSH_PORT="${AXIS1_SSH_PORT:-2665}"
REMOTE_WORK="${AXIS1_REMOTE_WORK:-~/work}"
REMOTE_REPO="$REMOTE_WORK/Simple-carla-WAM"
REMOTE_DATASET="${AXIS1_REMOTE_DATASET:-/home/ucloud/datasets/maram_groot_carla_frames}"
LOCAL_REPO="${AXIS1_LOCAL_REPO:-$ROOT_DIR/external/Simple-carla-WAM}"
LOCAL_DATASET_ZIP="${AXIS1_LOCAL_DATASET_ZIP:-$ROOT_DIR/exports/maram_groot_carla_frames.zip}"
LOCAL_DATASET_DIR="${AXIS1_LOCAL_DATASET_DIR:-$ROOT_DIR/exports/maram_groot_carla_frames}"
LOCAL_OUT="${AXIS1_LOCAL_OUT:-$ROOT_DIR/artifacts/axis1_training/axis1_live_2665}"
LOG_FILE="${AXIS1_LOG_FILE:-$ROOT_DIR/logs/axis1_auto_fetch_2665.log}"
PID_FILE="${AXIS1_PID_FILE:-$ROOT_DIR/logs/axis1_auto_fetch_2665.pid}"
POLL_SECONDS="${AXIS1_POLL_SECONDS:-300}"

SSH=(
  ssh
  -p "$SSH_PORT"
  -o "StrictHostKeyChecking=accept-new"
  -o "ConnectTimeout=20"
  -o "ServerAliveInterval=10"
  -o "ServerAliveCountMax=3"
  "$SSH_HOST"
)
RSYNC_RSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"

if [[ ! -d "$LOCAL_REPO" ]]; then
  echo "[axis1-start] Missing local repo: $LOCAL_REPO" >&2
  exit 1
fi
if [[ ! -f "$LOCAL_DATASET_ZIP" && ! -d "$LOCAL_DATASET_DIR" ]]; then
  echo "[axis1-start] Missing dataset zip/dir: $LOCAL_DATASET_ZIP or $LOCAL_DATASET_DIR" >&2
  exit 1
fi

echo "[axis1-start] cleaning local failed/live output: $LOCAL_OUT"
rm -rf "$LOCAL_OUT"
mkdir -p "$LOCAL_OUT" "$ROOT_DIR/logs"

echo "[axis1-start] checking VM connectivity"
"${SSH[@]}" "hostname; date; mkdir -p $REMOTE_WORK ~/datasets"

echo "[axis1-start] installing remote base packages"
"${SSH[@]}" "python3 --version; if ! python3 -m pip --version >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y python3-pip; fi; if ! command -v tmux >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y tmux; fi; python3 -m pip install --user -U pip wheel setuptools"

echo "[axis1-start] uploading repo -> $REMOTE_REPO"
"${SSH[@]}" "rm -rf $REMOTE_REPO && mkdir -p $REMOTE_REPO"
rsync -az --delete --info=progress2 -e "$RSYNC_RSH" \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='logdir/' \
  --exclude='outputs/' \
  --exclude='*.pyc' \
  "$LOCAL_REPO/" "$SSH_HOST:$REMOTE_REPO/"

echo "[axis1-start] uploading dataset -> $REMOTE_DATASET"
if [[ -f "$LOCAL_DATASET_ZIP" ]]; then
  rsync -az --info=progress2 -e "$RSYNC_RSH" \
    "$LOCAL_DATASET_ZIP" "$SSH_HOST:~/datasets/maram_groot_carla_frames.zip"
  "${SSH[@]}" "rm -rf $REMOTE_DATASET && mkdir -p $(dirname "$REMOTE_DATASET") && python3 - <<'PY'
import zipfile
from pathlib import Path
zip_path = Path.home() / 'datasets' / 'maram_groot_carla_frames.zip'
out_root = Path('$REMOTE_DATASET').parent
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(out_root)
print('unzipped', zip_path, 'to', out_root)
PY"
else
  "${SSH[@]}" "rm -rf $REMOTE_DATASET && mkdir -p $REMOTE_DATASET"
  rsync -az --delete --info=progress2 -e "$RSYNC_RSH" \
    "$LOCAL_DATASET_DIR/" "$SSH_HOST:$REMOTE_DATASET/"
fi

echo "[axis1-start] remote dependency install"
"${SSH[@]}" "cd $REMOTE_REPO && python3 - <<'PY'
import importlib.util
import subprocess
import sys

base_packages = [
    'numpy',
    'pillow',
    'pandas',
    'matplotlib',
    'tensorboard',
    'sentence-transformers',
    'moviepy',
    'imageio',
    'imageio-ffmpeg',
    'opencv-python',
    'einops',
    'ruamel.yaml',
    'protobuf',
]

def run(cmd):
    print('+', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd)

run([sys.executable, '-m', 'pip', 'install', '--user', '-U', *base_packages])

def torch_cuda_ok():
    try:
        import torch
        print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
        print('cuda available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            x = torch.randn(1, 3, 64, 64, device='cuda')
            conv = torch.nn.Conv2d(3, 8, 3).cuda()
            y = conv(x)
            torch.cuda.synchronize()
            print('cuda conv ok:', tuple(y.shape), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
        return True
    except Exception as exc:
        print('torch cuda test failed:', repr(exc), flush=True)
        return False

if not torch_cuda_ok():
    run([sys.executable, '-m', 'pip', 'install', '--user', '-U', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu121'])
    if not torch_cuda_ok():
        run([sys.executable, '-m', 'pip', 'install', '--user', '-U', '--pre', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/nightly/cu128'])
        if not torch_cuda_ok():
            raise SystemExit('PyTorch CUDA test still failed after stable and nightly installs')
PY"

echo "[axis1-start] remote sanity check"
"${SSH[@]}" "cd $REMOTE_REPO && DATASET_ROOT=$REMOTE_DATASET MANIFEST_PATH=$REMOTE_DATASET/manifest.jsonl COT_PATH=$REMOTE_REPO/data/danger_cot/cot_dataset.jsonl python3 test_setup.py --dataset_root $REMOTE_DATASET --manifest_path $REMOTE_DATASET/manifest.jsonl"

echo "[axis1-start] launching remote tmux training"
"${SSH[@]}" "cd $REMOTE_REPO && tmux kill-session -t axis1 2>/dev/null || true && mkdir -p logdir outputs && tmux new-session -d -s axis1 'DATASET_ROOT=$REMOTE_DATASET MANIFEST_PATH=$REMOTE_DATASET/manifest.jsonl COT_PATH=$REMOTE_REPO/data/danger_cot/cot_dataset.jsonl WAM_STEPS=500000 COT_EPISODES=1000 bash scripts/run_axis1_server.sh'"

echo "[axis1-start] starting local auto mirror"
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    kill "$old_pid" || true
  fi
fi
AXIS1_SSH_HOST="$SSH_HOST" \
AXIS1_SSH_PORT="$SSH_PORT" \
AXIS1_REMOTE_REPO="$REMOTE_REPO" \
AXIS1_LOCAL_OUT="$LOCAL_OUT" \
AXIS1_POLL_SECONDS="$POLL_SECONDS" \
  nohup bash "$ROOT_DIR/scripts/mirror_axis1_vm_progress.sh" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo
echo "[axis1-start] launched"
echo "[axis1-start] remote: $SSH_HOST:$REMOTE_REPO"
echo "[axis1-start] local mirror: $LOCAL_OUT"
echo "[axis1-start] mirror log: $LOG_FILE"
echo "[axis1-start] mirror pid: $(cat "$PID_FILE")"
