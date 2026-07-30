#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_HOST="${AXIS1_SSH_HOST:-ucloud@ssh.cloud.sdu.dk}"
SSH_PORT="${AXIS1_SSH_PORT:-2014}"
SSH_CONNECT_TIMEOUT="${AXIS1_SSH_CONNECT_TIMEOUT:-20}"
REMOTE_REPO="${AXIS1_REMOTE_REPO:-~/work/Simple-carla-WAM}"
RUN_NAME="${AXIS1_RUN_NAME:-carla_500000}"
COT_RUN_NAME="${AXIS1_COT_RUN_NAME:-cot_1000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOCAL_OUT="${AXIS1_LOCAL_OUT:-$ROOT_DIR/artifacts/axis1_training/axis1_${STAMP}}"
FETCH_ALL_CHECKPOINTS=0

usage() {
  cat <<EOF
Usage: bash scripts/fetch_axis1_training_results.sh [options]

Fetch useful Axis 1 WAM + Dreamer-PPO training artifacts from the GPU VM.

Options:
  --all-checkpoints       Also download every WAM checkpoint_*.pt file.
                          Default only downloads checkpoint_final.pt if present,
                          plus the latest numbered checkpoint as fallback/progress.
  --local-out DIR         Destination directory.
  --host USER@HOST        SSH host. Default: $SSH_HOST
  --port PORT             SSH port. Default: $SSH_PORT
  --remote-repo PATH      Remote repo path. Default: $REMOTE_REPO
  --help                  Show this help.

Environment overrides:
  AXIS1_SSH_HOST, AXIS1_SSH_PORT, AXIS1_REMOTE_REPO, AXIS1_LOCAL_OUT,
  AXIS1_RUN_NAME, AXIS1_COT_RUN_NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-checkpoints)
      FETCH_ALL_CHECKPOINTS=1
      shift
      ;;
    --local-out)
      LOCAL_OUT="$2"
      shift 2
      ;;
    --host)
      SSH_HOST="$2"
      shift 2
      ;;
    --port)
      SSH_PORT="$2"
      shift 2
      ;;
    --remote-repo)
      REMOTE_REPO="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[axis1-fetch] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SSH=(
  ssh
  -p "$SSH_PORT"
  -o "StrictHostKeyChecking=accept-new"
  -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT"
  -o "ServerAliveInterval=10"
  -o "ServerAliveCountMax=3"
  "$SSH_HOST"
)
RSYNC_RSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o ServerAliveInterval=10 -o ServerAliveCountMax=3"

remote() {
  "${SSH[@]}" "cd $REMOTE_REPO && $*"
}

remote_exists() {
  remote "test -e '$1'"
}

fetch_file() {
  local remote_rel="$1"
  local local_rel="$2"
  local local_path="$LOCAL_OUT/$local_rel"

  if remote_exists "$remote_rel"; then
    mkdir -p "$(dirname "$local_path")"
    echo "[axis1-fetch] fetching $remote_rel"
    rsync -av --partial --info=progress2 -e "$RSYNC_RSH" \
      "$SSH_HOST:$REMOTE_REPO/$remote_rel" "$local_path"
  else
    echo "[axis1-fetch] skip missing $remote_rel"
  fi
}

fetch_dir_filtered() {
  local remote_rel="$1"
  local local_rel="$2"
  shift 2

  if remote_exists "$remote_rel"; then
    mkdir -p "$LOCAL_OUT/$local_rel"
    echo "[axis1-fetch] fetching filtered directory $remote_rel"
    rsync -av --partial --info=progress2 -e "$RSYNC_RSH" \
      --include='*/' "$@" --exclude='*' \
      "$SSH_HOST:$REMOTE_REPO/$remote_rel/" "$LOCAL_OUT/$local_rel/"
  else
    echo "[axis1-fetch] skip missing $remote_rel"
  fi
}

mkdir -p "$LOCAL_OUT"

echo "[axis1-fetch] host=$SSH_HOST port=$SSH_PORT"
echo "[axis1-fetch] remote_repo=$REMOTE_REPO"
echo "[axis1-fetch] local_out=$LOCAL_OUT"

echo "[axis1-fetch] collecting remote status"
{
  echo "# Axis 1 Remote Status"
  echo "Fetched at: $(date)"
  echo "Remote: $SSH_HOST:$REMOTE_REPO"
  echo
  remote "pwd; echo; tmux ls 2>/dev/null || true; echo; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null || true; echo; tail -20 logdir/$RUN_NAME.log 2>/dev/null || true; echo; find logdir/$RUN_NAME -maxdepth 1 -type f -printf '%f %s bytes\n' 2>/dev/null | sort || true; echo; find outputs -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | sort || true"
} > "$LOCAL_OUT/remote_status.txt"

fetch_file "logdir/$RUN_NAME.log" "logs/$RUN_NAME.log"
fetch_file "logdir/$RUN_NAME/checkpoint_final.pt" "wam/checkpoint_final.pt"

LATEST_CKPT="$(remote "ls -1 logdir/$RUN_NAME/checkpoint_*.pt 2>/dev/null | grep -v checkpoint_final.pt | sort | tail -1 | xargs -r basename" || true)"
if [[ -n "$LATEST_CKPT" ]]; then
  fetch_file "logdir/$RUN_NAME/$LATEST_CKPT" "wam/$LATEST_CKPT"
else
  echo "[axis1-fetch] no numbered WAM checkpoint found yet"
fi

if [[ "$FETCH_ALL_CHECKPOINTS" == "1" ]]; then
  echo "[axis1-fetch] fetching all WAM checkpoints; this can be very large"
  mkdir -p "$LOCAL_OUT/wam/all_checkpoints"
  rsync -av --partial --info=progress2 -e "$RSYNC_RSH" \
    --include='checkpoint_*.pt' --exclude='*' \
    "$SSH_HOST:$REMOTE_REPO/logdir/$RUN_NAME/" "$LOCAL_OUT/wam/all_checkpoints/"
fi

fetch_file "outputs/$COT_RUN_NAME.log" "logs/$COT_RUN_NAME.log"
fetch_file "outputs/${COT_RUN_NAME}_eval.log" "logs/${COT_RUN_NAME}_eval.log"
fetch_file "outputs/$COT_RUN_NAME/dreamer_ppo_policy.pt" "dreamer_ppo/dreamer_ppo_policy.pt"
fetch_file "outputs/$COT_RUN_NAME/dreamer_ppo_worldmodel.pt" "dreamer_ppo/dreamer_ppo_worldmodel.pt"

fetch_dir_filtered "outputs/$COT_RUN_NAME" "dreamer_ppo/extra_outputs" \
  --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.txt' \
  --include='*.png' --include='*.jpg' --include='*.gif' --include='*.mp4' \
  --include='*.pt'

fetch_file "data/danger_cot/cot_dataset.jsonl" "data/cot_dataset.jsonl"

mkdir -p "$LOCAL_OUT/code_snapshot"
for f in \
  "train_carla.py" \
  "dreamer_ppo_train.py" \
  "evaluate_dreamer_ppo.py" \
  "evaluate_carla.py" \
  "build_cot_dataset.py" \
  "envs/carla_offline_cot_env.py" \
  "scripts/run_axis1_server.sh" \
  "requirements.txt"
do
  fetch_file "$f" "code_snapshot/$f"
done

cat > "$LOCAL_OUT/README_FETCHED_RESULTS.txt" <<EOF
Axis 1 training artifacts fetched from:
  $SSH_HOST:$REMOTE_REPO

Main files:
  logs/$RUN_NAME.log
    WAM training log.

  wam/checkpoint_final.pt
    Final WAM checkpoint, present only after the 500k-step WAM training finishes.

  wam/checkpoint_*.pt
    Latest numbered WAM checkpoint at fetch time. Useful if training is still running
    or if you want a progress snapshot.

  logs/$COT_RUN_NAME.log
    Dreamer-PPO CoT training log, present after the second phase starts.

  logs/${COT_RUN_NAME}_eval.log
    Dreamer-PPO evaluation log, present after evaluation finishes.

  dreamer_ppo/dreamer_ppo_policy.pt
    Trained Dreamer-PPO policy.

  dreamer_ppo/dreamer_ppo_worldmodel.pt
    Trained Dreamer-PPO world model.

  data/cot_dataset.jsonl
    Dangerous-driving CoT dataset used for Axis 1.

  code_snapshot/
    Exact scripts used on the VM for reproducibility.

  remote_status.txt
    Remote tmux/GPU/log/checkpoint status captured during fetch.

If the training is still running, run this script again later. It is safe to reuse.
For all intermediate WAM checkpoints, rerun with:
  bash scripts/fetch_axis1_training_results.sh --all-checkpoints
EOF

echo
echo "[axis1-fetch] done"
echo "[axis1-fetch] results: $LOCAL_OUT"
echo "[axis1-fetch] summary:"
find "$LOCAL_OUT" -maxdepth 3 -type f -printf "  %p (%s bytes)\n" | sort
