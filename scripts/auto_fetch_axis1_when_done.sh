#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_HOST="${AXIS1_SSH_HOST:-ucloud@ssh.cloud.sdu.dk}"
SSH_PORT="${AXIS1_SSH_PORT:-2014}"
SSH_CONNECT_TIMEOUT="${AXIS1_SSH_CONNECT_TIMEOUT:-20}"
REMOTE_REPO="${AXIS1_REMOTE_REPO:-~/work/Simple-carla-WAM}"
RUN_NAME="${AXIS1_RUN_NAME:-carla_500000}"
COT_RUN_NAME="${AXIS1_COT_RUN_NAME:-cot_1000}"
POLL_SECONDS="${AXIS1_POLL_SECONDS:-300}"
LOCAL_OUT="${AXIS1_LOCAL_OUT:-}"
FETCH_ALL_CHECKPOINTS="${AXIS1_FETCH_ALL_CHECKPOINTS:-0}"
WAIT_WAM_ONLY=0

usage() {
  cat <<EOF
Usage: bash scripts/auto_fetch_axis1_when_done.sh [options]

Wait until the remote Axis 1 training finishes, then fetch useful artifacts.

Options:
  --poll SECONDS          Poll interval. Default: $POLL_SECONDS
  --wait-wam-only         Fetch as soon as WAM checkpoint_final.pt exists.
                          Default waits for WAM + Dreamer-PPO policy/worldmodel.
  --all-checkpoints       Also fetch all intermediate WAM checkpoints.
  --local-out DIR         Destination directory passed to fetch script.
  --host USER@HOST        SSH host. Default: $SSH_HOST
  --port PORT             SSH port. Default: $SSH_PORT
  --remote-repo PATH      Remote repo path. Default: $REMOTE_REPO
  --help                  Show this help.

Environment overrides:
  AXIS1_SSH_HOST, AXIS1_SSH_PORT, AXIS1_REMOTE_REPO, AXIS1_POLL_SECONDS,
  AXIS1_LOCAL_OUT, AXIS1_RUN_NAME, AXIS1_COT_RUN_NAME,
  AXIS1_FETCH_ALL_CHECKPOINTS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --wait-wam-only)
      WAIT_WAM_ONLY=1
      shift
      ;;
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
      echo "[axis1-auto-fetch] Unknown argument: $1" >&2
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

remote() {
  "${SSH[@]}" "cd $REMOTE_REPO && $*"
}

remote_status() {
  remote "
    wam_final=0
    cot_policy=0
    cot_worldmodel=0
    cot_eval=0
    latest_step=unknown
    latest_ckpt=none

    test -f logdir/$RUN_NAME/checkpoint_final.pt && wam_final=1
    test -f outputs/$COT_RUN_NAME/dreamer_ppo_policy.pt && cot_policy=1
    test -f outputs/$COT_RUN_NAME/dreamer_ppo_worldmodel.pt && cot_worldmodel=1
    test -f outputs/${COT_RUN_NAME}_eval.log && cot_eval=1

    if test -f logdir/$RUN_NAME.log; then
      latest_step=\$(grep -o 'Step[[:space:]]*[0-9]*/[0-9]*' logdir/$RUN_NAME.log | tail -1 | tr -s ' ' | cut -d' ' -f2)
    fi
    latest_ckpt=\$(ls -1 logdir/$RUN_NAME/checkpoint_*.pt 2>/dev/null | grep -v checkpoint_final.pt | sort | tail -1 | xargs -r basename)

    echo \"wam_final=\$wam_final\"
    echo \"cot_policy=\$cot_policy\"
    echo \"cot_worldmodel=\$cot_worldmodel\"
    echo \"cot_eval=\$cot_eval\"
    echo \"latest_step=\$latest_step\"
    echo \"latest_ckpt=\${latest_ckpt:-none}\"
    tmux ls 2>/dev/null | sed 's/^/tmux=/' || true
  "
}

is_done() {
  local status="$1"
  local wam_final cot_policy cot_worldmodel

  wam_final="$(awk -F= '$1=="wam_final"{print $2}' <<<"$status")"
  cot_policy="$(awk -F= '$1=="cot_policy"{print $2}' <<<"$status")"
  cot_worldmodel="$(awk -F= '$1=="cot_worldmodel"{print $2}' <<<"$status")"

  if [[ "$WAIT_WAM_ONLY" == "1" ]]; then
    [[ "$wam_final" == "1" ]]
  else
    [[ "$wam_final" == "1" && "$cot_policy" == "1" && "$cot_worldmodel" == "1" ]]
  fi
}

echo "[axis1-auto-fetch] host=$SSH_HOST port=$SSH_PORT"
echo "[axis1-auto-fetch] remote_repo=$REMOTE_REPO"
echo "[axis1-auto-fetch] poll=${POLL_SECONDS}s"
if [[ "$WAIT_WAM_ONLY" == "1" ]]; then
  echo "[axis1-auto-fetch] waiting for WAM final checkpoint only"
else
  echo "[axis1-auto-fetch] waiting for WAM final + Dreamer-PPO outputs"
fi

while true; do
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  status="$(remote_status || true)"

  echo
  echo "[$now] remote status:"
  echo "$status"

  if is_done "$status"; then
    echo
    echo "[axis1-auto-fetch] training artifacts are ready; fetching now"

    fetch_args=()
    if [[ -n "$LOCAL_OUT" ]]; then
      fetch_args+=(--local-out "$LOCAL_OUT")
    fi
    if [[ "$FETCH_ALL_CHECKPOINTS" == "1" ]]; then
      fetch_args+=(--all-checkpoints)
    fi
    fetch_args+=(--host "$SSH_HOST" --port "$SSH_PORT" --remote-repo "$REMOTE_REPO")

    bash "$ROOT_DIR/scripts/fetch_axis1_training_results.sh" "${fetch_args[@]}"
    echo "[axis1-auto-fetch] complete"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
