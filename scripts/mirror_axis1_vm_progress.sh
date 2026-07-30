#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_HOST="${AXIS1_SSH_HOST:-ucloud@ssh.cloud.sdu.dk}"
SSH_PORT="${AXIS1_SSH_PORT:-2665}"
REMOTE_REPO="${AXIS1_REMOTE_REPO:-~/work/Simple-carla-WAM}"
LOCAL_OUT="${AXIS1_LOCAL_OUT:-$ROOT_DIR/artifacts/axis1_training/axis1_live_2665}"
POLL_SECONDS="${AXIS1_POLL_SECONDS:-300}"

mkdir -p "$LOCAL_OUT"

echo "[axis1-mirror] host=$SSH_HOST port=$SSH_PORT"
echo "[axis1-mirror] remote_repo=$REMOTE_REPO"
echo "[axis1-mirror] local_out=$LOCAL_OUT"
echo "[axis1-mirror] poll=${POLL_SECONDS}s"

while true; do
  echo
  echo "[axis1-mirror] sync attempt $(date '+%Y-%m-%d %H:%M:%S')"
  AXIS1_SSH_HOST="$SSH_HOST" \
  AXIS1_SSH_PORT="$SSH_PORT" \
  AXIS1_REMOTE_REPO="$REMOTE_REPO" \
  AXIS1_LOCAL_OUT="$LOCAL_OUT" \
    bash "$ROOT_DIR/scripts/fetch_axis1_training_results.sh" --local-out "$LOCAL_OUT" --host "$SSH_HOST" --port "$SSH_PORT" --remote-repo "$REMOTE_REPO" || true

  if bash "$ROOT_DIR/scripts/watch_axis1_vm_training.sh" >/tmp/axis1_watch_status.txt 2>/tmp/axis1_watch_status.err; then
    cat /tmp/axis1_watch_status.txt
    if grep -q '^wam_final=1$' /tmp/axis1_watch_status.txt \
      && grep -q '^cot_policy=1$' /tmp/axis1_watch_status.txt \
      && grep -q '^cot_worldmodel=1$' /tmp/axis1_watch_status.txt; then
      echo "[axis1-mirror] final artifacts detected; one last sync and exit"
      AXIS1_SSH_HOST="$SSH_HOST" \
      AXIS1_SSH_PORT="$SSH_PORT" \
      AXIS1_REMOTE_REPO="$REMOTE_REPO" \
      AXIS1_LOCAL_OUT="$LOCAL_OUT" \
        bash "$ROOT_DIR/scripts/fetch_axis1_training_results.sh" --local-out "$LOCAL_OUT" --host "$SSH_HOST" --port "$SSH_PORT" --remote-repo "$REMOTE_REPO" || true
      exit 0
    fi
  else
    cat /tmp/axis1_watch_status.err || true
  fi

  sleep "$POLL_SECONDS"
done
