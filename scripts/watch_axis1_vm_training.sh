#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${AXIS1_SSH_HOST:-ucloud@ssh.cloud.sdu.dk}"
SSH_PORT="${AXIS1_SSH_PORT:-2665}"
REMOTE_REPO="${AXIS1_REMOTE_REPO:-~/work/Simple-carla-WAM}"
RUN_NAME="${AXIS1_RUN_NAME:-carla_500000}"
COT_RUN_NAME="${AXIS1_COT_RUN_NAME:-cot_1000}"
CONNECT_TIMEOUT="${AXIS1_SSH_CONNECT_TIMEOUT:-15}"

SSH=(
  ssh
  -p "$SSH_PORT"
  -o "StrictHostKeyChecking=accept-new"
  -o "ConnectTimeout=$CONNECT_TIMEOUT"
  -o "ServerAliveInterval=10"
  -o "ServerAliveCountMax=3"
  "$SSH_HOST"
)

"${SSH[@]}" "cd $REMOTE_REPO 2>/dev/null || { echo 'remote repo missing: $REMOTE_REPO'; exit 0; }
printf '[axis1-watch] remote=%s\n' \"\$(pwd)\"
echo
echo 'tmux:'
tmux ls 2>/dev/null || true
echo
echo 'gpu:'
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null || true
echo
wam_final=0
cot_policy=0
cot_worldmodel=0
cot_eval=0
test -f logdir/$RUN_NAME/checkpoint_final.pt && wam_final=1
test -f outputs/$COT_RUN_NAME/dreamer_ppo_policy.pt && cot_policy=1
test -f outputs/$COT_RUN_NAME/dreamer_ppo_worldmodel.pt && cot_worldmodel=1
test -f outputs/${COT_RUN_NAME}_eval.log && cot_eval=1
echo \"wam_final=\$wam_final\"
echo \"cot_policy=\$cot_policy\"
echo \"cot_worldmodel=\$cot_worldmodel\"
echo \"cot_eval=\$cot_eval\"
echo
echo 'latest WAM:'
grep -o 'Step[[:space:]]*[0-9]*/[0-9]*' logdir/$RUN_NAME.log 2>/dev/null | tail -1 || true
ls -lh logdir/$RUN_NAME/checkpoint_*.pt 2>/dev/null | tail -5 || true
echo
echo 'last WAM log:'
tail -n 12 logdir/$RUN_NAME.log 2>/dev/null || true
echo
echo 'last CoT log:'
tail -n 12 outputs/$COT_RUN_NAME.log 2>/dev/null || true
echo
echo 'local output candidates on remote:'
find logdir/$RUN_NAME outputs/$COT_RUN_NAME -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -20 || true"
