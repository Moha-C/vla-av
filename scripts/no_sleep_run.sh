#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  scripts/no_sleep_run.sh COMMAND [ARG...]

Example:
  scripts/no_sleep_run.sh bash scripts/run_maram_dreamer_carla_training.sh

This keeps the desktop from idling/sleeping/shutting down through logind while
the command is running. It cannot protect against real power loss or a kernel
crash.
EOF
  exit 2
fi

exec systemd-inhibit \
  --what=sleep:shutdown:idle:handle-lid-switch \
  --who="vla-av training" \
  --why="Long CARLA/Dreamer training must not be interrupted" \
  --mode=block \
  "$@"
