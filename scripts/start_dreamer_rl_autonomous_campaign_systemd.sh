#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${DREAMER_RL_CAMPAIGN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$ROOT_DIR/logs/dreamer_rl_campaign/$RUN_ID"
LATEST_FILE="$ROOT_DIR/logs/dreamer_rl_campaign/latest_campaign.txt"
UNIT="vla-av-dreamer-rl-campaign-${RUN_ID}.service"
RUN_SCRIPT="$RUN_DIR/run_campaign_systemd.sh"
LOG_FILE="$RUN_DIR/campaign_stdout.log"
UNIT_FILE="$RUN_DIR/systemd_unit.txt"
UNIT_NAME_FILE="$RUN_DIR/systemd_unit_name.txt"

mkdir -p "$RUN_DIR" "$(dirname "$LATEST_FILE")"
echo "$RUN_DIR" > "$LATEST_FILE"

write_export() {
  local name="$1"
  local value="${!name-}"
  if [[ -n "$value" ]]; then
    printf 'export %s=%q\n' "$name" "$value"
  fi
}

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'source "$HOME/.bashrc" >/dev/null 2>&1 || true\n'
  write_export PATH
  write_export HOME
  write_export DISPLAY
  write_export XAUTHORITY
  write_export CARLA_ROOT
  write_export SUMO_HOME
  write_export PYTHONPATH
  write_export CONDA_EXE
  write_export CONDA_PREFIX
  write_export DREAMER_RL_DEVICE
  write_export DREAMER_RL_CAMPAIGN_MAX_ROUTES_PER_BUCKET
  write_export DREAMER_RL_CAMPAIGN_RL_EPISODES
  write_export DREAMER_RL_CAMPAIGN_WM_EPOCHS
  write_export DREAMER_RL_CAMPAIGN_VIDEO_QUALITY
  write_export DREAMER_RL_CAMPAIGN_SEED
  write_export DREAMER_RL_CAMPAIGN_TEACHER_MODE
  write_export DREAMER_RL_CAMPAIGN_SAMPLE_INTERVAL
  write_export DREAMER_RL_CAMPAIGN_MIN_GOOD_TRACES
  write_export DREAMER_RL_CAMPAIGN_COLLECT_RETRIES
  write_export DREAMER_RL_CAMPAIGN_COLLECT_MAX_WALL_SECONDS
  write_export DREAMER_RL_CAMPAIGN_RETRY_IF_SHORTER_THAN
  write_export DREAMER_RL_CAMPAIGN_ROUTE_COOLDOWN_SECONDS
  write_export DREAMER_RL_MIN_TRANSITIONS
  write_export DREAMER_RL_MIN_RUNS
  write_export DREAMER_RL_MIN_ROUTES
  write_export DREAMER_RL_MAX_STATIONARY_FRACTION
  write_export DREAMER_RL_MIN_RECOVERY_FRACTION
  write_export DREAMER_RL_CAMPAIGN_RL_MAX_EPISODE_STEPS
  write_export DREAMER_RL_CAMPAIGN_RL_ROLLOUT_SIZE
  write_export DREAMER_RL_CAMPAIGN_RL_EVAL_INTERVAL
  write_export CARLA_QUALITY
  printf 'cd %q\n' "$ROOT_DIR"
  printf 'exec python3 %q --run-id %q >> %q 2>&1\n' \
    "$ROOT_DIR/scripts/run_dreamer_rl_autonomous_campaign.py" "$RUN_ID" "$LOG_FILE"
} > "$RUN_SCRIPT"
chmod +x "$RUN_SCRIPT"

systemd-run --user \
  --unit "$UNIT" \
  --collect \
  --property=WorkingDirectory="$ROOT_DIR" \
  "$RUN_SCRIPT" | tee "$UNIT_FILE"
echo "$UNIT" > "$UNIT_NAME_FILE"

echo "[dreamer-rl-campaign] launched via systemd"
echo "[dreamer-rl-campaign] unit=$UNIT"
echo "[dreamer-rl-campaign] run_dir=$RUN_DIR"
echo "[dreamer-rl-campaign] watch: bash scripts/watch_dreamer_rl_autonomous_campaign.sh"
