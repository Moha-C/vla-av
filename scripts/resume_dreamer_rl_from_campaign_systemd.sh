#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN_RUN_DIR="${1:-$(cat "$ROOT_DIR/logs/dreamer_rl_campaign/latest_campaign.txt")}"
CAMPAIGN_ID="$(basename "$CAMPAIGN_RUN_DIR")"
RUN_ID="${DREAMER_RL_RESUME_ID:-${CAMPAIGN_ID}_resume_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$ROOT_DIR/logs/dreamer_rl_resume/$RUN_ID"
RUN_SCRIPT="$RUN_DIR/run_resume.sh"
LOG_FILE="$RUN_DIR/resume_stdout.log"
UNIT="vla-av-dreamer-rl-resume-${RUN_ID}.service"

PPO_WARMSTART="${DREAMER_RL_PPO_WARMSTART:-$ROOT_DIR/logs/dreamer_rl_warmstart/ppo/$CAMPAIGN_ID/best_world_model.pt}"
SDBS_WARMSTART="${DREAMER_RL_SDBS_WARMSTART:-$ROOT_DIR/logs/dreamer_rl_warmstart/sdbs/$CAMPAIGN_ID/best_world_model.pt}"

if [[ ! -s "$PPO_WARMSTART" ]]; then
  echo "[dreamer-rl-resume] missing PPO warm-start: $PPO_WARMSTART" >&2
  exit 1
fi
if [[ ! -s "$SDBS_WARMSTART" ]]; then
  echo "[dreamer-rl-resume] missing SDBS warm-start: $SDBS_WARMSTART" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

write_export() {
  local name="$1"
  local value="${!name-}"
  if [[ -n "$value" ]]; then
    printf 'export %s=%q\n' "$name" "$value"
  fi
}

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -u\n'
  printf 'source "$HOME/.bashrc" >/dev/null 2>&1 || true\n'
  write_export PATH
  write_export HOME
  write_export DISPLAY
  write_export XAUTHORITY
  write_export CARLA_ROOT
  write_export PYTHONPATH
  write_export CONDA_EXE
  write_export CONDA_PREFIX
  write_export DREAMER_RL_DEVICE
  write_export DREAMER_RL_EPISODES
  write_export DREAMER_RL_MAX_EPISODE_STEPS
  write_export DREAMER_RL_ROLLOUT_SIZE
  write_export DREAMER_RL_EVAL_INTERVAL
  write_export CARLA_QUALITY
  printf 'exec >> %q 2>&1\n' "$LOG_FILE"
  printf 'cd %q\n' "$ROOT_DIR"
  printf 'echo "[dreamer-rl-resume] campaign=%q run_id=%q"\n' "$CAMPAIGN_ID" "$RUN_ID"
  printf 'echo "[dreamer-rl-resume] PPO warm-start=%q"\n' "$PPO_WARMSTART"
  printf 'echo "[dreamer-rl-resume] SDBS warm-start=%q"\n' "$SDBS_WARMSTART"
  printf 'run_one() {\n'
  printf '  local kind="$1"\n'
  printf '  local warmstart="$2"\n'
  printf '  local suffix="$3"\n'
  printf '  echo "[dreamer-rl-resume] starting ${kind}"\n'
  printf '  DREAMER_RL_KIND="$kind" \\\n'
  printf '  DREAMER_RL_RUN_ID=%q_"$suffix" \\\n' "$RUN_ID"
  printf '  DREAMER_RL_INIT_WORLD_MODEL="$warmstart" \\\n'
  printf '  DREAMER_RL_FOREGROUND=1 \\\n'
  printf '  DREAMER_RL_INSTALL_LATEST=1 \\\n'
  printf '  RESTART_EXISTING=1 \\\n'
  printf '  bash scripts/start_dreamer_rl_noguard_training.sh\n'
  printf '  local status=$?\n'
  printf '  echo "$status" > %q/"${kind}.exit"\n' "$RUN_DIR"
  printf '  echo "[dreamer-rl-resume] ${kind} exit=${status}"\n'
  printf '  return 0\n'
  printf '}\n'
  printf 'run_one ppo %q ppo_rl\n' "$PPO_WARMSTART"
  printf 'run_one sdbs %q sdbs_rl\n' "$SDBS_WARMSTART"
  printf 'date -Iseconds > %q/resume.done\n' "$RUN_DIR"
} > "$RUN_SCRIPT"
chmod +x "$RUN_SCRIPT"

echo "$RUN_DIR" > "$ROOT_DIR/logs/dreamer_rl_resume/latest_resume.txt"
echo "$UNIT" > "$RUN_DIR/systemd_unit_name.txt"

systemd-run --user \
  --unit "$UNIT" \
  --collect \
  --property=WorkingDirectory="$ROOT_DIR" \
  "$RUN_SCRIPT" | tee "$RUN_DIR/systemd_unit.txt"

echo "[dreamer-rl-resume] launched via systemd"
echo "[dreamer-rl-resume] unit=$UNIT"
echo "[dreamer-rl-resume] run_dir=$RUN_DIR"
echo "[dreamer-rl-resume] log=$LOG_FILE"
