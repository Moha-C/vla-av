#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${RESIDUAL_DREAMERV3_OUTPUT:-${ROOT}/checkpoints/residual_dreamerv3/candidate}"
PID_FILE="${OUTPUT}/training.pid"
LOG_FILE="${OUTPUT}/training.log"

mkdir -p "${OUTPUT}"
if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "[residual-dreamerv3] training already running: pid=${old_pid}"
    echo "[residual-dreamerv3] watch: bash scripts/watch_residual_dreamerv3_training.sh"
    exit 0
  fi
fi

cd "${ROOT}"
nohup setsid --wait env \
  RESIDUAL_DREAMERV3_OUTPUT="${OUTPUT}" \
  RESIDUAL_DREAMERV3_PHASE="${RESIDUAL_DREAMERV3_PHASE:-all}" \
  RESIDUAL_DREAMERV3_DEVICE="${RESIDUAL_DREAMERV3_DEVICE:-cuda}" \
  RESIDUAL_DREAMERV3_WORLD_EPOCHS="${RESIDUAL_DREAMERV3_WORLD_EPOCHS:-40}" \
  RESIDUAL_DREAMERV3_ACTOR_EPOCHS="${RESIDUAL_DREAMERV3_ACTOR_EPOCHS:-40}" \
  RESIDUAL_DREAMERV3_MAX_WINDOWS="${RESIDUAL_DREAMERV3_MAX_WINDOWS:-0}" \
  bash "${ROOT}/scripts/run_residual_dreamerv3_pipeline.sh" "$@" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${PID_FILE}"
sleep 1
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "[residual-dreamerv3] training failed to start" >&2
  tail -80 "${LOG_FILE}" >&2 || true
  exit 1
fi
echo "[residual-dreamerv3] started pid=${pid}"
echo "[residual-dreamerv3] output=${OUTPUT}"
echo "[residual-dreamerv3] watch: bash scripts/watch_residual_dreamerv3_training.sh"
