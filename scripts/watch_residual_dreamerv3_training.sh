#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${RESIDUAL_DREAMERV3_OUTPUT:-${ROOT}/checkpoints/residual_dreamerv3/candidate}"
PID_FILE="${OUTPUT}/training.pid"
LOG_FILE="${OUTPUT}/training.log"

echo "=== Residual DreamerV3 ==="
echo "output: ${OUTPUT}"
if [[ -s "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "process: RUNNING pid=${pid}"
  else
    echo "process: STOPPED last_pid=${pid}"
  fi
else
  echo "process: not started through durable launcher"
fi

for gate in world_model_gate_validation.json world_model_gate_test.json world_model_gate_combined.json; do
  if [[ -s "${OUTPUT}/${gate}" ]]; then
    echo
    echo "--- ${gate} ---"
    jq '{passed, improvement, checks}' "${OUTPUT}/${gate}" 2>/dev/null || true
  fi
done

if [[ -s "${OUTPUT}/actor_history.json" ]]; then
  echo
  echo "--- actor candidate ---"
  jq '{best_validation_objective, latest: .history[-1]}' "${OUTPUT}/actor_history.json"
elif [[ -s "${OUTPUT}/world_model_history.json" ]]; then
  echo
  echo "--- world model ---"
  jq '{best_validation_total, latest: .history[-1]}' "${OUTPUT}/world_model_history.json"
fi

if [[ -s "${LOG_FILE}" ]]; then
  echo
  echo "--- latest output ---"
  tail -40 "${LOG_FILE}"
fi
