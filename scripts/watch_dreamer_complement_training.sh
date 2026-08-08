#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST_FILE="$ROOT_DIR/logs/dreamer_complement_training/latest_complement_training.txt"

if [[ -n "${DREAMER_COMPLEMENT_RUN_DIR:-}" ]]; then
  RUN_DIR="$DREAMER_COMPLEMENT_RUN_DIR"
elif [[ -s "$LATEST_FILE" ]]; then
  RUN_DIR="$(cat "$LATEST_FILE")"
else
  echo "[dreamer-complement-watch] no complement training run found yet"
  exit 1
fi

echo "=== Dreamer Complement Training ==="
echo "run_dir: $RUN_DIR"
echo

for kind in ppo sdbs; do
  OUT_DIR="$RUN_DIR/$kind"
  SUMMARY="$OUT_DIR/summary.json"
  SOURCE="$ROOT_DIR/external/simlingo/checkpoints/dreamer_${kind}_complement/latest_world_model_source.txt"
  CHECKPOINT="$ROOT_DIR/external/simlingo/checkpoints/dreamer_${kind}_complement/latest_world_model.pt"
  if [[ "$kind" == "sdbs" ]]; then
    SOURCE="$ROOT_DIR/external/simlingo/checkpoints/dreamer_sdbs_complement/latest_world_model_source.txt"
    CHECKPOINT="$ROOT_DIR/external/simlingo/checkpoints/dreamer_sdbs_complement/latest_world_model.pt"
  fi

  echo "--- $kind ---"
  if [[ -s "$CHECKPOINT" ]]; then
    echo "checkpoint: $CHECKPOINT"
  else
    echo "checkpoint: missing"
  fi
  if [[ -s "$SOURCE" ]]; then
    sed -n '1,12p' "$SOURCE"
  fi
  if [[ -s "$SUMMARY" ]]; then
    python3 - "$SUMMARY" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
best = data.get("best") or {}
print(f"transitions={data.get('transitions')}")
print(
    "best="
    f"epoch {best.get('epoch')} "
    f"loss {best.get('loss')} "
    f"risk_mae {best.get('risk_mae')} "
    f"state_mae_norm {best.get('state_mae_norm')}"
)
PY
  else
    echo "summary: missing"
  fi
  echo
done
