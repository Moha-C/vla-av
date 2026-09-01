#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="${CARDREAMER_REPO_ID:-ucd-dare/CarDreamer}"
RELATIVE_PATH="CarDreamer_checkpoints/overtake.ckpt"
DEST_DIR="${CARDREAMER_CHECKPOINT_DIR:-$ROOT_DIR/external/cardreamer_checkpoints}"
CHECKPOINT="$DEST_DIR/$RELATIVE_PATH"
EXPECTED_SHA256="123525828488d596e80dad0fad0681767cec937adcc04bf0d5aa8ee972aa8058"
PYTHON_BIN="${CARDREAMER_DOWNLOAD_PYTHON:-$HOME/miniconda3/envs/simlingo/bin/python}"

mkdir -p "$DEST_DIR"

if command -v hf >/dev/null 2>&1; then
  hf download "$REPO_ID" "$RELATIVE_PATH" --local-dir "$DEST_DIR"
elif [[ -x "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" - "$REPO_ID" "$RELATIVE_PATH" "$DEST_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, local_dir = sys.argv[1:]
print(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir))
PY
else
  echo "[cardreamer-download] Install the 'hf' CLI or create the simlingo environment first." >&2
  exit 1
fi

printf '%s  %s\n' "$EXPECTED_SHA256" "$CHECKPOINT" | sha256sum --check -
echo "[cardreamer-download] checkpoint=$CHECKPOINT"
