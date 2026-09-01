#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DEEPACCIDENT_PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${DEEPACCIDENT_PYTHON:-python3}"
fi

FILE_ID="1NXC_-zTWFdHj-30g3zSUfNFMp4Mk_7Hh"
RAW_DIR="${DEEPACCIDENT_RAW_DIR:-${ROOT}/data/deepaccident/raw}"
EXTRACT_DIR="${DEEPACCIDENT_EXTRACT_DIR:-${ROOT}/data/deepaccident/extracted/mini}"
PROCESSED_DIR="${DEEPACCIDENT_PROCESSED_DIR:-${ROOT}/data/deepaccident/processed/mini}"
ARCHIVE="${RAW_DIR}/DeepAccident_mini.zip"
TOOLS_DIR="${ROOT}/.cache/deepaccident_tools"

mkdir -p "${RAW_DIR}" "${EXTRACT_DIR}" "${PROCESSED_DIR}" "${TOOLS_DIR}"

if ! PYTHONPATH="${TOOLS_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c 'import gdown' >/dev/null 2>&1; then
  echo "[deepaccident] installing isolated downloader into ${TOOLS_DIR}"
  "${PYTHON_BIN}" -m pip install --target "${TOOLS_DIR}" 'gdown==5.2.0'
fi

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "[deepaccident] downloading official 9 GB mini dataset"
else
  echo "[deepaccident] resuming/verifying ${ARCHIVE}"
fi
PYTHONPATH="${TOOLS_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m gdown \
  --id "${FILE_ID}" \
  --continue \
  --output "${ARCHIVE}"

unzip -tq "${ARCHIVE}" >/dev/null
if [[ ! -f "${EXTRACT_DIR}/.extracted.ok" ]]; then
  echo "[deepaccident] extracting into ${EXTRACT_DIR}"
  unzip -q "${ARCHIVE}" -d "${EXTRACT_DIR}"
  touch "${EXTRACT_DIR}/.extracted.ok"
fi

echo "[deepaccident] indexing ordered ego front-camera sequences"
"${PYTHON_BIN}" "${ROOT}/scripts/prepare_deepaccident.py" \
  --dataset-root "${EXTRACT_DIR}" \
  --output-dir "${PROCESSED_DIR}" \
  --train-ratio "${DEEPACCIDENT_TRAIN_RATIO:-0.625}" \
  --validation-ratio "${DEEPACCIDENT_VALIDATION_RATIO:-0.25}" \
  --test-ratio "${DEEPACCIDENT_TEST_RATIO:-0.125}"

echo "[deepaccident] ready: ${PROCESSED_DIR}/audit.json"
