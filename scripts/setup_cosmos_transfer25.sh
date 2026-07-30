#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p external
cd external

if [ ! -d cosmos-transfer2.5 ]; then
  git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
fi

cd cosmos-transfer2.5
uv python install 3.10
printf "3.10\n" > .python-version
uv venv --python 3.10 --clear
uv sync --python 3.10 --extra=cu128
uv tool install -U "huggingface_hub[cli]"
hf auth login

echo "[setup] Cosmos-Transfer2.5 is ready in external/cosmos-transfer2.5"
