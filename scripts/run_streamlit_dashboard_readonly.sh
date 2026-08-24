#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${STREAMLIT_VENV_DIR:-$ROOT_DIR/.venv-streamlit}"
PORT="${STREAMLIT_PORT:-8501}"

cd "$ROOT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[streamlit-share] Creating isolated environment: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

if ! "$VENV_DIR/bin/python" -c "import streamlit" >/dev/null 2>&1; then
  echo "[streamlit-share] Installing presentation-only dependencies."
  "$VENV_DIR/bin/python" -m pip install -r streamlit_share/requirements.txt
fi

"$VENV_DIR/bin/python" scripts/export_streamlit_dashboard.py

echo "[streamlit-share] Read-only local URL: http://127.0.0.1:$PORT"
echo "[streamlit-share] LAN bind: 0.0.0.0:$PORT"
exec "$VENV_DIR/bin/python" -m streamlit run streamlit_share/app.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
