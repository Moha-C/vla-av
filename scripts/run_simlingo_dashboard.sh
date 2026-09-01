#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${SIMLINGO_ENV_NAME:-simlingo}"
BASE_PORT="${SIMLINGO_DASHBOARD_PORT:-8765}"
BIND_HOST="${SIMLINGO_DASHBOARD_HOST:-127.0.0.1}"

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

PORT="$(python - "$BASE_PORT" "$BIND_HOST" <<'PY'
import socket
import sys

base = int(sys.argv[1])
host = sys.argv[2]
for port in range(base, base + 20):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit(f"No free localhost port found from {base} to {base + 19}")
PY
)"

if [[ "$PORT" != "$BASE_PORT" ]]; then
  echo "[simlingo-dashboard] Port $BASE_PORT busy; using $PORT instead."
fi

export SIMLINGO_DASHBOARD_PORT="$PORT"
export SIMLINGO_DASHBOARD_HOST="$BIND_HOST"
export SIMLINGO_DASHBOARD_SHOW_EXPERIMENTAL="${SIMLINGO_DASHBOARD_SHOW_EXPERIMENTAL:-1}"
export CARLA_ROOT="${CARLA_ROOT:-$HOME/carla_simulator}"
cd "$ROOT_DIR"
exec python scripts/simlingo_dashboard.py
