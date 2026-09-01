#!/usr/bin/env bash
set -euo pipefail

CARLA_VERSION="0.9.15"
CARLA_ROOT="${CARLA_ROOT:-$HOME/carla_simulator}"
DOWNLOAD_DIR="${CARLA_DOWNLOAD_DIR:-$HOME/Downloads/vla-av}"
CARLA_URL="${CARLA_URL:-https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz}"
ARCHIVE="$DOWNLOAD_DIR/CARLA_${CARLA_VERSION}.tar.gz"

if [[ -x "$CARLA_ROOT/CarlaUE4.sh" && "${FORCE_CARLA_INSTALL:-0}" != "1" ]]; then
  echo "[carla-install] CARLA already installed at $CARLA_ROOT"
  echo "[carla-install] Set FORCE_CARLA_INSTALL=1 to extract it again."
else
  mkdir -p "$DOWNLOAD_DIR" "$CARLA_ROOT"
  echo "[carla-install] Downloading CARLA $CARLA_VERSION"
  echo "[carla-install] source=$CARLA_URL"
  echo "[carla-install] archive=$ARCHIVE"
  wget -c "$CARLA_URL" -O "$ARCHIVE"

  if [[ -n "${CARLA_ARCHIVE_SHA256:-}" ]]; then
    printf '%s  %s\n' "$CARLA_ARCHIVE_SHA256" "$ARCHIVE" | sha256sum --check -
  fi

  echo "[carla-install] Extracting to $CARLA_ROOT"
  tar -xzf "$ARCHIVE" -C "$CARLA_ROOT"
  chmod +x "$CARLA_ROOT/CarlaUE4.sh"
  [[ -f "$CARLA_ROOT/ImportAssets.sh" ]] && chmod +x "$CARLA_ROOT/ImportAssets.sh"
fi

if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
  echo "[carla-install] CARLA executable missing after extraction: $CARLA_ROOT/CarlaUE4.sh" >&2
  exit 1
fi

echo "[carla-install] CARLA $CARLA_VERSION is ready."
echo "[carla-install] export CARLA_ROOT=$CARLA_ROOT"

if [[ "${INSTALL_ADDITIONAL_MAPS:-0}" == "1" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CARLA_ROOT="$CARLA_ROOT" bash "$SCRIPT_DIR/install_carla_additional_maps_0915.sh"
else
  echo "[carla-install] Bench2Drive towns require the additional maps:"
  echo "  CARLA_ROOT=$CARLA_ROOT bash scripts/install_carla_additional_maps_0915.sh"
fi
