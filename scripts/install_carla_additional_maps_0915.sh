#!/usr/bin/env bash
set -euo pipefail

CARLA_ROOT="${CARLA_ROOT:-$HOME/carla_simulator}"
URL="${CARLA_ADDITIONAL_MAPS_URL:-https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz}"
ARCHIVE="$CARLA_ROOT/Import/AdditionalMaps_0.9.15.tar.gz"

if [[ ! -x "$CARLA_ROOT/ImportAssets.sh" ]]; then
  echo "[carla-maps] CARLA_ROOT does not look like a CARLA 0.9.15 install: $CARLA_ROOT" >&2
  exit 1
fi

mkdir -p "$CARLA_ROOT/Import"
echo "[carla-maps] CARLA_ROOT=$CARLA_ROOT"
echo "[carla-maps] Downloading additional maps archive..."
wget -c "$URL" -O "$ARCHIVE"

echo "[carla-maps] Importing maps. This can take several minutes."
cd "$CARLA_ROOT"
bash ImportAssets.sh

echo "[carla-maps] Installed towns:"
find "$CARLA_ROOT/CarlaUE4/Content/Carla/Maps" -maxdepth 1 -type f -name 'Town*.umap' \
  -printf '%f\n' | sed 's/\.umap$//' | sort
