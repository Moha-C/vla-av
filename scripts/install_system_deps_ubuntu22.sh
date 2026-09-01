#!/usr/bin/env bash
set -euo pipefail

echo "[vla-av-setup] Installing Ubuntu 22.04 system dependencies."
echo "[vla-av-setup] This script uses sudo and does not install CARLA itself."

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  git-lfs \
  libgl1 \
  libglib2.0-0 \
  libjpeg-dev \
  libomp5 \
  libpng-dev \
  libsdl2-2.0-0 \
  libsm6 \
  libvulkan1 \
  libxext6 \
  libxrender1 \
  mesa-utils \
  net-tools \
  nodejs \
  npm \
  openssh-client \
  python3-pip \
  python3-venv \
  rsync \
  sumo \
  sumo-tools \
  unzip \
  wget \
  x11-apps

git lfs install

echo "[vla-av-setup] Done."
echo "[vla-av-setup] If SUMO is installed under /usr/share/sumo, export:"
echo "  export SUMO_HOME=/usr/share/sumo"
