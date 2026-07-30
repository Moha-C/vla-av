# Fresh Install Checklist

This guide is for a new Ubuntu machine that wants to run the VLA-AV SimLingo
pipeline from GitHub.

## What GitHub Contains

The repository contains the project source needed to run the current pipeline:

- SimLingo/Bench2Drive integration.
- Dashboard launchers and web UI.
- Dreamer PPO and Dreamer SDBS runtime adapters.
- CARLA/SUMO mirror scripts.
- TwinSentinel attack-console integration.
- SAFE-DREAM KPI dashboard code.
- Small Dreamer runtime checkpoints.
- Python dependency files and setup scripts.

The repository intentionally does not contain:

- CARLA binaries.
- Large SimLingo Hugging Face model weights.
- Large pretrained VLM caches.
- Generated logs, videos, datasets, exports, VM backups, and failed experiments.

Those files are installed or regenerated locally.

## 1. Clone

```bash
cd ~/Desktop
git clone https://github.com/Moha-C/vla-av.git
cd vla-av
git lfs pull
```

## 2. Install System Dependencies

```bash
cd ~/Desktop/vla-av
bash scripts/install_system_deps_ubuntu22.sh
```

Then make sure SUMO is visible:

```bash
export SUMO_HOME=/usr/share/sumo
```

Add it to `~/.bashrc` if needed.

## 3. Install CARLA 0.9.15

Install CARLA 0.9.15 outside the Git repository, for example:

```text
~/carla_simulator
```

The expected executable is:

```text
~/carla_simulator/CarlaUE4.sh
```

Then export:

```bash
export CARLA_ROOT=$HOME/carla_simulator
```

Add it to `~/.bashrc` if needed.

CARLA must not be committed to Git because it is too large.

## 4. Create The Conda Environment

```bash
cd ~/Desktop/vla-av
conda env create -f environment.simlingo.yml
conda activate simlingo
```

If dependency solving fails on a different machine, use the lock/provenance files
as references:

```text
environment.simlingo-lock.yml
requirements.freeze.txt
```

## 5. Download SimLingo Weights

```bash
cd ~/Desktop/vla-av
conda activate simlingo
bash scripts/download_simlingo_model.sh
```

Expected file:

```text
models/simlingo_hf/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
```

## 6. Validate The Installation

```bash
cd ~/Desktop/vla-av
bash scripts/check_fresh_install.sh
```

The script checks Python, CARLA path, SUMO, key model files, Dreamer checkpoints,
and important imports without launching a full simulation.

## 7. Run The Dashboard

```bash
cd ~/Desktop/vla-av
conda activate simlingo
bash scripts/stop_simlingo_dashboard.sh
bash scripts/run_simlingo_dashboard.sh
```

Open:

```text
http://127.0.0.1:8765
```

## Notes

If the dashboard opens but a simulation does not start, check first:

- `CARLA_ROOT` points to the right CARLA 0.9.15 folder.
- `SUMO_HOME=/usr/share/sumo`.
- The SimLingo model was downloaded.
- The NVIDIA driver is installed and `nvidia-smi` works.
- Port `2000` is free or old CARLA processes were stopped.
