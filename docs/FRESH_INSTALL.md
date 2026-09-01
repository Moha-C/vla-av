# Fresh Install On Ubuntu 22.04

This procedure reconstructs the active VLA-AV runtime on a new NVIDIA Ubuntu
22.04 workstation without copying credentials, datasets, logs, or caches from
the original machine.

## Hardware And Disk

- NVIDIA GPU and a working proprietary driver (`nvidia-smi`).
- At least 32 GB RAM recommended.
- At least 45 GB free before model caches and recordings.
- Ports 2000/2001 for CARLA, 8765 for the local dashboard, and 8501 for the
  optional read-only Streamlit presentation.

## 1. Clone Source And LFS Artifacts

```bash
cd ~/Desktop
git clone https://github.com/Moha-C/vla-av.git
cd vla-av
git lfs install
git lfs pull
```

For a private repository, GitHub authentication is required only for cloning.
Never place a personal access token in `.env` or any project file.

## 2. Install Ubuntu Dependencies

```bash
cd ~/Desktop/vla-av
bash scripts/install_system_deps_ubuntu22.sh
```

This installs SUMO, `sumo-gui`, Git LFS, FFmpeg, graphical runtime libraries,
Node/npm for TwinSentinel, and Python venv support.

Verify SUMO:

```bash
sumo --version
sumo-gui --version
export SUMO_HOME=/usr/share/sumo
```

## 3. Install CARLA 0.9.15 And Bench2Drive Maps

```bash
cd ~/Desktop/vla-av
INSTALL_ADDITIONAL_MAPS=1 bash scripts/install_carla_0915.sh
export CARLA_ROOT="$HOME/carla_simulator"
```

The script downloads the official CARLA 0.9.15 Linux archive and official
AdditionalMaps archive, then runs `ImportAssets.sh`. To use an existing CARLA
installation, skip the installer and set `CARLA_ROOT` to the folder containing
`CarlaUE4.sh`.

Persist the runtime paths if desired:

```bash
printf '\nexport CARLA_ROOT="$HOME/carla_simulator"\nexport SUMO_HOME=/usr/share/sumo\n' >> ~/.bashrc
```

The repository already contains Bench2Drive, ScenarioRunner, and all 220 route
XML files under `external/simlingo/`; no additional Bench2Drive clone is needed.

## 4. Install Miniconda

Skip this section when `conda --version` already works.

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
```

Open a new shell after `conda init`, or keep sourcing `conda.sh` in the current
shell.

## 5. Create The SimLingo Environment

```bash
cd ~/Desktop/vla-av
conda env create -f environment.simlingo.yml
conda activate simlingo
```

The supported runtime is Python 3.8.18. `environment.simlingo-lock.yml` and
`requirements.freeze.txt` preserve the local environment provenance when an
exact dependency comparison is needed.

## 6. Download The Public SimLingo Weights

```bash
cd ~/Desktop/vla-av
conda activate simlingo
bash scripts/download_simlingo_model.sh
```

Expected artifact:

```text
models/simlingo_hf/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
```

The script uses the public Hugging Face repository `RenzKa/simlingo`. If the
Hub asks for authentication, run `hf auth login` with the new user's own token.
The token must never be committed or sent by another developer.

The hidden legacy CarDreamer mirror is optional. To reproduce that backend too,
download its public upstream checkpoint with:

```bash
bash scripts/download_cardreamer_overtake_checkpoint.sh
```

The downloader rejects the file unless its SHA-256 matches the runtime contract.

## 7. Validate Everything

```bash
cd ~/Desktop/vla-av
conda activate simlingo
export CARLA_ROOT="$HOME/carla_simulator"
export SUMO_HOME=/usr/share/sumo
bash scripts/check_fresh_install.sh
bash scripts/audit_repository_for_publish.sh
```

The checker verifies CARLA, SUMO, route XML files, Python imports, dashboard
syntax, TwinSentinel source, and that promoted Git LFS checkpoints are real
binary files rather than unresolved pointer files.

## 8. Launch The Full Local Application

```bash
cd ~/Desktop/vla-av
conda activate simlingo
bash scripts/stop_simlingo_dashboard.sh
bash scripts/run_simlingo_dashboard.sh
```

Open `http://127.0.0.1:8765`.

This interactive dashboard can launch CARLA/SimLingo, SUMO mirror mode,
recording/replay, promoted Dreamer modes, KPI refresh, and the TwinSentinel
attack console.

## 9. Launch Only The Shareable Frontend

```bash
cd ~/Desktop/vla-av
python3 scripts/export_streamlit_dashboard.py
bash scripts/run_streamlit_dashboard_readonly.sh
```

Open `http://127.0.0.1:8501`. This app serves a committed, sanitized static
snapshot. It contains the current KPI values but has no process-control backend
and cannot launch or modify anything, even if a visitor clicks a control.

For Streamlit Community Cloud, select `streamlit_share/app.py` as the app file.

## Common Failures

- **Checkpoint is a short text file:** run `git lfs pull`.
- **Town not installed:** rerun
  `CARLA_ROOT="$HOME/carla_simulator" bash scripts/install_carla_additional_maps_0915.sh`.
- **`sumolib` or `traci` missing:** export `SUMO_HOME=/usr/share/sumo`.
- **Dashboard opens but CARLA times out:** stop old processes, verify port 2000,
  and run `nvidia-smi`.
- **SimLingo model missing:** rerun `bash scripts/download_simlingo_model.sh`.
- **Hydra cannot import `simlingo_training.models`:** launch from the repository
  scripts; they set the required `PYTHONPATH` for the vendored SimLingo tree.
