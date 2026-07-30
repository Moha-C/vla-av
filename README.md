# VLA-AV SimLingo World

Research pipeline for evaluating and extending SimLingo in CARLA/Bench2Drive:

- Native SimLingo closed-loop driving with Pygame POV replay.
- Dreamer PPO and Dreamer SDBS runtime adapters.
- RL no-guard Dreamer training scaffolding with protected checkpoints.
- CARLA/SUMO mirror and TwinSentinel attack console integration.
- SAFE-DREAM KPI dashboard for native SimLingo, Dreamer PPO, Dreamer SDBS, and RL no-guard variants.

The current working entrypoint is the local dashboard:

```bash
cd ~/Desktop/vla-av
bash scripts/stop_simlingo_dashboard.sh
bash scripts/run_simlingo_dashboard.sh
```

Open:

```text
http://127.0.0.1:8765
```

## Repository Layout

```text
scripts/                         Launchers, dashboard, CARLA/SUMO bridge, RL helpers
external/simlingo/                Vendored SimLingo + Bench2Drive integration
external/simlingo/team_code/      SimLingo agent and Dreamer runtime adapter
experiments/dreamer_ppo_carla/    Dreamer PPO v1 training code
experiments/dreamer_ppo_carla_sdbs_fresh/
                                  Dreamer SDBS training code
experiments/TwinSentinel_Project/ TwinSentinel attack console integration
docs/                             Project handoffs and validation notes
```

Large/private local artifacts are intentionally ignored by Git: logs, videos,
datasets, VM backups, downloaded models, pretrained VLM caches, credentials, and
CARLA itself. A fresh clone does not give anyone access to Mohammed's PC, tokens,
private data, local videos, or backups.

## Runtime Modes

The dashboard Dreamer selector maps to these checkpoints:

| Dashboard mode | Checkpoint | Notes |
| --- | --- | --- |
| Off - native SimLingo | none | SimLingo baseline |
| Dreamer PPO | `external/simlingo/checkpoints/dreamer_guard/best_world_model.pt` | guarded runtime that currently works |
| Dreamer SDBS | `external/simlingo/checkpoints/dreamer_sdbs_fresh/best_world_model.pt` | guarded SDBS runtime |
| Dreamer PPO RL no guard | `external/simlingo/checkpoints/dreamer_ppo_rl_noguard/latest_rl_model.pt` | separate seed checkpoint now; replace after validated RL training |
| Dreamer SDBS RL no guard | `external/simlingo/checkpoints/dreamer_sdbs_rl_noguard/latest_rl_model.pt` | separate seed checkpoint now; replace after validated RL training |

Note: internally, the runtime environment variable is still named
`SIMLINGO_DREAMER_GUARD`. Setting it to `0` disables the Dreamer adapter entirely.
The `RL no guard` dashboard modes therefore keep the adapter enabled, but disable
the added recovery/collision-shield layers and route the mode to its own
`latest_rl_model.pt` checkpoint.

The guarded checkpoints are backed up in:

```text
external/simlingo/checkpoints/guarded_before_rl_20260730_142415/
```

## System Requirements

Tested local setup:

- Ubuntu 22.04
- NVIDIA GPU with working driver
- CARLA 0.9.15 installed at `~/carla_simulator`
- SUMO 1.18.0 installed from Ubuntu packages
- Miniconda
- Python 3.8.18 in the `simlingo` env

For a new machine, follow:

```text
docs/FRESH_INSTALL.md
```

Install system dependencies:

```bash
cd ~/Desktop/vla-av
bash scripts/install_system_deps_ubuntu22.sh
```

Install CARLA 0.9.15 manually, then set:

```bash
export CARLA_ROOT=$HOME/carla_simulator
export SUMO_HOME=/usr/share/sumo
```

CARLA is not committed to GitHub because it is too large.

## Python Environment

Recommended:

```bash
conda env create -f environment.simlingo.yml
conda activate simlingo
```

If you already have the env:

```bash
conda activate simlingo
pip install -r requirements.txt
```

For exact provenance of the current local environment, the repository also
contains:

```text
environment.simlingo-lock.yml
requirements.freeze.txt
```

Use those as reference/lock files if another machine needs to match this setup
as closely as possible.

## SimLingo Model Weights

The SimLingo checkpoint is expected at:

```text
models/simlingo_hf/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
```

Download it with:

```bash
cd ~/Desktop/vla-av
bash scripts/download_simlingo_model.sh
```

The large Hugging Face model directory is ignored by Git.

If Hugging Face authentication is ever needed, each user must use their own
account/token. Do not commit tokens into this repository.

## Run The Dashboard

```bash
cd ~/Desktop/vla-av
bash scripts/stop_simlingo_dashboard.sh
bash scripts/run_simlingo_dashboard.sh
```

The dashboard can launch:

- CARLA POV SimLingo runs.
- CARLA + SUMO mirror runs.
- Dreamer PPO / SDBS guarded runs.
- TwinSentinel attack console.
- SAFE-DREAM KPI comparison.

## RL No-Guard Training

Prepare protected seed checkpoints:

```bash
cd ~/Desktop/vla-av
bash scripts/prepare_dreamer_rl_noguard_checkpoints.sh
```

Start PPO RL no-guard training:

```bash
DREAMER_RL_KIND=ppo DREAMER_RL_DEVICE=cuda DREAMER_RL_EPISODES=100 \
  bash scripts/start_dreamer_rl_noguard_training.sh
```

Watch PPO:

```bash
DREAMER_RL_KIND=ppo bash scripts/watch_dreamer_rl_noguard_training.sh
```

Start SDBS RL no-guard training:

```bash
DREAMER_RL_KIND=sdbs DREAMER_RL_DEVICE=cuda DREAMER_RL_EPISODES=100 \
  bash scripts/start_dreamer_rl_noguard_training.sh
```

Watch SDBS:

```bash
DREAMER_RL_KIND=sdbs bash scripts/watch_dreamer_rl_noguard_training.sh
```

The current `latest_rl_model.pt` files are separate seed copies so the dashboard
options are wired without touching the guarded modes. After a real run is
visually and metrically validated, install its best checkpoint for the dashboard:

```bash
DREAMER_RL_KIND=ppo bash scripts/install_dreamer_rl_noguard_checkpoint.sh
DREAMER_RL_KIND=sdbs bash scripts/install_dreamer_rl_noguard_checkpoint.sh
```

By default, training does not overwrite dashboard runtime checkpoints.

## GitHub Notes

This project contains very large local artifacts. The `.gitignore` excludes
logs, recordings, datasets, VM backups, downloaded models, and CARLA binaries.

Use Git LFS for binary checkpoints if you decide to track them:

```bash
git lfs install
git lfs track "*.pt" "*.ckpt" "*.safetensors" "*.mp4" "*.zip"
```

If a nested upstream repo should be committed as normal files, remove or move its
nested `.git` folder first. Otherwise Git will treat it as an embedded repository
or submodule.
