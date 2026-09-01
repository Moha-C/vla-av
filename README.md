# VLA-AV SimLingo World

Closed-loop research platform for SimLingo, CARLA 0.9.15, Bench2Drive, SUMO,
TwinSentinel attacks, and residual Dreamer/RSSM policies.

The repository contains the complete active frontend and backend source. The
only components downloaded after cloning are third-party binaries or public
weights that are too large or cannot legally be redistributed here: CARLA, the
SimLingo Hugging Face checkpoint, and optional research datasets.

## Current Pipeline

- Native SimLingo evaluation on the 220 Bench2Drive route XML files.
- Pygame ego/chase view, recording, replay, and live model overlays.
- SimLingo + guarded Dreamer PPO complement.
- SimLingo + report-aligned Dreamer/RSSM D with learned residual authority.
- Bidirectional CARLA/SUMO mirror and TwinSentinel attack console.
- SAFE-DREAM and Bench2Drive KPI comparison.
- Static read-only Streamlit presentation containing the current KPI snapshot.
- Isolated DreamerV3/RSSM audit and training pipeline with explicit kill gates.

The local dashboard currently exposes only the three validated presentation
choices:

| UI choice | Runtime artifact | Role |
| --- | --- | --- |
| Off - native SimLingo | SimLingo checkpoint | Closed-loop baseline |
| Dreamer PPO | `external/simlingo/checkpoints/dreamer_guard/best_world_model.pt` | Guarded residual complement |
| Dreamer RSSM D - learned alpha | `checkpoints/report_aligned_dreamer/production/report_dreamer.pt` | Report-aligned RSSM complement |

Older SDBS, no-guard RL, CarDreamer mirror, and ablation backends remain in the
source tree for reproducibility, but they are deliberately hidden from the
main selector until they pass the frozen evaluation protocol.

The optional official CarDreamer overtake checkpoint is reproducibly downloaded
and checksum-verified with `bash scripts/download_cardreamer_overtake_checkpoint.sh`.

## Repository Layout

```text
configs/                         Dreamer/RSSM and training protocols
data/README.md                   Dataset inventory; raw data stays local
docs/                            Architecture, attribution, experiment protocols
external/simlingo/               SimLingo, Bench2Drive, ScenarioRunner, agent code
external/cardreamer_upstream/    Flattened upstream CarDreamer source and licenses
experiments/dreamer_ppo_carla/   Dreamer PPO v1 training/evaluation source
experiments/TwinSentinel_Project TwinSentinel attack console and adapters
scripts/                         Launch, mirror, collection, training, audit, export
src/                             Report Dreamer, DeepAccident, residual DreamerV3
streamlit_share/                 Static presentation app and KPI snapshot
tests/                           Runtime, reward, dashboard, and world-model tests
```

## Fresh Ubuntu 22.04 Installation

The detailed checklist is in [docs/FRESH_INSTALL.md](docs/FRESH_INSTALL.md).
The short path is:

```bash
cd ~/Desktop
git clone https://github.com/Moha-C/vla-av.git
cd vla-av
git lfs install
git lfs pull

bash scripts/install_system_deps_ubuntu22.sh
INSTALL_ADDITIONAL_MAPS=1 bash scripts/install_carla_0915.sh

conda env create -f environment.simlingo.yml
conda activate simlingo
bash scripts/download_simlingo_model.sh

export CARLA_ROOT="$HOME/carla_simulator"
export SUMO_HOME=/usr/share/sumo
bash scripts/check_fresh_install.sh
```

Bench2Drive, ScenarioRunner, route XML files, CARLA/SUMO synchronization code,
and the web frontend are already vendored in this repository. They must not be
cloned separately.

## Interactive Dashboard

```bash
cd ~/Desktop/vla-av
conda activate simlingo
bash scripts/stop_simlingo_dashboard.sh
bash scripts/run_simlingo_dashboard.sh
```

Open `http://127.0.0.1:8765`. This local process can launch CARLA, SUMO,
SimLingo, Dreamer modes, recordings, and TwinSentinel attacks.

## Read-Only Streamlit Share

The Streamlit app is intentionally a static presentation. It embeds no backend,
does not connect to the local dashboard, and cannot launch, stop, replay, attack,
or update a checkpoint.

Refresh its sanitized KPI snapshot before publishing:

```bash
cd ~/Desktop/vla-av
python3 scripts/export_streamlit_dashboard.py
bash scripts/run_streamlit_dashboard_readonly.sh
```

Local Streamlit URL: `http://127.0.0.1:8501`.

For Streamlit Community Cloud, use `streamlit_share/app.py` as the entrypoint.
The committed files are:

- `streamlit_share/dashboard_snapshot.html`: self-contained frontend.
- `streamlit_share/kpi_snapshot.json`: inspectable copy of the current KPI data.

## CARLA, SUMO, And Bench2Drive

This project is pinned to CARLA 0.9.15. The installer downloads the official
Linux package and, with `INSTALL_ADDITIONAL_MAPS=1`, the official additional map
archive needed by Bench2Drive towns:

```bash
INSTALL_ADDITIONAL_MAPS=1 bash scripts/install_carla_0915.sh
```

SUMO is installed from Ubuntu packages and expected at `/usr/share/sumo`.
The dashboard's SUMO mirror starts from the same CARLA world and mirrors actors,
traffic lights, and ego state. TwinSentinel attacks target that active bridge;
they do not start an unrelated SUMO simulation.

## Model Weights

The large SimLingo model is downloaded from `RenzKa/simlingo` into:

```text
models/simlingo_hf/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
```

Run `bash scripts/download_simlingo_model.sh`. If Hugging Face authentication is
ever required, every user must authenticate with their own account. No token is
stored in Git.

Small promoted Dreamer checkpoints are versioned with Git LFS. A fresh clone
must run `git lfs pull`; otherwise the files remain text pointers and the fresh
install checker will reject them.

## Dreamer Research Branches

The report-aligned branch keeps SimLingo as the reference controller and learns
a residual action plus continuous authority. Its protocol is documented in:

- [Report-aligned architecture](docs/REPORT_ALIGNED_DREAMER_ARCHITECTURE.md)
- [Frozen validation protocol](docs/REPORT_ALIGNED_DREAMER_PROTOCOL.md)
- [Residual DreamerV3 protocol](docs/RESIDUAL_DREAMERV3_PROTOCOL.md)
- [Residual DreamerV3 v2 audit result](docs/RESIDUAL_DREAMERV3_AUDIT_20260901.md)
- [Third-party attribution](docs/THIRD_PARTY_DREAMER_ATTRIBUTION.md)

Run the report branch software tests with:

```bash
bash scripts/test_report_dreamer.sh
```

Run the isolated residual DreamerV3 data audit, baseline comparison, training,
and gate with:

```bash
bash scripts/run_residual_dreamerv3_pipeline.sh
```

A candidate is never substituted for the promoted runtime model automatically.
Prediction gates and frozen closed-loop evaluation must pass first.
The full residual DreamerV3 v2 run from 2026-09-01 failed its frozen
world-model gate, so no actor was trained from that candidate.

## Privacy And Artifact Policy

Git intentionally excludes:

- `.env`, Streamlit secrets, API tokens, SSH keys, and credentials;
- CARLA binaries and Hugging Face caches;
- logs, videos, replay buffers, datasets, exports, and workstation backups;
- failed/candidate checkpoints and optimizer state;
- historical Alpamayo, Maram, and VM side experiments not used at runtime.

This does not prevent reproduction: installation scripts retrieve public
dependencies, collection scripts regenerate training data, and promoted small
runtime checkpoints are included through Git LFS. It prevents a clone from
receiving private workstation data.

Before every publication, run:

```bash
bash scripts/audit_repository_for_publish.sh
```

The audit scans every indexed file for common token/private-key signatures,
credential filenames, and oversized Git blobs.
