# Current Pipeline

This repository is now focused on the active autonomous-driving pipeline:

1. CARLA provides simulation, ego state, RGB/semantic/depth sensors, traffic, and actors.
2. Alpamayo 1.5 is the active VLA planner.
3. `src/models/alpamayo_adapter.py` runs Alpamayo as a sidecar worker and converts the predicted trajectory to CARLA controls.
4. Cosmos-Transfer2.5 is kept for photorealistic CARLA render transfer.

## Kept In Main Tree

- `start.sh`
- `scripts/demo.py`
- `scripts/alpamayo_worker.py`
- `scripts/collect_data.py`
- `scripts/cosmos_transfer_real.py`
- `scripts/cosmos_transfer_compare.py`
- `scripts/cosmos_transfer_demo.py`
- `scripts/prepare_alpamayo_transfer_dataset.py`
- `scripts/run_local_transfer_dataset.sh`
- `scripts/train_local_action_adapter.py`
- `scripts/export_reference_frame.py`
- `scripts/setup_cosmos_transfer25.sh`
- `src/carla_env/`
- `src/data/cosmos_transfer.py`
- `src/models/alpamayo_adapter.py`
- `src/models/local_action_adapter.py`
- `external/alpamayo1.5/`
- `external/cosmos-transfer2.5/`
- `data/synthetic/transferred_real/`

## Active Data Plan

Step 2 is documented in:

```text
docs/STEP2_TRANSFER_DATASET.md
```

The new data path is CARLA expert autopilot labels -> Cosmos-Transfer2.5
photorealistic frames -> Alpamayo manifest with actions and local future
trajectory labels.

## Backup

Everything from the previous Qwen/GR00T/Cosmos-Predict experiments, old
non-Transfer generated images, smoke fine-tuning images, and cloud-only B200
artifacts was moved to:

```text
backup_1ere_version/
```

This backup is intentionally a directory, not a compressed archive, so files can
be restored quickly if a past experiment needs to be checked.
