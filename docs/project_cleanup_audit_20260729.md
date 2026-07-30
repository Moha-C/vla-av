# Project Cleanup Audit - 2026-07-29

Scope: keep the active CARLA / SimLingo / SUMO / TwinSentinel workflow clean while avoiding destructive moves.

## Archived Now

Moved to:

`/home/mohm/Desktop/vla-av-archived-unused-20260729/legacy_sumo_project/`

Contents:

- `vla-av/SUMO_Project/`
- `vla-av/scripts/run_sumo_all_red_attack_demo.sh`
- `vla-av/scripts/sumo_all_red_attack_demo.py`
- `vla-av-simlingo-dev/SUMO_Project/`
- `vla-av-simlingo-dev/scripts/check_sumo_attack_configs.sh`
- `vla-av-simlingo-dev/scripts/run_sumo_all_red_attack_demo.sh`
- `vla-av-simlingo-dev/scripts/sumo_all_red_attack_demo.py`
- `vla-av-simlingo-stable-20260526/SUMO_Project/`
- `vla-av-simlingo-stable-20260526/scripts/run_sumo_all_red_attack_demo.sh`
- `vla-av-simlingo-stable-20260526/scripts/sumo_all_red_attack_demo.py`

Reason: these were legacy SUMO-only attack assets. The current active attack flow now uses `experiments/TwinSentinel_Project` and the VLA-AV CARLA-owned SUMO mirror.

## Active TwinSentinel Flow

Keep:

- `experiments/TwinSentinel_Project/`
- `scripts/run_twinsentinel_attack_console.sh`
- `scripts/carla_sumo_mirror.py`
- `scripts/run_simlingo_with_sumo_mirror.sh`
- `scripts/simlingo_dashboard.py`
- `generated_sumo_nets/`
- `.conda_node/`

The TwinSentinel dashboard now runs in VLA-AV live bridge mode and writes commands to `logs/sumo_mirror/attack_commands.jsonl`.

## Large Folders To Review Later

Do not move automatically without checking what needs to be preserved:

- `vm_backups/` - about 290 GB, likely VM/training backups.
- `logs/` - about 51 GB, active and historical simulation logs.
- `exports/` - about 36 GB, recovery bundles and handoff zips.
- `backup_1ere_version/` - about 3.9 GB, older dataset/project backup.
- `trash/` - about 3.6 GB, previous archived experiments.

Recommended next cleanup step: produce a manifest for these folders, then archive only dated failed runs and already-exported duplicates.
