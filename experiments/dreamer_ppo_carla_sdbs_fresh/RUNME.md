# How to run & evaluate — quick runbook

This project trains **end-to-end with a single command per variant** — there is
nothing to run "part by part". The world model, auxiliary heads, PPO policy,
curriculum, and dreaming are all set up and trained inside one loop. The
"Step 2–5" / "Tier 1–3" labels in the history are *build milestones*, not a
sequence you have to execute.

There are three variants (each writes its **own** CSV under `logs/`):

| Variant            | Command flag | What it is                                      |
|--------------------|--------------|-------------------------------------------------|
| PPO baseline       | `--baseline` | reference: no world model, no dreaming          |
| Dreamer-PPO        | *(none)*     | world model + greedy one-step dreaming          |
| SDBS Dreamer-PPO   | `--sdbs`     | full version: diverse beam search + curriculum  |

----

## 0. One-time check (no CARLA needed)

```bash
python tests/test_mock.py                     # full pipeline smoke test, ~seconds
python tests/test_integration_diagnostics.py   # dimension/horizon guardrails
```

## 1. (Optional) mock smoke test — confirms it runs without a CARLA server

```bash
python -m training.dreamer_ppo --mock --baseline --episodes 3
python -m training.dreamer_ppo --mock            --episodes 3
python -m training.dreamer_ppo --mock --sdbs     --episodes 3
```
> In `--mock` mode the agents move as random noise, so the *numbers* are
> meaningless — this only proves the code runs. Real numbers need CARLA.

## 2. Real training (needs a running CARLA 0.9.15 server)

Run each variant once. Each writes its own log automatically:
`logs/baseline.csv`, `logs/dreamer.csv`, `logs/sdbs.csv`.

```bash
python -m training.dreamer_ppo --baseline      # -> logs/baseline.csv
python -m training.dreamer_ppo                 # -> logs/dreamer.csv
python -m training.dreamer_ppo --sdbs          # -> logs/sdbs.csv
```

Notes:
- `--sdbs` is compute-heavy. For a first run, add `--episodes N` to keep it
  short.
- Add `--device cuda` if a GPU is available.
- Override the filename with `--log-name myrun.csv` if you want.

## 3. Inspect the per-variant logs

Each run writes one row per episode to its own CSV under `logs/`. Open the
three files (`baseline.csv`, `dreamer.csv`, `sdbs.csv`) directly, or load them
with any tool (pandas, a spreadsheet, `plot_results.py`). The columns separate
**VRU safety (primary)** from **vehicle safety (secondary)**:

- **VRU safety:** `vru_collisions`, `vru_near_misses`, `min_ttc_vru`,
  `avg_distance_to_vru`.
- **Vehicle safety:** `vehicle_collisions`, `vehicle_near_misses`,
  `rear_incidents`.
- **Performance:** `return`, `route_completion`.

A per-episode safety-progression table is printed during SDBS runs and can be
regenerated from a CSV via `training.logger.Logger.create_summary_table`.

## 4. Before integrating a checkpoint elsewhere — check dimensions first

A silent state-vector dimension/order mismatch between training and inference
is the most common cause of a checkpoint "not working" downstream. The default
state is the original v1 28-dim layout, so run the dimension check first:

```bash
python -m utils.checkpoint_check checkpoints/best_model.pt --state-dim 28
```
(Use `--state-dim 48` if trained with Tier-1, `55` with Tier-1 + Tier-2.)

If **route_completion stays at 0% across ALL variants** (baseline included),
that is an env/reward wiring bug, not a policy-quality issue — the trainers
print a `ProgressMonitor` warning when the `route_progress` signal never
advances. See the "Integrating this checkpoint elsewhere" section in
[README.md](README.md) for the full checklist (including starting S-DBS at
`sdbs_force_fixed_params=True, horizon=1, groups=1`).
