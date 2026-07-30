# Dreamer-PPO for CARLA (VRU-safe urban driving)

A Dreamer-style world model combined with a PPO policy for CARLA urban
driving, focused on the safety of Vulnerable Road Users (VRUs — pedestrians
and cyclists). The whole pipeline runs in a **mock mode** with no CARLA
installed, so every component can be developed and tested locally.

There are three variants, each trained end-to-end by a single command and
each writing its own CSV log under `logs/`:

| Variant           | Flag         | What it is                                     |
|-------------------|--------------|------------------------------------------------|
| PPO baseline      | `--baseline` | reference: no world model, no dreaming         |
| Dreamer-PPO       | *(none)*     | world model + greedy one-step dreaming         |
| SDBS Dreamer-PPO  | `--sdbs`     | full version: diverse beam search + curriculum |

See [RUNME.md](RUNME.md) for the exact commands and log-inspection steps.

## The state-vector contract (default is 28 dims — the v1 layout)

The **default** trainable/exportable state is the original **28-dim v1
layout**, the checkpoint proven to work in the downstream
SimLingo/`dreamer_guard.py` integration:

```
ego(6) + lane(4) + traffic(3) + vehicle_ahead(5) + 2 VRUs x 5 = 28
```

The expanded safety features are **opt-in** and are NOT part of the default
exported checkpoint until independently validated against a real downstream
adapter:

- **Tier 1** (`config.enable_tier1_state`) — rear/side/nearest vehicle
  awareness, inserting four extra vehicle blocks (28 → 48). This shifts the VRU
  block from index 18 to 38.
- **Tier 2** (`config.enable_tier2_state`) — map-agnostic generalization
  features appended at the end (+7).

`config.state_dim` is derived from these flags (28 / 35 / 48 / 55); the layout
index positions come from `rewards.vru_reward.resolve_layout(config)` — never
hardcode the moving VRU indices. `env.carla_env.expected_state_dim(config)`
returns the full dimension for a config, and
`validate_state_vector(state, expected_dim)` raises immediately on a shape
mismatch — it **never** silently pads, truncates, or reshapes. It is called at
the start of `CarlaEnv.reset()`/`step()`, `SDBSPlanner.plan()`, and
`ActorCritic.act()`/`forward()`, so a wrong dimension fails loudly at the first
model call instead of silently producing garbage.

## Integrating this checkpoint elsewhere

Before wiring a trained checkpoint into ANY external runtime/adapter
(e.g. a CARLA/SimLingo adapter):

1. **Run the dimension check.** The default checkpoint is 28-dim:

   ```bash
   python -m utils.checkpoint_check checkpoints/best_model.pt --state-dim 28
   ```

   (Use `--state-dim 48` if you trained with Tier-1, or `55` with Tier-1 +
   Tier-2.) A `FAIL` here means the checkpoint and your state vector disagree —
   it will produce meaningless output until fixed.

2. **Print the state shape in the adapter.** Add a state-vector shape print
   right before every model call and confirm it matches 28 (or your configured
   dim). A silent dimension/order mismatch between training and inference is the
   most common cause of a checkpoint "not working".

3. **Start S-DBS at horizon=1, groups=1.** Set
   `sdbs_force_fixed_params=True, sdbs_fixed_horizon=1, sdbs_fixed_groups=1`
   (behaviourally equivalent to greedy one-step dreaming) and confirm the
   result matches the plain dreamer variant before enabling full multi-step
   search. Then increase `sdbs_fixed_horizon` gradually and watch for where
   behaviour degrades — this isolates whether the problem is the search depth
   or something upstream.

4. **If route_completion stays at 0% across ALL variants** (baseline
   included), this is an env/reward wiring bug, not a policy-quality issue.
   The trainers run a `ProgressMonitor` that prints a warning when the
   `route_progress` signal never advances — check that output before
   concluding anything about which dreamer variant performs better.

## Tests (no CARLA needed)

```bash
python tests/test_mock.py                     # full mock pipeline
python tests/test_sdbs.py                      # S-DBS core logic
python tests/test_safe_dream_metrics.py        # SAFE-DREAM safety metrics
python tests/test_integration_diagnostics.py   # integration guardrails
python tests/test_tier1_safety.py
python tests/test_tier2_generalization.py
python tests/test_tier3_interpretability.py
```

## SAFE-DREAM Safety Evaluation

Based on Prof. Ben Yahia's **SAFE-DREAM** framework (July 2026). This module
implements **Dimension 4 (Action Dreaming Quality)**, **Dimension 5 (Decision
Quality)**, and a partial **Dimension 6 (Safety Assurance via
logging/traceability)** — the dimensions genuinely applicable to a world-model +
RL system that has no perception stack and no LLM reasoning layer.

Implemented metrics:

- **Decision Quality / Safety (§3.2):** Collision Rate, Near-Collision Rate,
  Min/P5 TTC, Min Safety Distance, Safety-Envelope Violations, Disengagement
  Rate, Traffic-Rule Compliance, Driving Comfort (jerk).
- **Dreaming-specific KPIs (§4, dreamer/sdbs only):** Counterfactual Coverage,
  Future Diversity, Unsafe-Maneuver Rejection Rate, Risk-Prediction Accuracy,
  Hazard-Anticipation Horizon.

**Explicitly out of scope (documented, not silently skipped):**

- **Dimension 1 (Perception Quality)** — CARLA ground truth is used; there is no
  perception stack in this repo.
- **Dimension 2 (Semantic World Model)** — the world model predicts state
  vectors, not semantic scene graphs.
- **Dimension 3 (Reasoning Quality)** — no LLM/causal reasoning module.
- **Hallucination Rate, Physical Realism, EU-CEM Traceability/Scenario
  Coverage** — require a generative visual world model and a formal scenario
  database; both are future work.

Usage (reads the per-variant CSV logs produced by training):

```bash
python -m scripts.safe_dream_report \
    baseline=logs/baseline.csv dreamer=logs/dreamer.csv sdbs=logs/sdbs.csv \
    --plot logs/plots/safe_dream.png
```

**Important caveat:** `distance_per_episode_km` is an **estimate** (default
`0.15` km/episode) because exact traveled distance isn't tracked yet. Collision
Rate, Near-Collision Rate, and Disengagement Rate are therefore approximate
until real odometry is logged; every other metric is exact.
