# Report-aligned SimLingo + Dreamer architecture

This document is an implementation contract. It does not replace or modify the
internship report. The code must continue to support the following scientific
description:

> SimLingo is the reference driver. A Dreamer world model based on an RSSM
> learns temporal environment dynamics, imagines candidate futures, and trains
> an actor and critic in those imagined trajectories. Dreamer only proposes an
> assistance. A continuous authority `alpha_t` blends that proposal with the
> native SimLingo command.

## Repository diagnosis

### SimLingo action interface

`external/simlingo/team_code/agent_simlingo.py` predicts geometric route
waypoints and temporal speed waypoints. `control_pid()` converts them into a
CARLA `VehicleControl`; the existing Dreamer integrations are applied after
that conversion.

The report-aligned branch therefore intervenes at the **post-PID control
level**, using `[steer, throttle, brake]` as the physical action. This is the
least invasive compatible level for the current data and checkpoints:

- candidate 0 can be the exact native SimLingo control;
- `alpha=0` can be verified bit-for-bit against the native baseline;
- the native waypoint prediction and native PID are not modified;
- existing temporal traces already contain these controls and their outcomes.

Intervening before the PID would require a learned trajectory decoder, new
trajectory-labelled training data, and a separate validation campaign. It is
not silently approximated here. The chosen control-level residual remains a
bounded correction around SimLingo, not an autonomous replacement policy.

The RSSM transition receives only that three-dimensional physical command.
`alpha_t` is stored as authority/reward metadata, but it is not appended as a
fourth dynamics action: its physical effect is already present in the blended
command sent to CARLA. This avoids exposing the model at imagination time to
an artificial fourth input that is always zero in native Phase-1 data.

### Existing Dreamer code

The repository already contains useful prototypes in
`dreamer_world_models.py`, `dreamer_guard.py`, and the RSSM training scripts.
They establish that categorical latent dynamics and pairwise calibration can
run in the SimLingo Python 3.8 environment. They also mix several historical
guards, recovery rules, schemas, and policy variants in a monolithic runtime.
They remain untouched and available as the legacy ablation.

The report-aligned implementation lives independently under `src/world_model`
and has its own configuration and checkpoints.

### CarDreamer and DreamerV3 components

The vendored CarDreamer tree is pinned by its Git commit. CarDreamer itself is
licensed for non-commercial research use. Its bundled DreamerV3 implementation
is MIT licensed. The new PyTorch implementation adapts the following
DreamerV3 mechanisms to the compact SimLingo state rather than importing the
CarDreamer environment or autonomous agent:

- recurrent deterministic state plus categorical stochastic state;
- prior/posterior latent transition and straight-through categorical samples;
- balanced representation/dynamics KL with free nats;
- prediction heads trained jointly with the world model;
- latent imagination with continuation weights;
- actor and critic optimization on imagined returns;
- temporally ordered replay sequences.

No CarDreamer BEV, CARLA task, route logic, or complete policy is used.
Attribution and license boundaries are recorded in
`docs/THIRD_PARTY_DREAMER_ATTRIBUTION.md`.

The inspected technical correspondences are explicit:

| Upstream reference | Mechanism adapted locally | Local implementation |
| --- | --- | --- |
| `dreamerv3/nets.py:RSSM` | deterministic recurrent state, categorical stochastic state, prior/posterior and imagination | `src/world_model/rssm.py` |
| `dreamerv3/agent.py:WorldModel.imagine` | latent rollout with continuation-aware horizon | `src/world_model/planning.py` |
| `dreamerv3/behaviors.py:ActorCritic` | actor/critic learning from imagined returns | `src/world_model/training.py` |
| `dreamerv3/embodied/replay/` | ordered temporal sequence sampling | `src/world_model/dataset.py` |

These are adapted algorithmic mechanisms, not imported runtime classes. The
upstream implementation is JAX/Python 3.10 and expects CarDreamer observations;
the local implementation is PyTorch/Python 3.8 and consumes the report-defined
compact observation. This boundary avoids replacing SimLingo or its
Bench2Drive environment.

## Runtime invariants

1. The native condition never constructs or calls a report-aligned module.
2. Candidate zero is the exact native SimLingo control.
3. Shadow mode computes predictions but returns the native control unchanged.
4. Fixed authority and learned authority are separate ablations.
5. `alpha=0` returns the native control without round-tripping through a
   different longitudinal representation.
6. Candidate generation contains no geometric safety veto. Safety must come
   from learned risk/value predictions; the legacy guard remains a distinct
   comparison condition.
7. Pairwise calibration is optional and cannot change candidate features or
   world-model weights.
8. A checkpoint is never labelled production-ready merely because it loads.
   Validation and test metrics remain separate from closed-loop driving KPIs.
9. A candidate rollout evaluates its proposed first physical command, then
   uses the learned residual actor for latent continuation. It does not claim
   access to unknown future SimLingo commands.

## Target modules

| File | Responsibility |
| --- | --- |
| `src/world_model/config.py` | Versioned YAML loading and validation |
| `src/world_model/observation.py` | Named compact observation, normalization, pure builder |
| `src/world_model/rssm.py` | Categorical RSSM and separate physical heads |
| `src/world_model/policy.py` | Residual actor and latent critic |
| `src/world_model/planning.py` | Candidate generation, latent rollout, utility scoring |
| `src/world_model/authority.py` | Learned continuous authority and temporal smoothing |
| `src/world_model/pairwise.py` | Optional separately trained pairwise calibrator |
| `src/world_model/reward.py` | Configurable report reward and component logging |
| `src/world_model/dataset.py` | Trace conversion, temporal windows, seed-disjoint splits |
| `src/world_model/agent.py` | `SimLingoDreamerAgent` orchestration and checkpoint API |
| `external/simlingo/team_code/report_dreamer_adapter.py` | CARLA/SimLingo context adapter only |
| `configs/dreamer_report_aligned.yaml` | Architecture, reward, authority, training, ablations |
| `scripts/train_report_dreamer.py` | Inspect, world-model, actor/critic and pairwise phases |
| `scripts/evaluate_report_dreamer.py` | Frozen split and prediction-head evaluation |
| `tests/test_report_aligned_dreamer.py` | Progressive tests that do not require live CARLA |

## Ablation contract

- **A native**: SimLingo only.
- **B legacy guards**: existing guarded Dreamer runtime, unchanged.
- **C RSSM fixed**: RSSM candidate planning with a configured fixed alpha.
- **D RSSM learned**: RSSM, imagined actor/critic, learned continuous alpha.
- **E RSSM pairwise**: D plus optional pairwise utility residual.

All conditions use the same route XML, seed, weather, CARLA version, and stop
criteria. Alpamayo remains an external comparison.

## Dataset and scientific split

Input traces are parsed as ordered episodes. Every record stores, when
available, town, route, scenario, seed, weather, native action, proposed
action, alpha, final action, reward components, incidents, TTC, progression,
and RSSM predictions. Missing metadata is recorded as unknown and is never
invented.

Native Phase-1 traces persist the named normalized 32D observation at capture
time. Training consumes that exact vector when complete; rebuilding from raw
context is only a compatibility fallback for diagnostic historical traces.

Splits are made by seed group before windowing. Train, validation, and test seed
sets are disjoint, with at least two distinct seeds in each split. Validation
selects hyperparameters and checkpoints. Test is read only by the final
evaluator. Pairwise examples inherit the same split and cannot pair records
across it.

## Validation order

The implementation is accepted progressively: tensor forward, small world
model fit, latent rollout, imagined actor/critic update, shadow mode, exact
`alpha=0`, fixed low alpha, learned authority, pairwise mode, then a short live
CARLA run. The first nine are automated. The live test requires the installed
CARLA server and is reported as pending unless actually executed.

World-model errors and closed-loop driving scores are reported separately.
No counterfactual or SAFE-DREAM KPI is emitted unless its required prediction
and ground-truth fields are present.
