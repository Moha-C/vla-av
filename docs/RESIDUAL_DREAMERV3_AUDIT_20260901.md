# Residual DreamerV3 v2 World-Model Audit

## Decision

**Rejected at Gate 2 on 2026-09-01.** The actor/critic stage was not run and
the production SimLingo, guarded Dreamer, SUMO bridge, dashboard, and promoted
report RSSM checkpoint were not modified.

The kill criterion was fixed before training: the RSSM had to beat both frozen
baselines on unseen, seed-disjoint trajectories. It did not beat the best
baseline for open-loop observation prediction, so training an actor inside
this model would not be scientifically justified.

## Frozen Data

- 12 accepted native SimLingo Bench2Drive episodes;
- 8,707 ordered transitions from Town10HD, Town12, and Town13;
- six train seeds, three validation seeds, and three test seeds;
- no seed overlap between splits;
- one incomplete trace rejected for missing Bench2Drive ground truth;
- 32-dimensional state, physically applied action, reward, continuation,
  collision, off-road, route, scenario, town, and seed provenance.

## Training

- schema: `residual_dreamerv3_v2`;
- device: NVIDIA RTX A6000 / CUDA;
- 40 world-model epochs over all available temporal windows;
- best checkpoint: epoch 25, validation composite loss `4.462917`;
- actor epochs executed: 0.

## Validation Results

| Horizon | RSSM observation MSE | Persistence | Action-conditioned ridge |
| ---: | ---: | ---: | ---: |
| 1 | 0.179957 | 0.177938 | **0.159966** |
| 5 | 0.548550 | **0.444487** | 0.575055 |
| 10 | 1.139400 | **0.749805** | 0.879800 |
| 20 | 2.525274 | **1.102437** | 1.146179 |

Aggregate comparison against the best baseline:

- observation MSE improvement: `-77.53%` (failed);
- reward MAE improvement: `+42.94%` (passed);
- risk Brier improvement: `+41.24%` (passed);
- action collapse check: passed;
- action sensitivity spread check: passed;
- finite-value check: passed.

The failure is specifically long-horizon state drift. Reward and risk heads
learned useful signals, but those successes do not compensate for an inaccurate
imagined state trajectory.

## Local Evidence

The private workstation artifacts are preserved outside Git at:

```text
checkpoints/residual_dreamerv3/rejected/20260901_v2_open_loop_gate_fail/
```

Artifact SHA-256 digests:

```text
554ef261e60263d3be5e05b5c6f8a4e8a4c27ff2cdca71cc77200544d29707eb  world_model_candidate.pt
630b19d3c5ddd9c28bf5c093b20f61714a82b1bbe522b931f6af247dbd83bcd4  world_model_gate_validation.json
2c7ed05d6e0edb4a436fa9112fe7f3be839c256e9dfb584f77c1e9e99987211f  dataset_manifest.json
```

These files are intentionally excluded from Git because candidate and rejected
checkpoints, raw traces, and local paths are private generated artifacts. The
pipeline, configuration, tests, and this result report remain versioned.

## Next Scientific Step

Do not tune the actor, weaken the gate, or expose this checkpoint in the web
interface. A new attempt requires a materially different world-model approach
or substantially broader native SimLingo transition coverage. It must be
evaluated against the same frozen persistence and ridge baselines before any
imagined actor training is allowed.
