# SimLingo + Dreamer-PPO Offline Validation

Date: 2026-06-17

## Goal

Evaluate whether `youma2003/dreamer_ppo_carla` can be used as a Dreamer-style world-model component around SimLingo, with the objective of improving SimLingo-VLA-AV without replacing the stable SimLingo baseline.

## Current Position

The external Dreamer-PPO repository is not a drop-in replacement for SimLingo on Bench2Drive/CARLA. It is better suited as a learned predictive module that scores candidate actions.

The viable architecture is therefore:

- SimLingo remains the main VLA driving policy.
- Dreamer learns short-horizon consequences from collected SimLingo driving samples.
- A conservative Dreamer Guard may override SimLingo only when it predicts a clear risk reduction.

## Data Used

Source:

`/home/mohm/Desktop/vla-av-simlingo-dev/data/action_dreaming_adv/filtered/batch_town12_13_20260616_filtered.jsonl`

Converted dataset:

`data/simlingo_ad_filtered_20260616.npz`

Conversion summary:

- Input rows: 2528
- Transitions: 2509
- Runs: 19
- Town12 transitions: 1103
- Town13 transitions: 1406
- Skipped rows: 18 route-boundary rows

## World Model Training

Checkpoint:

`outputs/simlingo_world_model_20260616/best_world_model.pt`

Training summary:

- Transitions: 2509
- Train samples: 2007
- Validation samples: 502
- Best epoch: 27
- Best validation loss: 0.9370
- State MAE, normalized: 0.0690
- Risk MAE: 0.1227
- Progress MAE, normalized: 0.2021

This is good enough for a first offline action-scoring test, but not enough to trust the Dreamer as an autonomous replacement policy.

## Pure Dreamer Replacement Test

Output:

`outputs/simlingo_vs_dreamer_benchmark_wrisk2/summary.json`

Result:

- Samples: 2528
- Dreamer selected the same candidate as SimLingo: 33.94%
- Dreamer improved its own learned score on average.
- But compared to original candidate metadata, Dreamer often selected weaker actions.
- Mean metadata score delta vs SimLingo: -0.1000
- Mean metadata progress delta vs SimLingo: -0.0609

Conclusion: pure Dreamer replacement is not safe yet. It is too free and does not consistently agree with the original action-quality metadata.

## Dreamer Guard Test

Strict guard:

`outputs/simlingo_dreamer_guard_rm005_md005/summary.json`

Settings:

- Risk margin: 0.05
- Max metadata score drop: 0.05

Result:

- Samples: 2528
- Overrides: 103
- Override rate: 4.07%
- Mean Dreamer risk delta vs SimLingo: -0.00389
- Mean metadata score delta vs SimLingo: -0.00106

Slightly more permissive guard:

`outputs/simlingo_dreamer_guard_rm003_md003/summary.json`

Settings:

- Risk margin: 0.03
- Max metadata score drop: 0.03

Result:

- Samples: 2528
- Overrides: 137
- Override rate: 5.42%
- Mean Dreamer risk delta vs SimLingo: -0.00400
- Mean metadata score delta vs SimLingo: -0.00101

Conclusion: Dreamer Guard is the correct next direction. It keeps SimLingo as the main driver and only proposes limited risk-reduction overrides.

## Decision

We should not replace SimLingo with Dreamer-PPO.

We should continue with a conservative SimLingo + Dreamer Guard architecture:

1. SimLingo predicts its normal action and waypoint trajectory.
2. A candidate action set is generated around the SimLingo action.
3. Dreamer predicts short-horizon risk/progress for each candidate.
4. The guard accepts an override only if risk decreases clearly and metadata/action quality does not degrade.

This preserves the working SimLingo baseline while creating a legitimate research contribution: a learned world-model safety guard around a VLA driving policy.

## Next Step

Integrate the strict Dreamer Guard into the dev SimLingo pipeline behind an explicit flag, for example:

`SIMLINGO_DREAMER_GUARD=1`

The stable SimLingo backup must remain untouched.
