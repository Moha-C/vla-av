# DeepAccident Risk Encoder Protocol

## Scope

DeepAccident is used only to pretrain a temporal visual hazard representation.
It does not replace Bench2Drive traces, does not provide expert control actions,
and does not directly train the SimLingo residual policy.

The active SimLingo/RSSM checkpoints remain untouched until the new risk signal
passes frozen offline and paired CARLA evaluations.

## Data provenance

- Official project: <https://deepaccident.github.io/>
- Official mini dataset: 20 scenarios, approximately 9 GB
- Official full dataset: 691 scenarios, approximately 313 GB
- Sensor used initially: `Camera_Front` of the actor identified by the
  official `colliding agents` metadata
- Native frame rate: 10 Hz
- Temporal input: 3 frames sampled every 5 native frames (one second history)
- Prediction target: collision within the next two seconds

The published data layout does not expose low-level expert
`steer/throttle/brake` commands. The final frame of an accident sequence is
therefore treated as a weak collision-time proxy. This assumption is written
into `audit.json` and every exported checkpoint.

## Preparation

```bash
cd ~/Desktop/vla-av
bash scripts/prepare_deepaccident_mini.sh
```

The script downloads the official archive, validates it, extracts it, preserves
scenario order, and writes:

- `data/deepaccident/processed/mini/frames.jsonl`
- `data/deepaccident/processed/mini/scenarios.jsonl`
- `data/deepaccident/processed/mini/audit.json`

Accident and normal counterparts with the same scenario identifier are an
atomic split group. The mini protocol uses 5 groups for training, 2 for
validation, and 1 for its final test. No actor viewpoint or paired scene from
one group may occur in more than one split. Runs whose accident metadata says
`none none` are rejected instead of receiving a fabricated collision target.

## Risk pretraining

```bash
cd ~/Desktop/vla-av
/home/mohm/miniconda3/envs/vla-av/bin/python \
  scripts/train_deepaccident_risk_encoder.py \
  --processed-dir data/deepaccident/processed/mini \
  --output-dir checkpoints/deepaccident/risk_encoder_mini_actor_aligned
```

The model is a MobileNetV3 frame encoder followed by a GRU. It exports a
128-dimensional temporal embedding and separate risk/TTC heads. The checkpoint
contains the complete backbone and never downloads weights at runtime.

Every completed run writes `promotion_decision.json`. The offline gate requires
at least 30 independent source groups plus validation/test discrimination,
recall, and false-positive checks. Passing only makes the model eligible for
paired CARLA evaluation; it never modifies the active SimLingo checkpoint.

## Promotion gate

The encoder remains isolated until all of the following hold:

1. Test average precision and ROC-AUC exceed a constant risk baseline.
2. The selected threshold is fixed from validation data only.
3. Early-warning false positives and TTC error are reported by scenario.
4. A camera-only Bench2Drive adapter improves collision/off-road outcomes over
   at least three fixed seeds without reducing route completion materially.
5. The same route, seed, weather, and SimLingo checkpoint are used for paired
   baseline/RSSM comparisons.

Only after this gate may the frozen embedding be appended to the RSSM
observation. PPO and action-conditioned dynamics remain trained on native
Bench2Drive transitions where real controls and rewards are available.

## Scientific limitations

- The terminal-frame collision time is weak supervision and must be audited.
- The mini archive leaves only eight usable independent accident/normal
  groups after actor alignment; it is a feasibility test, not a publishable
  performance estimate.
- Accident-heavy data can be poorly calibrated against normal driving.
- V2X views may be used as a privileged teacher, but production inference must
  stay ego-camera-only unless explicitly reported otherwise.
- Dataset redistribution and publication terms must be verified with the
  authors before sharing archives or derived data.

## Mini feasibility outcome

The actor-aligned mini experiment was completed on 2026-08-28 and did not pass
the offline promotion gate. See
[DEEPACCIDENT_MINI_FEASIBILITY_20260828.md](DEEPACCIDENT_MINI_FEASIBILITY_20260828.md)
for the frozen metrics and interpretation. This negative result is retained to
prevent accidental installation of an under-supported risk encoder.
