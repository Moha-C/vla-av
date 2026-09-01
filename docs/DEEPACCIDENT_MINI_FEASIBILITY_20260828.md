# DeepAccident Mini Feasibility Result

## Decision

**Hold, not promoted.** The DeepAccident mini archive is useful for validating
the ingestion and evaluation protocol, but it is not sufficient to train a
reliable camera-only hazard encoder for the SimLingo runtime.

No active SimLingo, PPO, SDBS, RSSM, or CarDreamer checkpoint was modified.

## Audited Data

- Official DeepAccident mini archive, downloaded from the project page.
- 10 accident/normal source pairs inspected.
- 2 pairs rejected because their official accident metadata says
  `colliding agents: none none`.
- 8 independent source groups retained.
- 12 colliding-actor front-camera accident tracks and their 12 paired normal
  tracks retained.
- 1,932 ordered frames and 252 two-second positive-horizon frames.
- Atomic split: 5 source groups train, 2 validation, 1 final test.
- The actor named by `colliding agents` determines the camera viewpoint. This
  prevents a collision involving another instrumented vehicle from being
  assigned to an unrelated `ego_vehicle/Camera_Front` stream.

## Frozen Result

Model: ImageNet-pretrained MobileNetV3-Small per frame, GRU temporal encoder,
128-dimensional embedding, risk head, and TTC head. The best checkpoint was
selected at epoch 1 using validation average precision only.

| Metric | Validation | Test |
| --- | ---: | ---: |
| Positive prevalence | 0.1667 | 0.1544 |
| Average precision | 0.1488 | 0.1430 |
| ROC-AUC | 0.4735 | 0.4345 |
| F1 at validation threshold | 0.3119 | 0.1748 |
| TTC MAE | 0.533 s | 0.496 s |

The average precision is below the positive prevalence on both frozen splits.
The validation false-positive rate is 0.676. These values do not demonstrate
out-of-scenario hazard discrimination.

## Interpretation

The failure is not evidence that the complete DeepAccident dataset is useless.
It shows that eight independent source groups cannot support this proposed
camera-only transfer. DeepAccident was designed as a multi-view, multi-agent,
V2X motion and accident prediction benchmark, while SimLingo currently uses a
single forward camera. The mini archive is therefore a data-pipeline smoke set,
not a sound basis for runtime promotion.

## Next Gate

The next defensible experiment is to run the same actor-aligned protocol on a
substantially larger official subset or the full dataset, without changing the
frozen Bench2Drive test protocol. A candidate must first pass
`promotion_decision.json`, then improve paired CARLA collision/off-road results
across fixed routes and seeds before its frozen embedding can be appended to an
RSSM observation.

Official sources:

- <https://deepaccident.github.io/>
- <https://deepaccident.github.io/data.html>
- <https://arxiv.org/abs/2304.01168>
