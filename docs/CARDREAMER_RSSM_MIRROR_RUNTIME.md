# CarDreamer RSSM Mirror Runtime

## Purpose

This runtime evaluates the official CarDreamer `overtake.ckpt` as an active
complement to native SimLingo. SimLingo remains the primary controller. A
separate Python 3.10 process reconstructs CarDreamer's privileged full-state
BEV, runs the official DreamerV3/RSSM policy, and publishes a discrete control
proposal. The Python 3.8 SimLingo agent then applies a bounded residual blend.

## Mirror Convention

The upstream overtake policy predominantly proposes the side used in its
training task. For the Bench2Drive accident layout used here, the privileged
BEV is flipped horizontally before inference and the predicted steering sign
is inverted when mapped back to CARLA. The checkpoint weights are unchanged.

## Runtime Contract

- Dashboard mode: `CarDreamer RSSM mirror + traffic gate`
- Sidecar: `scripts/cardreamer_shadow_sidecar.py --runtime-mode residual`
- Consumer: `external/simlingo/team_code/cardreamer_residual.py`
- Checkpoint: `external/cardreamer_checkpoints/CarDreamer_checkpoints/overtake.ckpt`
- Expected SHA256: `123525828488d596e80dad0fad0681767cec937adcc04bf0d5aa8ee972aa8058`
- Default blend: 35% CarDreamer proposal, 65% current SimLingo command
- Authority scope: blocked-lane overtaking only
- Traffic gate: adjacent-lane clearance, rear TTC, and oncoming TTC

Throttle and brake are represented as one signed longitudinal command. After
two consecutive positive lateral RSSM proposals while a stationary vehicle is
detected ahead, the adapter can enter a temporal engagement phase. Entry is
accepted only when the adjacent lane exists, has at least 5 m clearance, has a
rear TTC of at least 5 s, and has an oncoming TTC of at least 7 s. During the
engagement, a closing rear or oncoming vehicle immediately ends CarDreamer's
authority and hands the untouched command back to SimLingo. Outside an active
overtaking episode, CarDreamer has no control authority.

Transport checks reject stale, malformed, non-mirrored, or unexpected
checkpoint messages. The deterministic traffic gate additionally rejects an
unsafe entry or interrupts an unsafe committed maneuver. Pygame shows the
proposal, native SimLingo command, final command, rear/oncoming TTC, gate phase,
and rejection reason.

## Scientific Status

This mode is experimental and not safety validated. It uses privileged CARLA
actor/map state to build the full-state BEV expected by CarDreamer, so it must
not be described as camera-only. The horizontal mirror transforms the complete
BEV, including all road geometry and actors; it is not limited to the blocking
vehicle.

The official `overtake.ckpt` task uses Town04 with one slow lead vehicle. It
does not train with rear-left traffic or opposing traffic during the overtake.
The traffic gate therefore compensates for a documented distribution gap. It
is deterministic runtime arbitration and must not be reported as learned RSSM
safety. A learned solution requires retraining the checkpoint with these actor
configurations.

## Multi-Scenario Campaign

The campaign selects real Bench2Drive XML routes covering Town10HD/Town12
accidents, VRU crossings, traffic flow, Town13 urban scenes, red-light
intersections, blocked lanes, and oncoming traffic. It runs sequentially and
continues after a failed route.

```bash
cd ~/Desktop/vla-av
python3 scripts/run_cardreamer_rssm_campaign.py \
  --max-per-bucket 2 \
  --max-routes 18 \
  --route-timeout 900 \
  --residual-alpha 0.35 \
  --quality Low
```

Use `--dry-run` to inspect route selection without starting CARLA and
`--record` to retain videos. Campaign artifacts are written under
`logs/cardreamer_rssm_campaign/`. Completed Bench2Drive results are classified
in the dashboard as `SimLingo + CarDreamer RSSM mirror`.
