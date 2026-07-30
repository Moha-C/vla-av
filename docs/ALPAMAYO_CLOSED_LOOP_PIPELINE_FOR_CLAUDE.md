# Alpamayo-R1 Closed-Loop CARLA Pipeline

Use this document as a prompt/context block for Claude when generating the project documentation.

## Context

The project started with a fine-tuned NVIDIA Alpamayo-R1 model on a CARLA-derived dataset. Alpamayo-R1 is not, by itself, a complete closed-loop CARLA driver in the same sense as SimLingo. It predicts future trajectory/waypoints from camera frames, ego history and textual navigation/context, but it needs an external navigation wrapper around it to decide where the vehicle should go at intersections and how to convert predicted trajectory into CARLA controls.

SimLingo is used as the reference baseline and design inspiration, not as the final project model. SimLingo works well in Bench2Drive because it receives route-level information from the benchmark: target points, high-level commands, route progress and scenario context. The same principle is now being applied around Alpamayo-R1.

## Current Alpamayo-R1 Pipeline

1. CARLA runs a live ego vehicle with an RGB front camera.
2. The local demo process captures camera frames and ego state.
3. The Alpamayo-R1 sidecar worker receives:
   - recent RGB image history,
   - ego pose/speed history,
   - a navigation prompt,
   - live traffic-rule context such as traffic light state, stop sign distance and safety monitor messages.
4. Alpamayo-R1 predicts a future trajectory.
5. The local adapter converts the predicted trajectory to low-level CARLA control:
   - pure-pursuit-like steering,
   - speed control for throttle/brake,
   - smoothing,
   - safety braking for red/yellow lights, stop signs, off-road detection and excessive speed.

## What Was Missing

The original local Alpamayo test only gave broad text such as “follow the lane” or “turn left”. That is not enough for robust closed-loop driving at intersections. From a single camera view, “go left / straight / right” is underspecified unless the model also gets a route target or route planner signal.

This explains why a smaller model such as SimLingo can outperform a larger Alpamayo-R1 in this local CARLA setup: SimLingo was designed and evaluated with Bench2Drive route conditioning, while Alpamayo-R1 was initially used as a standalone trajectory predictor.

## New Closed-Loop Navigation Adapter

The new adapter adds a SimLingo-style local route target point to Alpamayo-R1 without retraining.

At every frame:

1. CARLA map geometry is queried from the ego vehicle location.
2. The adapter scans the current lane ahead.
3. If a command is requested (`left`, `right`, or `straight`), it searches for the next branch/junction and selects the branch matching that command.
4. It converts the selected target waypoint into the ego frame:
   - `x` = meters forward,
   - `y` = meters left,
   - distance = Euclidean distance to target.
5. The runtime prompt sent to Alpamayo-R1 includes this target:

   `Closed-loop route target point, inspired by SimLingo/Bench2Drive: local target is x=... m forward and y=... m left. Planner command: turn left/right/straight/follow lane. Treat this target point as the route objective.`

This turns Alpamayo-R1 from a pure local trajectory predictor into a route-conditioned closed-loop agent.

Because the current Alpamayo-R1 checkpoint was not explicitly trained to consume target points, the wrapper also includes an optional deterministic route-target steering blend. This is not the old lane assist. It is the closed-loop navigation adapter: Alpamayo still predicts the trajectory, but the external route target can gently bias steering toward the intended route branch when the text prompt alone is not enough.

The CARLA-to-Alpamayo coordinate conversion is important. CARLA yaw=0 points along world +x and world +y lies on the vehicle's right, while Alpamayo/AV trajectory coordinates expect positive lateral values to mean left. The wrapper therefore converts world coordinates into an ego frame where `x` is forward and `y` is left before sending history or target-point information to Alpamayo.

## Implementation Files

- `scripts/demo.py`
  - Adds `--route-target-nav`.
  - Adds `--route-target-distance`.
  - Adds `--route-target-scan-distance`.
  - Adds `--route-target-steer-blend`.
  - Computes the local target point from CARLA waypoint geometry.
  - Injects target point + command into the runtime Alpamayo prompt.
  - Optionally blends Alpamayo steering with the target-point steering controller.

- `scripts/run_alpamayo_r1_local_test.sh`
  - Enables route-target navigation by default for Alpamayo-R1 tests.
  - Exposes environment variables:
    - `ROUTE_TARGET_NAV=1`
    - `ROUTE_TARGET_DISTANCE=18`
    - `ROUTE_TARGET_SCAN_DISTANCE=55`
    - `ROUTE_TARGET_STEER_BLEND=0.35`
    - `NAV_MANEUVER=follow_lane|auto|straight|left|right`

## Experiment Plan

Step 1 was SimLingo as a functional baseline. It validates the closed-loop CARLA setup and proves that route-conditioned VLA driving is the right design pattern.

Step 2 is the new Alpamayo-R1 closed-loop wrapper. It adds target points and route commands around the already fine-tuned Alpamayo-R1 checkpoint.

Step 3 is testing Alpamayo-R1 with target points without retraining. This will show whether prompt-level route conditioning is enough.

Step 4, only if needed, will be route-conditioned fine-tuning. The dataset should include the same kind of navigation signals used by SimLingo:

- target point in ego frame,
- next target point,
- high-level command,
- route progress,
- optional commentary/reasoning text,
- RGB camera history,
- ego speed and pose history,
- ground-truth future trajectory.

Step 5 is the final project framing:

`Alpamayo-R1 fine-tuned + CARLA closed-loop navigation adapter`

The final model is not just the base Alpamayo checkpoint. It is the fine-tuned Alpamayo-R1 planner plus the custom CARLA wrapper that provides route conditioning, converts trajectories to controls, and enforces safety.

## Important Limitation

The current no-retraining adapter gives Alpamayo target information only through text. If Alpamayo-R1 does not reliably use this textual target point, route-conditioned fine-tuning will be necessary. That fine-tuning should make target point and high-level command part of the training distribution, matching the SimLingo/Bench2Drive closed-loop recipe more closely.
