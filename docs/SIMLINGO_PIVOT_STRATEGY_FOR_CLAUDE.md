# SimLingo Pivot Strategy and Alpamayo-R1 Decision

Use this document as context/prompt for Claude to generate the project documentation.

## Executive Decision

The project should pivot its main demonstrable autonomous-driving baseline to SimLingo, while keeping Alpamayo-R1 as a secondary research attempt.

This is not because Alpamayo-R1 is a bad model. It is because Alpamayo-R1 is not a drop-in closed-loop CARLA/Bench2Drive driver in our current setup. It is better understood as an open autonomous-driving reasoning and trajectory-planning foundation model that still requires an external closed-loop driving stack: route planner, target points, high-level commands, trajectory-to-control adapter, safety monitor, and CARLA-specific integration.

SimLingo, by contrast, was designed and released as a CARLA closed-loop agent. Its repository includes the agent code, Bench2Drive evaluation interface, route files, data generation code, training code, language label generation, and model checkpoints. For the current internship objective, adapting SimLingo is more realistic, more demonstrable, and more directly aligned with CARLA.

## Sources to Cite

- SimLingo GitHub repository: https://github.com/RenzKa/simlingo
- SimLingo paper: https://arxiv.org/abs/2503.09594
- SimLingo paper states that SimLingo handles closed-loop driving, vision-language understanding, and language-action alignment in CARLA/Bench2Drive.
- SimLingo GitHub README states that the repository includes PDM-lite expert, data collection, language label generation, dreaming data generation, training, and closed-loop evaluation.
- SimLingo is Apache-2.0 licensed, allowing modification and redistribution under the license conditions.
- NVIDIA Alpamayo developer page: https://developer.nvidia.com/drive/alpamayo
- NVIDIA describes Alpamayo as a foundation for reasoning and trajectory planning, not as our ready-to-use CARLA closed-loop application stack.

## Why Alpamayo-R1 Was Difficult in Our CARLA Setup

The initial Alpamayo-R1 fine-tuning did not solve closed-loop driving because the training/inference setup did not include the full missing navigation context.

The model could see camera frames and ego history and could produce future trajectories, but in CARLA intersections this is underspecified. A vehicle needs to know:

- which route branch to follow,
- what high-level command applies next,
- where the next target point is in ego coordinates,
- how far the route target is,
- whether the traffic light/stop sign/crosswalk blocks the route,
- how to translate a predicted trajectory into steer/throttle/brake,
- how to recover from cumulative closed-loop errors.

SimLingo and Bench2Drive already provide this style of closed-loop route-conditioned setup. Alpamayo-R1 would need a large amount of custom engineering and probably a new route-conditioned fine-tuning dataset to match this behavior.

Therefore, spending more time on a generic Alpamayo full-backbone fine-tuning is not recommended. It risks repeating the same problem: training a strong model without giving it the exact closed-loop CARLA signals needed for the task.

## Ethical and Technical Positioning

We should not claim that the original SimLingo base model was created from scratch by us.

The correct positioning is:

> We build a VLA-AV closed-loop autonomous-driving platform based on an open-source SimLingo foundation agent, adapted and extended for our own CARLA scenarios, visualization, adversarial traffic perturbations, metrics, and user-facing control interface.

This is legitimate because SimLingo is open-source, Apache-2.0 licensed, and intended for research extension. The contribution is the system adaptation and experimental framework, not ownership of the original base checkpoint.

## Recommended SimLingo Adaptations for Our Project

### 1. VLA-AV Scenario Control Layer

Build a project-specific scenario launcher around SimLingo:

- map selector,
- route selector,
- random compatible route,
- weather selector,
- traffic density presets,
- pedestrian/VRU density presets,
- bicycle/scooter/vehicle scenario filtering,
- reproducible seed control,
- Pygame/chase/wheel camera mode.

This makes SimLingo usable as a controlled experimental platform instead of only a raw benchmark script.

Project value: this turns an academic repo into a practical CARLA testbed.

### 2. VLA-AV Safety and Evaluation Dashboard

Add a metrics and logging layer around SimLingo:

- route completion,
- collisions,
- red-light infractions,
- stop-sign infractions,
- off-road events,
- blocked-agent events,
- average speed,
- scenario type,
- route ID,
- seed,
- weather,
- model latency,
- video capture and screenshots.

Project value: this creates measurable outputs for the internship report and makes experiments comparable.

### 3. Red-Team / SUMO Attack Integration

Integrate the future work with Mehdi's SUMO/red-attack system:

- external traffic-light perturbation events,
- malicious or abnormal traffic-flow injection,
- sudden congestion,
- emergency braking events,
- adversarial VRU/crosswalk events,
- before/after comparison of SimLingo behavior.

This can become the strongest original contribution:

> A CARLA closed-loop VLA evaluation platform for testing the robustness of autonomous-driving VLA agents under adversarial traffic perturbations.

Project value: this goes beyond vanilla SimLingo and aligns the model with our specific internship theme.

## Optional Later Adaptation: SimLingo-VLA-AV Fork

Once the baseline is stable, create a named fork/system:

`SimLingo-VLA-AV`

This fork can include:

- our launcher,
- our dashboard,
- our Pygame visualization modes,
- our route/scenario filtering,
- our metrics exporter,
- our SUMO red-team bridge,
- our generated evaluation reports,
- optional fine-tuning on a small custom dataset of CARLA/SUMO red-attack scenes.

This is a defensible “our model/system” framing:

> SimLingo-VLA-AV is our adapted closed-loop VLA autonomous-driving system built on the open-source SimLingo foundation agent.

## What to Do With Alpamayo-R1

Alpamayo-R1 should not be the main deliverable unless there is enough time for a proper route-conditioned training pipeline.

It can remain as a secondary experimental track:

1. Keep the current Alpamayo-R1 fine-tuned checkpoint.
2. Keep the CARLA closed-loop navigation adapter already implemented.
3. Test it on simple lane-following scenes.
4. Document the limitation: without route-conditioned training, Alpamayo-R1 struggles with CARLA closed-loop intersections.
5. Present it as a research comparison against SimLingo.

This is valuable academically because it explains why a strong foundation model may fail in closed-loop CARLA if it lacks the correct integration signals.

## Final Recommendation

For the internship deliverable:

Main path:

`SimLingo open-source VLA + VLA-AV scenario/control/evaluation/red-team extensions`

Secondary path:

`Alpamayo-R1 fine-tuned + experimental closed-loop navigation adapter`

This is safer, more honest, and more likely to produce a working demo and report-quality results.

The final project should emphasize:

- closed-loop CARLA evaluation,
- VLA behavior under traffic scenarios,
- interaction with pedestrians/VRUs,
- robustness under red-team perturbations,
- comparison between SimLingo and Alpamayo-R1,
- explanation of why route-conditioned closed-loop design matters.
