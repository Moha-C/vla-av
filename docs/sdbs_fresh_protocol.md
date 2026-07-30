# Fresh SDBS Dreamer Protocol

This branch is the clean restart for `youma2003/dreamer_ppo_carla`.

## Current State

- Fresh repo clone: `experiments/dreamer_ppo_carla_sdbs_fresh`
- Old SDBS work archived under: `trash/sdbs_archive_20260715_114154`
- Fresh checkpoint slot: `external/simlingo/checkpoints/dreamer_sdbs_fresh`
- Working teacher: `Youma v1 accident/overtake recovery`

Do not use the `Youma SDBS fresh accident/overtake` dashboard mode until a fresh checkpoint exists.

## Teacher Collection

Use the dashboard with:

- Launch mode: `CARLA POV + Action Dreaming collect`
- Dreamer mode: `Youma v1 accident/overtake recovery`
- Scenario: `Accident`
- Priority routes:
  - `148 | Town10HD | Accident`
  - `06 | Town12 | AccidentTwoWays`
  - `32 | Town12 | Accident`
  - `33 | Town12 | Accident`
  - `36 | Town12 | AccidentTwoWays`

Let each run pass the accident and return to the lane before stopping. If a teacher run causes a collision, offroad, or unsafe overtake, mark that run as bad and do not use it for training.

Collected traces are written under:

```text
logs/action_dreaming_collect/
```

The latest trace path is:

```text
logs/action_dreaming_collect/latest_trace.txt
```

## Training Target

The SDBS fresh model should learn from the successful SimLingo + Dreamer v1 traces:

- state: 28D SimLingo/Dreamer vector
- action: chosen teacher control
- next state: next collected state
- risk/progress targets: derived from Dreamer status

After training, install the checkpoint as:

```text
external/simlingo/checkpoints/dreamer_sdbs_fresh/best_world_model.pt
```

Then the dashboard mode `Youma SDBS fresh accident/overtake` can be tested against native SimLingo and Youma v1.
