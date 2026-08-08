# Dreamer Complement Training

This project uses Dreamer as a complement to SimLingo, not as a standalone
CARLA driving policy.

## Correct Objective

SimLingo remains the base driver and produces the control command every tick.
The Dreamer module observes the same closed-loop context and scores a small set
of SimLingo-relative candidate actions:

- keep SimLingo's native action;
- brake or hold when the immediate future looks unsafe;
- cautiously adjust steer/throttle;
- overtake or finish a pass only when the guard logic says the manoeuvre is
  physically plausible.

Training for this objective should update the Dreamer world-model scorer used by
`external/simlingo/team_code/dreamer_guard.py`.

## What Went Wrong With `*_rl_noguard`

The `dreamer_ppo_rl_noguard` and `dreamer_sdbs_rl_noguard` experiments trained a
policy inside the generic Dreamer `CarlaEnv`. That makes Dreamer act as an
autonomous policy rather than a SimLingo complement, so the resulting checkpoint
does not learn the desired problem.

Those modes were removed from the dashboard/backend and archived outside the
active project tree. They should not be used as the main SimLingo+Dreamer
evaluation path.

## Correct Command

Use converted SimLingo+Dreamer traces and train the complement scorer:

```bash
cd ~/Desktop/vla-av
DREAMER_COMPLEMENT_KIND=both bash scripts/train_dreamer_complement_from_traces.sh
```

The script installs separate checkpoints:

- `external/simlingo/checkpoints/dreamer_ppo_complement/latest_world_model.pt`
- `external/simlingo/checkpoints/dreamer_sdbs_complement/latest_world_model.pt`

These are exposed in the dashboard as:

- `Dreamer PPO complement-trained`
- `Dreamer SDBS complement-trained`

The stable guarded checkpoints are not overwritten.
