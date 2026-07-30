# Dreamer RL Protocol

This project keeps SimLingo as the primary closed-loop driver. The Dreamer does
not replace SimLingo end to end; it learns to score or intervene around
SimLingo's proposed control when the scene is risky.

## What Stays Safe

- `dreamer_ppo` uses `external/simlingo/checkpoints/dreamer_guard/best_world_model.pt`.
- `dreamer_sdbs` uses `external/simlingo/checkpoints/dreamer_sdbs_fresh/best_world_model.pt`.
- These guarded checkpoints are the stable demo modes and must not be overwritten
  by RL experiments.

## Experimental Modes

- `dreamer_ppo_rl_noguard`
- `dreamer_sdbs_rl_noguard`

Both modes have separate checkpoints under:

- `external/simlingo/checkpoints/dreamer_ppo_rl_noguard/`
- `external/simlingo/checkpoints/dreamer_sdbs_rl_noguard/`

They start as copies of the guarded checkpoints, then get replaced only after
validated RL training.

## Recommended Pipeline

1. Collect traces from real SimLingo runs using `CARLA POV + Action Dreaming collect`.
2. Build a strict RL dataset from the JSONL traces.
3. Audit the dataset before training.
4. Train an offline world-model warm-start from the audited dataset.
5. Launch small closed-loop RL smoke tests.
6. Launch longer closed-loop PPO/SDBS training.
7. Install a checkpoint only after KPI validation.

## Commands

Build and audit a dataset from all local Action Dreaming traces:

```bash
cd ~/Desktop/vla-av
bash scripts/build_dreamer_rl_dataset.sh
```

Build from explicit traces:

```bash
cd ~/Desktop/vla-av
bash scripts/build_dreamer_rl_dataset.sh \
  logs/action_dreaming_collect/action_dreaming_20260715_150111.jsonl \
  logs/action_dreaming_collect/action_dreaming_20260715_154252.jsonl
```

Train an offline warm-start world model for PPO:

```bash
cd ~/Desktop/vla-av
DREAMER_RL_KIND=ppo bash scripts/train_dreamer_rl_world_model_warmstart.sh
```

Train an offline warm-start world model for SDBS:

```bash
cd ~/Desktop/vla-av
DREAMER_RL_KIND=sdbs bash scripts/train_dreamer_rl_world_model_warmstart.sh
```

Launch a small real CARLA smoke test from a warm-start:

```bash
cd ~/Desktop/vla-av
DREAMER_RL_KIND=ppo \
DREAMER_RL_EPISODES=10 \
DREAMER_RL_INIT_WORLD_MODEL=/path/to/best_world_model.pt \
bash scripts/start_dreamer_rl_noguard_training.sh
```

Watch progress:

```bash
cd ~/Desktop/vla-av
DREAMER_RL_KIND=ppo bash scripts/watch_dreamer_rl_noguard_training.sh
```

Install the best RL checkpoint into the experimental dashboard slot only after
manual validation:

```bash
cd ~/Desktop/vla-av
DREAMER_RL_KIND=ppo bash scripts/install_dreamer_rl_noguard_checkpoint.sh
```

## Important Caveat

The current runtime Dreamer adapter scores candidate actions around SimLingo. It
can load world-model checkpoints produced by offline training or PPO training.
A future runtime policy adapter would be needed if we want to execute the PPO
actor directly. That is intentionally not the default, because the project goal
is a SimLingo-complement Dreamer, not a standalone replacement driver.
