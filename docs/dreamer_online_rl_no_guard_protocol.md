# Dreamer Online RL No-Guard Protocol

This file fixes the vocabulary for the SimLingo + Dreamer RL work.

## Target

Both RL variants are complements to SimLingo:

- SimLingo always produces the base closed-loop control.
- The Dreamer policy observes the world state, SimLingo's proposed command, the
  blocked counter, and adjacent/oncoming-lane context (44 values total).
- It predicts an absolute target control plus a learned intervention gate. The
  executed command is the continuous blend between SimLingo and that target.
- No runtime guard, shield, recovery heuristic, gap rule, TTC veto, or hard-coded safety override is allowed in the RL no-guard variants.
- Safety is learned through rewards and penalties, not through hand-coded action filtering.

The two target modes are:

- `dreamer_ppo_rl_noguard`: SimLingo + Dreamer PPO RL no-guard.
- `dreamer_sdbs_rl_noguard`: SimLingo + Dreamer SDBS RL no-guard.

The guarded production modes remain separate:

- `dreamer_ppo`: SimLingo + Dreamer PPO with runtime guard.
- `dreamer_sdbs`: SimLingo + Dreamer SDBS with runtime guard.

## PPO Bootstrap

The PPO no-guard checkpoint is initialized by offline distillation from three
clean, fully completed Dreamer-v1 guarded routes. The teacher guards are used
only to create successful supervision and are absent from the exported runtime.
The bootstrap is accepted only when a held-out split passes gate, recovery,
defer-to-SimLingo, and active-control error thresholds.

```bash
cd ~/Desktop/vla-av
conda run -n simlingo python scripts/pretrain_dreamer_rl_from_v1.py \
  --checkpoint external/simlingo/checkpoints/dreamer_ppo_rl_noguard/latest_rl_model.pt
```

This is behavior-cloning initialization followed by online PPO, not a claim
that the guard-generated demonstrations are themselves reinforcement learning.

## Online RL Meaning

A normal dashboard launch is evaluation only unless an online trainer is running.

In online RL, each Bench2Drive/SimLingo simulation is an episode or part of a rollout batch:

1. Launch a route/scenario in CARLA through SimLingo.
2. At each tick, record state, SimLingo base action, Dreamer chosen action, log-probability, value estimate, next state, and done flag.
3. Compute reward from the real environment outcome and step-level safety/progress signals.
4. Update the PPO/SDBS policy and world model from the new rollout.
5. Save the updated checkpoint back to the RL no-guard checkpoint slot.
6. Continue with the next route/scenario.

Training episodes that remain blocked for the configured tick threshold are
terminated as failed episodes and still sent to PPO. This changes only the
episode boundary; it never changes the vehicle command and is not a guard.

So repeated simulations become training only when the online trainer performs steps 2-5.

## Local Commands

Start a short PPO-only smoke run:

```bash
cd ~/Desktop/vla-av
DREAMER_ONLINE_RL_KIND=ppo DREAMER_ONLINE_RL_MAX_ROUTES=1 DREAMER_ONLINE_RL_MAX_WALL_SECONDS=300 \
  bash scripts/start_dreamer_online_rl_training.sh
```

Start the normal online run over PPO and SDBS:

```bash
cd ~/Desktop/vla-av
DREAMER_ONLINE_RL_KIND=both DREAMER_ONLINE_RL_MAX_ROUTES=6 DREAMER_ONLINE_RL_MAX_ROUTES_PER_BUCKET=1 \
  bash scripts/start_dreamer_online_rl_training.sh
```

Watch progress:

```bash
cd ~/Desktop/vla-av
bash scripts/watch_dreamer_online_rl_training.sh
```

The trainer backs up the current RL checkpoints in:

```text
logs/dreamer_online_rl/<run_id>/checkpoint_backups/
```

Then each episode updates:

```text
external/simlingo/checkpoints/dreamer_ppo_rl_noguard/latest_rl_model.pt
external/simlingo/checkpoints/dreamer_sdbs_rl_noguard/latest_rl_model.pt
```

## Reward Contract

Positive rewards:

- route progress;
- clean overtake completion: blocked front lane, safe pass, return to route/lane, no collision, no off-road;
- smooth control;
- traffic-rule compliance;
- recovery from blocked/stuck situations without infractions.

Negative rewards:

- collision with vehicles, pedestrians, cyclists, static objects;
- off-road / outside-route / lane departure;
- red-light or stop-sign violation;
- blocked-agent/stuck behavior;
- unsafe overtake attempt with close rear/side/oncoming traffic;
- harsh steering, throttle, or braking when unnecessary;
- route failure or timeout.

Guards are not part of the RL action decision. The only safety pressure in RL no-guard comes from these rewards.
