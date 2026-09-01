# Dreamer PPO RL Curriculum Protocol

This protocol protects the validated 8 August checkpoint from all training.
The dashboard reads `production_model.pt`; learning writes only
`candidate_model.pt`. The immutable, hash-addressed source remains under
`protected_snapshots/` even after a future promotion.

## Training Contract

1. Distill only clean, completed SimLingo + guarded Dreamer-v1 demonstrations.
2. Freeze one candidate hash for every four-episode collection batch.
3. Balance clean success, justified waiting, collision, unsafe failure, and
   partial progress at episode level.
4. Bound rewards and use conservative PPO with a frozen-policy trust term.
5. Fit a map-invariant, normalized world model on real transitions while one
   whole episode is held out. Run short differentiable imagined rollouts only
   when held-out state, risk, and progress errors pass fixed thresholds.
6. Progress through simple accidents, oncoming traffic, dense traffic, then
   VRU scenarios.
7. On identical routes and seeds, compare native SimLingo against the same
   SimLingo assisted by the no-guard Dreamer candidate. The Dreamer is never
   evaluated as a standalone driver.
8. Roll back a stage unless the assisted system improves strictly over native
   SimLingo without adding collisions, off-road events, or rule violations.
9. Promote only after the candidate passes the full fixed multi-route,
   multi-seed suite. Equality is not sufficient, and a failed candidate never
   changes production.

## Commands

Start the detached autonomous campaign:

```bash
bash scripts/start_dreamer_curriculum_training.sh
```

Inspect progress without changing the run:

```bash
bash scripts/watch_dreamer_curriculum_training.sh
```

Stop the campaign and all of its CARLA children explicitly:

```bash
bash scripts/stop_dreamer_curriculum_training.sh
```

The final decision and every trace, metric, rollback, model hash, batch
manifest, and imagined-rollout summary are stored under
`logs/dreamer_curriculum/<run-id>/`.

`production_model.pt` remains the dashboard model throughout the campaign and
is used only to initialize or restore the candidate. Scientific performance is
measured against native SimLingo. Even a successfully trained candidate is not
copied to production unless all eight paired native/assisted route and seed
comparisons pass the promotion gate.
