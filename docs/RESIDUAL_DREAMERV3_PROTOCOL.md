# Residual DreamerV3 for SimLingo

## Status

This is a new isolated branch. It does not replace the working SimLingo,
Dreamer guard, SUMO bridge, dashboard, or historical checkpoints. It is not
exposed as a driving option until it passes every gate below.

The implementation is a PyTorch/Python 3.8 adaptation of DreamerV3 mechanisms,
not a copy of the official JAX agent and not a claim that the current candidate
already outperforms SimLingo.

## Intended controller

SimLingo remains the reference driver and produces
`a_native = [steer, throttle, brake]`. The learned actor emits two bounded
residuals and a continuous authority:

```text
delta_t, alpha_t = actor(RSSM_state_t, observation_t)
a_dreamer_t       = physical_control(a_native_t + delta_t)
a_final_t         = physical_control(a_native_t + alpha_t * delta_t)
```

`alpha_t` starts at 0.02. There is no geometric veto, turn guard, forced gap,
hard-coded overtake state machine, or threshold that silently changes the
action. Safety must be learned through real transition outcomes and imagined
returns. Candidate checkpoints are physically shadow-only.

## Data contract

Only ordered, physically executed, native SimLingo transitions are accepted:

- exact normalized 32D observation captured at time `t`;
- final CARLA control applied at `t`;
- observation and progress at `t+1`;
- synchronized collision/off-road events when available;
- authoritative Bench2Drive terminal metrics;
- route, town, scenario and seed provenance;
- SHA-256 digest for every source trace.

Splits are made by seed before temporal windows are created. Train, validation,
and test are disjoint. The current audited matrix contains 12 accepted episodes
and 8,707 transitions across Town10HD, Town12 and Town13. One incomplete route
is rejected instead of being silently treated as a success.

During training only, synchronized incident windows receive sampling weight 32
and high-risk windows weight 4. Validation and test remain unweighted. This
addresses rare-event imbalance without inventing a collision, changing its
timestamp, or modifying the runtime policy.

To inspect the frozen split:

```bash
cd ~/Desktop/vla-av
RESIDUAL_DREAMERV3_PHASE=inspect \
  bash scripts/run_residual_dreamerv3_pipeline.sh
```

Additional native trace trees can be supplied with repeated `--trace` options.
They must satisfy the same provenance checks.

## Architecture

The world model contains:

- symlog MLP encoder and decoder;
- GRU deterministic recurrent state;
- 16 x 16 categorical stochastic state with straight-through samples;
- prior conditioned on the physically applied action;
- posterior conditioned on the next observation;
- 1% uniform mixture, balanced KL and free nats;
- two-hot reward head;
- continuation, collision and off-road heads.

The behavior module contains a stochastic residual actor, two-hot critic and
slow critic. Actor and critic train only in futures imagined by a frozen world
model that has passed validation.

## Mandatory gates

### Gate 1: data

- no unknown policy source;
- no non-native control;
- no missing Bench2Drive ground truth;
- no incomplete 32D observation;
- no seed overlap;
- enough seeds in every split.

### Gate 2: world model

On frozen horizons 1, 5, 10 and 20, the RSSM must beat both:

1. observation persistence with train-set scalar priors;
2. an action-conditioned ridge dynamics model.

The comparison uses train-set normalization and open-loop action rollouts. The
RSSM must improve normalized observation MSE by at least 2%, not regress reward
MAE, remain finite, and react measurably to brake/accelerate/left/right action
counterfactuals. A failed gate forbids actor training.

### Gate 3: actor candidate

The actor trains in latent imagination with continuation-aware lambda returns,
return normalization, entropy, intervention cost and action-change cost. The
world model stays frozen. The output remains `status=candidate` and
`control_allowed=false`.

An explicit `allow_candidate_evaluation=True` runtime exists solely for the
paired closed-loop campaign. It is not the dashboard default and is reported
as `evaluation_only=true` on every step. Shadow and evaluation modes are
mutually exclusive.

### Gate 4: closed loop promotion

Promotion requires at least six paired seeds under identical routes and
conditions. Compared with native SimLingo, driving score and route completion
must not decrease, and collision/km and off-road rate must not increase. A
candidate loaded in shadow returns the native SimLingo action regardless of its
proposal, so it cannot accidentally control CARLA.

## Training

Durable offline launch:

```bash
cd ~/Desktop/vla-av
bash scripts/start_residual_dreamerv3_training.sh
```

Progress:

```bash
cd ~/Desktop/vla-av
bash scripts/watch_residual_dreamerv3_training.sh
```

Short CPU diagnostic, which is expected to fail the model gate:

```bash
cd ~/Desktop/vla-av
RESIDUAL_DREAMERV3_DEVICE=cpu \
RESIDUAL_DREAMERV3_WORLD_EPOCHS=1 \
RESIDUAL_DREAMERV3_MAX_WINDOWS=64 \
RESIDUAL_DREAMERV3_PHASE=world-model \
RESIDUAL_DREAMERV3_OUTPUT=/tmp/residual_dreamerv3_smoke \
  bash scripts/run_residual_dreamerv3_pipeline.sh
```

Manual evaluation of an existing world-model candidate:

```bash
~/miniconda3/envs/simlingo/bin/python scripts/train_residual_dreamerv3.py evaluate \
  --world-checkpoint checkpoints/residual_dreamerv3/candidate/world_model_candidate.pt \
  --output checkpoints/residual_dreamerv3/candidate \
  --device cuda
```

Promotion is deliberately separate:

```bash
~/miniconda3/envs/simlingo/bin/python scripts/train_residual_dreamerv3.py promote \
  --actor-checkpoint checkpoints/residual_dreamerv3/candidate/actor_candidate.pt \
  --closed-loop-report PATH/closed_loop_eval.json \
  --promoted-output checkpoints/residual_dreamerv3/production/residual_dreamerv3.pt \
  --output checkpoints/residual_dreamerv3/candidate
```

The report must follow
`configs/residual_dreamerv3_closed_loop_eval.example.json`. Editing the status
inside a checkpoint is not a valid promotion.

## Tests

```bash
~/miniconda3/envs/simlingo/bin/python -m unittest -v \
  tests.test_residual_dreamerv3
```

The tests cover transforms, tensor losses, physical controls, initial
authority, consecutive windows, seed leakage, frozen baselines, action-collapse
rejection, continuation-aware returns, shadow isolation and promotion rules.

## Honest limitations

- 8,707 transitions are enough to validate the pipeline, not enough to claim a
  robust general driving world model.
- Only five current episodes contain synchronized collision events; more clean
  and failed native trajectories are needed for calibrated rare-event heads.
- Compact privileged state is used. This is not camera-only Dreamer.
- Offline prediction superiority does not prove better driving. Only the paired
  closed-loop campaign can establish that result.
- Until a checkpoint is promoted, the dashboard and production SimLingo path
  must remain unchanged.

## References

- Official DreamerV3 implementation: https://github.com/danijar/dreamerv3
- DreamerV3 paper: https://arxiv.org/abs/2301.04104
