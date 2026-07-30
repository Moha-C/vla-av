# Step 20 - Alpamayo CARLA Planner

Alpamayo replaces the GR00T low-level action policy for autonomous-driving demos.
It is used as a native AV trajectory planner:

1. Alpamayo reads the front camera history, ego-motion history, and the driving instruction.
2. Alpamayo predicts a future ego trajectory.
3. `AlpamayoCarlaAdapter` tracks that trajectory with pure pursuit plus a speed controller.
4. CARLA still receives `steer/throttle/brake`, because that is CARLA's actuator API.

The CARLA actuator adapter is unavoidable, but the learned policy is no longer a
direct `steer/throttle/brake` imitation model.

## Install

```bash
cd ~/Desktop/vla-av/external
git clone https://github.com/NVlabs/alpamayo1.5.git
cd alpamayo1.5
uv python install 3.12
uv venv a1_5_venv --python 3.12
source a1_5_venv/bin/activate
uv sync --active --no-install-package flash-attn
hf auth login
```

The CARLA demo keeps running in the normal `vla-av`/`vla-av-step18` environment.
Alpamayo runs as a sidecar worker through `ALPAMAYO_PYTHON`, so we do not need
CARLA's Python package inside the Python 3.12 Alpamayo environment.

If the repo uses `a1_5_venv` instead of `.venv`, `start.sh` supports both.

`flash-attn` is skipped here on purpose because the demo defaults to eager
attention (`--alpamayo-attn-implementation eager`). Alpamayo 1.5 does not
currently dispatch through PyTorch SDPA in Transformers, and eager avoids needing
a local CUDA toolkit build setup.

## First Smoke Test

Use a larger camera than the old 224x224 Qwen/GR00T demos.

```bash
cd ~/Desktop/vla-av
CARLA_QUALITY=High ./start.sh --real --vla-control \
  --model alpamayo \
  --alpamayo-repo external/alpamayo1.5 \
  --alpamayo-model-path nvidia/Alpamayo-1.5-10B \
  --lane-assist 0 \
  --spawn-preset traffic_law \
  --camera-width 640 \
  --camera-height 360 \
  --camera-fov 95 \
  --target-speed-kmh 12 \
  --max-vla-throttle 0.30 \
  --alpamayo-plan-horizon 8 \
  --alpamayo-lookahead-index 8 \
  --instruction "Drive safely, follow the lane, obey red lights and stop signs, yield to pedestrians, cyclists, scooters, and other vehicles."
```

Overlay expectation:

```text
Alpamayo-1.5: trajectory ...ms | ... pts | refreshed/cached, queue=...
```

## Tuning Knobs

- If steering is too nervous: increase `--alpamayo-steering-smoothing 0.45`.
- If it is too slow: raise `--target-speed-kmh` and `--max-vla-throttle` slightly.
- If it cuts corners: raise `--alpamayo-lookahead-index`.
- If inference is too slow: lower `--alpamayo-max-generation-length` or increase `--alpamayo-plan-horizon`.

## Why An Adapter Still Exists

Alpamayo outputs a trajectory, not physical simulator pedal/steering commands.
CARLA can only move the ego vehicle through control APIs such as
`VehicleControl(steer, throttle, brake)` or similar actuation calls. The adapter
is therefore the simulator bridge, not the intelligence layer.
