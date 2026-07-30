# Step 2: CARLA -> Cosmos Transfer -> Alpamayo Dataset

Goal: rebuild the fine-tuning data from clean CARLA expert runs, then train or
adapt Alpamayo on photorealistic Cosmos-Transfer2.5 frames instead of raw CARLA
graphics.

## What Is Kept

The validated Transfer2.5 videos are still in the main tree:

```text
data/synthetic/transferred_real/transfer25_20260506_155946/transfer_output/
data/synthetic/transferred_real/transfer25_hood_pov_480/transfer_output/
```

Old fine-tuning images, old raw CARLA episodes, Cosmos-Predict generations, and
GR00T/Qwen-era datasets were moved to `backup_1ere_version/`.

## Local Capture Command

Use one base CARLA map/session, high-quality CARLA, hood/windshield camera, and
expert autopilot labels. This records RGB, semantic, depth, per-frame
instruction/action metadata, then optionally runs Transfer2.5.

```bash
cd ~/Desktop/vla-av

CARLA_QUALITY=Epic PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/cosmos_transfer_real.py \
  --run-name transfer25_dataset_base_001 \
  --camera-preset hood \
  --frames 49 \
  --fps 10 \
  --width 960 \
  --height 540 \
  --spawn-preset traffic_law \
  --vehicles 30 \
  --two-wheelers 16 \
  --walkers 70 \
  --pedestrian-cross-factor 0.95 \
  --traffic-speed-difference 20 \
  --ego-speed-difference 10 \
  --weather "clear natural daytime urban dashcam, realistic exposure, real city buildings, neutral colors, no cyberpunk, no handheld camera" \
  --instruction "Drive like a safe autonomous vehicle in an urban environment. Follow the current lane and road markings, keep a smooth centered trajectory, respect speed limits, obey red lights, green lights, stop signs, lane arrows, crosswalks, priority rules, and right-of-way. Yield to pedestrians, cyclists, scooters, motorbikes, parked cars pulling out, and other vehicles. Stop when the path is blocked, wait until it is clear, then continue smoothly without leaving the drivable lane." \
  --negative-prompt "CGI, video game, cyberpunk, neon glow, handheld camera, phone in hand, dashcam holder, cartoon, anime, oversaturated colors, distorted buildings, warped lane markings, melted texture, flicker, motion smear, low quality" \
  --guidance 6 \
  --seg-weight 1.0 \
  --depth-weight 0 \
  --vis-weight 0 \
  --transfer-resolution 480 \
  --transfer-max-frames 49 \
  --transfer-num-steps 24 \
  --no-keep-input-resolution \
  --run-transfer
```

For more diversity, repeat with different `--run-name`, `--spawn-index` or
`--scenario-seed`, and weather prompts such as:

```text
clear natural daytime urban dashcam, realistic exposure, real city buildings
cloudy morning realistic dashcam, neutral colors, real road texture
rainy day realistic dashcam, wet asphalt, realistic reflections
wet road after rain natural dashcam, overcast sky, realistic exposure
foggy early morning natural dashcam, mild haze, visible road markings
golden hour natural dashcam, warm sunlight, realistic shadows
night urban dashcam, realistic headlights, realistic street lights, no neon cyberpunk
```

## Build The Alpamayo Manifest

After Transfer2.5 finishes, extract the photoreal frames and align them with the
CARLA autopilot labels and ego trajectories:

```bash
python scripts/prepare_alpamayo_transfer_dataset.py \
  --runs-dir data/synthetic/transferred_real \
  --run-glob "transfer25_dataset_*" \
  --output-dir data/alpamayo_transfer_dataset \
  --history-steps 16 \
  --future-steps 64 \
  --dt 0.1 \
  --camera-index 1
```

This writes:

```text
data/alpamayo_transfer_dataset/manifest.jsonl
data/alpamayo_transfer_dataset/summary.json
data/alpamayo_transfer_dataset/images/<run-name>/frame_000000.jpg
```

Each manifest row contains:

- photoreal image path
- the policy driving instruction used for training
- CARLA autopilot `steering`, `throttle`, `brake`
- traffic-light / stop-sign metadata
- ego pose and speed
- local-frame `ego_history_xyz` and `ego_future_xyz`

The VLA should be trained on the Cosmos-Transfer frame as the visual input, not
the raw CARLA frame. CARLA remains the expert that provides aligned actions,
traffic-rule metadata, and future ego motion.

## Fine-Tuning Decision

For Alpamayo-1.5-10B, a real fine-tune is better on the B200 cloud than on the
local workstation. Locally, we can validate the data and run inference. On the
B200 cloud, the likely best path is:

1. Upload `data/alpamayo_transfer_dataset/`.
2. Use the Alpamayo repo environment plus the manifest.
3. Train an adapter/LoRA or a trajectory head against `ego_future_xyz`, keeping
   the base model mostly frozen unless the cloud budget allows a larger run.
4. Export the checkpoint back under `checkpoints/alpamayo_transfer_v1/`.

Once the cloud UI/VM details are known, the dataset produced here is the payload
we should move there.

## B200 VM Workflow

The cloud workflow is prepared in:

```text
cloud_b200/
```

Local machine:

```bash
cd ~/Desktop/vla-av
bash cloud_b200/pack_project_for_b200.sh
rsync -avP artifacts/vla-av-b200-project_*.tar.zst USER@VM_HOST:~/
```

VM:

```bash
ssh USER@VM_HOST
sudo apt update
sudo apt install -y zstd
tar --use-compress-program=unzstd -xf ~/vla-av-b200-project_*.tar.zst -C ~/
cd ~/vla-av
export HF_TOKEN="hf_xxx"
bash cloud_b200/setup_b200_vm.sh
```

Smoke test:

```bash
RUNS=2 FRAMES=49 bash cloud_b200/run_big_dataset.sh
```

Serious run:

```bash
RUNS=300 FRAMES=49 bash cloud_b200/run_big_dataset.sh
```

Large run:

```bash
RUNS=800 FRAMES=49 bash cloud_b200/run_big_dataset.sh
```

Outputs to retrieve:

```text
artifacts/alpamayo_transfer_dataset_b200.tar.zst
data/alpamayo_transfer_dataset_b200/
```
