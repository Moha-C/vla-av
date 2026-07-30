"""Live CARLA camera demo for the first VLA-AV end-to-end milestone."""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.carla_env.carla_client import CarlaClient, CarlaClientConfig
from src.carla_env.data_collector import DataCollector, EpisodeConfig
from src.carla_env.sensors import RGBCameraConfig, RGBCameraSensor
from src.carla_env.visualizer import PygameVisualizer, VisualizerConfig
from src.models import VLAConfig, VLAModel, VLMBackboneConfig, apply_lora_adapters


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for CARLA connection, camera, and display settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        default=8000,
        help="CARLA traffic manager port.",
    )
    parser.add_argument("--map", dest="map_name", default=None, help="Optional map name.")
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=None,
        help="Optional deterministic ego spawn point index.",
    )
    parser.add_argument(
        "--spawn-preset",
        default=None,
        choices=("straight", "straight_turn", "junction", "traffic_law"),
        help="Choose an automatic spawn preset, e.g. a long straight road, a turn, or traffic-law scene.",
    )
    parser.add_argument(
        "--demo-vehicles",
        type=int,
        default=0,
        help="Spawn this many NPC vehicles in the live demo.",
    )
    parser.add_argument(
        "--demo-two-wheelers",
        type=int,
        default=0,
        help="Spawn this many bikes/motorcycles/scooters in the live demo.",
    )
    parser.add_argument(
        "--demo-walkers",
        type=int,
        default=0,
        help="Spawn this many pedestrian walkers in the live demo.",
    )
    parser.add_argument(
        "--demo-pedestrian-cross-factor",
        type=float,
        default=0.35,
        help="CARLA pedestrian crossing probability used by live demo walkers.",
    )
    parser.add_argument(
        "--demo-traffic-speed-difference",
        type=float,
        default=15.0,
        help="Traffic Manager speed difference percentage for live demo NPCs.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run CARLA in synchronous mode with a fixed delta time.",
    )
    parser.add_argument(
        "--tick-timeout",
        type=float,
        default=0.1,
        help="Max seconds to wait for one CARLA async tick before refreshing pygame.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run. Use 0 to run until the window is closed.",
    )
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--camera-tick", type=float, default=0.05)
    parser.add_argument(
        "--instruction",
        default="Drive safely and follow the lane.",
        help="Instruction text shown in the overlay.",
    )
    parser.add_argument(
        "--backbone",
        default="dummy",
        choices=(
            "dummy",
            "qwen2_vl",
            "qwen2-vl",
            "qwen",
            "cosmos_reason",
            "cosmos-reason",
            "reason2",
            "qwen3_vl",
            "qwen3-vl",
        ),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"),
        help="Backbone dtype. auto uses bf16 for Cosmos-Reason and fp16 otherwise.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for the VLA model. Defaults to cuda for qwen2_vl and auto for dummy.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face models from the local cache only.",
    )
    parser.add_argument(
        "--sync-inference",
        action="store_true",
        help="Run VLA inference on the pygame thread. Useful only for debugging.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint containing action_head_state_dict.",
    )
    parser.add_argument(
        "--model",
        default="vla",
        choices=("vla", "groot", "isaac_groot", "alpamayo", "alpamayo_r1"),
        help=(
            "Model architecture: vla=MLP action head, groot=local dual-system diffusion, "
            "isaac_groot=official Isaac GR00T policy adapter, "
            "alpamayo=NVIDIA Alpamayo 1.5 AV trajectory planner, "
            "alpamayo_r1=fine-tuned official NVIDIA Alpamayo R1 planner."
        ),
    )
    parser.add_argument("--groot-model-path", default="checkpoints/isaac_groot_carla_v1")
    parser.add_argument("--groot-repo", default="external/Isaac-GR00T-git")
    parser.add_argument("--groot-embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--groot-device", default=None)
    parser.add_argument("--groot-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--groot-action-key", default="vehicle_control")
    parser.add_argument("--groot-action-horizon-index", type=int, default=0)
    parser.add_argument("--groot-execute-horizon", type=int, default=4)
    parser.add_argument("--groot-max-steering-delta", type=float, default=0.08)
    parser.add_argument("--groot-max-throttle-delta", type=float, default=0.08)
    parser.add_argument("--groot-max-brake-delta", type=float, default=0.18)
    parser.add_argument("--groot-steering-scale", type=float, default=1.0)
    parser.add_argument("--groot-throttle-scale", type=float, default=1.0)
    parser.add_argument("--groot-brake-scale", type=float, default=1.0)
    parser.add_argument("--alpamayo-model-path", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--alpamayo-repo", default="external/alpamayo1.5")
    parser.add_argument("--alpamayo-python", default=None)
    parser.add_argument(
        "--alpamayo-r1-model-path",
        default="vm_backups/official_sft/intermediate/stage2/checkpoint-10528",
    )
    parser.add_argument("--alpamayo-r1-repo", default="external/alpamayo_official")
    parser.add_argument("--alpamayo-r1-python", default=None)
    parser.add_argument(
        "--alpamayo-r1-attn-implementation",
        default="eager",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument("--alpamayo-device", default=None)
    parser.add_argument(
        "--alpamayo-dtype",
        default="bfloat16",
        choices=("float16", "fp16", "bfloat16", "bf16", "float32", "fp32"),
    )
    parser.add_argument(
        "--alpamayo-attn-implementation",
        default="eager",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument("--alpamayo-num-frames", type=int, default=4)
    parser.add_argument("--alpamayo-history-steps", type=int, default=16)
    parser.add_argument("--alpamayo-plan-horizon", type=int, default=8)
    parser.add_argument("--alpamayo-lookahead-index", type=int, default=8)
    parser.add_argument("--alpamayo-num-traj-samples", type=int, default=1)
    parser.add_argument("--alpamayo-max-generation-length", type=int, default=256)
    parser.add_argument("--alpamayo-temperature", type=float, default=0.6)
    parser.add_argument("--alpamayo-top-p", type=float, default=0.98)
    parser.add_argument("--alpamayo-nav-text", default=None)
    parser.add_argument(
        "--alpamayo-use-instruction-as-nav",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--alpamayo-target-speed-kmh", type=float, default=None)
    parser.add_argument("--alpamayo-max-throttle", type=float, default=None)
    parser.add_argument("--alpamayo-max-brake", type=float, default=0.70)
    parser.add_argument("--alpamayo-steering-smoothing", type=float, default=0.35)
    parser.add_argument("--alpamayo-throttle-smoothing", type=float, default=0.25)
    parser.add_argument("--alpamayo-brake-smoothing", type=float, default=0.20)
    parser.add_argument(
        "--alpamayo-action-adapter-checkpoint",
        default=None,
        help="Optional local action-adapter checkpoint trained from CARLA/Cosmos labels.",
    )
    parser.add_argument(
        "--alpamayo-action-adapter-blend",
        type=float,
        default=0.0,
        help="Blend between deterministic Alpamayo tracking and local action adapter.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=10,
        help="DDPM denoising steps for --model groot.",
    )
    parser.add_argument(
        "--system2-cache-every",
        type=int,
        default=5,
        help="Refresh the VLM/System 2 embedding every N VLA predictions for --model groot.",
    )
    parser.add_argument("--diffusion-hidden-dim", type=int, default=256)
    parser.add_argument("--diffusion-layers", type=int, default=2)
    parser.add_argument("--diffusion-heads", type=int, default=4)
    parser.add_argument("--action-smoothing", type=float, default=0.25)
    parser.add_argument(
        "--stochastic-system1",
        action="store_true",
        help="Use random DDPM sampling instead of deterministic warm-start for System 1.",
    )
    parser.add_argument(
        "--vla-control",
        action="store_true",
        help="Apply VLA actions to the CARLA ego vehicle instead of shadow inference.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show side-by-side VLA versus autopilot expert actions.",
    )
    parser.add_argument(
        "--safety-max-speed",
        type=float,
        default=80.0,
        help="Safety brake threshold in km/h for VLA control.",
    )
    parser.add_argument(
        "--off-road-distance",
        type=float,
        default=3.0,
        help="Safety brake threshold in meters from nearest driving waypoint.",
    )
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=10.0,
        help="Seconds between VLA control metric reports.",
    )
    parser.add_argument(
        "--brake-deadzone",
        type=float,
        default=0.05,
        help="Brake values below this are treated as zero in VLA control.",
    )
    parser.add_argument(
        "--throttle-deadzone",
        type=float,
        default=0.03,
        help="Throttle values below this are treated as zero in VLA control.",
    )
    parser.add_argument(
        "--min-start-throttle",
        type=float,
        default=0.45,
        help="Minimum throttle applied while VLA starts from near-zero speed.",
    )
    parser.add_argument(
        "--release-stuck-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let the demo gently release VLA brake after a stale stop when the path is clear.",
    )
    parser.add_argument(
        "--release-after-seconds",
        type=float,
        default=1.5,
        help="Seconds to wait before release-stuck-stop can override a stale VLA brake.",
    )
    parser.add_argument(
        "--release-throttle",
        type=float,
        default=0.35,
        help="Throttle used by release-stuck-stop when green/clear.",
    )
    parser.add_argument(
        "--release-max-speed-kmh",
        type=float,
        default=4.0,
        help="Only release stale stops while the ego car is below this speed.",
    )
    parser.add_argument(
        "--release-hazard-distance",
        type=float,
        default=9.0,
        help="Do not release stale stops when a vehicle, bike, or pedestrian is this close ahead.",
    )
    parser.add_argument(
        "--bootstrap-throttle",
        type=float,
        default=0.0,
        help="Optional throttle used until the first real model action is ready.",
    )
    parser.add_argument(
        "--disable-bootstrap-action",
        action="store_true",
        help="Do not use the VLA action-head prior while waiting for real model inference.",
    )
    parser.add_argument(
        "--warmup-autopilot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep CARLA autopilot active until the first real async VLA action arrives.",
    )
    parser.add_argument(
        "--target-speed-kmh",
        type=float,
        default=25.0,
        help="Target cruise speed used by the VLA safety speed governor.",
    )
    parser.add_argument(
        "--autopilot-demo-speed-difference",
        type=float,
        default=45.0,
        help=(
            "Traffic Manager speed difference percentage for the ego vehicle when "
            "autopilot is used for a VLA-style assisted demo. Higher means slower."
        ),
    )
    parser.add_argument(
        "--autopilot-demo-hesitation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add occasional light braking while autopilot drives, for a cautious VLA-style demo.",
    )
    parser.add_argument(
        "--autopilot-demo-label",
        default="CARLA autopilot",
        help="Overlay label shown when the CARLA autopilot is used as the assisted driver.",
    )
    parser.add_argument(
        "--nav-maneuver",
        default="follow_lane",
        choices=("auto", "follow_lane", "straight", "left", "right"),
        help=(
            "High-level navigation intent injected into the VLA prompt. "
            "Use left/right/straight to test a specific maneuver."
        ),
    )
    parser.add_argument(
        "--route-target-nav",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject a SimLingo-style local route target point into the VLA prompt. "
            "This gives Alpamayo an explicit navigation objective without retraining."
        ),
    )
    parser.add_argument(
        "--route-target-distance",
        type=float,
        default=18.0,
        help="Meters ahead for the local route target point used by --route-target-nav.",
    )
    parser.add_argument(
        "--route-target-scan-distance",
        type=float,
        default=55.0,
        help="Meters to scan for the next branch/junction when selecting left/right/straight.",
    )
    parser.add_argument(
        "--route-target-steer-blend",
        type=float,
        default=0.0,
        help=(
            "Blend Alpamayo steering with a controller that points at the local route "
            "target. This is the closed-loop navigation adapter, separate from lane assist."
        ),
    )
    parser.add_argument(
        "--max-vla-throttle",
        type=float,
        default=0.45,
        help="Maximum throttle allowed while VLA control is active.",
    )
    parser.add_argument(
        "--lane-assist",
        type=float,
        default=0.65,
        help="Blend waypoint steering into VLA steering. Use 0 for pure VLA control.",
    )
    parser.add_argument(
        "--lane-lookahead",
        type=float,
        default=8.0,
        help="Waypoint lookahead distance in meters for lane assist.",
    )
    parser.add_argument(
        "--start-speed-threshold",
        type=float,
        default=1.0,
        help="Speed below which min-start-throttle launch assist can activate.",
    )
    parser.add_argument(
        "--red-team-host",
        default="0.0.0.0",
        help="UDP host used by the SUMO red-team attack listener.",
    )
    parser.add_argument(
        "--red-team-port",
        type=int,
        default=5005,
        help="UDP port used by the SUMO red-team attack listener.",
    )
    parser.add_argument(
        "--red-team-ttl",
        type=float,
        default=5.0,
        help="Seconds before a SUMO attack message expires. Use 0 to keep it active.",
    )
    parser.add_argument(
        "--disable-red-team-listener",
        action="store_true",
        help="Disable the UDP listener for live SUMO red-team attacks.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=896,
        help="Pygame window width in pixels.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=672,
        help="Pygame window height in pixels.",
    )
    parser.add_argument(
        "--record-video",
        default=None,
        help="Optional path to save the pygame demo window as an MP4 video.",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=30.0,
        help="FPS metadata for --record-video.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def build_carla_client(args: argparse.Namespace) -> CarlaClient:
    """Create the configured CARLA lifecycle manager."""

    config = CarlaClientConfig(
        host=args.host,
        port=args.port,
        map_name=args.map_name,
        synchronous_mode=args.sync,
        tick_timeout_seconds=args.tick_timeout,
        traffic_manager_port=args.traffic_manager_port,
        ego_spawn_index=args.spawn_index,
        ego_spawn_preset=args.spawn_preset,
        autopilot=True,
    )
    return CarlaClient(config)


def build_camera(args: argparse.Namespace, client: CarlaClient) -> RGBCameraSensor:
    """Create the RGB camera wrapper used by the visualizer."""

    config = RGBCameraConfig(
        width=args.camera_width,
        height=args.camera_height,
        fov=args.camera_fov,
        sensor_tick=args.camera_tick,
    )
    return RGBCameraSensor(config, client=client)


def build_visualizer(args: argparse.Namespace) -> PygameVisualizer:
    """Create the pygame window used for live inspection."""

    config = VisualizerConfig(window_size=(args.window_width, args.window_height))
    return PygameVisualizer(config)


def _open_demo_video_writer(args: argparse.Namespace) -> tuple[Any, Any]:
    """Create an OpenCV writer for recording the pygame window."""

    import cv2

    output_path = Path(args.record_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        float(args.record_fps),
        (int(args.window_width), int(args.window_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    LOGGER.info("Recording pygame demo window to %s", output_path)
    return writer, cv2


def build_demo_scenario_spawner(
    args: argparse.Namespace,
    client: CarlaClient,
) -> Optional[DataCollector]:
    """Reuse the training scenario spawner to populate live demo scenes."""

    if (
        args.demo_vehicles <= 0
        and args.demo_two_wheelers <= 0
        and args.demo_walkers <= 0
    ):
        return None

    episode_config = EpisodeConfig(
        n_episodes=1,
        frames_per_episode=1,
        instruction=args.instruction,
        seed=42,
        traffic_manager_port=args.traffic_manager_port,
        npc_vehicles=max(0, args.demo_vehicles),
        npc_two_wheelers=max(0, args.demo_two_wheelers),
        npc_walkers=max(0, args.demo_walkers),
        pedestrian_cross_factor=args.demo_pedestrian_cross_factor,
        traffic_speed_difference=args.demo_traffic_speed_difference,
        weather_presets=(),
    )
    return DataCollector(episode_config, carla_config=client.config)


def _is_cosmos_reason_backbone(backbone: str) -> bool:
    return backbone in {"cosmos_reason", "cosmos-reason", "reason2", "qwen3_vl", "qwen3-vl"}


def _is_real_transformer_backbone(backbone: str) -> bool:
    return backbone in {"qwen2_vl", "qwen2-vl", "qwen"} or _is_cosmos_reason_backbone(backbone)


def _resolve_backbone_dtype(backbone: str, dtype: str) -> str:
    if dtype != "auto":
        return dtype
    return "bfloat16" if _is_cosmos_reason_backbone(backbone) else "float16"


def _resolve_model_name(backbone: str, model_name: str) -> str:
    if _is_cosmos_reason_backbone(backbone) and model_name == "Qwen/Qwen2-VL-7B-Instruct":
        return "nvidia/Cosmos-Reason2-2B"
    return model_name


def build_vla_model(args: argparse.Namespace) -> Any:
    """Create either the dummy VLA model or a real transformer VLM model."""

    if args.model == "isaac_groot":
        from src.models.isaac_groot_adapter import (
            IsaacGrootAdapterConfig,
            IsaacGrootCarlaAdapter,
        )

        device = args.groot_device or args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Isaac GR00T demo requires CUDA for practical use, but "
                f"torch.cuda.is_available()={torch.cuda.is_available()}."
            )
        config = IsaacGrootAdapterConfig(
            model_path=args.groot_model_path,
            repo_path=args.groot_repo,
            embodiment_tag=args.groot_embodiment_tag,
            device=str(device),
            strict=bool(args.groot_strict),
            action_key=args.groot_action_key,
            action_horizon_index=args.groot_action_horizon_index,
            execute_action_horizon=args.groot_execute_horizon,
            action_smoothing=args.action_smoothing,
            max_steering_delta=args.groot_max_steering_delta,
            max_throttle_delta=args.groot_max_throttle_delta,
            max_brake_delta=args.groot_max_brake_delta,
            steering_scale=args.groot_steering_scale,
            throttle_scale=args.groot_throttle_scale,
            brake_scale=args.groot_brake_scale,
            max_throttle=args.max_vla_throttle,
        )
        LOGGER.info(
            "Building official Isaac GR00T policy adapter: model=%s repo=%s embodiment=%s",
            config.model_path,
            config.repo_path,
            config.embodiment_tag,
        )
        return IsaacGrootCarlaAdapter(config)

    if args.model == "alpamayo":
        from src.models.alpamayo_adapter import (
            AlpamayoAdapterConfig,
            AlpamayoCarlaAdapter,
        )

        device = args.alpamayo_device or args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Alpamayo demo requires CUDA for practical use, but "
                f"torch.cuda.is_available()={torch.cuda.is_available()}."
            )
        target_speed = (
            float(args.alpamayo_target_speed_kmh)
            if args.alpamayo_target_speed_kmh is not None
            else float(args.target_speed_kmh)
        )
        max_throttle = (
            float(args.alpamayo_max_throttle)
            if args.alpamayo_max_throttle is not None
            else float(args.max_vla_throttle)
        )
        config = AlpamayoAdapterConfig(
            model_path=args.alpamayo_model_path,
            repo_path=args.alpamayo_repo,
            python_path=args.alpamayo_python,
            device=str(device),
            dtype=args.alpamayo_dtype,
            attn_implementation=args.alpamayo_attn_implementation,
            num_frames_per_camera=args.alpamayo_num_frames,
            num_history_steps=args.alpamayo_history_steps,
            num_traj_samples=args.alpamayo_num_traj_samples,
            max_generation_length=args.alpamayo_max_generation_length,
            top_p=args.alpamayo_top_p,
            temperature=args.alpamayo_temperature,
            nav_text=args.alpamayo_nav_text,
            use_instruction_as_nav=bool(args.alpamayo_use_instruction_as_nav),
            plan_horizon=args.alpamayo_plan_horizon,
            lookahead_index=args.alpamayo_lookahead_index,
            target_speed_kmh=target_speed,
            max_throttle=max_throttle,
            max_brake=float(args.alpamayo_max_brake),
            steering_smoothing=float(args.alpamayo_steering_smoothing),
            throttle_smoothing=float(args.alpamayo_throttle_smoothing),
            brake_smoothing=float(args.alpamayo_brake_smoothing),
            action_adapter_checkpoint=args.alpamayo_action_adapter_checkpoint,
            action_adapter_blend=float(args.alpamayo_action_adapter_blend),
        )
        LOGGER.info(
            "Building Alpamayo AV planner adapter: model=%s repo=%s",
            config.model_path,
            config.repo_path,
        )
        return AlpamayoCarlaAdapter(config)

    if args.model == "alpamayo_r1":
        from src.models.alpamayo_r1_adapter import (
            AlpamayoR1AdapterConfig,
            AlpamayoR1CarlaAdapter,
        )

        device = args.alpamayo_device or args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Alpamayo R1 demo requires CUDA for practical use, but "
                f"torch.cuda.is_available()={torch.cuda.is_available()}."
            )
        target_speed = (
            float(args.alpamayo_target_speed_kmh)
            if args.alpamayo_target_speed_kmh is not None
            else float(args.target_speed_kmh)
        )
        max_throttle = (
            float(args.alpamayo_max_throttle)
            if args.alpamayo_max_throttle is not None
            else float(args.max_vla_throttle)
        )
        config = AlpamayoR1AdapterConfig(
            model_path=args.alpamayo_r1_model_path,
            repo_path=args.alpamayo_r1_repo,
            python_path=args.alpamayo_r1_python,
            device=str(device),
            dtype=args.alpamayo_dtype,
            attn_implementation=args.alpamayo_r1_attn_implementation,
            num_frames_per_camera=args.alpamayo_num_frames,
            num_history_steps=args.alpamayo_history_steps,
            num_traj_samples=args.alpamayo_num_traj_samples,
            max_generation_length=args.alpamayo_max_generation_length,
            top_p=args.alpamayo_top_p,
            temperature=args.alpamayo_temperature,
            nav_text=args.alpamayo_nav_text,
            use_instruction_as_nav=bool(args.alpamayo_use_instruction_as_nav),
            plan_horizon=args.alpamayo_plan_horizon,
            lookahead_index=args.alpamayo_lookahead_index,
            target_speed_kmh=target_speed,
            max_throttle=max_throttle,
            max_brake=float(args.alpamayo_max_brake),
            steering_smoothing=float(args.alpamayo_steering_smoothing),
            throttle_smoothing=float(args.alpamayo_throttle_smoothing),
            brake_smoothing=float(args.alpamayo_brake_smoothing),
            action_adapter_checkpoint=args.alpamayo_action_adapter_checkpoint,
            action_adapter_blend=float(args.alpamayo_action_adapter_blend),
        )
        LOGGER.info(
            "Building fine-tuned Alpamayo R1 planner adapter: model=%s repo=%s",
            config.model_path,
            config.repo_path,
        )
        return AlpamayoR1CarlaAdapter(config)

    device = args.device
    if device is None:
        device = "cuda" if _is_real_transformer_backbone(args.backbone) else "auto"
    if _is_real_transformer_backbone(args.backbone) and not torch.cuda.is_available():
        raise RuntimeError(
            "Real VLM demo requires CUDA, but this Python environment has "
            f"torch.cuda.is_available()={torch.cuda.is_available()} and "
            f"torch.version.cuda={torch.version.cuda!r}. Repair the vla-av PyTorch "
            "CUDA install before running --real."
        )
    dtype = _resolve_backbone_dtype(args.backbone, args.dtype)
    model_name = _resolve_model_name(args.backbone, args.model_name)

    checkpoint = None
    checkpoint_path: Optional[Path] = None
    effective_model = args.model
    checkpoint_head_config = {}
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        LOGGER.info("Loading checkpoint metadata from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_head_type = str(checkpoint.get("action_head_type", "")).lower()
        checkpoint_head_config = checkpoint.get("action_head_config", {}) or {}
        if args.model == "vla" and checkpoint_head_type in {"diffusion", "groot"}:
            effective_model = "groot"
            LOGGER.info("Checkpoint uses diffusion action head; enabling --model groot automatically.")
    embedding_dim = int(checkpoint_head_config.get("embedding_dim", args.embedding_dim))
    diffusion_steps = int(checkpoint_head_config.get("n_steps", args.diffusion_steps))
    diffusion_hidden_dim = int(checkpoint_head_config.get("hidden_dim", args.diffusion_hidden_dim))
    diffusion_layers = int(checkpoint_head_config.get("n_layers", args.diffusion_layers))
    diffusion_heads = int(checkpoint_head_config.get("n_heads", args.diffusion_heads))

    config = VLAConfig(
        backbone=VLMBackboneConfig(
            backend=args.backbone,
            model_name=model_name,
            device=device,
            dtype=dtype,
            embedding_dim=embedding_dim,
            freeze=True,
            pooling="eos",
            add_generation_prompt=False,
            local_files_only=args.local_files_only,
            trust_remote_code=_is_cosmos_reason_backbone(args.backbone),
        ),
        action_head_type="diffusion" if effective_model == "groot" else "mlp",
        diffusion_steps=diffusion_steps,
        diffusion_hidden_dim=diffusion_hidden_dim,
        diffusion_layers=diffusion_layers,
        diffusion_heads=diffusion_heads,
        system2_cache_every=args.system2_cache_every,
        action_smoothing=args.action_smoothing,
        deterministic_system1=not args.stochastic_system1,
    )
    LOGGER.info(
        "Building VLA model: architecture=%s backend=%s model=%s dtype=%s local_files_only=%s",
        effective_model,
        args.backbone,
        model_name,
        dtype,
        args.local_files_only,
    )
    model = VLAModel.from_config(config)
    LOGGER.info("VLA backbone and action head initialized.")

    if checkpoint is not None and checkpoint_path is not None:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        if "backbone_lora_state_dict" in checkpoint:
            LOGGER.info("Applying LoRA adapters from checkpoint.")
            apply_lora_adapters(
                model.backbone,
                checkpoint.get("lora_config", {}),
                state_dict=checkpoint["backbone_lora_state_dict"],
                trainable=False,
            )
            LOGGER.info("Loaded LoRA adapters from %s", checkpoint_path)

        state_dict = checkpoint.get("action_head_state_dict", checkpoint)
        try:
            model.action_head.load_state_dict(state_dict)
        except RuntimeError as exc:
            LOGGER.warning(
                "Could not load action head from %s into %s model: %s",
                checkpoint_path,
                effective_model,
                exc.__class__.__name__,
            )
        else:
            LOGGER.info("Loaded action head checkpoint from %s", checkpoint_path)

    model.eval()
    return model


def run_demo(args: argparse.Namespace) -> None:
    """Run CARLA autopilot, camera capture, and pygame rendering together."""

    client = build_carla_client(args)
    camera: Optional[RGBCameraSensor] = None
    visualizer = build_visualizer(args)
    vla_model: Optional[Any] = None
    inference_worker: Optional[AsyncVLAInference] = None
    red_team_server: Optional[Any] = None
    scenario_spawner: Optional[DataCollector] = None
    video_writer: Optional[Any] = None
    record_cv2: Optional[Any] = None
    scenario_actors: Dict[str, List[Any]] = {
        "vehicles": [],
        "walkers": [],
        "walker_controllers": [],
    }

    try:
        if args.backbone == "dummy" and args.model != "isaac_groot":
            render_model_loading_screen(args, visualizer)
        else:
            LOGGER.info(
                "Loading real VLA model before opening pygame to avoid a frozen window."
            )
        vla_model = build_vla_model(args)
        LOGGER.info("Initialized %s VLA model for shadow inference.", args.model)
        if _uses_async_inference(args) and not args.sync_inference:
            inference_worker = AsyncVLAInference(vla_model, args.instruction)
            inference_worker.start()
            LOGGER.info("Real VLM inference will run asynchronously for pygame responsiveness.")
        if args.record_video:
            video_writer, record_cv2 = _open_demo_video_writer(args)

        if not args.disable_red_team_listener:
            try:
                from src.evaluation.red_team_attacks import SUMORedTeamAttackServer

                red_team_server = SUMORedTeamAttackServer(
                    host=args.red_team_host,
                    port=args.red_team_port,
                    ttl_seconds=args.red_team_ttl,
                )
                red_team_server.start()
            except ImportError as exc:
                LOGGER.warning(
                    "SUMO red-team listener disabled because optional dependencies "
                    "are missing: %s",
                    exc,
                )
                red_team_server = None
            except OSError as exc:
                LOGGER.warning("Could not start SUMO red-team UDP listener: %s", exc)
                red_team_server = None

        world = client.connect()
        ego_vehicle = client.spawn_ego_vehicle()
        _configure_autopilot_demo_behavior(client, args)
        scenario_spawner = build_demo_scenario_spawner(args, client)
        if scenario_spawner is not None:
            scenario_actors = scenario_spawner._spawn_scenario_actors(
                client,
                world,
                ego_vehicle,
            )
        vla_control_requested = bool(args.vla_control)
        warmup_with_autopilot = (
            vla_control_requested
            and args.warmup_autopilot
            and inference_worker is not None
        )
        vla_control_active = vla_control_requested and not warmup_with_autopilot
        if vla_control_active:
            client.set_autopilot(False)
            LOGGER.info("VLA control enabled; CARLA autopilot disabled.")
        elif warmup_with_autopilot:
            client.set_autopilot(True)
            _configure_autopilot_demo_behavior(client, args)
            LOGGER.info(
                "VLA control requested; keeping autopilot active until the first real model action."
            )

        camera = build_camera(args, client)
        camera.spawn(world, ego_vehicle)

        initial_frame = None
        try:
            initial_frame = camera.wait_for_frame(timeout_seconds=5.0)
            LOGGER.info("Received first RGB camera frame: %s", initial_frame.frame_id)
        except TimeoutError:
            LOGGER.warning(
                "No RGB camera frame received within 5 seconds. "
                "Continuing with the waiting screen."
            )

        LOGGER.info("Demo running. Close the pygame window or press Q/Escape to stop.")
        started_at = time.monotonic()
        last_fps_report_at = started_at
        frames_since_report = 0
        inference_count = 0
        inference_seconds_total = 0.0
        last_inference_ms = 0.0
        last_metrics_report_at = started_at
        steering_error_total = 0.0
        steering_error_count = 0
        speed_total = 0.0
        speed_count = 0
        off_road_events = 0
        previous_off_road = False
        collisions = 0
        latest_vla_action = (
            None if warmup_with_autopilot else _make_bootstrap_vla_action(vla_model, args)
        )
        latest_model_stats: Optional[Dict[str, Any]] = None
        stale_stop_started_at: Optional[float] = None
        if latest_vla_action is not None:
            LOGGER.info("Using VLA bootstrap action until the first real model inference finishes.")
        latest_inference_seq = -1
        real_vla_ready = False

        while True:
            if not visualizer.process_events():
                break

            if visualizer.consume_emergency_autopilot_request():
                if vla_control_requested:
                    if vla_control_active:
                        client.set_autopilot(True)
                        _configure_autopilot_demo_behavior(client, args)
                        vla_control_active = False
                        warmup_with_autopilot = False
                        LOGGER.warning(
                            "Space pressed: CARLA autopilot now controls the vehicle."
                        )
                    elif real_vla_ready and latest_vla_action is not None:
                        client.set_autopilot(False)
                        vla_control_active = True
                        warmup_with_autopilot = False
                        LOGGER.warning(
                            "Space pressed: VLA control resumed."
                        )
                    else:
                        client.set_autopilot(True)
                        _configure_autopilot_demo_behavior(client, args)
                        vla_control_active = False
                        warmup_with_autopilot = True
                        LOGGER.warning(
                            "Space pressed before first VLA action; autopilot keeps control."
                        )
                else:
                    client.set_autopilot(True)
                    _configure_autopilot_demo_behavior(client, args)
                    vla_control_active = False
                    LOGGER.warning("Emergency autopilot takeover requested from pygame Space key.")

            client.tick(timeout_seconds=args.tick_timeout)
            frame = camera.get_latest_frame(copy=True) or initial_frame
            state = client.get_vehicle_state()
            vla_action = None
            expert_action = _expert_action_from_state(state)
            steering_error = None
            safety_status = client.get_safety_status(
                max_speed_kmh=args.safety_max_speed,
                off_road_distance_m=args.off_road_distance,
            )
            if safety_status.off_road and not previous_off_road:
                off_road_events += 1
            previous_off_road = safety_status.off_road
            speed_total += float(state.speed_kmh)
            speed_count += 1

            attack_status = (
                red_team_server.get_status_text()
                if red_team_server is not None
                else None
            )

            if frame is None:
                visualizer.render(
                    None,
                    state,
                    instruction=args.instruction,
                    vla_action=vla_action,
                    attack_status=attack_status,
                    control_mode=_control_mode_label(vla_control_active, state, args),
                    vla_applied=bool(vla_control_active),
                    safety_status=safety_status,
                    waiting_text="Waiting for camera...",
                )
            else:
                image_for_model = frame.image
                if red_team_server is not None:
                    image_for_model = red_team_server.apply_to_image(
                        frame.image,
                        model=vla_model,
                        instruction=args.instruction,
                        target_action=_expert_action_from_state(state),
                    )

                runtime_instruction = _compose_runtime_instruction(
                    args,
                    client,
                    state,
                    safety_status,
                )
                if inference_worker is not None:
                    inference_worker.submit(
                        image_for_model,
                        state=state,
                        instruction=runtime_instruction,
                    )
                    result = inference_worker.latest_result()
                    if result is not None:
                        seq, async_action, inference_seconds, model_stats = result
                        if seq != latest_inference_seq:
                            latest_inference_seq = seq
                            latest_vla_action = async_action
                            latest_model_stats = model_stats
                            real_vla_ready = True
                            inference_count += 1
                            inference_seconds_total += inference_seconds
                            last_inference_ms = inference_seconds * 1000.0
                            if warmup_with_autopilot:
                                client.set_autopilot(False)
                                vla_control_active = True
                                warmup_with_autopilot = False
                                LOGGER.info(
                                    "First real model action ready; VLA now controls the vehicle."
                                )
                    vla_action = latest_vla_action
                else:
                    inference_started_at = time.perf_counter()
                    vla_action = _predict_action(
                        vla_model,
                        image_for_model,
                        runtime_instruction,
                        state=state,
                    )
                    inference_seconds = time.perf_counter() - inference_started_at
                    inference_count += 1
                    inference_seconds_total += inference_seconds
                    last_inference_ms = inference_seconds * 1000.0
                    latest_vla_action = vla_action
                    latest_model_stats = _model_runtime_stats(vla_model)
                    real_vla_ready = True

                vla_applied = False
                if vla_action is not None:
                    vla_control = _vla_action_to_control(vla_action)
                    steering_error = abs(float(vla_control["steering"]) - float(expert_action[0]))
                    steering_error_total += steering_error
                    steering_error_count += 1

                    if vla_control_active:
                        vla_control = _stabilize_vla_control(vla_control, state, args)
                        vla_control, stale_stop_started_at = _release_stale_stop_if_clear(
                            vla_control,
                            state,
                            client,
                            args,
                            now=time.monotonic(),
                            stale_stop_started_at=stale_stop_started_at,
                        )
                        vla_control = _blend_route_target_steering(vla_control, client, args)
                        vla_control = _blend_lane_assist(vla_control, client, args)
                        safety_status = client.apply_vla_control(
                            steering=vla_control["steering"],
                            throttle=vla_control["throttle"],
                            brake=vla_control["brake"],
                            max_speed_kmh=args.safety_max_speed,
                            off_road_distance_m=args.off_road_distance,
                            safety_status=safety_status,
                        )
                        if safety_status.forced_brake:
                            vla_control["throttle"] = 0.0
                            vla_control["brake"] = 1.0
                        vla_action = [
                            vla_control["steering"],
                            vla_control["throttle"],
                            vla_control["brake"],
                        ]
                        latest_vla_action = vla_action
                        state = client.get_vehicle_state()
                        vla_applied = True

                if not vla_control_active:
                    _apply_autopilot_demo_hesitation(client, args, now=time.monotonic(), state=state)

                status_text = None
                if warmup_with_autopilot:
                    status_text = (
                        "Warmup: CARLA autopilot controls until first Alpamayo-R1 action "
                        f"(lane assist configured: {float(args.lane_assist):.2f})"
                    )
                elif vla_control_active and not real_vla_ready:
                    status_text = "VLA BOOTSTRAP: waiting for first real model action"

                if args.compare:
                    visualizer.render_compare(
                        image_for_model,
                        state,
                        instruction=runtime_instruction,
                        frame_id=frame.frame_id,
                        vla_action=vla_action,
                        expert_action=expert_action,
                        steering_error=steering_error,
                        attack_status=attack_status or status_text,
                        safety_status=safety_status,
                        model_stats=latest_model_stats,
                    )
                else:
                    visualizer.render(
                        image_for_model,
                        state,
                        instruction=runtime_instruction,
                        frame_id=frame.frame_id,
                        vla_action=vla_action,
                        attack_status=attack_status or status_text,
                        control_mode=_control_mode_label(vla_control_active, state, args),
                        vla_applied=bool(vla_applied),
                        safety_status=safety_status,
                        model_stats=latest_model_stats,
                    )
                initial_frame = None

            if video_writer is not None and record_cv2 is not None:
                frame_rgb = visualizer.capture_frame_rgb()
                if frame_rgb is not None:
                    video_writer.write(record_cv2.cvtColor(frame_rgb, record_cv2.COLOR_RGB2BGR))

            frames_since_report += 1
            now = time.monotonic()
            if now - last_fps_report_at >= 5.0:
                elapsed = now - last_fps_report_at
                fps = frames_since_report / elapsed if elapsed > 0.0 else 0.0
                print(f"[demo] FPS: {fps:.1f}", flush=True)
                if inference_count > 0:
                    avg_inference_ms = (inference_seconds_total / inference_count) * 1000.0
                    print(
                        "[demo] VLA inference: "
                        f"avg={avg_inference_ms:.2f} ms "
                        f"last={last_inference_ms:.2f} ms "
                        f"samples={inference_count}",
                        flush=True,
                    )
                else:
                    if inference_worker is not None and inference_worker.is_busy:
                        print("[demo] VLA inference: running asynchronously", flush=True)
                    else:
                        print("[demo] VLA inference: waiting for camera frames", flush=True)
                last_fps_report_at = now
                frames_since_report = 0
                inference_count = 0
                inference_seconds_total = 0.0

            if now - last_metrics_report_at >= args.metrics_interval:
                avg_steering_error = (
                    steering_error_total / steering_error_count
                    if steering_error_count > 0
                    else 0.0
                )
                avg_speed = speed_total / speed_count if speed_count > 0 else 0.0
                print(
                    "[vla] "
                    f"avg_steering_error={avg_steering_error:.3f} | "
                    f"avg_speed={avg_speed:.1f} km/h | "
                    f"off_road_events={off_road_events} | "
                    f"collisions={collisions}",
                    flush=True,
                )
                last_metrics_report_at = now
                steering_error_total = 0.0
                steering_error_count = 0
                speed_total = 0.0
                speed_count = 0
                off_road_events = 0

            if args.duration > 0.0 and now - started_at >= args.duration:
                break

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        if video_writer is not None:
            video_writer.release()
        if inference_worker is not None:
            inference_worker.stop()
        if red_team_server is not None:
            red_team_server.stop()
        if camera is not None:
            camera.destroy()
        if (
            scenario_spawner is not None
            and client.world is not None
            and any(scenario_actors.values())
        ):
            scenario_spawner._destroy_scenario_actors(client, client.world, scenario_actors)
        visualizer.close()
        client.cleanup()


class AsyncVLAInference:
    """Run slow VLA inference off the pygame thread and keep only the latest result."""

    def __init__(self, model: Any, instruction: str) -> None:
        self.model = model
        self.instruction = instruction
        self._condition = threading.Condition()
        self._request = None
        self._closed = False
        self._busy = False
        self._latest_result: Optional[tuple[int, object, float, Dict[str, Any]]] = None
        self._sequence = 0
        self._thread = threading.Thread(
            target=self._run,
            name="vla-inference",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def submit(
        self,
        image: object,
        *,
        state: Optional[object] = None,
        instruction: Optional[str] = None,
    ) -> bool:
        """Queue one image if the worker is idle; drop it otherwise."""

        with self._condition:
            if self._closed or self._busy or self._request is not None:
                return False
            image_copy = image.copy() if hasattr(image, "copy") else image
            state_copy = state.to_dict() if hasattr(state, "to_dict") else state
            self._request = (image_copy, state_copy, instruction or self.instruction)
            self._condition.notify()
            return True

    def latest_result(self) -> Optional[tuple[int, object, float, Dict[str, Any]]]:
        """Return the newest completed inference result, if any."""

        with self._condition:
            return self._latest_result

    @property
    def is_busy(self) -> bool:
        with self._condition:
            return self._busy or self._request is not None

    def stop(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._request is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                image, state, instruction = self._request
                self._request = None
                self._busy = True

            started_at = time.perf_counter()
            try:
                action = _predict_action(self.model, image, instruction, state=state)
            except Exception:
                LOGGER.exception("VLA inference failed in background thread.")
                action = None
            inference_seconds = time.perf_counter() - started_at
            model_stats = _model_runtime_stats(self.model)

            with self._condition:
                self._busy = False
                if action is not None:
                    self._sequence += 1
                    self._latest_result = (
                        self._sequence,
                        action,
                        inference_seconds,
                        model_stats,
                    )


def render_model_loading_screen(
    args: argparse.Namespace,
    visualizer: PygameVisualizer,
) -> None:
    """Open pygame early so long model loads do not look like a dead process."""

    if not visualizer.process_events():
        raise KeyboardInterrupt("Pygame window closed while loading the VLA model.")
    if args.model == "isaac_groot":
        waiting_text = "Loading Isaac GR00T policy..."
    elif args.model == "alpamayo_r1":
        waiting_text = "Loading fine-tuned Alpamayo R1 planner..."
    elif args.model == "alpamayo":
        waiting_text = "Loading Alpamayo planner..."
    elif _is_cosmos_reason_backbone(args.backbone):
        waiting_text = "Loading Cosmos-Reason System 2..."
    elif args.backbone in {"qwen2_vl", "qwen2-vl", "qwen"}:
        waiting_text = "Loading Qwen2-VL model..."
    else:
        waiting_text = "Loading VLA model..."
    visualizer.render(
        None,
        instruction=args.instruction,
        waiting_text=waiting_text,
    )
    time.sleep(0.1)


def _expert_action_from_state(state: object) -> list[float]:
    if hasattr(state, "to_dict"):
        state = state.to_dict()
    if not isinstance(state, dict):
        return [0.0, 0.0, 0.0]

    control = state.get("control", {})
    return [
        float(control.get("steering", control.get("steer", 0.0))),
        float(control.get("throttle", 0.0)),
        float(control.get("brake", 0.0)),
    ]


def _uses_async_inference(args: argparse.Namespace) -> bool:
    return args.model in {"isaac_groot", "alpamayo", "alpamayo_r1"} or _is_real_transformer_backbone(args.backbone)


def _predict_action(
    model: Any,
    image: object,
    instruction: str,
    *,
    state: Optional[object] = None,
) -> object:
    try:
        return model.predict_action(image, instruction, state=state)
    except TypeError:
        return model.predict_action(image, instruction)


def _model_runtime_stats(model: Optional[Any]) -> Dict[str, Any]:
    if model is None or not hasattr(model, "get_runtime_stats"):
        return {}
    try:
        return dict(model.get_runtime_stats())
    except Exception:
        return {}


def _make_bootstrap_vla_action(
    model: Optional[Any],
    args: argparse.Namespace,
) -> Optional[list[float]]:
    """Return a safe action-head prior used before the first slow VLM frame finishes."""

    if model is None or args.disable_bootstrap_action:
        return None

    try:
        device = _module_device(model.action_head)
        dtype = _module_dtype(model.action_head)
        embedding_dim = int(getattr(model.action_head.config, "embedding_dim", args.embedding_dim))
        embedding = torch.zeros((1, embedding_dim), device=device, dtype=dtype)
        with torch.no_grad():
            prior = model.action_head(embedding)[0].detach().cpu().tolist()
        control = _vla_action_to_control(prior)
    except Exception as exc:
        LOGGER.warning("Could not compute VLA bootstrap prior, using launch assist: %s", exc)
        control = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}

    control["steering"] = _clamp(control.get("steering", 0.0), -0.15, 0.15)
    control["throttle"] = max(
        _clamp(control.get("throttle", 0.0), 0.0, 1.0),
        _clamp(args.bootstrap_throttle, 0.0, 1.0),
    )
    control["brake"] = 0.0

    return [control["steering"], control["throttle"], control["brake"]]


def _vla_action_to_control(action: object) -> dict[str, float]:
    if hasattr(action, "detach"):
        action = action.detach().cpu().flatten().tolist()
    values = list(action)
    if len(values) < 3:
        raise ValueError("VLA action must contain steering, throttle, and brake.")
    return {
        "steering": _clamp(values[0], -1.0, 1.0),
        "throttle": _clamp(values[1], 0.0, 1.0),
        "brake": _clamp(values[2], 0.0, 1.0),
    }


def _stabilize_vla_control(
    control: dict[str, float],
    state: object,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Remove tiny pedal noise and keep VLA driving at a presentation-safe speed."""

    stabilized = dict(control)
    if stabilized["brake"] < args.brake_deadzone:
        stabilized["brake"] = 0.0
    if stabilized["throttle"] < args.throttle_deadzone:
        stabilized["throttle"] = 0.0

    speed_kmh = _speed_from_state(state)
    target_speed = max(1.0, float(args.target_speed_kmh))
    max_throttle = _clamp(args.max_vla_throttle, 0.0, 1.0)
    stabilized["throttle"] = min(stabilized["throttle"], max_throttle)

    can_launch = (
        speed_kmh < args.start_speed_threshold
        and stabilized["brake"] <= 0.0
        and stabilized["throttle"] > 0.0
    )
    if can_launch:
        stabilized["throttle"] = max(
            stabilized["throttle"],
            _clamp(args.min_start_throttle, 0.0, 1.0),
        )

    if speed_kmh >= target_speed:
        overspeed = speed_kmh - target_speed
        stabilized["throttle"] = 0.0
        if overspeed > 5.0:
            stabilized["brake"] = max(stabilized["brake"], min(0.45, overspeed / 25.0))
    elif speed_kmh >= target_speed * 0.75:
        stabilized["throttle"] = min(stabilized["throttle"], max_throttle * 0.55)

    if stabilized["brake"] > 0.25:
        stabilized["throttle"] = 0.0

    return stabilized


def _release_stale_stop_if_clear(
    control: dict[str, float],
    state: object,
    client: CarlaClient,
    args: argparse.Namespace,
    *,
    now: float,
    stale_stop_started_at: Optional[float],
) -> tuple[dict[str, float], Optional[float]]:
    """Optionally release over-conservative stops without touching steering."""

    if not args.release_stuck_stop:
        return control, None

    speed_kmh = _speed_from_state(state)
    stale_brake = (
        speed_kmh <= max(0.1, float(args.release_max_speed_kmh))
        and control.get("brake", 0.0) > max(args.brake_deadzone, 0.10)
        and control.get("throttle", 0.0) <= args.throttle_deadzone
    )
    if not stale_brake:
        return control, None

    if stale_stop_started_at is None:
        stale_stop_started_at = now
        return control, stale_stop_started_at

    if now - stale_stop_started_at < max(0.0, float(args.release_after_seconds)):
        return control, stale_stop_started_at

    if _blocked_by_red_or_yellow_light(client):
        return control, stale_stop_started_at
    if _has_forward_hazard(client, max_distance_m=float(args.release_hazard_distance)):
        return control, stale_stop_started_at

    released = dict(control)
    released["brake"] = 0.0
    released["throttle"] = min(
        _clamp(args.max_vla_throttle, 0.0, 1.0),
        max(released.get("throttle", 0.0), _clamp(args.release_throttle, 0.0, 1.0)),
    )
    return released, stale_stop_started_at


def _blocked_by_red_or_yellow_light(client: CarlaClient) -> bool:
    try:
        vehicle = client.ego_vehicle
        if vehicle is None or not vehicle.is_at_traffic_light():
            return False
        traffic_light = vehicle.get_traffic_light()
        if traffic_light is None:
            return False
        state = str(traffic_light.get_state()).split(".")[-1].lower()
        return state in {"red", "yellow"}
    except RuntimeError:
        return False


def _has_forward_hazard(client: CarlaClient, *, max_distance_m: float) -> bool:
    try:
        world = client.world
        ego = client.ego_vehicle
        if world is None or ego is None:
            return False
        ego_transform = ego.get_transform()
        ego_location = ego_transform.location
        yaw = math.radians(float(ego_transform.rotation.yaw))
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        right_x = -forward_y
        right_y = forward_x
        ego_id = int(ego.id)
        actors = list(world.get_actors().filter("vehicle.*"))
        actors.extend(world.get_actors().filter("walker.pedestrian.*"))
    except RuntimeError:
        return False

    for actor in actors:
        try:
            if int(actor.id) == ego_id:
                continue
            location = actor.get_location()
            dx = float(location.x - ego_location.x)
            dy = float(location.y - ego_location.y)
            forward_distance = dx * forward_x + dy * forward_y
            lateral_distance = abs(dx * right_x + dy * right_y)
            if 0.0 < forward_distance <= max_distance_m and lateral_distance <= 3.5:
                return True
        except RuntimeError:
            continue
    return False


def _blend_lane_assist(
    control: dict[str, float],
    client: CarlaClient,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Blend VLA steering with a CARLA waypoint correction for stable demos."""

    amount = _clamp(getattr(args, "lane_assist", 0.0), 0.0, 1.0)
    if amount <= 0.0:
        return control

    lane_steer = client.get_lane_follow_steering(
        lookahead_m=float(getattr(args, "lane_lookahead", 8.0))
    )
    if lane_steer is None:
        return control

    blended = dict(control)
    blended["steering"] = _clamp(
        (1.0 - amount) * float(control["steering"]) + amount * lane_steer,
        -1.0,
        1.0,
    )
    return blended


def _configure_autopilot_demo_behavior(client: CarlaClient, args: argparse.Namespace) -> None:
    """Slow down the ego autopilot for a cautious VLA-style assisted demo."""

    try:
        ego = getattr(client, "ego_vehicle", None)
        carla_client = getattr(client, "client", None)
        if ego is None or carla_client is None:
            return
        traffic_manager = carla_client.get_trafficmanager(int(args.traffic_manager_port))
        traffic_manager.vehicle_percentage_speed_difference(
            ego,
            float(getattr(args, "autopilot_demo_speed_difference", 45.0)),
        )
        traffic_manager.auto_lane_change(ego, False)
        traffic_manager.distance_to_leading_vehicle(ego, 3.5)
    except Exception as exc:
        LOGGER.debug("Could not configure ego autopilot demo behavior: %s", exc)


def _apply_autopilot_demo_hesitation(
    client: CarlaClient,
    args: argparse.Namespace,
    *,
    now: float,
    state: object,
) -> None:
    """Inject occasional gentle hesitation while autopilot remains the low-level driver."""

    if not bool(getattr(args, "autopilot_demo_hesitation", True)):
        return
    if _speed_from_state(state) < 6.0:
        return
    try:
        ego = getattr(client, "ego_vehicle", None)
        if ego is None:
            return
        phase = now % 11.0
        if 3.2 <= phase <= 3.65 or 7.4 <= phase <= 7.75:
            import carla

            brake = 0.10 + 0.05 * random.random()
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=brake, steer=0.0))
    except Exception as exc:
        LOGGER.debug("Could not apply autopilot hesitation: %s", exc)


def _compose_runtime_instruction(
    args: argparse.Namespace,
    client: CarlaClient,
    state: object,
    safety_status: object,
) -> str:
    """Inject route intent and live traffic-control context into the VLA prompt."""

    parts = [str(args.instruction).strip()]
    nav = str(getattr(args, "nav_maneuver", "follow_lane")).lower()
    if nav == "left":
        parts.append(
            "Navigation command: turn LEFT at the next legal junction or road branch. "
            "Prepare early, follow the left-turn lane/arc, and do not continue straight "
            "if the route requires a legal left turn."
        )
    elif nav == "right":
        parts.append(
            "Navigation command: turn RIGHT at the next legal junction or road branch. "
            "Prepare early, follow the right-turn lane/arc, and do not continue straight "
            "if the route requires a legal right turn."
        )
    elif nav == "straight":
        parts.append(
            "Navigation command: continue STRAIGHT through the next legal junction when "
            "traffic rules and right-of-way allow it."
        )
    elif nav == "auto":
        route_hint = _route_hint_from_lane_geometry(client, args)
        parts.append(f"Navigation command: {route_hint}")
    else:
        parts.append(
            "Navigation command: follow the current drivable lane and road center through "
            "curves. If the lane bends, steer with the lane; never leave the road by "
            "continuing straight across a sidewalk, curb, wall, or non-drivable area."
        )

    route_target = _route_target_context(client, args)
    if route_target is not None:
        command = str(route_target["command"])
        x_m = float(route_target["x_m"])
        y_m = float(route_target["y_m"])
        distance_m = float(route_target["distance_m"])
        branch_found = bool(route_target["branch_found"])
        lane_note = "branch selected" if branch_found else "lane-follow target"
        parts.append(
            "Closed-loop route target point, inspired by SimLingo/Bench2Drive: "
            f"local target is x={x_m:.1f} m forward and y={y_m:.1f} m left "
            f"(distance {distance_m:.1f} m, {lane_note}). "
            f"Planner command: {command}. Treat this target point as the route objective; "
            "at intersections, choose the legal branch that moves toward it instead of "
            "blindly continuing straight."
        )

    status = safety_status.to_dict() if hasattr(safety_status, "to_dict") else safety_status
    if isinstance(status, dict):
        light_state = status.get("traffic_light_state")
        if light_state:
            parts.append(
                f"Live traffic light state: {str(light_state).upper()}. "
                "Red or yellow means brake and stop before the stop line; green means "
                "proceed only if the junction and crosswalk are clear."
            )
        stop_distance = status.get("stop_sign_distance_m")
        if stop_distance is not None:
            try:
                parts.append(
                    f"Live stop-sign context: stop sign about {float(stop_distance):.1f} m ahead. "
                    "Make a complete stop, wait briefly, then proceed only when priority is clear."
                )
            except (TypeError, ValueError):
                pass
        message = status.get("message")
        if message:
            parts.append(f"Safety monitor: {message}.")

    if hasattr(state, "to_dict"):
        state = state.to_dict()
    if isinstance(state, dict):
        parts.append(f"Current speed: {float(state.get('speed_kmh', 0.0)):.1f} km/h.")
    return " ".join(part for part in parts if part)


def _route_target_context(client: CarlaClient, args: argparse.Namespace) -> Optional[Dict[str, object]]:
    """Build a SimLingo-style local target point from CARLA lane geometry."""

    if not bool(getattr(args, "route_target_nav", False)):
        return None
    try:
        world = getattr(client, "world", None)
        ego = getattr(client, "ego_vehicle", None)
        if world is None or ego is None:
            return None

        import carla

        carla_map = world.get_map()
        transform = ego.get_transform()
        location = transform.location
        waypoint = carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return None

        command = str(getattr(args, "nav_maneuver", "follow_lane")).lower()
        target_wp, branch_found = _select_route_target_waypoint(
            waypoint,
            ego_transform=transform,
            command=command,
            target_distance_m=float(getattr(args, "route_target_distance", 18.0)),
            scan_distance_m=float(getattr(args, "route_target_scan_distance", 55.0)),
        )
        if target_wp is None:
            return None

        target_location = target_wp.transform.location
        x_m, y_m = _location_to_ego_xy(transform, target_location)
        distance_m = math.hypot(x_m, y_m)
        if distance_m < 1e-3:
            return None
        return {
            "command": _route_command_text(command),
            "x_m": x_m,
            "y_m": y_m,
            "distance_m": distance_m,
            "branch_found": branch_found,
        }
    except Exception as exc:
        LOGGER.debug("Could not build route target context: %s", exc)
        return None


def _select_route_target_waypoint(
    waypoint: object,
    *,
    ego_transform: object,
    command: str,
    target_distance_m: float,
    scan_distance_m: float,
) -> tuple[Optional[object], bool]:
    """Select a waypoint target, scanning ahead for a requested branch."""

    target_distance_m = max(4.0, float(target_distance_m))
    scan_distance_m = max(target_distance_m, float(scan_distance_m))
    step_m = 3.0
    wants_branch = command in {"left", "right", "straight"}

    current = waypoint
    lane_follow_target: Optional[object] = None
    distance_walked = 0.0
    previous_yaw = float(current.transform.rotation.yaw)

    while distance_walked < scan_distance_m:
        try:
            candidates = list(current.next(step_m))
        except RuntimeError:
            break
        if not candidates:
            break

        if wants_branch and len(candidates) > 1:
            branch = _choose_branch_candidate(candidates, ego_transform, command)
            branch_target = _advance_waypoint(branch, max(6.0, target_distance_m * 0.45))
            return branch_target or branch, True

        next_wp = min(
            candidates,
            key=lambda candidate: abs(
                _angle_delta_degrees(
                    previous_yaw,
                    float(candidate.transform.rotation.yaw),
                )
            ),
        )
        distance_walked += step_m
        current = next_wp
        previous_yaw = float(current.transform.rotation.yaw)
        lane_follow_target = current
        if distance_walked >= target_distance_m and not wants_branch:
            return lane_follow_target, False

    if lane_follow_target is None:
        lane_follow_target = _advance_waypoint(waypoint, target_distance_m)
    return lane_follow_target, False


def _choose_branch_candidate(candidates: List[object], ego_transform: object, command: str) -> object:
    scored = []
    for candidate in candidates:
        x_m, y_m = _location_to_ego_xy(ego_transform, candidate.transform.location)
        heading_delta = abs(
            _angle_delta_degrees(
                float(ego_transform.rotation.yaw),
                float(candidate.transform.rotation.yaw),
            )
        )
        scored.append((candidate, x_m, y_m, heading_delta))

    if command == "left":
        return max(scored, key=lambda item: (item[2], item[1]))[0]
    if command == "right":
        return min(scored, key=lambda item: (item[2], -item[1]))[0]
    return min(scored, key=lambda item: (abs(item[2]), item[3]))[0]


def _advance_waypoint(waypoint: object, distance_m: float) -> Optional[object]:
    current = waypoint
    remaining = max(0.0, float(distance_m))
    while remaining > 0.0:
        step = min(3.0, remaining)
        try:
            candidates = list(current.next(max(1.0, step)))
        except RuntimeError:
            return current
        if not candidates:
            return current
        previous_yaw = float(current.transform.rotation.yaw)
        current = min(
            candidates,
            key=lambda candidate: abs(
                _angle_delta_degrees(
                    previous_yaw,
                    float(candidate.transform.rotation.yaw),
                )
            ),
        )
        remaining -= step
    return current


def _location_to_ego_xy(ego_transform: object, target_location: object) -> tuple[float, float]:
    ego_location = ego_transform.location
    dx = float(target_location.x - ego_location.x)
    dy = float(target_location.y - ego_location.y)
    yaw = math.radians(float(ego_transform.rotation.yaw))
    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)
    # CARLA yaw=0 points along +x, with +y on the vehicle's right.
    # Alpamayo/AV coordinates expect positive lateral values to mean left.
    left_x = math.sin(yaw)
    left_y = -math.cos(yaw)
    x_m = dx * forward_x + dy * forward_y
    y_m = dx * left_x + dy * left_y
    return float(x_m), float(y_m)


def _route_command_text(command: str) -> str:
    if command == "left":
        return "turn left at the next legal branch"
    if command == "right":
        return "turn right at the next legal branch"
    if command == "straight":
        return "go straight through the next legal branch"
    if command == "auto":
        return "follow the lane geometry chosen by the local route planner"
    return "follow the current lane geometry"


def _angle_delta_degrees(first: float, second: float) -> float:
    return (second - first + 180.0) % 360.0 - 180.0


def _blend_route_target_steering(
    control: Dict[str, float],
    client: CarlaClient,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Blend Alpamayo steering with the local route target controller."""

    amount = _clamp(getattr(args, "route_target_steer_blend", 0.0), 0.0, 1.0)
    if amount <= 0.0 or not bool(getattr(args, "route_target_nav", False)):
        return control

    route_target = _route_target_context(client, args)
    if route_target is None:
        return control

    x_m = float(route_target["x_m"])
    y_m = float(route_target["y_m"])
    if x_m < 1.0:
        return control

    # Positive local y is left of the vehicle, while CARLA positive steer turns right.
    heading_error = math.atan2(y_m, max(1.0, x_m))
    target_steer = _clamp(-heading_error / math.radians(45.0), -1.0, 1.0)

    blended = dict(control)
    blended["steering"] = _clamp(
        (1.0 - amount) * float(control["steering"]) + amount * target_steer,
        -1.0,
        1.0,
    )
    return blended


def _route_hint_from_lane_geometry(client: CarlaClient, args: argparse.Namespace) -> str:
    try:
        lane_steer = client.get_lane_follow_steering(
            lookahead_m=float(getattr(args, "lane_lookahead", 8.0))
        )
    except Exception:
        lane_steer = None
    if lane_steer is None:
        return "follow the current lane and road geometry."
    if lane_steer > 0.18:
        return "the current lane geometry bends RIGHT; follow the bend and stay on road."
    if lane_steer < -0.18:
        return "the current lane geometry bends LEFT; follow the bend and stay on road."
    return "follow the current lane straight ahead while remaining centered."


def _control_mode_label(vla_control_active: bool, state: object, args: argparse.Namespace) -> str:
    raw_state = state.to_dict() if hasattr(state, "to_dict") else state
    autopilot_enabled = (
        isinstance(raw_state, dict)
        and bool(raw_state.get("control", {}).get("autopilot", False))
    )
    if autopilot_enabled:
        return str(getattr(args, "autopilot_demo_label", "CARLA autopilot"))
    if vla_control_active:
        route_blend = _clamp(getattr(args, "route_target_steer_blend", 0.0), 0.0, 1.0)
        if bool(getattr(args, "route_target_nav", False)) and route_blend > 0.0:
            return f"VLA + route target {route_blend:.2f}"
        lane_assist = _clamp(getattr(args, "lane_assist", 0.0), 0.0, 1.0)
        if lane_assist > 0.0:
            return f"VLA + lane assist {lane_assist:.2f}"
        return "VLA"
    return "Manual"


def _speed_from_state(state: object) -> float:
    if hasattr(state, "to_dict"):
        state = state.to_dict()
    if isinstance(state, dict):
        return float(state.get("speed_kmh", 0.0))
    return float(getattr(state, "speed_kmh", 0.0))


def _clamp(value: object, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.float32


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_demo(args)


if __name__ == "__main__":
    main()
