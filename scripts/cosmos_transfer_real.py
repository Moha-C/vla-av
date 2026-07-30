"""Collect CARLA control videos and run real NVIDIA Cosmos-Transfer2.5 inference."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.carla_env.carla_client import CarlaClient, CarlaClientConfig
from src.carla_env.data_collector import DataCollector, EpisodeCollector, EpisodeConfig
from src.carla_env.sensors import (
    CameraFrame,
    DepthCameraConfig,
    DepthCameraSensor,
    RGBCameraConfig,
    RGBCameraSensor,
    SemanticSegmentationCameraConfig,
    SemanticSegmentationCameraSensor,
)
from src.data.cosmos_transfer import CosmosTransfer, CosmosTransferConfig


LOGGER = logging.getLogger(__name__)


DEFAULT_DRIVING_INSTRUCTION = (
    "Drive like a safe autonomous vehicle in an urban environment. Follow the "
    "current lane and road markings, keep a smooth centered trajectory, respect "
    "speed limits, obey red lights, green lights, stop signs, lane arrows, "
    "crosswalks, priority rules, and right-of-way. Yield to pedestrians, cyclists, "
    "scooters, motorbikes, parked cars pulling out, and other vehicles. Stop when "
    "the path is blocked, wait until it is clear, then continue smoothly without "
    "leaving the drivable lane."
)


DEFAULT_TRANSFER_NEGATIVE_PROMPT = (
    "CGI, video game, cyberpunk, neon glow, handheld camera, phone in hand, "
    "dashcam holder, visible dashcam, visible camera, gopro, action camera, "
    "camera rig, camera mount, suction cup, car hood, dashboard, windshield frame, "
    "windshield wipers, cartoon, anime, oversaturated colors, distorted buildings, "
    "warped lane markings, broken lane markings, melted lane markings, missing lane markings, "
    "melted texture, flicker, motion smear, low quality"
)


CAMERA_PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "location": (1.5, 0.0, 2.4),
        "rotation": (-15.0, 0.0, 0.0),
        "fov": 90.0,
    },
    "hood": {
        "location": (1.15, 0.0, 1.38),
        "rotation": (-7.0, 0.0, 0.0),
        "fov": 82.0,
    },
    "windshield": {
        "location": (0.65, 0.0, 1.62),
        "rotation": (-5.0, 0.0, 0.0),
        "fov": 84.0,
    },
    "bumper": {
        "location": (2.35, 0.0, 0.82),
        "rotation": (-3.0, 0.0, 0.0),
        "fov": 90.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--map", dest="map_name", default=None)
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument(
        "--spawn-top-k",
        type=int,
        default=1,
        help="When using a spawn preset, choose among the top K scored spawn points using scenario seed.",
    )
    parser.add_argument(
        "--spawn-preset",
        default="straight",
        choices=("straight", "straight_turn", "junction", "traffic_law", "traffic_light", "stop_or_light"),
    )
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument(
        "--camera-preset",
        default="hood",
        choices=tuple(CAMERA_PRESETS),
        help="CARLA camera mount. hood/windshield make the output feel more vehicle-mounted.",
    )
    parser.add_argument(
        "--camera-location",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override CARLA camera location relative to the ego vehicle.",
    )
    parser.add_argument(
        "--camera-rotation",
        type=float,
        nargs=3,
        default=None,
        metavar=("PITCH", "YAW", "ROLL"),
        help="Override CARLA camera rotation relative to the ego vehicle.",
    )
    parser.add_argument("--output-dir", default="data/synthetic/transferred_real")
    parser.add_argument(
        "--reuse-run-dir",
        default=None,
        help="Reuse an existing run directory containing carla_rgb.mp4/carla_seg.mp4/carla_depth.mp4.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--weather",
        default="clear natural daytime urban dashcam with realistic exposure",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=DEFAULT_TRANSFER_NEGATIVE_PROMPT)
    parser.add_argument(
        "--instruction",
        default=DEFAULT_DRIVING_INSTRUCTION,
        help="Driving instruction saved with every expert action label.",
    )
    parser.add_argument("--vehicles", type=int, default=25)
    parser.add_argument("--two-wheelers", type=int, default=12)
    parser.add_argument("--walkers", type=int, default=45)
    parser.add_argument("--pedestrian-cross-factor", type=float, default=0.85)
    parser.add_argument("--traffic-speed-difference", type=float, default=20.0)
    parser.add_argument("--ego-speed-difference", type=float, default=10.0)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument(
        "--save-frame-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save raw CARLA RGB/seg/depth frames beside the videos. Videos are always saved.",
    )
    parser.add_argument("--metadata-name", default="episode.jsonl")
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--transfer-seed", type=int, default=None)
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0.0,
        help="Cosmos Transfer2.5 edge-control weight. Use this to preserve lane markings and road geometry.",
    )
    parser.add_argument(
        "--edge-threshold",
        default="medium",
        choices=("very_low", "low", "medium", "high", "very_high"),
        help="Canny edge threshold preset for on-the-fly edge control. Lower keeps more lane-paint detail.",
    )
    parser.add_argument("--seg-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=0.0)
    parser.add_argument("--vis-weight", type=float, default=0.0)
    parser.add_argument("--transfer-resolution", default="480", choices=("480", "720"))
    parser.add_argument("--transfer-max-frames", type=int, default=49)
    parser.add_argument("--transfer-chunk-frames", type=int, default=None)
    parser.add_argument("--transfer-num-steps", type=int, default=24)
    parser.add_argument(
        "--keep-input-resolution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resize Transfer2.5 output back to the CARLA input resolution. "
            "Use this with 1920x1080 CARLA captures when building 1080p datasets."
        ),
    )
    parser.add_argument("--disable-guardrails", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-transfer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--transfer-repo-dir", default="external/cosmos-transfer2.5")
    parser.add_argument("--transfer-python", default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--launch-carla", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--carla-path", default="~/carla_simulator/CarlaUE4.sh")
    parser.add_argument("--carla-quality", default="Epic")
    parser.add_argument(
        "--carla-streaming-port",
        type=int,
        default=None,
        help="Optional CARLA streaming port for running multiple local CARLA servers.",
    )
    parser.add_argument(
        "--carla-primary-port",
        type=int,
        default=None,
        help="Optional CARLA primary port for multi-server runs.",
    )
    parser.add_argument(
        "--carla-secondary-port",
        type=int,
        default=None,
        help="Optional CARLA secondary port for multi-server runs.",
    )
    parser.add_argument(
        "--carla-offscreen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Launch CARLA with -RenderOffScreen for headless cloud GPUs.",
    )
    parser.add_argument("--carla-start-timeout", type=float, default=120.0)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


class VideoWriter:
    def __init__(self, path: Path, fps: int, width: int, height: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.path = path
        self.writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")

    def write_rgb(self, image: np.ndarray) -> None:
        image = np.asarray(image)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def close(self) -> None:
        self.writer.release()


def is_carla_ready(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        import carla

        client = carla.Client(host, port)
        client.set_timeout(timeout)
        client.get_world()
        return True
    except Exception:
        return False


def maybe_launch_carla(args: argparse.Namespace) -> Optional[subprocess.Popen[Any]]:
    if is_carla_ready(args.host, args.port):
        LOGGER.info("CARLA is already reachable on %s:%s.", args.host, args.port)
        return None
    if not args.launch_carla:
        return None

    carla_path = Path(args.carla_path).expanduser()
    if not carla_path.exists():
        raise RuntimeError(f"CARLA launcher not found at {carla_path}")

    LOGGER.info("Starting CARLA from %s", carla_path)
    carla_command = [str(carla_path), f"-quality-level={args.carla_quality}", "-nosound"]
    if args.port != 2000:
        carla_command.append(f"-carla-rpc-port={args.port}")
    if args.carla_streaming_port is not None:
        carla_command.append(f"-carla-streaming-port={int(args.carla_streaming_port)}")
    if args.carla_primary_port is not None:
        carla_command.append(f"-carla-primary-port={int(args.carla_primary_port)}")
    if args.carla_secondary_port is not None:
        carla_command.append(f"-carla-secondary-port={int(args.carla_secondary_port)}")
    if args.carla_offscreen:
        carla_command.append("-RenderOffScreen")
    process = subprocess.Popen(
        carla_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + args.carla_start_timeout
    while time.monotonic() < deadline:
        if is_carla_ready(args.host, args.port):
            LOGGER.info("CARLA became reachable.")
            return process
        time.sleep(2.0)
    raise RuntimeError("CARLA did not become reachable in time.")


def build_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    return (
        "Real-world photorealistic forward-facing automotive perception footage "
        "from a fixed vehicle camera. The camera itself is not visible. No visible "
        "camera rig, no GoPro, no car hood, no dashboard, no windshield frame, and "
        "no recording device in the image. "
        f"Weather and lighting: {args.weather}. "
        "Natural camera exposure, realistic asphalt, concrete, glass, trees, traffic "
        "lights, signs, sidewalks, lane arrows, stop lines, crosswalks, parked cars, "
        "moving cars, pedestrians, cyclists, scooters, and urban buildings. Preserve "
        "the exact road geometry, lane boundaries, lane arrows, crosswalks, stop lines, "
        "curbs, traffic-light positions, signs, vehicle positions, and pedestrian "
        "positions from the control video. Lane paint must remain crisp, continuous, "
        "geometrically straight where straight, and faithful to the source control. "
        "Make the scene look like real city driving footage, not a simulator."
    )


def make_client(args: argparse.Namespace) -> CarlaClient:
    return CarlaClient(
        CarlaClientConfig(
            host=args.host,
            port=args.port,
            timeout_seconds=60.0,
            tick_timeout_seconds=2.0,
            map_name=args.map_name,
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / float(args.fps),
            traffic_manager_port=args.traffic_manager_port,
            seed=args.scenario_seed,
            traffic_manager_seed=args.scenario_seed,
            ego_spawn_index=args.spawn_index,
            ego_spawn_preset=args.spawn_preset,
            ego_spawn_top_k=args.spawn_top_k,
            autopilot=True,
        )
    )


def make_camera_configs(args: argparse.Namespace) -> tuple[RGBCameraConfig, SemanticSegmentationCameraConfig, DepthCameraConfig]:
    preset = CAMERA_PRESETS[args.camera_preset]
    location = tuple(args.camera_location) if args.camera_location is not None else preset["location"]
    rotation = tuple(args.camera_rotation) if args.camera_rotation is not None else preset["rotation"]
    fov = args.fov if args.fov is not None else preset["fov"]
    common = {
        "width": args.width,
        "height": args.height,
        "fov": fov,
        "sensor_tick": 1.0 / float(args.fps),
        "location": location,
        "rotation": rotation,
    }
    LOGGER.info(
        "Using camera preset=%s location=%s rotation=%s fov=%s",
        args.camera_preset,
        location,
        rotation,
        fov,
    )
    return (
        RGBCameraConfig(**common),
        SemanticSegmentationCameraConfig(**common),
        DepthCameraConfig(**common),
    )


def save_capture_metadata(args: argparse.Namespace, run_dir: Path) -> None:
    preset = CAMERA_PRESETS[args.camera_preset]
    location = tuple(args.camera_location) if args.camera_location is not None else preset["location"]
    rotation = tuple(args.camera_rotation) if args.camera_rotation is not None else preset["rotation"]
    fov = args.fov if args.fov is not None else preset["fov"]
    metadata = {
        "camera_preset": args.camera_preset,
        "camera_location": location,
        "camera_rotation": rotation,
        "fov": fov,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "frames": args.frames,
        "map": args.map_name,
        "spawn_index": args.spawn_index,
        "spawn_top_k": args.spawn_top_k,
        "spawn_preset": args.spawn_preset,
        "carla_quality": args.carla_quality,
        "carla_offscreen": args.carla_offscreen,
        "instruction": args.instruction,
        "scenario_seed": args.scenario_seed,
        "scenario_actors": {
            "vehicles": args.vehicles,
            "two_wheelers": args.two_wheelers,
            "walkers": args.walkers,
            "pedestrian_cross_factor": args.pedestrian_cross_factor,
            "traffic_speed_difference": args.traffic_speed_difference,
            "ego_speed_difference": args.ego_speed_difference,
        },
    }
    with (run_dir / "capture_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def wait_for_triplet(
    rgb_camera: RGBCameraSensor,
    semantic_camera: SemanticSegmentationCameraSensor,
    depth_camera: DepthCameraSensor,
    timeout_seconds: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_frame, semantic_frame, depth_frame = wait_for_frame_triplet(
        rgb_camera,
        semantic_camera,
        depth_camera,
        timeout_seconds=timeout_seconds,
    )
    return rgb_frame.image, semantic_frame.image, depth_frame.image


def wait_for_frame_triplet(
    rgb_camera: RGBCameraSensor,
    semantic_camera: SemanticSegmentationCameraSensor,
    depth_camera: DepthCameraSensor,
    timeout_seconds: float = 10.0,
) -> tuple[CameraFrame, CameraFrame, CameraFrame]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        rgb = rgb_camera.get_latest_frame(copy=True)
        semantic = semantic_camera.get_latest_frame(copy=True)
        depth = depth_camera.get_latest_frame(copy=True)
        if rgb is not None and semantic is not None and depth is not None:
            return rgb, semantic, depth
        time.sleep(0.03)
    raise TimeoutError("Timed out waiting for RGB/semantic/depth camera frames.")


def make_label_collector(args: argparse.Namespace) -> EpisodeCollector:
    return EpisodeCollector(
        EpisodeConfig(
            instruction=args.instruction,
            seed=args.scenario_seed,
            frame_timeout_seconds=5.0,
            traffic_manager_port=args.traffic_manager_port,
        )
    )


def make_scenario_collector(args: argparse.Namespace) -> DataCollector:
    episode_config = EpisodeConfig(
        n_episodes=1,
        frames_per_episode=args.frames,
        instruction=args.instruction,
        seed=args.scenario_seed,
        traffic_manager_port=args.traffic_manager_port,
        npc_vehicles=max(0, int(args.vehicles)),
        npc_two_wheelers=max(0, int(args.two_wheelers)),
        npc_walkers=max(0, int(args.walkers)),
        pedestrian_cross_factor=float(args.pedestrian_cross_factor),
        traffic_speed_difference=float(args.traffic_speed_difference),
        ego_speed_difference=float(args.ego_speed_difference),
        spawn_focus=args.spawn_preset,
    )
    return DataCollector(
        episode_config=episode_config,
        carla_config=CarlaClientConfig(
            host=args.host,
            port=args.port,
            timeout_seconds=60.0,
            tick_timeout_seconds=2.0,
            traffic_manager_port=args.traffic_manager_port,
            traffic_manager_seed=args.scenario_seed,
            autopilot=True,
        ),
    )


def enrich_frame_metadata(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    client: CarlaClient,
    label_collector: EpisodeCollector,
    ego_vehicle: Any,
    rgb_frame: CameraFrame,
    frame_idx: int,
) -> dict[str, Any]:
    metadata = label_collector._build_metadata(ego_vehicle, rgb_frame)
    state = client.get_vehicle_state().to_dict()
    transfer_frame_path = f"photoreal_frames/frame_{frame_idx:06d}.jpg"
    metadata.update(
        {
            "sample_index": int(frame_idx),
            "image_path": transfer_frame_path,
            "photoreal_frame_path": transfer_frame_path,
            "carla_rgb_frame_path": (
                f"carla_rgb_frames/frame_{frame_idx:06d}.jpg" if args.save_frame_images else None
            ),
            "carla_seg_frame_path": (
                f"carla_seg_frames/frame_{frame_idx:06d}.png" if args.save_frame_images else None
            ),
            "carla_depth_frame_path": (
                f"carla_depth_frames/frame_{frame_idx:06d}.png" if args.save_frame_images else None
            ),
            "carla_rgb_video": str((run_dir / "carla_rgb.mp4").name),
            "carla_seg_video": str((run_dir / "carla_seg.mp4").name),
            "carla_depth_video": str((run_dir / "carla_depth.mp4").name),
            "camera_preset": args.camera_preset,
            "weather_prompt": args.weather,
            "map": args.map_name,
            "spawn_index": args.spawn_index,
            "spawn_preset": args.spawn_preset,
            "ego_state": state,
            "action": {
                "steering": float(metadata["steering"]),
                "throttle": float(metadata["throttle"]),
                "brake": float(metadata["brake"]),
            },
        }
    )
    return metadata


def save_optional_frame_images(
    run_dir: Path,
    frame_idx: int,
    rgb: np.ndarray,
    semantic: np.ndarray,
    depth: np.ndarray,
) -> None:
    paths = {
        "carla_rgb_frames": (rgb, ".jpg"),
        "carla_seg_frames": (semantic, ".png"),
        "carla_depth_frames": (depth, ".png"),
    }
    for folder, (image, suffix) in paths.items():
        target_dir = run_dir / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(target_dir / f"frame_{frame_idx:06d}{suffix}")


def save_preview(run_dir: Path, rgb: np.ndarray, semantic: np.ndarray, depth: np.ndarray) -> None:
    preview_dir = run_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(preview_dir / "rgb_first.png")
    Image.fromarray(semantic).save(preview_dir / "semantic_first.png")
    Image.fromarray(depth).save(preview_dir / "depth_first.png")


def collect_control_videos(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, Path]:
    carla_process = maybe_launch_carla(args)
    client = make_client(args)
    rgb_camera: Optional[RGBCameraSensor] = None
    semantic_camera: Optional[SemanticSegmentationCameraSensor] = None
    depth_camera: Optional[DepthCameraSensor] = None
    world: Optional[Any] = None
    scenario_collector: Optional[DataCollector] = None
    scenario_actors: dict[str, list[Any]] = {
        "vehicles": [],
        "walkers": [],
        "walker_controllers": [],
    }

    rgb_path = run_dir / "carla_rgb.mp4"
    seg_path = run_dir / "carla_seg.mp4"
    depth_path = run_dir / "carla_depth.mp4"
    metadata_path = run_dir / args.metadata_name
    rgb_writer = VideoWriter(rgb_path, args.fps, args.width, args.height)
    seg_writer = VideoWriter(seg_path, args.fps, args.width, args.height)
    depth_writer = VideoWriter(depth_path, args.fps, args.width, args.height)
    save_capture_metadata(args, run_dir)
    metadata_path.write_text("", encoding="utf-8")

    try:
        world = client.connect()
        ego_vehicle = client.spawn_ego_vehicle()
        scenario_collector = make_scenario_collector(args)
        if args.vehicles > 0 or args.two_wheelers > 0 or args.walkers > 0:
            scenario_actors = scenario_collector._spawn_scenario_actors(
                client,
                world,
                ego_vehicle,
            )
        rgb_config, semantic_config, depth_config = make_camera_configs(args)
        rgb_camera = RGBCameraSensor(rgb_config, client=client)
        semantic_camera = SemanticSegmentationCameraSensor(semantic_config, client=client)
        depth_camera = DepthCameraSensor(depth_config, client=client)
        rgb_camera.spawn(world, ego_vehicle)
        semantic_camera.spawn(world, ego_vehicle)
        depth_camera.spawn(world, ego_vehicle)

        for _ in range(3):
            client.tick(timeout_seconds=2.0)
        rgb_frame, semantic_frame, depth_frame = wait_for_frame_triplet(
            rgb_camera,
            semantic_camera,
            depth_camera,
        )
        save_preview(run_dir, rgb_frame.image, semantic_frame.image, depth_frame.image)

        LOGGER.info("Collecting %s frames at %sx%s/%sfps", args.frames, args.width, args.height, args.fps)
        label_collector = make_label_collector(args)
        with metadata_path.open("a", encoding="utf-8") as metadata_file:
            for frame_idx in range(args.frames):
                client.tick(timeout_seconds=2.0)
                rgb_frame, semantic_frame, depth_frame = wait_for_frame_triplet(
                    rgb_camera,
                    semantic_camera,
                    depth_camera,
                )
                rgb_writer.write_rgb(rgb_frame.image)
                seg_writer.write_rgb(semantic_frame.image)
                depth_writer.write_rgb(depth_frame.image)
                metadata = enrich_frame_metadata(
                    args=args,
                    run_dir=run_dir,
                    client=client,
                    label_collector=label_collector,
                    ego_vehicle=ego_vehicle,
                    rgb_frame=rgb_frame,
                    frame_idx=frame_idx,
                )
                metadata_file.write(json.dumps(metadata) + "\n")
                metadata_file.flush()
                if args.save_frame_images:
                    save_optional_frame_images(
                        run_dir,
                        frame_idx,
                        rgb_frame.image,
                        semantic_frame.image,
                        depth_frame.image,
                    )
                if (frame_idx + 1) % 10 == 0 or frame_idx + 1 == args.frames:
                    LOGGER.info("Captured %s/%s frames", frame_idx + 1, args.frames)

        return rgb_path, seg_path, depth_path
    finally:
        for writer in (rgb_writer, seg_writer, depth_writer):
            writer.close()
        for camera in (rgb_camera, semantic_camera, depth_camera):
            if camera is not None:
                camera.destroy()
        if world is not None and scenario_collector is not None:
            scenario_collector._destroy_scenario_actors(client, world, scenario_actors)
        client.cleanup()
        if carla_process is not None:
            LOGGER.info("Stopping CARLA to free GPU memory before Transfer2.5.")
            carla_process.terminate()
            try:
                carla_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                carla_process.kill()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.reuse_run_dir:
        run_dir = Path(args.reuse_run_dir).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"reuse run dir does not exist: {run_dir}")
        run_name = args.run_name or run_dir.name
        rgb_path = run_dir / "carla_rgb.mp4"
        seg_path = run_dir / "carla_seg.mp4"
        depth_path = run_dir / "carla_depth.mp4"
        for path in (rgb_path, seg_path, depth_path):
            if not path.exists():
                raise FileNotFoundError(f"Expected existing control video: {path}")
    else:
        run_name = args.run_name or time.strftime("transfer25_%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        rgb_path, seg_path, depth_path = collect_control_videos(args, run_dir)

    transfer = CosmosTransfer(
        CosmosTransferConfig(
            backend="transfer2.5",
            transfer_repo_dir=args.transfer_repo_dir,
            transfer_python=args.transfer_python,
            disable_guardrails=args.disable_guardrails,
        )
    )
    params_path = transfer.write_transfer2_5_spec(
        name=run_name,
        prompt=build_prompt(args),
        video_path=rgb_path,
        params_path=run_dir / "transfer25_params.json",
        negative_prompt=args.negative_prompt,
        edge_weight=args.edge_weight,
        edge_threshold=args.edge_threshold,
        seg_control_path=seg_path,
        depth_control_path=depth_path,
        guidance=args.guidance,
        seed=args.transfer_seed,
        seg_weight=args.seg_weight,
        depth_weight=args.depth_weight,
        vis_weight=args.vis_weight,
        resolution=args.transfer_resolution,
        max_frames=args.transfer_max_frames,
        num_video_frames_per_chunk=args.transfer_chunk_frames,
        num_steps=args.transfer_num_steps,
        keep_input_resolution=args.keep_input_resolution,
    )
    LOGGER.info("Wrote Transfer2.5 params to %s", params_path)

    if args.run_transfer:
        transfer.run_transfer2_5(params_path, output_dir=run_dir / "transfer_output", num_gpus=args.num_gpus)
    else:
        LOGGER.info(
            "Prepared real Transfer2.5 inputs. Run with --run-transfer when external/cosmos-transfer2.5 is installed."
        )


if __name__ == "__main__":
    main()
