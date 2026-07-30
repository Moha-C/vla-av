"""CARLA semantic/depth to Cosmos-Transfer photorealistic preview demo."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError as exc:  # pragma: no cover - only hit without pygame installed.
    pygame = None
    _PYGAME_IMPORT_ERROR = exc
else:
    _PYGAME_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.carla_env.carla_client import CarlaClient, CarlaClientConfig
from src.carla_env.sensors import (
    DepthCameraConfig,
    DepthCameraSensor,
    RGBCameraConfig,
    RGBCameraSensor,
    SemanticSegmentationCameraConfig,
    SemanticSegmentationCameraSensor,
)
from src.data.cosmos_transfer import CosmosTransfer, CosmosTransferConfig


LOGGER = logging.getLogger(__name__)

WEATHER_VARIANTS = [
    "clear day urban",
    "rainy day urban",
    "heavy rain at night",
    "foggy night urban",
    "snowy dusk urban",
    "wet sunset urban",
    "cloudy morning urban",
    "bright noon urban",
    "stormy evening urban",
    "misty sunrise urban",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--map", dest="map_name", default=None)
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument("--spawn-preset", default="straight", choices=("straight", "straight_turn", "junction", "traffic_law"))
    parser.add_argument("--weather", default="heavy rain at night")
    parser.add_argument("--backend", default="stylized", choices=("stylized", "mock", "local", "transfer2.5", "predict2.5"))
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means until closed.")
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--camera-tick", type=float, default=0.05)
    parser.add_argument("--window-width", type=int, default=1200)
    parser.add_argument("--window-height", type=int, default=520)
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--transfer-every", type=int, default=1, help="Refresh transferred preview every N display frames.")
    parser.add_argument("--output-dir", default="data/synthetic/transferred")
    parser.add_argument(
        "--save-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save 10 weather variants from the same captured CARLA trajectory.",
    )
    parser.add_argument(
        "--variant-frames",
        type=int,
        default=30,
        help="Number of trajectory frames to save for each weather variant.",
    )
    parser.add_argument(
        "--variant-every",
        type=int,
        default=5,
        help="Save one transferred trajectory frame every N display frames.",
    )
    parser.add_argument(
        "--launch-carla",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start ~/carla_simulator/CarlaUE4.sh if CARLA is not reachable.",
    )
    parser.add_argument("--carla-path", default="~/carla_simulator/CarlaUE4.sh")
    parser.add_argument("--carla-quality", default="Low")
    parser.add_argument("--carla-start-timeout", type=float, default=90.0)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def ensure_pygame_available() -> None:
    if _PYGAME_IMPORT_ERROR is not None:
        raise RuntimeError("pygame is required for cosmos_transfer_demo.py") from _PYGAME_IMPORT_ERROR


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
        LOGGER.warning("CARLA is not reachable; continuing without launching it.")
        return None

    carla_path = Path(args.carla_path).expanduser()
    if not carla_path.exists():
        LOGGER.warning("CARLA launcher not found at %s; start CARLA manually.", carla_path)
        return None

    LOGGER.info("Starting CARLA from %s", carla_path)
    process = subprocess.Popen(
        [str(carla_path), f"-quality-level={args.carla_quality}", "-nosound"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + args.carla_start_timeout
    while time.monotonic() < deadline:
        if is_carla_ready(args.host, args.port):
            LOGGER.info("CARLA became reachable.")
            return process
        time.sleep(2.0)
    LOGGER.warning("CARLA did not become reachable within %.0fs.", args.carla_start_timeout)
    return process


def build_client(args: argparse.Namespace) -> CarlaClient:
    config = CarlaClientConfig(
        host=args.host,
        port=args.port,
        timeout_seconds=30.0,
        map_name=args.map_name,
        traffic_manager_port=args.traffic_manager_port,
        ego_spawn_index=args.spawn_index,
        ego_spawn_preset=args.spawn_preset,
        autopilot=True,
    )
    return CarlaClient(config)


def make_camera_configs(args: argparse.Namespace) -> tuple[RGBCameraConfig, SemanticSegmentationCameraConfig, DepthCameraConfig]:
    common = {
        "width": args.camera_width,
        "height": args.camera_height,
        "fov": args.camera_fov,
        "sensor_tick": args.camera_tick,
    }
    return (
        RGBCameraConfig(**common),
        SemanticSegmentationCameraConfig(**common),
        DepthCameraConfig(**common),
    )


def wait_for_triplet(
    rgb_camera: RGBCameraSensor,
    semantic_camera: SemanticSegmentationCameraSensor,
    depth_camera: DepthCameraSensor,
    timeout_seconds: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        rgb = rgb_camera.get_latest_image(copy=True)
        semantic = semantic_camera.get_latest_image(copy=True)
        depth = depth_camera.get_latest_image(copy=True)
        if rgb is not None and semantic is not None and depth is not None:
            return rgb, semantic, depth
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for synchronized RGB/semantic/depth camera frames.")


def draw_three_columns(
    screen: Any,
    font: Any,
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: tuple[str, str, str],
) -> None:
    screen.fill((8, 10, 12))
    width, height = screen.get_size()
    label_h = 34
    column_w = width // 3
    for idx, (image, label) in enumerate(zip(images, labels)):
        x = idx * column_w
        bounds = pygame.Rect(x, label_h, column_w, height - label_h)
        draw_image(screen, image, bounds)
        pygame.draw.rect(screen, (24, 28, 34), pygame.Rect(x, 0, column_w, label_h))
        text = font.render(label, True, (235, 240, 245))
        screen.blit(text, (x + 12, 9))
        if idx > 0:
            pygame.draw.line(screen, (34, 38, 44), (x, 0), (x, height), 2)
    pygame.display.flip()


def draw_image(screen: Any, image: np.ndarray, bounds: Any) -> None:
    image = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
    surface = pygame.surfarray.make_surface(np.swapaxes(image, 0, 1))
    target = fit_rect(image.shape[1], image.shape[0], bounds)
    if surface.get_size() != target.size:
        surface = pygame.transform.smoothscale(surface, target.size)
    screen.blit(surface, target)


def fit_rect(src_w: int, src_h: int, bounds: Any) -> Any:
    scale = min(bounds.width / max(src_w, 1), bounds.height / max(src_h, 1))
    width = max(1, int(src_w * scale))
    height = max(1, int(src_h * scale))
    x = bounds.x + (bounds.width - width) // 2
    y = bounds.y + (bounds.height - height) // 2
    return pygame.Rect(x, y, width, height)


def save_weather_variants(
    transfer: CosmosTransfer,
    rgb: np.ndarray,
    semantic: np.ndarray,
    depth: np.ndarray,
    output_dir: Path,
    frame_index: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_name = f"frame_{frame_index:05d}.png"
    for folder_name, image in (
        ("carla_rgb", rgb),
        ("carla_semantic", semantic),
        ("carla_depth", depth),
    ):
        folder = output_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(folder / frame_name)

    for weather in WEATHER_VARIANTS:
        transferred = transfer.transfer(semantic, depth, weather, reference_image=rgb)
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in weather).strip("_")
        folder = output_dir / slug
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(transferred).save(folder / frame_name)


def run_demo(args: argparse.Namespace) -> None:
    ensure_pygame_available()
    carla_process = maybe_launch_carla(args)
    client = build_client(args)
    rgb_camera: Optional[RGBCameraSensor] = None
    semantic_camera: Optional[SemanticSegmentationCameraSensor] = None
    depth_camera: Optional[DepthCameraSensor] = None

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((args.window_width, args.window_height))
    pygame.display.set_caption("VLA-AV Cosmos Transfer")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    transfer = CosmosTransfer(
        CosmosTransferConfig(
            backend=args.backend,
            output_size=(args.camera_width, args.camera_height),
            cache_dir=args.output_dir,
        )
    )

    try:
        world = client.connect()
        ego_vehicle = client.spawn_ego_vehicle()
        rgb_config, semantic_config, depth_config = make_camera_configs(args)
        rgb_camera = RGBCameraSensor(rgb_config, client=client)
        semantic_camera = SemanticSegmentationCameraSensor(semantic_config, client=client)
        depth_camera = DepthCameraSensor(depth_config, client=client)
        rgb_camera.spawn(world, ego_vehicle)
        semantic_camera.spawn(world, ego_vehicle)
        depth_camera.spawn(world, ego_vehicle)

        rgb, semantic, depth = wait_for_triplet(rgb_camera, semantic_camera, depth_camera)
        transferred = transfer.transfer(semantic, depth, args.weather, reference_image=rgb)

        started_at = time.monotonic()
        frame_idx = 0
        saved_variant_frames = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

            if args.duration > 0 and time.monotonic() - started_at >= args.duration:
                running = False

            client.tick(timeout_seconds=0.1)
            latest_rgb = rgb_camera.get_latest_image(copy=True)
            latest_semantic = semantic_camera.get_latest_image(copy=True)
            latest_depth = depth_camera.get_latest_image(copy=True)
            if latest_rgb is not None:
                rgb = latest_rgb
            if latest_semantic is not None:
                semantic = latest_semantic
            if latest_depth is not None:
                depth = latest_depth

            if frame_idx % max(1, args.transfer_every) == 0:
                transferred = transfer.transfer(semantic, depth, args.weather, reference_image=rgb)

            if (
                args.save_variants
                and saved_variant_frames < max(0, args.variant_frames)
                and frame_idx % max(1, args.variant_every) == 0
            ):
                save_weather_variants(
                    transfer,
                    rgb,
                    semantic,
                    depth,
                    Path(args.output_dir),
                    saved_variant_frames,
                )
                saved_variant_frames += 1
                if saved_variant_frames == max(0, args.variant_frames):
                    LOGGER.info(
                        "Saved %s frames for each of %s weather variants to %s",
                        saved_variant_frames,
                        len(WEATHER_VARIANTS),
                        args.output_dir,
                    )
            frame_idx += 1

            draw_three_columns(
                screen,
                font,
                (rgb, semantic, transferred),
                ("CARLA RGB", "CARLA Semantic", f"Transfer preview: {args.weather}"),
            )
            clock.tick(args.target_fps)

    finally:
        for camera in (rgb_camera, semantic_camera, depth_camera):
            if camera is not None:
                camera.destroy()
        client.cleanup()
        pygame.quit()
        if carla_process is not None and args.launch_carla:
            carla_process.terminate()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_demo(args)


if __name__ == "__main__":
    main()
