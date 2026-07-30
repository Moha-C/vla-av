"""Collect supervised CARLA expert episodes for VLA action-head training."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.carla_env.carla_client import CarlaClientConfig
from src.carla_env.data_collector import DataCollector, EpisodeConfig
from src.carla_env.sensors import RGBCameraConfig


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse data collection settings from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes.")
    parser.add_argument("--frames", type=int, default=200, help="Frames per episode.")
    parser.add_argument("--output-dir", default="data/raw", help="Dataset output root.")
    parser.add_argument(
        "--instruction",
        default="Drive safely and follow the lane.",
        help="Instruction saved with every frame.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="CARLA RPC timeout in seconds. CARLA can need >10s right after startup.",
    )
    parser.add_argument(
        "--connect-retries",
        type=int,
        default=5,
        help="Number of CARLA connection attempts before failing.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=5.0,
        help="Seconds to wait between CARLA connection attempts.",
    )
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
        help="Optional deterministic first spawn point index.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic collection seed.")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run CARLA in synchronous mode while collecting.",
    )
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--camera-tick", type=float, default=0.05)
    parser.add_argument(
        "--recovery",
        action="store_true",
        help=(
            "Collect lane-recovery examples by periodically nudging the vehicle "
            "off-center/yawed and recording autopilot corrections."
        ),
    )
    parser.add_argument(
        "--recovery-every",
        type=int,
        default=40,
        help="Inject one recovery perturbation every N collected frames.",
    )
    parser.add_argument(
        "--recovery-frames",
        type=int,
        default=24,
        help="Number of frames tagged as recovery after each perturbation.",
    )
    parser.add_argument(
        "--lateral-offsets",
        nargs="+",
        type=float,
        default=[-1.2, -0.7, 0.7, 1.2],
        help="Meters left/right from lane center used for recovery perturbations.",
    )
    parser.add_argument(
        "--yaw-offsets",
        nargs="+",
        type=float,
        default=[-10.0, -5.0, 5.0, 10.0],
        help="Yaw offsets in degrees used for recovery perturbations.",
    )
    parser.add_argument(
        "--vehicles",
        type=int,
        default=0,
        help="NPC autopilot cars/trucks spawned in each episode.",
    )
    parser.add_argument(
        "--two-wheelers",
        type=int,
        default=0,
        help="NPC bicycles/motorcycles/scooters spawned in each episode when blueprints exist.",
    )
    parser.add_argument(
        "--walkers",
        type=int,
        default=0,
        help="Pedestrian VRU walkers spawned in each episode.",
    )
    parser.add_argument(
        "--pedestrian-cross-factor",
        type=float,
        default=0.35,
        help="CARLA pedestrian crossing probability/factor when supported.",
    )
    parser.add_argument(
        "--traffic-speed-difference",
        type=float,
        default=10.0,
        help="Traffic Manager percentage speed difference for NPC vehicles; positive is slower.",
    )
    parser.add_argument(
        "--ego-speed-difference",
        type=float,
        default=0.0,
        help="Traffic Manager percentage speed difference for the expert ego vehicle; positive is slower.",
    )
    parser.add_argument(
        "--spawn-focus",
        default="random",
        choices=("random", "junction", "traffic_light", "stop_or_light", "traffic_law"),
        help="Bias ego spawn points toward traffic-law situations such as junctions, lights, and stops.",
    )
    parser.add_argument(
        "--weather-presets",
        nargs="+",
        default=["ClearNoon"],
        help="CARLA WeatherParameters preset names cycled across episodes.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    episode_config = EpisodeConfig(
        n_episodes=args.episodes,
        frames_per_episode=args.frames,
        output_dir=args.output_dir,
        instruction=args.instruction,
        seed=args.seed,
        traffic_manager_port=args.traffic_manager_port,
        recovery_mode=args.recovery,
        recovery_every_n_frames=args.recovery_every,
        recovery_frames=args.recovery_frames,
        recovery_lateral_offsets_m=tuple(args.lateral_offsets),
        recovery_yaw_offsets_deg=tuple(args.yaw_offsets),
        npc_vehicles=args.vehicles,
        npc_two_wheelers=args.two_wheelers,
        npc_walkers=args.walkers,
        pedestrian_cross_factor=args.pedestrian_cross_factor,
        traffic_speed_difference=args.traffic_speed_difference,
        ego_speed_difference=args.ego_speed_difference,
        spawn_focus=args.spawn_focus,
        weather_presets=tuple(args.weather_presets),
    )
    carla_config = CarlaClientConfig(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout,
        map_name=args.map_name,
        seed=args.seed,
        synchronous_mode=args.sync,
        traffic_manager_port=args.traffic_manager_port,
        traffic_manager_seed=args.seed,
        ego_spawn_index=args.spawn_index,
        autopilot=True,
    )
    camera_config = RGBCameraConfig(
        width=args.camera_width,
        height=args.camera_height,
        fov=args.camera_fov,
        sensor_tick=args.camera_tick,
    )

    wait_for_carla(args)

    total_frames = args.episodes * args.frames
    with tqdm(total=total_frames, desc="Collecting frames", unit="frame") as progress:
        collector = DataCollector(
            episode_config,
            carla_config=carla_config,
            camera_config=camera_config,
            on_frame_collected=lambda _: progress.update(1),
        )
        episode_paths = collector.collect()

    collected_frames = len(episode_paths) * args.frames
    mode = "recovery" if args.recovery else "standard"
    print(f"Collected {len(episode_paths)} episodes, {collected_frames} frames total ({mode} mode)")


def wait_for_carla(args: argparse.Namespace) -> None:
    """Poll CARLA before starting the long collection job."""

    from carla import Client

    last_error: BaseException | None = None
    for attempt in range(1, args.connect_retries + 1):
        try:
            client = Client(args.host, args.port)
            client.set_timeout(args.timeout)
            world = client.get_world()
            LOGGER.info(
                "CARLA ready on %s:%s with map %s",
                args.host,
                args.port,
                world.get_map().name,
            )
            return
        except BaseException as exc:
            last_error = exc
            LOGGER.warning(
                "CARLA not ready yet (%s/%s): %s",
                attempt,
                args.connect_retries,
                exc,
            )
            if attempt < args.connect_retries:
                time.sleep(args.retry_sleep)

    raise RuntimeError(
        f"CARLA was not ready on {args.host}:{args.port} after "
        f"{args.connect_retries} attempts. Start/restart CARLA and retry."
    ) from last_error


if __name__ == "__main__":
    main()
