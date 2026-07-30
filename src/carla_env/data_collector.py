"""Autopilot expert data collection for supervised VLA training."""

from __future__ import annotations

import json
import logging
import math
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from src.carla_env.carla_client import CarlaClient, CarlaClientConfig, carla
from src.carla_env.sensors import CameraFrame, RGBCameraConfig, RGBCameraSensor


LOGGER = logging.getLogger(__name__)
FrameCallback = Optional[Callable[[Dict[str, Any]], None]]


@dataclass(frozen=True)
class EpisodeConfig:
    """Configuration for collecting CARLA expert-driving episodes."""

    n_episodes: int = 5
    frames_per_episode: int = 200
    output_dir: str = "data/raw"
    instruction: str = "Drive safely and follow the lane."
    seed: int = 42
    frame_timeout_seconds: float = 5.0
    writer_queue_size: int = 1024
    traffic_manager_port: int = 8000

    recovery_mode: bool = False
    recovery_every_n_frames: int = 40
    recovery_frames: int = 24
    recovery_lateral_offsets_m: Tuple[float, ...] = (-1.2, -0.7, 0.7, 1.2)
    recovery_yaw_offsets_deg: Tuple[float, ...] = (-10.0, -5.0, 5.0, 10.0)
    recovery_settle_ticks: int = 4

    npc_vehicles: int = 0
    npc_two_wheelers: int = 0
    npc_walkers: int = 0
    pedestrian_cross_factor: float = 0.35
    traffic_speed_difference: float = 10.0
    ego_speed_difference: float = 0.0
    spawn_focus: str = "random"
    weather_presets: Tuple[str, ...] = ("ClearNoon",)


@dataclass(frozen=True)
class _SaveTask:
    image: np.ndarray
    metadata: Dict[str, Any]
    image_path: Path


class _AsyncEpisodeWriter:
    """Write PNG frames and JSONL metadata on a background thread."""

    def __init__(self, episode_dir: Path, max_queue_size: int) -> None:
        self.episode_dir = episode_dir
        self.frames_dir = episode_dir / "frames"
        self.jsonl_path = episode_dir / "episode.jsonl"
        self._queue: "queue.Queue[Optional[_SaveTask]]" = queue.Queue(max_queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._error: Optional[BaseException] = None

    def start(self) -> None:
        """Create output folders and start the background writer thread."""

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.write_text("")
        self._thread.start()

    def enqueue(self, image: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Queue one frame for async PNG and JSONL writing."""

        self._raise_if_failed()
        image_path = self.episode_dir / metadata["image_path"]
        self._queue.put(_SaveTask(image=image.copy(), metadata=metadata, image_path=image_path))

    def close(self) -> None:
        """Flush all queued writes, stop the writer, and surface writer errors."""

        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._raise_if_failed()

    def _run(self) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as jsonl_file:
            while True:
                task = self._queue.get()
                try:
                    if task is None:
                        return
                    self._write_task(task, jsonl_file)
                except BaseException as exc:  # pragma: no cover - depends on IO failure.
                    if self._error is None:
                        self._error = exc
                finally:
                    self._queue.task_done()

    def _write_task(self, task: _SaveTask, jsonl_file: Any) -> None:
        image = np.asarray(task.image, dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(task.image_path)
        jsonl_file.write(json.dumps(task.metadata) + "\n")
        jsonl_file.flush()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Async episode writer failed.") from self._error


class EpisodeCollector:
    """Collect one episode worth of camera frames and expert autopilot actions."""

    def __init__(
        self,
        config: Optional[EpisodeConfig] = None,
        *,
        on_frame_collected: FrameCallback = None,
    ) -> None:
        self.config = config or EpisodeConfig()
        self.output_dir = Path(self.config.output_dir)
        self.on_frame_collected = on_frame_collected
        self._next_episode_index: Optional[int] = None
        self._rng = random.Random(self.config.seed + 12_345)

    def collect_episode(self, world: Any, vehicle: Any, camera: RGBCameraSensor) -> Path:
        """Record frames_per_episode unique camera frames for one ego vehicle."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        episode_dir = self._allocate_episode_dir()
        writer = _AsyncEpisodeWriter(episode_dir, self.config.writer_queue_size)
        writer.start()

        LOGGER.info("Collecting episode into %s", episode_dir)
        collected = 0
        last_frame_id: Optional[int] = None
        recovery_remaining = 0
        recovery_context: Optional[Dict[str, Any]] = None

        try:
            while collected < self.config.frames_per_episode:
                if self._should_inject_recovery(collected, recovery_remaining):
                    recovery_context = self._inject_recovery_pose(world, vehicle)
                    recovery_remaining = self.config.recovery_frames

                self._advance_world(world)
                frame = self._wait_for_new_frame(camera, last_frame_id)
                last_frame_id = frame.frame_id

                metadata = self._build_metadata(
                    vehicle,
                    frame,
                    recovery_context=recovery_context,
                )
                writer.enqueue(frame.image, metadata)
                collected += 1
                if recovery_remaining > 0:
                    recovery_remaining -= 1
                    if recovery_remaining == 0:
                        recovery_context = None

                if self.on_frame_collected is not None:
                    self.on_frame_collected(metadata)
        finally:
            writer.close()

        LOGGER.info("Collected %s frames in %s", collected, episode_dir)
        return episode_dir

    def _allocate_episode_dir(self) -> Path:
        if self._next_episode_index is None:
            self._next_episode_index = self._find_next_episode_index()

        while True:
            episode_dir = self.output_dir / f"episode_{self._next_episode_index:03d}"
            self._next_episode_index += 1
            try:
                episode_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return episode_dir

    def _find_next_episode_index(self) -> int:
        if not self.output_dir.exists():
            return 0

        max_index = -1
        for path in self.output_dir.glob("episode_*"):
            if not path.is_dir():
                continue
            suffix = path.name.removeprefix("episode_")
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
        return max_index + 1

    def _advance_world(self, world: Any) -> None:
        settings = world.get_settings()
        if settings.synchronous_mode:
            world.tick()
        else:
            world.wait_for_tick(self.config.frame_timeout_seconds)

    def _wait_for_new_frame(
        self,
        camera: RGBCameraSensor,
        last_frame_id: Optional[int],
    ) -> CameraFrame:
        deadline = time.monotonic() + self.config.frame_timeout_seconds
        while time.monotonic() < deadline:
            frame = camera.get_latest_frame(copy=True)
            if frame is not None and frame.frame_id != last_frame_id:
                return frame
            time.sleep(0.002)

        raise TimeoutError(
            "Timed out waiting for a new RGB camera frame after "
            f"{self.config.frame_timeout_seconds:.1f}s."
        )

    def _should_inject_recovery(self, collected: int, recovery_remaining: int) -> bool:
        if not self.config.recovery_mode:
            return False
        if recovery_remaining > 0 or collected <= 0:
            return False
        interval = max(1, int(self.config.recovery_every_n_frames))
        return collected % interval == 0

    def _inject_recovery_pose(self, world: Any, vehicle: Any) -> Dict[str, Any]:
        """Nudge the ego off-center/yawed so autopilot labels recovery actions."""

        if carla is None:
            raise RuntimeError("CARLA recovery collection requires the carla Python package.")

        carla_map = world.get_map()
        current_location = vehicle.get_location()
        waypoint = carla_map.get_waypoint(
            current_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            LOGGER.warning("Skipping recovery injection because no driving waypoint was found.")
            return {
                "recovery_active": False,
                "recovery_lateral_offset_m": 0.0,
                "recovery_yaw_offset_deg": 0.0,
            }

        lateral_offsets = self.config.recovery_lateral_offsets_m or (0.8,)
        yaw_offsets = self.config.recovery_yaw_offsets_deg or (6.0,)
        lateral_offset = self._rng.choice(lateral_offsets)
        yaw_offset = self._rng.choice(yaw_offsets)

        base_transform = waypoint.transform
        lane_yaw_rad = math.radians(float(base_transform.rotation.yaw))
        right_x = math.cos(lane_yaw_rad + math.pi / 2.0)
        right_y = math.sin(lane_yaw_rad + math.pi / 2.0)

        location = carla.Location(
            x=float(base_transform.location.x) + right_x * float(lateral_offset),
            y=float(base_transform.location.y) + right_y * float(lateral_offset),
            z=float(base_transform.location.z) + 0.2,
        )
        rotation = carla.Rotation(
            pitch=float(base_transform.rotation.pitch),
            yaw=float(base_transform.rotation.yaw) + float(yaw_offset),
            roll=float(base_transform.rotation.roll),
        )
        transform = carla.Transform(location, rotation)

        try:
            vehicle.set_autopilot(False, self.config.traffic_manager_port)
            vehicle.set_transform(transform)
            zero = carla.Vector3D(0.0, 0.0, 0.0)
            vehicle.set_target_velocity(zero)
            vehicle.set_target_angular_velocity(zero)
            vehicle.apply_control(carla.VehicleControl())
            self._settle_world(world, self.config.recovery_settle_ticks)
            vehicle.set_autopilot(True, self.config.traffic_manager_port)
        except RuntimeError as exc:
            raise RuntimeError("Failed to inject CARLA recovery pose.") from exc

        LOGGER.debug(
            "Injected recovery pose: lateral=%+.2fm yaw=%+.1fdeg",
            lateral_offset,
            yaw_offset,
        )
        return {
            "recovery_active": True,
            "recovery_lateral_offset_m": float(lateral_offset),
            "recovery_yaw_offset_deg": float(yaw_offset),
        }

    def _settle_world(self, world: Any, ticks: int) -> None:
        for _ in range(max(0, int(ticks))):
            self._advance_world(world)

    def _build_metadata(
        self,
        vehicle: Any,
        frame: CameraFrame,
        *,
        recovery_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        control = vehicle.get_control()
        velocity = vehicle.get_velocity()
        speed_mps = math.sqrt(
            velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
        )
        relative_image_path = f"frames/frame_{frame.frame_id}.png"
        metadata = {
            "frame_id": int(frame.frame_id),
            "timestamp": float(frame.timestamp),
            "steering": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "speed_kmh": float(speed_mps * 3.6),
            "instruction": self.config.instruction,
            "image_path": relative_image_path,
        }
        metadata.update(self._traffic_rule_metadata(vehicle))
        if recovery_context is not None:
            metadata.update(recovery_context)
        else:
            metadata.update(
                {
                    "recovery_active": False,
                    "recovery_lateral_offset_m": 0.0,
                    "recovery_yaw_offset_deg": 0.0,
                }
            )
        return metadata

    def _traffic_rule_metadata(self, vehicle: Any) -> Dict[str, Any]:
        """Record traffic-light, speed-limit, and nearby sign context."""

        metadata: Dict[str, Any] = {
            "at_traffic_light": False,
            "traffic_light_state": "None",
            "traffic_light_id": None,
            "speed_limit_kmh": None,
            "near_stop_sign": False,
            "near_stop_sign_distance_m": None,
            "hazards": [],
        }
        hazards: List[str] = []

        try:
            metadata["speed_limit_kmh"] = float(vehicle.get_speed_limit())
        except RuntimeError:
            pass

        try:
            at_light = bool(vehicle.is_at_traffic_light())
            metadata["at_traffic_light"] = at_light
            traffic_light = vehicle.get_traffic_light() if at_light else None
            if traffic_light is not None:
                metadata["traffic_light_id"] = int(traffic_light.id)
                state = str(traffic_light.get_state()).split(".")[-1]
                metadata["traffic_light_state"] = state
                if state.lower() in {"red", "yellow"}:
                    hazards.append(f"traffic_light_{state.lower()}")
        except RuntimeError:
            pass

        stop_distance = self._distance_to_stop_sign(vehicle)
        if stop_distance is not None:
            metadata["near_stop_sign"] = True
            metadata["near_stop_sign_distance_m"] = float(stop_distance)
            hazards.append("stop_sign")

        metadata["hazards"] = hazards
        return metadata

    def _distance_to_stop_sign(self, vehicle: Any) -> Optional[float]:
        """Best-effort OpenDRIVE landmark lookup for nearby stop signs."""

        if carla is None:
            return None
        try:
            world = vehicle.get_world()
            carla_map = world.get_map()
            location = vehicle.get_location()
            waypoint = carla_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None or not hasattr(waypoint, "get_landmarks"):
                return None
            landmarks = waypoint.get_landmarks(14.0, True)
        except RuntimeError:
            return None

        best_distance: Optional[float] = None
        for landmark in landmarks:
            name = str(getattr(landmark, "name", "")).lower()
            landmark_type = str(getattr(landmark, "type", "")).lower()
            if "stop" not in name and landmark_type not in {"206", "stop"}:
                continue
            transform = getattr(landmark, "transform", None)
            if transform is None:
                continue
            landmark_location = transform.location
            dx = float(location.x - landmark_location.x)
            dy = float(location.y - landmark_location.y)
            dz = float(location.z - landmark_location.z)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            best_distance = distance if best_distance is None else min(best_distance, distance)
        return best_distance


class DataCollector:
    """Orchestrate multiple autopilot episodes with random ego respawns."""

    def __init__(
        self,
        episode_config: Optional[EpisodeConfig] = None,
        *,
        carla_config: Optional[CarlaClientConfig] = None,
        camera_config: Optional[RGBCameraConfig] = None,
        on_frame_collected: FrameCallback = None,
    ) -> None:
        self.episode_config = episode_config or EpisodeConfig()
        self.carla_config = carla_config or CarlaClientConfig(autopilot=True)
        self.camera_config = camera_config or RGBCameraConfig()
        self.on_frame_collected = on_frame_collected
        self._rng = random.Random(self.episode_config.seed)

    def collect(self) -> List[Path]:
        """Collect n_episodes and return the created episode directories."""

        client = CarlaClient(self.carla_config)
        episode_paths: List[Path] = []
        used_spawn_indices: Set[int] = set()
        world: Optional[Any] = None
        scenario_actors: Dict[str, List[Any]] = {
            "vehicles": [],
            "walkers": [],
            "walker_controllers": [],
        }

        try:
            world = client.connect()
            scenario_actors = self._spawn_scenario_actors(client, world, ego_vehicle=None)
            episode_collector = EpisodeCollector(
                self.episode_config,
                on_frame_collected=self.on_frame_collected,
            )

            for episode_idx in range(self.episode_config.n_episodes):
                self._apply_episode_weather(world, episode_idx)
                vehicle = self._spawn_vehicle(client, world, used_spawn_indices)
                camera = RGBCameraSensor(self.camera_config, client=client)
                camera.spawn(world, vehicle)

                try:
                    LOGGER.info("Starting episode %s/%s", episode_idx + 1, self.episode_config.n_episodes)
                    episode_paths.append(
                        episode_collector.collect_episode(world, vehicle, camera)
                    )
                finally:
                    self._destroy_episode_actors(client, world, vehicle, camera)

        finally:
            if world is not None:
                self._destroy_scenario_actors(client, world, scenario_actors)
            client.cleanup()

        return episode_paths

    def _spawn_vehicle(
        self,
        client: CarlaClient,
        world: Any,
        used_spawn_indices: Set[int],
    ) -> Any:
        spawn_points = list(world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError("CARLA map has no available vehicle spawn points.")

        candidate_indices = self._candidate_spawn_indices(
            world,
            spawn_points,
            used_spawn_indices,
        )
        blueprint = self._make_ego_blueprint(world)

        for index in candidate_indices[: self.carla_config.max_spawn_attempts]:
            vehicle = world.try_spawn_actor(blueprint, spawn_points[index])
            if vehicle is None:
                continue

            used_spawn_indices.add(index)
            client.ego_vehicle = vehicle
            client.register_actor(vehicle)
            client.set_autopilot(True)
            self._configure_traffic_manager_actor(
                client.traffic_manager,
                vehicle,
                speed_difference=self.episode_config.ego_speed_difference,
            )
            LOGGER.info("Spawned episode ego vehicle at spawn point %s", index)
            return vehicle

        raise RuntimeError("Failed to spawn ego vehicle for data collection episode.")

    def _candidate_spawn_indices(
        self,
        world: Any,
        spawn_points: List[Any],
        used_spawn_indices: Set[int],
    ) -> List[int]:
        available = [idx for idx in range(len(spawn_points)) if idx not in used_spawn_indices]
        if not available:
            LOGGER.warning("All spawn points were used; reusing spawn points from this episode onward.")
            used_spawn_indices.clear()
            available = list(range(len(spawn_points)))

        first = self._choose_first_spawn_index(world, spawn_points, available)
        unused_remaining = [idx for idx in available if idx != first]
        used_fallback = [idx for idx in range(len(spawn_points)) if idx not in available]
        self._rng.shuffle(unused_remaining)
        self._rng.shuffle(used_fallback)
        return [first] + unused_remaining + used_fallback

    def _choose_first_spawn_index(
        self,
        world: Any,
        spawn_points: List[Any],
        available: List[int],
    ) -> int:
        focus = self.episode_config.spawn_focus.lower().strip()
        if focus in {"", "random", "none"}:
            return self._rng.choice(available)

        scored = [
            (
                idx,
                self._spawn_focus_score(world, spawn_points[idx], focus)
                + self._rng.random() * 0.05,
            )
            for idx in available
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        positive = [item for item in scored if item[1] > 0.05]
        if not positive:
            LOGGER.debug("No spawn points matched focus=%s; using random spawn.", focus)
            return self._rng.choice(available)

        candidate_count = max(1, min(len(positive), max(8, len(positive) // 4)))
        return self._rng.choice([idx for idx, _ in positive[:candidate_count]])

    def _spawn_focus_score(self, world: Any, transform: Any, focus: str) -> float:
        score = 0.0
        if focus in {"junction", "traffic_law", "stop_or_light", "traffic_light"}:
            score += self._junction_score(world, transform)
        if focus in {"traffic_light", "traffic_law", "stop_or_light"}:
            score += self._traffic_light_score(world, transform)
        if focus in {"stop_or_light", "traffic_law"}:
            score += self._stop_sign_score(world, transform)
        return score

    def _junction_score(self, world: Any, transform: Any) -> float:
        if carla is None:
            return 0.0
        try:
            waypoint = world.get_map().get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            return 0.0
        if waypoint is None:
            return 0.0
        if bool(getattr(waypoint, "is_junction", False)):
            return 2.0

        score = 0.0
        frontier = [waypoint]
        for step in range(1, 7):
            next_frontier: List[Any] = []
            for candidate in frontier:
                try:
                    next_frontier.extend(candidate.next(8.0))
                except RuntimeError:
                    continue
            if not next_frontier:
                break
            if any(bool(getattr(item, "is_junction", False)) for item in next_frontier):
                score = max(score, 1.0 / float(step))
                break
            frontier = next_frontier[:4]
        return score

    def _traffic_light_score(self, world: Any, transform: Any) -> float:
        try:
            lights = list(world.get_actors().filter("traffic.traffic_light*"))
        except RuntimeError:
            return 0.0
        if not lights:
            return 0.0

        best_distance: Optional[float] = None
        for light in lights:
            try:
                distance = self._location_distance(transform.location, light.get_location())
            except RuntimeError:
                continue
            best_distance = distance if best_distance is None else min(best_distance, distance)
        if best_distance is None or best_distance > 80.0:
            return 0.0
        return max(0.0, 1.5 * (1.0 - best_distance / 80.0))

    def _stop_sign_score(self, world: Any, transform: Any) -> float:
        if carla is None:
            return 0.0
        try:
            waypoint = world.get_map().get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None or not hasattr(waypoint, "get_landmarks"):
                return 0.0
            landmarks = waypoint.get_landmarks(45.0, True)
        except RuntimeError:
            return 0.0

        for landmark in landmarks:
            name = str(getattr(landmark, "name", "")).lower()
            landmark_type = str(getattr(landmark, "type", "")).lower()
            if "stop" in name or landmark_type in {"206", "stop"}:
                return 1.5
        return 0.0

    def _make_ego_blueprint(self, world: Any) -> Any:
        blueprints = list(world.get_blueprint_library().filter(self.carla_config.ego_vehicle_filter))
        if not blueprints:
            raise RuntimeError(
                f"No vehicle blueprint matches filter: {self.carla_config.ego_vehicle_filter}"
            )

        blueprint = self._rng.choice(blueprints)
        blueprint.set_attribute("role_name", self.carla_config.ego_role_name)
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", self._rng.choice(colors))
        return blueprint

    def _apply_episode_weather(self, world: Any, episode_idx: int) -> None:
        presets = tuple(preset for preset in self.episode_config.weather_presets if preset)
        if not presets or carla is None:
            return

        preset_name = presets[episode_idx % len(presets)]
        weather = getattr(carla.WeatherParameters, preset_name, None)
        if weather is None:
            LOGGER.warning("Unknown CARLA weather preset ignored: %s", preset_name)
            return
        world.set_weather(weather)
        LOGGER.info("Applied weather preset: %s", preset_name)

    def _spawn_scenario_actors(
        self,
        client: CarlaClient,
        world: Any,
        ego_vehicle: Optional[Any],
    ) -> Dict[str, List[Any]]:
        """Populate the map with vehicles, two-wheelers, and VRU walkers."""

        actors: Dict[str, List[Any]] = {
            "vehicles": [],
            "walkers": [],
            "walker_controllers": [],
        }
        if carla is None:
            return actors

        if self.episode_config.npc_vehicles > 0 or self.episode_config.npc_two_wheelers > 0:
            actors["vehicles"].extend(
                self._spawn_npc_vehicles(client, world, ego_vehicle, two_wheelers=False)
            )
            actors["vehicles"].extend(
                self._spawn_npc_vehicles(client, world, ego_vehicle, two_wheelers=True)
            )

        if self.episode_config.npc_walkers > 0:
            walkers, controllers = self._spawn_walkers(world)
            actors["walkers"].extend(walkers)
            actors["walker_controllers"].extend(controllers)

        total = sum(len(value) for value in actors.values())
        if total:
            LOGGER.info(
                "Spawned scenario actors: %s vehicles, %s walkers.",
                len(actors["vehicles"]),
                len(actors["walkers"]),
            )
            self._settle_world(world, 8)
        return actors

    def _spawn_npc_vehicles(
        self,
        client: CarlaClient,
        world: Any,
        ego_vehicle: Optional[Any],
        *,
        two_wheelers: bool,
    ) -> List[Any]:
        target_count = (
            self.episode_config.npc_two_wheelers
            if two_wheelers
            else self.episode_config.npc_vehicles
        )
        if target_count <= 0:
            return []

        blueprints = self._vehicle_blueprints(world, two_wheelers=two_wheelers)
        if not blueprints:
            LOGGER.warning("No %s blueprints found.", "two-wheeler" if two_wheelers else "vehicle")
            return []

        spawn_points = list(world.get_map().get_spawn_points())
        self._rng.shuffle(spawn_points)
        ego_location = ego_vehicle.get_location() if ego_vehicle is not None else None
        spawned: List[Any] = []

        for transform in spawn_points:
            if len(spawned) >= target_count:
                break
            if ego_location is not None and self._location_distance(ego_location, transform.location) < 10.0:
                continue

            blueprint = self._rng.choice(blueprints)
            self._randomize_vehicle_blueprint(blueprint)
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            actor.set_autopilot(True, self.carla_config.traffic_manager_port)
            self._configure_traffic_manager_actor(
                client.traffic_manager,
                actor,
                speed_difference=self.episode_config.traffic_speed_difference,
            )
            spawned.append(actor)
        return spawned

    def _vehicle_blueprints(self, world: Any, *, two_wheelers: bool) -> List[Any]:
        blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
        if not two_wheelers:
            return [
                blueprint
                for blueprint in blueprints
                if not self._is_two_wheeler_blueprint(blueprint)
            ]
        return [
            blueprint
            for blueprint in blueprints
            if self._is_two_wheeler_blueprint(blueprint)
        ]

    @staticmethod
    def _is_two_wheeler_blueprint(blueprint: Any) -> bool:
        type_id = str(getattr(blueprint, "id", "")).lower()
        keywords = (
            "bike",
            "bicycle",
            "crossbike",
            "century",
            "omafiets",
            "motorcycle",
            "harley",
            "kawasaki",
            "yamaha",
            "vespa",
            "scooter",
        )
        return any(keyword in type_id for keyword in keywords)

    def _randomize_vehicle_blueprint(self, blueprint: Any) -> None:
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", self._rng.choice(colors))
        if blueprint.has_attribute("driver_id"):
            drivers = blueprint.get_attribute("driver_id").recommended_values
            if drivers:
                blueprint.set_attribute("driver_id", self._rng.choice(drivers))
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")

    def _spawn_walkers(self, world: Any) -> Tuple[List[Any], List[Any]]:
        blueprint_library = world.get_blueprint_library()
        walker_blueprints = list(blueprint_library.filter("walker.pedestrian.*"))
        if not walker_blueprints:
            LOGGER.warning("No pedestrian walker blueprints found.")
            return [], []

        try:
            controller_bp = blueprint_library.find("controller.ai.walker")
        except RuntimeError:
            LOGGER.warning("No walker AI controller blueprint found.")
            return [], []

        if hasattr(world, "set_pedestrians_cross_factor"):
            world.set_pedestrians_cross_factor(
                float(self.episode_config.pedestrian_cross_factor)
            )

        walkers: List[Any] = []
        controllers: List[Any] = []
        for _ in range(self.episode_config.npc_walkers):
            location = world.get_random_location_from_navigation()
            if location is None:
                continue
            spawn_point = carla.Transform(location)
            walker_bp = self._rng.choice(walker_blueprints)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            walker = world.try_spawn_actor(walker_bp, spawn_point)
            if walker is None:
                continue
            controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
            if controller is None:
                walker.destroy()
                continue

            walkers.append(walker)
            controllers.append(controller)
            try:
                controller.start()
                destination = world.get_random_location_from_navigation()
                if destination is not None:
                    controller.go_to_location(destination)
                controller.set_max_speed(self._rng.uniform(0.8, 2.2))
            except RuntimeError as exc:
                LOGGER.debug("Could not start walker controller: %s", exc)
        return walkers, controllers

    def _configure_traffic_manager_actor(
        self,
        traffic_manager: Any,
        actor: Any,
        *,
        speed_difference: float,
    ) -> None:
        if traffic_manager is None:
            return
        self._safe_tm_call(traffic_manager, "ignore_lights_percentage", actor, 0.0)
        self._safe_tm_call(traffic_manager, "ignore_signs_percentage", actor, 0.0)
        self._safe_tm_call(traffic_manager, "ignore_walkers_percentage", actor, 0.0)
        self._safe_tm_call(traffic_manager, "vehicle_percentage_speed_difference", actor, speed_difference)
        self._safe_tm_call(traffic_manager, "distance_to_leading_vehicle", actor, 2.5)

    @staticmethod
    def _safe_tm_call(traffic_manager: Any, method_name: str, *args: Any) -> None:
        method = getattr(traffic_manager, method_name, None)
        if method is None:
            return
        try:
            method(*args)
        except RuntimeError as exc:
            LOGGER.debug("Ignoring Traffic Manager call failure %s: %s", method_name, exc)

    @staticmethod
    def _location_distance(first: Any, second: Any) -> float:
        dx = float(first.x - second.x)
        dy = float(first.y - second.y)
        dz = float(first.z - second.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _destroy_scenario_actors(
        client: CarlaClient,
        world: Any,
        actors: Dict[str, List[Any]],
    ) -> None:
        commands = []
        for group_name in ("walker_controllers", "walkers", "vehicles"):
            for actor in actors.get(group_name, []):
                actor_id = DataCollector._safe_actor_id(actor)
                if actor_id is not None:
                    commands.append(carla.command.DestroyActor(actor_id))
        if not commands:
            return
        try:
            client.client.apply_batch_sync(commands, True)
        except RuntimeError as exc:
            LOGGER.debug("Ignoring scenario actor batch destroy failure: %s", exc)
        DataCollector._settle_world(world, 2)

    @staticmethod
    def _safe_actor_id(actor: Any) -> Optional[int]:
        try:
            return int(actor.id)
        except RuntimeError:
            return None

    @staticmethod
    def _destroy_episode_actors(
        client: CarlaClient,
        world: Any,
        vehicle: Any,
        camera: RGBCameraSensor,
    ) -> None:
        camera.stop()
        DataCollector._settle_world(world, 2)
        camera.destroy()
        DataCollector._settle_world(world, 2)
        try:
            if client.ego_vehicle is vehicle:
                client.set_autopilot(False)
        except RuntimeError as exc:
            LOGGER.debug("Ignoring autopilot disable failure during collection cleanup: %s", exc)
        DataCollector._settle_world(world, 2)
        client.destroy_actor(vehicle)
        if client.ego_vehicle is vehicle:
            client.ego_vehicle = None
            client._autopilot_enabled = False
        DataCollector._settle_world(world, 2)

    @staticmethod
    def _settle_world(world: Any, ticks: int) -> None:
        for _ in range(max(0, int(ticks))):
            settings = world.get_settings()
            try:
                if settings.synchronous_mode:
                    world.tick()
                else:
                    world.wait_for_tick(1.0)
            except RuntimeError as exc:
                LOGGER.debug("Ignoring settle tick failure during collection cleanup: %s", exc)
                break
