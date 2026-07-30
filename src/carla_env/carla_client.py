"""CARLA client wrapper for connecting, spawning, and controlling the ego car."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import carla
except ImportError as exc:  # pragma: no cover - exercised only without CARLA installed.
    carla = None
    _CARLA_IMPORT_ERROR = exc
else:
    _CARLA_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CarlaClientConfig:
    """Connection and ego-vehicle settings for a local CARLA server."""

    host: str = "127.0.0.1"
    port: int = 2000
    timeout_seconds: float = 10.0
    tick_timeout_seconds: float = 0.1
    map_name: Optional[str] = None
    seed: int = 42

    synchronous_mode: bool = False
    fixed_delta_seconds: Optional[float] = 0.05

    traffic_manager_port: int = 8000
    traffic_manager_seed: int = 42

    ego_vehicle_filter: str = "vehicle.tesla.model3"
    ego_spawn_index: Optional[int] = None
    ego_spawn_preset: Optional[str] = None
    ego_spawn_top_k: int = 1
    ego_role_name: str = "ego"
    autopilot: bool = True
    max_spawn_attempts: int = 30


@dataclass(frozen=True)
class VehicleState:
    """Serializable snapshot of ego pose, speed, and current control values."""

    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    acceleration: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    speed_mps: float
    speed_kmh: float
    control: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict that can be logged or written as JSON metadata."""

        return asdict(self)


@dataclass(frozen=True)
class SafetyStatus:
    """Safety-monitor output used before applying model control."""

    speed_kmh: float
    waypoint_distance_m: Optional[float]
    speed_limit_exceeded: bool
    off_road: bool
    forced_brake: bool
    message: Optional[str] = None
    traffic_light_state: Optional[str] = None
    stop_sign_distance_m: Optional[float] = None
    traffic_control_blocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict for UI overlays and logging."""

        return asdict(self)


class CarlaClient:
    """Small lifecycle manager around CARLA's Python API for the ego vehicle."""

    def __init__(self, config: Optional[CarlaClientConfig] = None) -> None:
        self.config = config or CarlaClientConfig()
        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.traffic_manager: Optional[Any] = None
        self.ego_vehicle: Optional[Any] = None
        self.spawned_actors: List[Any] = []
        self._spawned_actor_ids: set[int] = set()
        self._original_world_settings: Optional[Any] = None
        self._autopilot_enabled = False
        self._synchronous_mode_active = False
        self._stop_sign_hold_until = 0.0
        self._stop_sign_release_until = 0.0

    def connect(self) -> Any:
        """Connect to CARLA, optionally load a map, and configure simulation timing."""

        self._ensure_carla_available()

        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout_seconds)

        self.world = self.client.get_world()
        if self.config.map_name:
            current_map = self.world.get_map().name
            if not current_map.endswith(self.config.map_name):
                LOGGER.info("Loading CARLA map %s", self.config.map_name)
                self.world = self.client.load_world(self.config.map_name)
                self._wait_until_world_ready(self.world, self.config.map_name)

        self.traffic_manager = self._get_traffic_manager()
        self.traffic_manager.set_random_device_seed(self.config.traffic_manager_seed)

        if self.config.synchronous_mode:
            self._enable_synchronous_mode()
        else:
            self._disable_stale_synchronous_mode()

        LOGGER.info(
            "Connected to CARLA at %s:%s", self.config.host, self.config.port
        )
        return self.world

    def spawn_ego_vehicle(self) -> Any:
        """Spawn the ego vehicle at a deterministic spawn point when possible."""

        world = self._require_world()
        blueprint = self._select_ego_blueprint()
        spawn_points = list(world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError("CARLA map has no available vehicle spawn points.")

        ordered_spawn_points = self._ordered_spawn_points(spawn_points)
        attempts = min(self.config.max_spawn_attempts, len(ordered_spawn_points))

        for transform in ordered_spawn_points[:attempts]:
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is None:
                continue

            self.ego_vehicle = vehicle
            self.register_actor(vehicle)
            self.set_autopilot(self.config.autopilot)
            LOGGER.info("Spawned ego vehicle: %s", vehicle.type_id)
            return vehicle

        raise RuntimeError(
            f"Failed to spawn ego vehicle after {attempts} attempts. "
            "Try another map, spawn index, or restart the CARLA world."
        )

    def set_autopilot(self, enabled: bool = True) -> None:
        """Enable or disable CARLA Traffic Manager autopilot for the ego vehicle."""

        vehicle = self._require_ego_vehicle()
        vehicle.set_autopilot(enabled, self.config.traffic_manager_port)
        self._autopilot_enabled = enabled

    def apply_manual_control(
        self,
        steering: float = 0.0,
        throttle: float = 0.0,
        brake: float = 0.0,
        *,
        hand_brake: bool = False,
        reverse: bool = False,
        disable_autopilot: bool = True,
    ) -> None:
        """Send normalized manual driving commands to the ego vehicle."""

        vehicle = self._require_ego_vehicle()
        if disable_autopilot and self._autopilot_enabled:
            self.set_autopilot(False)

        control = carla.VehicleControl(
            steer=self._clamp(steering, -1.0, 1.0),
            throttle=self._clamp(throttle, 0.0, 1.0),
            brake=self._clamp(brake, 0.0, 1.0),
            hand_brake=hand_brake,
            reverse=reverse,
        )
        vehicle.apply_control(control)

    def apply_vla_control(
        self,
        steering: float,
        throttle: float,
        brake: float,
        *,
        max_speed_kmh: float = 80.0,
        off_road_distance_m: float = 3.0,
        safety_status: Optional[SafetyStatus] = None,
    ) -> SafetyStatus:
        """Apply VLA control after enforcing speed and off-road safety limits."""

        status = safety_status or self.get_safety_status(
            max_speed_kmh=max_speed_kmh,
            off_road_distance_m=off_road_distance_m,
        )
        if status.forced_brake:
            throttle = 0.0
            brake = 1.0

        self.apply_manual_control(
            steering=steering,
            throttle=throttle,
            brake=brake,
            disable_autopilot=True,
        )
        return status

    def get_vehicle_state(self) -> VehicleState:
        """Read the ego vehicle state in a logging-friendly Python structure."""

        vehicle = self._require_ego_vehicle()
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        angular_velocity = vehicle.get_angular_velocity()
        control = vehicle.get_control()

        speed_mps = math.sqrt(
            velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
        )

        return VehicleState(
            location=self._vector_to_tuple(transform.location),
            rotation=(
                float(transform.rotation.pitch),
                float(transform.rotation.yaw),
                float(transform.rotation.roll),
            ),
            velocity=self._vector_to_tuple(velocity),
            acceleration=self._vector_to_tuple(acceleration),
            angular_velocity=self._vector_to_tuple(angular_velocity),
            speed_mps=float(speed_mps),
            speed_kmh=float(speed_mps * 3.6),
            control={
                "steering": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
                "hand_brake": bool(control.hand_brake),
                "reverse": bool(control.reverse),
                "manual_gear_shift": bool(control.manual_gear_shift),
                "gear": int(control.gear),
                "autopilot": self._autopilot_enabled,
            },
        )

    def get_safety_status(
        self,
        *,
        max_speed_kmh: float = 80.0,
        off_road_distance_m: float = 3.0,
    ) -> SafetyStatus:
        """Check speed and road proximity before applying autonomous commands."""

        state = self.get_vehicle_state()
        waypoint_distance = self._distance_to_driving_waypoint()
        vehicle = self._require_ego_vehicle()
        traffic_light_state = self._current_traffic_light_state(vehicle)
        stop_sign_distance = self._distance_to_stop_sign(vehicle)
        speed_limit_exceeded = state.speed_kmh > max_speed_kmh
        off_road = (
            waypoint_distance is not None
            and waypoint_distance > off_road_distance_m
        )
        red_or_yellow_light = str(traffic_light_state or "").lower() in {"red", "yellow"}
        stop_sign_blocked = self._stop_sign_should_block(
            speed_kmh=state.speed_kmh,
            distance_m=stop_sign_distance,
        )
        traffic_control_blocked = red_or_yellow_light or stop_sign_blocked
        forced_brake = speed_limit_exceeded or off_road or traffic_control_blocked

        message = None
        if red_or_yellow_light:
            message = f"SAFETY STOP: {str(traffic_light_state).upper()} LIGHT"
        elif stop_sign_blocked:
            message = "SAFETY STOP: STOP SIGN"
        elif off_road:
            message = "⚠️ OFF ROAD"
        elif speed_limit_exceeded:
            message = "⚠️ SPEED > 80 KM/H"

        return SafetyStatus(
            speed_kmh=state.speed_kmh,
            waypoint_distance_m=waypoint_distance,
            speed_limit_exceeded=speed_limit_exceeded,
            off_road=off_road,
            forced_brake=forced_brake,
            message=message,
            traffic_light_state=traffic_light_state,
            stop_sign_distance_m=stop_sign_distance,
            traffic_control_blocked=traffic_control_blocked,
        )

    def _current_traffic_light_state(self, vehicle: Any) -> Optional[str]:
        try:
            if not vehicle.is_at_traffic_light():
                return None
            traffic_light = vehicle.get_traffic_light()
            if traffic_light is None:
                return None
            return str(traffic_light.get_state()).split(".")[-1]
        except RuntimeError:
            return None

    def _stop_sign_should_block(self, *, speed_kmh: float, distance_m: Optional[float]) -> bool:
        now = time.monotonic()
        if distance_m is None or distance_m > 9.0:
            self._stop_sign_hold_until = 0.0
            self._stop_sign_release_until = 0.0
            return False
        if now < self._stop_sign_release_until:
            return False
        if speed_kmh > 0.8:
            return True
        if self._stop_sign_hold_until <= now:
            self._stop_sign_hold_until = now + 3.0
            return True
        if now < self._stop_sign_hold_until:
            return True
        self._stop_sign_release_until = now + 6.0
        return False

    def _distance_to_stop_sign(self, vehicle: Any) -> Optional[float]:
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

    def get_lane_follow_steering(
        self,
        lookahead_m: float = 8.0,
        *,
        heading_gain: float = 0.70,
        lateral_gain: float = 0.60,
    ) -> Optional[float]:
        """Estimate steering toward the lane center and next driving waypoint."""

        world = self._require_world()
        vehicle = self._require_ego_vehicle()
        carla_map = world.get_map()
        transform = vehicle.get_transform()
        location = transform.location

        try:
            waypoint = carla_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError as exc:
            LOGGER.debug("Could not query lane steering waypoint: %s", exc)
            return None

        if waypoint is None:
            return None

        next_waypoints = waypoint.next(max(1.0, float(lookahead_m)))
        target = (
            next_waypoints[0].transform.location
            if next_waypoints
            else waypoint.transform.location
        )

        dx = float(target.x - location.x)
        dy = float(target.y - location.y)
        distance = math.hypot(dx, dy)
        if distance < 1e-3:
            return 0.0

        dx /= distance
        dy /= distance
        yaw = math.radians(float(transform.rotation.yaw))
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        cross = forward_x * dy - forward_y * dx
        dot = forward_x * dx + forward_y * dy
        # CARLA steering uses positive values for right turns. The 2-D cross
        # product is positive when the target is left of the vehicle, so invert it.
        heading_error = math.atan2(-cross, dot)
        heading_steer = heading_error / math.radians(45.0)

        waypoint_location = waypoint.transform.location
        lane_yaw = math.radians(float(waypoint.transform.rotation.yaw))
        right_x = math.cos(lane_yaw + math.pi / 2.0)
        right_y = math.sin(lane_yaw + math.pi / 2.0)
        lateral_error = (
            float(location.x - waypoint_location.x) * right_x
            + float(location.y - waypoint_location.y) * right_y
        )
        lane_width = max(2.0, float(getattr(waypoint, "lane_width", 3.5) or 3.5))
        lateral_steer = -lateral_error / (lane_width * 0.5)

        steering = heading_gain * heading_steer + lateral_gain * lateral_steer
        return self._clamp(steering, -1.0, 1.0)

    def tick(self, timeout_seconds: Optional[float] = None) -> Any:
        """Advance one synchronous frame, or wait for the next async frame."""

        world = self._require_world()
        if self._synchronous_mode_active:
            return world.tick()

        timeout = (
            self.config.tick_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        try:
            return world.wait_for_tick(timeout)
        except RuntimeError as exc:
            LOGGER.debug("Timed out waiting for CARLA tick: %s", exc)
            return None

    def register_actor(self, actor: Any) -> None:
        """Track a spawned actor so cleanup can destroy it later."""

        if actor is None:
            return

        actor_id = int(actor.id)
        if actor_id not in self._spawned_actor_ids:
            self.spawned_actors.append(actor)
            self._spawned_actor_ids.add(actor_id)

    def unregister_actor(self, actor: Any) -> None:
        """Forget one actor without calling CARLA methods on it."""

        if actor is None:
            return

        try:
            actor_id = int(actor.id)
        except RuntimeError:
            actor_id = None

        self.spawned_actors = [
            existing
            for existing in self.spawned_actors
            if existing is not actor and self._safe_actor_id(existing) != actor_id
        ]
        if actor_id is not None:
            self._spawned_actor_ids.discard(actor_id)

    def destroy_actor(self, actor: Any) -> None:
        """Destroy one actor and remove it from this client's cleanup list."""

        if actor is None:
            return

        self.unregister_actor(actor)
        try:
            actor.destroy()
        except RuntimeError as exc:
            LOGGER.debug("Ignoring CARLA actor destroy failure: %s", exc)

    def cleanup(self) -> None:
        """Disable autopilot, destroy spawned actors, and restore world settings."""

        for actor in reversed(self.spawned_actors):
            try:
                if self.ego_vehicle is not None and actor is self.ego_vehicle:
                    actor.set_autopilot(False, self.config.traffic_manager_port)
                actor.destroy()
            except RuntimeError as exc:
                LOGGER.warning("Failed to destroy CARLA actor: %s", exc)

        self.spawned_actors.clear()
        self._spawned_actor_ids.clear()
        self.ego_vehicle = None
        self._autopilot_enabled = False

        if self.world is not None and self._original_world_settings is not None:
            self.world.apply_settings(self._original_world_settings)
            self._original_world_settings = None

        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(False)
        self._synchronous_mode_active = False

    def close(self) -> None:
        """Alias for cleanup to make the client usable in generic shutdown code."""

        self.cleanup()

    def __enter__(self) -> "CarlaClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()

    def _enable_synchronous_mode(self) -> None:
        world = self._require_world()
        self._original_world_settings = world.get_settings()

        settings = world.get_settings()
        settings.synchronous_mode = True
        if self.config.fixed_delta_seconds is not None:
            settings.fixed_delta_seconds = self.config.fixed_delta_seconds
        world.apply_settings(settings)

        self.traffic_manager.set_synchronous_mode(True)
        self._synchronous_mode_active = True

    def _disable_stale_synchronous_mode(self) -> None:
        """Recover from a previous run that left the CARLA world in sync mode."""

        world = self._require_world()
        settings = world.get_settings()
        if not settings.synchronous_mode:
            if self.traffic_manager is not None:
                self.traffic_manager.set_synchronous_mode(False)
            self._synchronous_mode_active = False
            return

        LOGGER.warning(
            "CARLA world was already in synchronous mode; switching it back to async."
        )
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(False)
        self._synchronous_mode_active = False

    def _wait_until_world_ready(self, world: Any, map_name: str) -> None:
        """Wait after a map load until CARLA responds to basic world queries."""

        deadline = time.monotonic() + max(30.0, self.config.timeout_seconds * 2.0)
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            try:
                loaded_map = world.get_map()
                spawn_points = loaded_map.get_spawn_points()
                if loaded_map.name.endswith(map_name) and spawn_points:
                    try:
                        world.wait_for_tick(2.0)
                    except RuntimeError:
                        pass
                    LOGGER.info(
                        "CARLA map %s ready with %s spawn points.",
                        loaded_map.name,
                        len(spawn_points),
                    )
                    return
            except RuntimeError as exc:
                last_error = exc
            time.sleep(2.0)

        raise RuntimeError(
            f"CARLA map {map_name} did not become ready after loading."
        ) from last_error

    def _get_traffic_manager(self) -> Any:
        """Get Traffic Manager with retries because map loading can stall RPCs."""

        attempts = max(3, int(self.config.timeout_seconds // 30) + 1)
        last_error: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                return self.client.get_trafficmanager(self.config.traffic_manager_port)
            except RuntimeError as exc:
                last_error = exc
                LOGGER.warning(
                    "Traffic Manager not ready yet (%s/%s): %s",
                    attempt,
                    attempts,
                    exc,
                )
                time.sleep(3.0)
        raise RuntimeError("CARLA Traffic Manager did not become ready.") from last_error

    def _select_ego_blueprint(self) -> Any:
        world = self._require_world()
        blueprints = list(world.get_blueprint_library().filter(self.config.ego_vehicle_filter))
        if not blueprints:
            raise RuntimeError(
                f"No vehicle blueprint matches filter: {self.config.ego_vehicle_filter}"
            )

        rng = random.Random(self.config.seed)
        blueprint = rng.choice(blueprints)
        blueprint.set_attribute("role_name", self.config.ego_role_name)

        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", rng.choice(colors))

        return blueprint

    def _ordered_spawn_points(self, spawn_points: Sequence[Any]) -> List[Any]:
        spawn_points = list(spawn_points)
        if self.config.ego_spawn_index is not None:
            index = self.config.ego_spawn_index
            if index < 0 or index >= len(spawn_points):
                raise IndexError(
                    f"ego_spawn_index={index} is outside available range "
                    f"0..{len(spawn_points) - 1}"
                )
            return [spawn_points[index]] + [
                point for idx, point in enumerate(spawn_points) if idx != index
            ]

        if self.config.ego_spawn_preset in {
            "straight",
            "straight_turn",
            "junction",
            "traffic_law",
            "traffic_light",
            "stop_or_light",
        }:
            scorers = {
                "straight": self._score_straight_spawn_point,
                "straight_turn": self._score_straight_then_turn_spawn_point,
                "junction": self._score_junction_spawn_point,
                "traffic_law": self._score_traffic_law_spawn_point,
                "traffic_light": lambda point: (
                    self._score_junction_spawn_point(point)
                    + 2.0 * self._score_near_traffic_light(point)
                ),
                "stop_or_light": self._score_traffic_law_spawn_point,
            }
            scorer = scorers[self.config.ego_spawn_preset]
            scored_points = [
                (scorer(point), idx, point)
                for idx, point in enumerate(spawn_points)
            ]
            scored_points.sort(key=lambda item: item[0], reverse=True)
            top_k = max(1, min(int(self.config.ego_spawn_top_k), len(scored_points)))
            selected_offset = 0
            if top_k > 1:
                selected_offset = random.Random(self.config.seed).randrange(top_k)
                selected = scored_points.pop(selected_offset)
                scored_points.insert(0, selected)
            best_score, best_index, _ = scored_points[0]
            LOGGER.info(
                "Selected %s spawn preset: index=%s score=%.2f top_k=%s",
                self.config.ego_spawn_preset,
                best_index,
                best_score,
                top_k,
            )
            return [point for _, _, point in scored_points]

        rng = random.Random(self.config.seed)
        rng.shuffle(spawn_points)
        return spawn_points

    def _score_straight_spawn_point(self, transform: Any) -> float:
        """Prefer spawn points with long, straight, non-junction road ahead."""

        world = self._require_world()
        carla_map = world.get_map()
        try:
            waypoint = carla_map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            return float("-inf")

        if waypoint is None:
            return float("-inf")

        score = 0.0
        current = waypoint
        previous_yaw = float(current.transform.rotation.yaw)
        if current.is_junction:
            score -= 250.0

        for step in range(1, 25):
            next_waypoints = current.next(2.5)
            if not next_waypoints:
                break

            next_waypoint = min(
                next_waypoints,
                key=lambda candidate: abs(
                    self._angle_delta_degrees(
                        previous_yaw,
                        float(candidate.transform.rotation.yaw),
                    )
                ),
            )
            yaw_delta = abs(
                self._angle_delta_degrees(
                    previous_yaw,
                    float(next_waypoint.transform.rotation.yaw),
                )
            )
            score += 2.5
            score -= yaw_delta * 0.7
            if next_waypoint.is_junction:
                score -= max(0.0, 120.0 - step * 5.0)

            current = next_waypoint
            previous_yaw = float(current.transform.rotation.yaw)

        return score

    def _score_junction_spawn_point(self, transform: Any) -> float:
        """Prefer spawn points at or near a junction."""

        world = self._require_world()
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
            return 3.0

        frontier = [waypoint]
        for step in range(1, 8):
            next_frontier: List[Any] = []
            for candidate in frontier:
                try:
                    next_frontier.extend(candidate.next(8.0))
                except RuntimeError:
                    continue
            if not next_frontier:
                break
            if any(bool(getattr(item, "is_junction", False)) for item in next_frontier):
                return 2.0 / float(step)
            frontier = next_frontier[:4]
        return 0.0

    def _score_traffic_law_spawn_point(self, transform: Any) -> float:
        """Prefer junctions that are close to traffic lights or stop signs."""

        return (
            self._score_junction_spawn_point(transform)
            + self._score_near_traffic_light(transform)
            + self._score_near_stop_sign(transform)
        )

    def _score_near_traffic_light(self, transform: Any) -> float:
        world = self._require_world()
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
        return max(0.0, 2.0 * (1.0 - best_distance / 80.0))

    def _score_near_stop_sign(self, transform: Any) -> float:
        world = self._require_world()
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
                return 2.0
        return 0.0

    def _score_straight_then_turn_spawn_point(self, transform: Any) -> float:
        """Prefer a straight launch segment followed by a turn or junction."""

        world = self._require_world()
        carla_map = world.get_map()
        try:
            waypoint = carla_map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            return float("-inf")

        if waypoint is None or waypoint.is_junction:
            return float("-inf")

        score = 0.0
        current = waypoint
        initial_yaw = float(current.transform.rotation.yaw)
        previous_yaw = initial_yaw
        saw_turn = False

        for step in range(1, 37):
            next_waypoints = current.next(2.5)
            if not next_waypoints:
                break

            branch_turn = max(
                abs(
                    self._angle_delta_degrees(
                        previous_yaw,
                        float(candidate.transform.rotation.yaw),
                    )
                )
                for candidate in next_waypoints
            )
            next_waypoint = min(
                next_waypoints,
                key=lambda candidate: abs(
                    self._angle_delta_degrees(
                        previous_yaw,
                        float(candidate.transform.rotation.yaw),
                    )
                ),
            )
            yaw_from_start = abs(
                self._angle_delta_degrees(
                    initial_yaw,
                    float(next_waypoint.transform.rotation.yaw),
                )
            )
            yaw_delta = abs(
                self._angle_delta_degrees(
                    previous_yaw,
                    float(next_waypoint.transform.rotation.yaw),
                )
            )

            if step <= 8:
                score += 5.0
                score -= yaw_delta * 1.5
                if next_waypoint.is_junction:
                    score -= 180.0
            elif next_waypoint.is_junction or branch_turn > 18.0 or yaw_from_start > 18.0:
                distance_m = step * 2.5
                distance_bonus = max(0.0, 50.0 - abs(distance_m - 45.0))
                turn_strength = max(branch_turn, yaw_from_start)
                score += 160.0 + distance_bonus + min(80.0, turn_strength)
                saw_turn = True
                break
            else:
                score += 1.5
                score -= yaw_delta * 0.5

            current = next_waypoint
            previous_yaw = float(current.transform.rotation.yaw)

        if not saw_turn:
            score -= 250.0

        return score

    @staticmethod
    def _location_distance(first: Any, second: Any) -> float:
        dx = float(first.x - second.x)
        dy = float(first.y - second.y)
        dz = float(first.z - second.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _angle_delta_degrees(first: float, second: float) -> float:
        """Return signed smallest angular difference in degrees."""

        return (second - first + 180.0) % 360.0 - 180.0

    def _require_world(self) -> Any:
        if self.world is None:
            raise RuntimeError("CARLA world is not connected. Call connect() first.")
        return self.world

    def _require_ego_vehicle(self) -> Any:
        if self.ego_vehicle is None:
            raise RuntimeError("Ego vehicle is not spawned. Call spawn_ego_vehicle() first.")
        return self.ego_vehicle

    def _distance_to_driving_waypoint(self) -> Optional[float]:
        world = self._require_world()
        vehicle = self._require_ego_vehicle()
        carla_map = world.get_map()
        location = vehicle.get_location()

        try:
            waypoint = carla_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError as exc:
            LOGGER.debug("Could not query driving waypoint: %s", exc)
            return None

        if waypoint is None:
            return None

        waypoint_location = waypoint.transform.location
        dx = float(location.x - waypoint_location.x)
        dy = float(location.y - waypoint_location.y)
        dz = float(location.z - waypoint_location.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _ensure_carla_available() -> None:
        if _CARLA_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The CARLA Python package is not installed. Activate the vla-av "
                "environment or install carla==0.9.15."
            ) from _CARLA_IMPORT_ERROR

    @staticmethod
    def _vector_to_tuple(vector: Any) -> Tuple[float, float, float]:
        return float(vector.x), float(vector.y), float(vector.z)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def _safe_actor_id(actor: Any) -> Optional[int]:
        try:
            return int(actor.id)
        except RuntimeError:
            return None
