#!/usr/bin/env python3
"""Run the official CarDreamer overtake policy as a CARLA proposal publisher.

The process reconstructs the privileged BEV observation expected by the
upstream ``overtake.ckpt`` and publishes proposed controls. It deliberately
does not own the simulator clock and never sends commands to a CARLA actor.
In residual mode, SimLingo consumes these proposals in its own Python process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ACCELERATIONS = (-2.0, 0.0, 2.0)
STEERING = (-0.6, -0.2, 0.0, 0.2, 0.6)
PREFERRED_EGO_ROLES = {"hero", "ego", "ego_vehicle", "hero0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--upstream", type=pathlib.Path, required=True)
    parser.add_argument("--route-file", type=pathlib.Path)
    parser.add_argument("--status-path", type=pathlib.Path, required=True)
    parser.add_argument("--trace-path", type=pathlib.Path, required=True)
    parser.add_argument("--bev-path", type=pathlib.Path)
    parser.add_argument("--interval-game-seconds", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--blocked-distance", type=float, default=18.0)
    parser.add_argument("--blocked-vehicle-speed", type=float, default=1.5)
    parser.add_argument("--minimum-clearance", type=float, default=5.0)
    parser.add_argument("--minimum-oncoming-ttc", type=float, default=7.0)
    parser.add_argument("--minimum-rear-ttc", type=float, default=5.0)
    parser.add_argument(
        "--runtime-mode",
        choices=("shadow", "residual"),
        default="shadow",
        help="Publish read-only diagnostics or proposals consumed by SimLingo.",
    )
    parser.add_argument(
        "--lateral-adapter",
        choices=("native", "mirror"),
        default="native",
        help=(
            "Coordinate convention only. 'mirror' horizontally mirrors the privileged BEV "
            "and maps the proposed steering back to the CARLA frame; checkpoint weights stay unchanged."
        ),
    )
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def speed_mps(actor) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def find_ego(world):
    vehicles = list(world.get_actors().filter("vehicle.*"))
    for actor in vehicles:
        if actor.attributes.get("role_name", "") in PREFERRED_EGO_ROLES:
            return actor
    for actor in vehicles:
        if actor.attributes.get("role_name", "") not in {"scenario", "background", "autopilot"}:
            return actor
    return None


def route_points(path: Optional[pathlib.Path]) -> List[Tuple[float, float]]:
    if path is None or not path.is_file():
        return []
    root = ET.parse(path).getroot()
    points = []
    for node in root.findall(".//waypoints/position"):
        points.append((float(node.attrib["x"]), float(node.attrib["y"])))
    return points


def local_lane_path(carla_map, ego, distance: float = 80.0, spacing: float = 2.0) -> List[Tuple[float, float]]:
    waypoint = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    if waypoint is None:
        return []
    points = []
    travelled = 0.0
    current = waypoint
    while travelled <= distance:
        location = current.transform.location
        points.append((location.x, location.y))
        candidates = current.next(spacing)
        if not candidates:
            break
        forward = current.transform.get_forward_vector()
        current = max(
            candidates,
            key=lambda item: (
                forward.x * item.transform.get_forward_vector().x
                + forward.y * item.transform.get_forward_vector().y
            ),
        )
        travelled += spacing
    return points


def actor_polygon(actor) -> List[List[float]]:
    transform = actor.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    length = actor.bounding_box.extent.x
    width = actor.bounding_box.extent.y
    local = np.array(
        [[length, width], [length, -width], [-length, -width], [-length, width]],
        dtype=np.float64,
    )
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    center = np.array([transform.location.x, transform.location.y], dtype=np.float64)
    return (local @ rotation.T + center).tolist()


class ReadOnlyWorldAdapter:
    """Minimal WorldManager contract backed only by CARLA getter calls."""

    def __init__(self, world):
        self.carla_world = world
        self.carla_map = world.get_map()
        self._actors = []
        self._transforms = {}
        self._polygons = {}

    def refresh(self) -> None:
        actors = [actor for actor in self.carla_world.get_actors().filter("vehicle.*") if actor.is_alive]
        self._actors = actors
        self._transforms = {actor.id: actor.get_transform() for actor in actors}
        self._polygons = {actor.id: actor_polygon(actor) for actor in actors}

    @property
    def actor_ids(self):
        return list(self._transforms)

    @property
    def actor_transforms(self):
        return self._transforms

    @property
    def actor_polygons(self):
        return self._polygons

    @property
    def actor_actions(self):
        # The official overtake task does not expose a Traffic Manager plan for
        # its scripted non-ego vehicle either.
        return {}

    def carla_actors(self, actor_type: str = ""):
        return [actor for actor in self.carla_world.get_actors() if actor_type in actor.type_id]


def same_lane(left, right) -> bool:
    return bool(left and right and left.road_id == right.road_id and left.lane_id == right.lane_id)


def driving_adjacent(waypoint, direction: str):
    if waypoint is None:
        return None
    candidate = waypoint.get_left_lane() if direction == "left" else waypoint.get_right_lane()
    if candidate is None:
        return None
    import carla

    return candidate if candidate.lane_type == carla.LaneType.Driving else None


@dataclass
class LaneGeometry:
    ego_speed_mps: float = 0.0
    front_vehicle_m: float = 80.0
    front_vehicle_speed_mps: float = 0.0
    left_lane_available: bool = False
    right_lane_available: bool = False
    left_front_m: float = 80.0
    left_rear_m: float = 80.0
    right_front_m: float = 80.0
    right_rear_m: float = 80.0
    left_clear_m: float = 80.0
    right_clear_m: float = 80.0
    left_oncoming_m: float = 80.0
    right_oncoming_m: float = 80.0
    left_oncoming_ttc_s: float = 99.0
    right_oncoming_ttc_s: float = 99.0
    left_rear_ttc_s: float = 99.0
    right_rear_ttc_s: float = 99.0


def observe_lane_geometry(carla_map, ego, vehicles: Iterable) -> LaneGeometry:
    result = LaneGeometry(ego_speed_mps=speed_mps(ego))
    ego_transform = ego.get_transform()
    ego_location = ego_transform.location
    forward_vector = ego_transform.get_forward_vector()
    right_vector = ego_transform.get_right_vector()
    ego_forward = np.array([forward_vector.x, forward_vector.y], dtype=np.float64)
    ego_right = np.array([right_vector.x, right_vector.y], dtype=np.float64)
    ego_velocity = ego.get_velocity()
    ego_velocity_xy = np.array([ego_velocity.x, ego_velocity.y], dtype=np.float64)
    ego_waypoint = carla_map.get_waypoint(ego_location, project_to_road=True)
    left_waypoint = driving_adjacent(ego_waypoint, "left")
    right_waypoint = driving_adjacent(ego_waypoint, "right")
    result.left_lane_available = left_waypoint is not None
    result.right_lane_available = right_waypoint is not None

    buckets = {
        "ego": [],
        "left": [],
        "right": [],
    }
    for actor in vehicles:
        if actor.id == ego.id or not actor.is_alive:
            continue
        transform = actor.get_transform()
        delta = np.array(
            [transform.location.x - ego_location.x, transform.location.y - ego_location.y],
            dtype=np.float64,
        )
        longitudinal = float(np.dot(delta, ego_forward))
        lateral = float(np.dot(delta, ego_right))
        if abs(longitudinal) > 85.0 or abs(lateral) > 16.0:
            continue
        actor_waypoint = carla_map.get_waypoint(transform.location, project_to_road=True)
        lane = None
        if same_lane(actor_waypoint, ego_waypoint):
            lane = "ego"
        elif same_lane(actor_waypoint, left_waypoint):
            lane = "left"
        elif same_lane(actor_waypoint, right_waypoint):
            lane = "right"
        elif -5.5 < lateral < -1.2:
            lane = "left"
        elif 1.2 < lateral < 5.5:
            lane = "right"
        elif abs(lateral) <= 1.8:
            lane = "ego"
        if lane is None:
            continue
        velocity = actor.get_velocity()
        velocity_xy = np.array([velocity.x, velocity.y], dtype=np.float64)
        actor_forward = transform.get_forward_vector()
        heading_dot = float(actor_forward.x * ego_forward[0] + actor_forward.y * ego_forward[1])
        relative_longitudinal_speed = float(np.dot(velocity_xy - ego_velocity_xy, ego_forward))
        distance = max(0.0, float(np.linalg.norm(delta)) - actor.bounding_box.extent.x - ego.bounding_box.extent.x)
        buckets[lane].append(
            {
                "longitudinal": longitudinal,
                "distance": distance,
                "speed": float(np.linalg.norm(velocity_xy)),
                "heading_dot": heading_dot,
                "relative_longitudinal_speed": relative_longitudinal_speed,
            }
        )

    ego_front = [item for item in buckets["ego"] if item["longitudinal"] > 0]
    if ego_front:
        nearest = min(ego_front, key=lambda item: item["distance"])
        result.front_vehicle_m = nearest["distance"]
        result.front_vehicle_speed_mps = nearest["speed"]

    for lane in ("left", "right"):
        items = buckets[lane]
        fronts = [item for item in items if item["longitudinal"] >= 0]
        rears = [item for item in items if item["longitudinal"] < 0]
        front_m = min((item["distance"] for item in fronts), default=80.0)
        rear_m = min((item["distance"] for item in rears), default=80.0)
        setattr(result, f"{lane}_front_m", front_m)
        setattr(result, f"{lane}_rear_m", rear_m)
        setattr(result, f"{lane}_clear_m", min(front_m, rear_m))

        oncoming = [item for item in fronts if item["heading_dot"] < -0.25]
        if oncoming:
            nearest = min(oncoming, key=lambda item: item["distance"])
            closing = max(0.1, -nearest["relative_longitudinal_speed"])
            setattr(result, f"{lane}_oncoming_m", nearest["distance"])
            setattr(result, f"{lane}_oncoming_ttc_s", nearest["distance"] / closing)

        approaching_rear = [item for item in rears if item["relative_longitudinal_speed"] > 0.1]
        if approaching_rear:
            nearest = min(approaching_rear, key=lambda item: item["distance"])
            setattr(
                result,
                f"{lane}_rear_ttc_s",
                nearest["distance"] / nearest["relative_longitudinal_speed"],
            )
    return result


def maneuver_from_control(steer: float, throttle: float, brake: float) -> str:
    if steer < -0.1:
        return "left"
    if steer > 0.1:
        return "right"
    if brake > 0:
        return "brake"
    if throttle > 0:
        return "straight_accelerate"
    return "straight_coast"


def decode_action(action_vector: np.ndarray, lateral_adapter: str = "native") -> Dict:
    index = int(np.argmax(np.asarray(action_vector).reshape(-1)))
    acceleration = ACCELERATIONS[index // len(STEERING)]
    discrete_steer = STEERING[index % len(STEERING)]
    policy_frame_steer = -discrete_steer
    carla_steer = -policy_frame_steer if lateral_adapter == "mirror" else policy_frame_steer
    throttle = max(0.0, acceleration / 3.0)
    brake = max(0.0, -acceleration / 3.0)
    return {
        "action_index": index,
        "acceleration": acceleration,
        "discrete_steering": discrete_steer,
        "lateral_adapter": lateral_adapter,
        "policy_frame_maneuver": maneuver_from_control(policy_frame_steer, throttle, brake),
        "maneuver": maneuver_from_control(carla_steer, throttle, brake),
        "proposed_control": {
            "throttle": throttle,
            "brake": brake,
            "steer": carla_steer,
        },
    }


def assess_proposal(
    action: Dict,
    geometry: LaneGeometry,
    blocked_ticks: int,
    blocked_distance: float,
    minimum_clearance: float,
    minimum_oncoming_ttc: float,
    minimum_rear_ttc: float,
    blocked_vehicle_speed: float = 1.5,
) -> Dict:
    blocked = (
        geometry.front_vehicle_m <= blocked_distance
        and geometry.front_vehicle_speed_mps <= blocked_vehicle_speed
    )
    left_safe = (
        geometry.left_lane_available
        and geometry.left_clear_m >= minimum_clearance
        and geometry.left_oncoming_ttc_s >= minimum_oncoming_ttc
        and geometry.left_rear_ttc_s >= minimum_rear_ttc
    )
    right_safe = (
        geometry.right_lane_available
        and geometry.right_clear_m >= minimum_clearance
        and geometry.right_oncoming_ttc_s >= minimum_oncoming_ttc
        and geometry.right_rear_ttc_s >= minimum_rear_ttc
    )
    opportunity = bool(blocked and left_safe)
    maneuver = action["maneuver"]
    unsafe = False
    label = "coherent_lane_follow"
    steer = float(action.get("proposed_control", {}).get("steer", 0.0))
    strong_lateral_intent = abs(steer) >= 0.45
    # A lateral proposal stays unsafe even after the ego has moved far enough
    # for the stationary obstacle to leave the ``blocked`` bucket.  The old
    # condition stopped reporting adjacent traffic precisely while the lane
    # change was still in progress.
    if maneuver == "left" and strong_lateral_intent and not left_safe:
        unsafe = True
        label = "unsafe_left_proposal"
    elif maneuver == "right" and strong_lateral_intent and not right_safe:
        unsafe = True
        label = "unsafe_right_proposal"
    elif opportunity and maneuver == "left":
        label = "coherent_overtake_proposal"
    elif opportunity and blocked_ticks >= 5:
        label = "missed_safe_overtake_opportunity"
    elif blocked and not left_safe and maneuver in {"brake", "straight_coast"}:
        label = "coherent_wait_for_gap"
    elif blocked and not left_safe and maneuver == "straight_accelerate":
        unsafe = True
        label = "unsafe_acceleration_into_blockage"
    return {
        "blocked": blocked,
        "left_safe": left_safe,
        "right_safe": right_safe,
        "safe_overtake_opportunity": opportunity,
        "unsafe": unsafe,
        "coherent": not unsafe and not label.startswith("missed_"),
        "coherence_label": label,
    }


class ShadowSpecEnv:
    """Space-only Gym environment used to instantiate the unchanged agent."""

    def __init__(self):
        import gym

        self.observation_space = gym.spaces.Dict(
            {
                "collision": gym.spaces.Box(0.0, np.inf, (1,), dtype=np.float32),
                "birdeye_wpt": gym.spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
            }
        )
        self.action_space = gym.spaces.Discrete(len(ACCELERATIONS) * len(STEERING))

    def close(self):
        pass


def build_policy(checkpoint: pathlib.Path, upstream: pathlib.Path, seed: int):
    import ruamel.yaml as yaml
    import car_dreamer
    import dreamerv3
    import embodied
    from dreamerv3.eval import wrap_env
    from embodied.envs import from_gym

    model_config_path = upstream / "dreamerv3" / "dreamerv3.yaml"
    model_configs = yaml.YAML(typ="safe").load(model_config_path.read_text())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})
    config = config.update(car_dreamer.load_task_configs("carla_overtake"))
    config = config.update(
        {
            "dreamerv3.seed": seed,
            "dreamerv3.logdir": str(checkpoint.parent / "shadow_runtime"),
            "dreamerv3.jax.platform": "gpu",
            "dreamerv3.jax.policy_devices": (0,),
            "dreamerv3.jax.train_devices": (0,),
            "dreamerv3.jax.prealloc": False,
        }
    )
    spec = from_gym.FromGym(ShadowSpecEnv())
    spec = wrap_env(spec, config.dreamerv3)
    step = embodied.Counter()
    agent = dreamerv3.Agent(spec.obs_space, spec.act_space, step, config.dreamerv3)
    loader = embodied.Checkpoint()
    loader.agent = agent
    loader.load(str(checkpoint), keys=["agent"])
    return agent


def make_observation(bev: np.ndarray, is_first: bool) -> Dict[str, np.ndarray]:
    return {
        "collision": np.zeros((1, 1), dtype=np.float32),
        "birdeye_wpt": bev[np.newaxis].astype(np.uint8),
        "reward": np.zeros((1,), dtype=np.float32),
        "is_first": np.array([is_first], dtype=bool),
        "is_last": np.array([False], dtype=bool),
        "is_terminal": np.array([False], dtype=bool),
    }


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.upstream.is_dir():
        raise FileNotFoundError(args.upstream)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    sys.path.insert(0, str(args.upstream))
    sys.path.insert(0, str(args.upstream / "dreamerv3"))

    import carla
    import car_dreamer
    from car_dreamer.toolkit.observer.handlers.birdeye_handler import BirdeyeHandler

    np.random.seed(args.seed)
    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    args.trace_path.parent.mkdir(parents=True, exist_ok=True)
    args.trace_path.write_text("", encoding="utf-8")
    fixed_route = route_points(args.route_file)
    checkpoint_hash = sha256(args.checkpoint)
    policy = build_policy(args.checkpoint, args.upstream, args.seed)

    stop = False

    def stop_handler(signum, frame):
        nonlocal stop
        del signum, frame
        stop = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    started = time.time()
    adapter = None
    handler = None
    ego = None
    ego_id = None
    world_name = None
    policy_state = None
    last_game_time = -math.inf
    blocked_ticks = 0
    decision_index = 0
    initial_status = {
        "schema_version": 1,
        "timestamp": time.time(),
        "state": "waiting_for_carla",
        "mode": args.runtime_mode,
        "shadow": args.runtime_mode == "shadow",
        "control_authority": "none_in_sidecar",
        "proposal_consumer": (
            "simlingo_residual" if args.runtime_mode == "residual" else "none"
        ),
        "privileged_information": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "lateral_adapter": args.lateral_adapter,
        "observation_transform": (
            "horizontal_mirror" if args.lateral_adapter == "mirror" else "identity"
        ),
    }
    atomic_json(args.status_path, initial_status)

    print(
        "[cardreamer-runtime] proposal publisher: no CARLA mutation in sidecar; "
        f"checkpoint={args.checkpoint} sha256={checkpoint_hash} "
        f"runtime_mode={args.runtime_mode} lateral_adapter={args.lateral_adapter}",
        flush=True,
    )
    while not stop and time.time() - started < args.timeout:
        try:
            world = client.get_world()
            current_world_name = world.get_map().name
            current_ego = find_ego(world)
            if current_ego is None:
                atomic_json(
                    args.status_path,
                    {**initial_status, "timestamp": time.time(), "state": "waiting_for_ego"},
                )
                time.sleep(0.5)
                continue

            reset = current_world_name != world_name or current_ego.id != ego_id or handler is None
            if reset:
                ego = current_ego
                ego_id = ego.id
                world_name = current_world_name
                adapter = ReadOnlyWorldAdapter(world)
                adapter.refresh()
                config = car_dreamer.load_task_configs("carla_overtake")
                handler = BirdeyeHandler(adapter, config.env.observation.birdeye_wpt)
                handler.reset(ego)
                policy_state = None
                last_game_time = -math.inf
                blocked_ticks = 0
                print(f"[cardreamer-runtime] observing {world_name} ego={ego_id}", flush=True)

            snapshot = world.get_snapshot()
            game_time = float(snapshot.timestamp.elapsed_seconds)
            if game_time - last_game_time + 1e-6 < args.interval_game_seconds:
                time.sleep(0.01)
                continue
            last_game_time = game_time
            adapter.refresh()
            if ego.id not in adapter.actor_ids:
                ego = None
                ego_id = None
                continue

            path = fixed_route or local_lane_path(adapter.carla_map, ego)
            observation, _ = handler.get_observation({"ego_waypoints": path})
            bev = observation["birdeye_wpt"]
            policy_bev = (
                np.ascontiguousarray(np.flip(bev, axis=1))
                if args.lateral_adapter == "mirror"
                else bev
            )
            policy_output, policy_state = policy.policy(
                make_observation(policy_bev, is_first=decision_index == 0 or reset),
                policy_state,
                mode="eval",
            )
            action = decode_action(policy_output["action"][0], args.lateral_adapter)
            geometry = observe_lane_geometry(adapter.carla_map, ego, adapter._actors)
            if (
                geometry.front_vehicle_m <= args.blocked_distance
                and geometry.front_vehicle_speed_mps <= args.blocked_vehicle_speed
            ):
                blocked_ticks += 1
            else:
                blocked_ticks = 0
            assessment = assess_proposal(
                action,
                geometry,
                blocked_ticks,
                args.blocked_distance,
                args.minimum_clearance,
                args.minimum_oncoming_ttc,
                args.minimum_rear_ttc,
                args.blocked_vehicle_speed,
            )
            payload = {
                "schema_version": 1,
                "timestamp": time.time(),
                "game_time": game_time,
                "frame": int(snapshot.frame),
                "decision_index": decision_index,
                "state": "observing",
                "mode": args.runtime_mode,
                "shadow": args.runtime_mode == "shadow",
                "control_authority": "none_in_sidecar",
                "proposal_consumer": (
                    "simlingo_residual" if args.runtime_mode == "residual" else "none"
                ),
                "privileged_information": True,
                "observation_contract": "CarDreamer full-observability birdeye_wpt 128x128",
                "observation_transform": (
                    "horizontal_mirror" if args.lateral_adapter == "mirror" else "identity"
                ),
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "town": world_name.split("/")[-1],
                "ego_id": ego.id,
                "route_file": str(args.route_file.resolve()) if args.route_file and args.route_file.exists() else None,
                "route_source": "bench2drive_xml" if fixed_route else "local_carla_lane",
                "blocked_ticks": blocked_ticks,
                **action,
                **asdict(geometry),
                **assessment,
            }
            atomic_json(args.status_path, payload)
            append_jsonl(args.trace_path, payload)
            if args.bev_path:
                args.bev_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.bev_path), cv2.cvtColor(policy_bev, cv2.COLOR_RGB2BGR))
            if decision_index % 20 == 0:
                print(
                    f"CARDREAMER_{args.runtime_mode.upper()}_PROPOSAL "
                    f"frame={payload['frame']} action={action['action_index']} "
                    f"maneuver={action['maneuver']} blocked={int(assessment['blocked'])} "
                    f"opportunity={int(assessment['safe_overtake_opportunity'])} "
                    f"coherence={assessment['coherence_label']}",
                    flush=True,
                )
            decision_index += 1
        except (RuntimeError, OSError) as exc:
            atomic_json(
                args.status_path,
                {
                    **initial_status,
                    "timestamp": time.time(),
                    "state": "waiting_after_error",
                    "error": str(exc),
                },
            )
            time.sleep(0.5)

    final = {
        **initial_status,
        "timestamp": time.time(),
        "state": "finished",
        "decisions": decision_index,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_json(args.status_path, final)
    print(f"[cardreamer-runtime] finished decisions={decision_index} trace={args.trace_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
