#!/usr/bin/env python3
"""Mirror a running CARLA simulation into SUMO GUI without driving CARLA's tick."""

import argparse
import json
import math
import os
import signal
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import carla
except ImportError as exc:
    raise SystemExit(
        "Could not import carla. Activate the simlingo env or set PYTHONPATH for CARLA."
    ) from exc


def ensure_sumo_tools() -> None:
    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    os.environ["SUMO_HOME"] = sumo_home
    tools = Path(sumo_home) / "tools"
    if tools.exists() and str(tools) not in sys.path:
        sys.path.insert(0, str(tools))


ensure_sumo_tools()
import traci  # pylint: disable=wrong-import-position,import-error
import sumolib  # pylint: disable=wrong-import-position,import-error


EGO_SUMO_COLOR = (255, 235, 0, 0)
BIKE_SUMO_COLOR = (120, 255, 40, 255)
EGO_MARKER_OUTER_ID = "simlingo_ego_rectangle_outline"
EGO_MARKER_INNER_ID = "simlingo_ego_rectangle"
EGO_MARKER_OUTER_COLOR = (255, 255, 255, 255)
EGO_MARKER_INNER_COLOR = (255, 235, 0, 255)
EGO_ROLE_NAMES = {"hero", "hero0", "ego", "ego_vehicle", "simlingo", "leaderboard"}
# Deliberately avoid dark and blue hues so actors remain visible on every SUMO theme.
VISIBLE_SUMO_VEHICLE_COLORS = (
    (255, 40, 180, 255),
    (120, 255, 40, 255),
    (255, 150, 20, 255),
    (255, 255, 255, 255),
    (255, 80, 70, 255),
    (210, 255, 40, 255),
    (255, 170, 220, 255),
    (160, 255, 180, 255),
)


def parse_net_offset(net_file: Path):
    root = ET.parse(net_file).getroot()
    location = root.find("location")
    if location is None:
        return (0.0, 0.0)
    raw = location.attrib.get("netOffset", "0,0").split(",")
    return (float(raw[0]), float(raw[1]))


def find_dummy_edge(net_file: Path) -> str:
    net = sumolib.net.readNet(str(net_file))
    for edge in net.getEdges():
        if edge.getFunction():
            continue
        lanes = edge.getLanes()
        if not lanes:
            continue
        return edge.getID()
    raise RuntimeError(f"Could not find a drivable edge in {net_file}")


def write_mirror_files(output_dir: Path, net_file: Path, additional_file: Path = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    dummy_edge = find_dummy_edge(net_file)
    routes_file = output_dir / "mirror.rou.xml"
    cfg_file = output_dir / "mirror.sumocfg"

    routes_file.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <vType id="mirror_ego" vClass="passenger" guiShape="passenger" length="4.8" width="1.9" color="255,235,0,0"/>
  <vType id="mirror_vehicle" vClass="passenger" guiShape="passenger" length="4.8" width="1.9" color="255,40,180"/>
  <vType id="mirror_bike" vClass="bicycle" guiShape="bicycle" length="1.8" width="0.6" color="120,255,40"/>
  <route id="mirror_dummy" edges="{dummy_edge}"/>
</routes>
""",
        encoding="utf-8",
    )

    additional_line = ""
    if additional_file:
        additional_line = f'    <additional-files value="{additional_file.resolve()}"/>\n'

    cfg_file.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
  <input>
    <net-file value="{net_file.resolve()}"/>
    <route-files value="{routes_file.resolve()}"/>
{additional_line}  </input>
  <time>
    <begin value="0"/>
    <end value="100000"/>
  </time>
</configuration>
""",
        encoding="utf-8",
    )
    return cfg_file


def connect_carla(host: str, port: int, retries: int, wait: float, timeout: float):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            world = client.get_world()
            print(
                f"[carla-sumo-mirror] connected to CARLA {host}:{port} map={world.get_map().name}",
                flush=True,
            )
            return client, world
        except RuntimeError as exc:
            last_error = exc
            print(
                f"[carla-sumo-mirror] waiting for CARLA {host}:{port} "
                f"attempt={attempt}/{retries}: {exc}",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"Could not connect to CARLA {host}:{port}: {last_error}")


def wait_for_vehicle_spawn(
    client,
    world,
    min_vehicles: int,
    timeout: float,
    poll: float,
    host: str,
    port: int,
    rpc_timeout: float,
):
    if min_vehicles <= 0:
        return world
    start = time.time()
    last_report = 0.0
    while True:
        try:
            world = client.get_world()
            vehicles = list(world.get_actors().filter("vehicle.*"))
            if len(vehicles) >= min_vehicles:
                print(
                    f"[carla-sumo-mirror] CARLA vehicles ready: {len(vehicles)} "
                    f"(min={min_vehicles}) map={world.get_map().name}",
                    flush=True,
                )
                return world
        except RuntimeError as exc:
            print(f"[carla-sumo-mirror] CARLA world not stable yet: {exc}", flush=True)
            client, world = connect_carla(host, port, 5, poll, rpc_timeout)
            vehicles = []

        elapsed = time.time() - start
        if timeout > 0 and elapsed >= timeout:
            print(
                f"[carla-sumo-mirror] warning: timed out waiting for "
                f"{min_vehicles} CARLA vehicle(s); continuing with {len(vehicles)}.",
                flush=True,
            )
            return world
        if elapsed - last_report >= 5.0:
            print(
                f"[carla-sumo-mirror] waiting for SimLingo actors: "
                f"vehicles={len(vehicles)} min={min_vehicles} elapsed={elapsed:.1f}s",
                flush=True,
            )
            last_report = elapsed
        time.sleep(max(0.05, poll))


def carla_to_sumo_transform(transform, offset):
    ox, oy = offset
    location = transform.location
    rotation = transform.rotation
    x = float(location.x) + ox
    y = -float(location.y) + oy
    angle = (float(rotation.yaw) + 90.0) % 360.0
    return x, y, angle


def rectangle_shape(x, y, angle, length, width):
    radians = math.radians(angle)
    forward_x, forward_y = math.sin(radians), math.cos(radians)
    right_x, right_y = math.cos(radians), -math.sin(radians)
    half_length = 0.5 * float(length)
    half_width = 0.5 * float(width)
    return [
        (
            x + longitudinal * forward_x + lateral * right_x,
            y + longitudinal * forward_y + lateral * right_y,
        )
        for longitudinal, lateral in (
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        )
    ]


def upsert_ego_marker(marker_id, shape, color, layer, active_marker_ids):
    if marker_id in active_marker_ids:
        try:
            traci.polygon.setShape(marker_id, shape)
            return
        except traci.TraCIException:
            active_marker_ids.discard(marker_id)

    traci.polygon.add(
        polygonID=marker_id,
        shape=shape,
        color=color,
        fill=True,
        polygonType="simlingo_ego",
        layer=layer,
    )
    active_marker_ids.add(marker_id)


def center_sumo_views_on_ego(x, y):
    try:
        view_ids = traci.gui.getIDList()
    except (AttributeError, traci.TraCIException):
        return
    for view_id in view_ids:
        try:
            traci.gui.setOffset(view_id, x, y)
        except traci.TraCIException:
            continue


def update_ego_rectangle(actor, offset, active_marker_ids, follow_in_gui=False):
    x, y, angle = carla_to_sumo_transform(actor.get_transform(), offset)
    marker_specs = (
        (
            EGO_MARKER_OUTER_ID,
            rectangle_shape(x, y, angle, length=7.0, width=3.4),
            EGO_MARKER_OUTER_COLOR,
            1000,
        ),
        (
            EGO_MARKER_INNER_ID,
            rectangle_shape(x, y, angle, length=6.2, width=2.6),
            EGO_MARKER_INNER_COLOR,
            1001,
        ),
    )
    for marker_id, shape, color, layer in marker_specs:
        upsert_ego_marker(marker_id, shape, color, layer, active_marker_ids)
    if follow_in_gui:
        center_sumo_views_on_ego(x, y)


def remove_ego_rectangle(active_marker_ids):
    for marker_id in tuple(active_marker_ids):
        try:
            traci.polygon.remove(marker_id)
        except traci.TraCIException:
            pass
        active_marker_ids.discard(marker_id)


def carla_speed(actor) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)


def is_ego_vehicle(actor) -> bool:
    role = actor.attributes.get("role_name", "").strip().lower()
    return role in EGO_ROLE_NAMES or role.startswith(("hero", "ego", "simlingo", "leaderboard"))


def vtype_for_actor(actor) -> str:
    type_id = actor.type_id.lower()
    if "bike" in type_id or "bicycle" in type_id or "crossbike" in type_id:
        return "mirror_bike"
    if is_ego_vehicle(actor):
        return "mirror_ego"
    return "mirror_vehicle"


def vehicle_color(actor):
    if is_ego_vehicle(actor):
        return EGO_SUMO_COLOR
    if vtype_for_actor(actor) == "mirror_bike":
        return BIKE_SUMO_COLOR
    return VISIBLE_SUMO_VEHICLE_COLORS[
        int(actor.id) % len(VISIBLE_SUMO_VEHICLE_COLORS)
    ]


def add_or_update_vehicle(actor, active_ids, offset):
    veh_id = f"carla_{actor.id}"
    transform = actor.get_transform()
    x, y, angle = carla_to_sumo_transform(transform, offset)
    speed = carla_speed(actor)

    if veh_id not in active_ids:
        traci.vehicle.add(
            vehID=veh_id,
            routeID="mirror_dummy",
            typeID=vtype_for_actor(actor),
            depart=str(max(0.0, traci.simulation.getTime())),
            departSpeed="0",
        )
        color = vehicle_color(actor)
        if color is not None:
            traci.vehicle.setColor(veh_id, color)
        try:
            # This SUMO instance is a CARLA-owned mirror. Disable SUMO's own
            # car-following / traffic-light reactions so the 2D vehicle never
            # diverges from the real CARLA actor between mirror updates.
            traci.vehicle.setSpeedMode(veh_id, 0)
            traci.vehicle.setLaneChangeMode(veh_id, 0)
        except Exception:
            pass
        active_ids.add(veh_id)

    traci.vehicle.moveToXY(
        vehID=veh_id,
        edgeID="",
        lane=-1,
        x=x,
        y=y,
        angle=angle,
        keepRoute=2,
    )
    traci.vehicle.setSpeed(veh_id, max(0.0, speed))
    return veh_id


def build_tl_link_map():
    mapping = {}
    for tlid in traci.trafficlight.getIDList():
        for logic in traci.trafficlight.getAllProgramLogics(tlid):
            params = logic.getParameters()
            for key, value in params.items():
                if not key.startswith("linkSignalID:"):
                    continue
                try:
                    link_index = int(key.split(":", 1)[1])
                except ValueError:
                    continue
                mapping.setdefault(str(value), []).append((tlid, link_index))
    print(f"[carla-sumo-mirror] traffic-light landmark mappings={len(mapping)}", flush=True)
    return mapping


def carla_light_odr_id(light):
    for attr_name in ("get_opendrive_id", "get_open_drive_id"):
        getter = getattr(light, attr_name, None)
        if callable(getter):
            try:
                value = getter()
                if value not in (None, ""):
                    return str(value)
            except RuntimeError:
                return None
    return None


def light_state_char(light):
    state = str(light.get_state()).rsplit(".", 1)[-1]
    if state == "Red":
        return "r"
    if state == "Yellow":
        return "y"
    if state == "Green":
        return "G"
    return "O"


def rotate_point(point, angle_degrees):
    angle = math.radians(angle_degrees)
    x_value = math.cos(angle) * point.x - math.sin(angle) * point.y
    y_value = math.sin(angle) * point.x + math.cos(angle) * point.y
    return carla.Vector3D(x_value, y_value, point.z)


def traffic_light_trigger_location(light):
    base_transform = light.get_transform()
    return base_transform.transform(light.trigger_volume.location)


def collect_red_light_stop_points(world):
    carla_map = world.get_map()
    stop_points = []
    seen = set()
    for light in world.get_actors().filter("traffic.traffic_light*"):
        try:
            if traffic_light_state_name(light) != "Red":
                continue
        except RuntimeError:
            continue

        getter = getattr(light, "get_stop_waypoints", None)
        if callable(getter):
            try:
                for wp in getter() or []:
                    key = (light.id, wp.road_id, wp.lane_id, round(wp.s, 1))
                    if key in seen:
                        continue
                    seen.add(key)
                    stop_points.append(
                        {
                            "light_id": light.id,
                            "location": wp.transform.location,
                            "road_id": wp.road_id,
                            "lane_id": wp.lane_id,
                        }
                    )
                if stop_points:
                    continue
            except RuntimeError:
                pass

        if not hasattr(light, "trigger_volume"):
            continue
        try:
            base_transform = light.get_transform()
            base_rot = base_transform.rotation.yaw
            area_loc = traffic_light_trigger_location(light)
            area_ext = light.trigger_volume.extent
            width = max(0.5, float(area_ext.x))
            sample_count = max(2, min(24, int(math.ceil(1.8 * width)) + 1))
            sampled_wps = []
            for index in range(sample_count):
                frac = 0.0 if sample_count == 1 else index / float(sample_count - 1)
                x_offset = -0.9 * width + frac * (1.8 * width)
                rotated = rotate_point(carla.Vector3D(x_offset, 0.0, area_ext.z), base_rot)
                point_location = area_loc + carla.Location(x=rotated.x, y=rotated.y)
                wp = carla_map.get_waypoint(
                    point_location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if wp is None:
                    continue
                if sampled_wps and sampled_wps[-1].road_id == wp.road_id and sampled_wps[-1].lane_id == wp.lane_id:
                    continue
                sampled_wps.append(wp)

            for wp in sampled_wps:
                for _ in range(80):
                    if wp.is_intersection:
                        break
                    next_wps = wp.next(0.5)
                    if not next_wps:
                        break
                    next_wp = next_wps[0]
                    if next_wp.is_intersection:
                        break
                    wp = next_wp
                key = (light.id, wp.road_id, wp.lane_id, round(wp.s, 1))
                if key in seen:
                    continue
                seen.add(key)
                stop_points.append(
                    {
                        "light_id": light.id,
                        "location": wp.transform.location,
                        "road_id": wp.road_id,
                        "lane_id": wp.lane_id,
                    }
                )
        except RuntimeError:
            continue
    return stop_points


def point_ahead_of_vehicle(vehicle_transform, point_location, max_distance, lateral_limit):
    vehicle_location = vehicle_transform.location
    forward = vehicle_transform.get_forward_vector()
    dx = point_location.x - vehicle_location.x
    dy = point_location.y - vehicle_location.y
    longitudinal = dx * forward.x + dy * forward.y
    if longitudinal < 0.5 or longitudinal > max_distance:
        return False, longitudinal, float("inf")
    lateral = abs(-forward.y * dx + forward.x * dy)
    return lateral <= lateral_limit, longitudinal, lateral


def red_light_ahead_for_vehicle(world, vehicle, stop_points, max_distance):
    try:
        if vehicle.is_at_traffic_light() and str(vehicle.get_traffic_light_state()).rsplit(".", 1)[-1] == "Red":
            return True, "carla_vehicle_api"
    except RuntimeError:
        pass

    try:
        carla_map = world.get_map()
        vehicle_transform = vehicle.get_transform()
        vehicle_wp = carla_map.get_waypoint(
            vehicle_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except RuntimeError:
        return False, "map_error"

    best = None
    for point in stop_points:
        point_location = point["location"]
        lane_match = (
            vehicle_wp is not None
            and point.get("road_id") == vehicle_wp.road_id
            and point.get("lane_id") == vehicle_wp.lane_id
        )
        lateral_limit = 7.0 if lane_match else 3.5
        ok, longitudinal, lateral = point_ahead_of_vehicle(
            vehicle_transform,
            point_location,
            max_distance,
            lateral_limit,
        )
        if not ok:
            continue
        if not lane_match and longitudinal > min(18.0, max_distance):
            continue
        if best is None or longitudinal < best[0]:
            best = (longitudinal, lateral, point.get("light_id"), lane_match)

    if best is None:
        return False, "no_red_stop_ahead"
    longitudinal, lateral, light_id, lane_match = best
    return (
        True,
        f"red_stop_ahead light={light_id} dist={longitudinal:.1f}m lateral={lateral:.1f}m lane={int(lane_match)}",
    )


def sync_traffic_lights(world, tl_mapping):
    if not tl_mapping:
        return 0
    states_by_tlid = {}
    changed = 0
    lights = world.get_actors().filter("traffic.traffic_light*")
    for light in lights:
        odr_id = carla_light_odr_id(light)
        if odr_id is None or odr_id not in tl_mapping:
            continue
        try:
            char = light_state_char(light)
        except RuntimeError:
            continue
        for tlid, link_index in tl_mapping[odr_id]:
            if tlid not in states_by_tlid:
                try:
                    states_by_tlid[tlid] = list(traci.trafficlight.getRedYellowGreenState(tlid))
                except traci.TraCIException:
                    continue
            if 0 <= link_index < len(states_by_tlid[tlid]):
                states_by_tlid[tlid][link_index] = char
                changed += 1
    for tlid, state in states_by_tlid.items():
        try:
            traci.trafficlight.setRedYellowGreenState(tlid, "".join(state))
        except traci.TraCIException:
            continue
    return changed


def open_sumo(args, cfg_file: Path):
    binary = "sumo-gui" if args.sumo_gui else "sumo"
    command = [
        binary,
        "-c",
        str(cfg_file),
        "--step-length",
        str(args.step_length),
        "--start",
        "--quit-on-end",
        "false",
    ]
    if args.no_warnings:
        command.extend(["--no-warnings", "true"])
    print("[carla-sumo-mirror] " + " ".join(command), flush=True)
    traci.start(command)


def write_summary(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def traffic_light_state_name(light) -> str:
    return str(light.get_state()).rsplit(".", 1)[-1]


def traffic_light_state_enum(name: str):
    return getattr(carla.TrafficLightState, name, carla.TrafficLightState.Red)


def red_sumo_state(tlid: str) -> str:
    try:
        state = traci.trafficlight.getRedYellowGreenState(tlid)
    except traci.TraCIException:
        return ""
    return "r" * len(state)


def restore_traffic_light_attack(world, attack) -> None:
    data = attack.get("data", {})
    for actor_id, state_name in data.get("carla_states", {}).items():
        try:
            light = world.get_actor(int(actor_id))
            if light is not None:
                light.set_state(traffic_light_state_enum(state_name))
                light.freeze(False)
        except RuntimeError:
            continue

    for tlid, state in data.get("sumo_states", {}).items():
        try:
            if state:
                traci.trafficlight.setRedYellowGreenState(tlid, state)
        except Exception:
            continue


def apply_traffic_light_all_red(world, attack) -> None:
    carla_count = 0
    for light in world.get_actors().filter("traffic.traffic_light*"):
        try:
            light.set_state(carla.TrafficLightState.Red)
            light.freeze(True)
            carla_count += 1
        except RuntimeError:
            continue

    sumo_count = 0
    for tlid in traci.trafficlight.getIDList():
        try:
            state = red_sumo_state(tlid)
            if state:
                traci.trafficlight.setRedYellowGreenState(tlid, state)
                sumo_count += 1
        except Exception:
            continue

    attack.setdefault("stats", {})["carla_lights"] = carla_count
    attack.setdefault("stats", {})["sumo_lights"] = sumo_count
    if "carla_stop_points" not in attack:
        stop_points = collect_red_light_stop_points(world)
        attack["carla_stop_points"] = stop_points
        attack.setdefault("stats", {})["carla_stop_points"] = len(stop_points)


def enforce_carla_red_light_stop(world, attack, traffic_manager=None, max_distance=42.0) -> int:
    """Make non-ego CARLA traffic physically respect the attacked red lights.

    SUMO is only a visual mirror in this pipeline, so SUMO cannot directly control
    CARLA actors. During all-red attacks we therefore apply a CARLA-side traffic
    constraint to background vehicles while leaving the SimLingo ego untouched.
    """
    stopped = 0
    candidates = 0
    stop_points = attack.get("carla_stop_points") or []
    reasons = []
    vehicles = world.get_actors().filter("vehicle.*")
    for vehicle in vehicles:
        if is_ego_vehicle(vehicle):
            continue
        try:
            if traffic_manager is not None:
                traffic_manager.ignore_lights_percentage(vehicle, 0.0)
        except RuntimeError:
            pass
        try:
            should_stop, reason = red_light_ahead_for_vehicle(world, vehicle, stop_points, max_distance)
            if not should_stop:
                continue
            candidates += 1
            if len(reasons) < 3:
                reasons.append(f"{vehicle.id}:{reason}")
            vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            stopped += 1
        except RuntimeError:
            continue
    attack.setdefault("stats", {})["carla_vehicles_red_stop_candidates"] = candidates
    if reasons:
        attack.setdefault("stats", {})["carla_stop_reason_sample"] = reasons
    return stopped


def get_ego_vehicle(world):
    for actor in world.get_actors().filter("vehicle.*"):
        try:
            if is_ego_vehicle(actor):
                return actor
        except RuntimeError:
            continue
    return None


def choose_attack_blueprint(world, color=None):
    blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    preferred = [
        bp
        for bp in blueprints
        if any(name in bp.id for name in ("lincoln", "audi", "tesla", "dodge", "mercedes"))
    ] or blueprints
    for bp in preferred:
        if bp.has_attribute("number_of_wheels"):
            try:
                wheels_attr = bp.get_attribute("number_of_wheels")
                wheels = wheels_attr.as_int() if hasattr(wheels_attr, "as_int") else int(str(wheels_attr))
                if wheels < 4:
                    continue
            except (TypeError, ValueError):
                pass
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "twinsentinel_attack")
        if color and bp.has_attribute("color"):
            colors = [c.strip() for c in bp.get_attribute("color").recommended_values]
            bp.set_attribute("color", color if color in colors else (colors[0] if colors else color))
        return bp
    return None


def transform_offset(transform, longitudinal: float, lateral: float):
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    location = carla.Location(
        x=transform.location.x + forward.x * longitudinal + right.x * lateral,
        y=transform.location.y + forward.y * longitudinal + right.y * lateral,
        z=transform.location.z + 0.25,
    )
    return carla.Transform(location, transform.rotation)


def project_spawn_transform(world, raw_transform):
    try:
        waypoint = world.get_map().get_waypoint(
            raw_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except RuntimeError:
        waypoint = None
    if waypoint is None:
        spawn = raw_transform
    else:
        spawn = waypoint.transform
    spawn.location.z += 0.35
    return spawn


def spawn_attack_vehicle(world, ego, prefix: str, index: int, longitudinal: float, lateral: float, speed: float, color=None):
    blueprint = choose_attack_blueprint(world, color=color)
    if blueprint is None or ego is None:
        return None
    ego_transform = ego.get_transform()
    offsets = [0.0, 2.5, -2.5, 5.0, -5.0, 8.0]
    actor = None
    for extra in offsets:
        candidate = project_spawn_transform(world, transform_offset(ego_transform, longitudinal + extra, lateral))
        try:
            actor = world.try_spawn_actor(blueprint, candidate)
        except RuntimeError:
            actor = None
        if actor is not None:
            break
    if actor is None:
        return None
    try:
        actor.set_simulate_physics(True)
    except RuntimeError:
        pass
    try:
        if speed <= 0.1:
            actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        else:
            forward = actor.get_transform().get_forward_vector()
            actor.set_target_velocity(carla.Vector3D(forward.x * speed, forward.y * speed, 0.0))
            actor.apply_control(carla.VehicleControl(throttle=0.45, brake=0.0, hand_brake=False))
    except RuntimeError:
        pass
    print(
        f"[carla-sumo-mirror] spawned {prefix} actor={actor.id} long={longitudinal:.1f} "
        f"lat={lateral:.1f} speed={speed:.1f}",
        flush=True,
    )
    return actor


def start_vehicle_injection_attack(world, command, sim_time: float, attack_type: str, mode: str):
    duration = max(0.1, float(command.get("duration", 30.0)))
    count = max(1, int(command.get("count", command.get("num_obstacles", 2))))
    ego = get_ego_vehicle(world)
    spawned = []
    if attack_type in {"fake_safety", "adversarial_sensor_spoofing"}:
        color = "255,255,0" if attack_type == "fake_safety" else "255,0,0"
        for index in range(count):
            actor = spawn_attack_vehicle(
                world,
                ego,
                prefix=attack_type,
                index=index,
                longitudinal=24.0 + 12.0 * index,
                lateral=0.0,
                speed=0.0,
                color=color,
            )
            if actor is not None:
                spawned.append(actor.id)
    elif attack_type == "sybil":
        for index in range(count):
            lateral = [-3.6, 3.6, 0.0, -7.2, 7.2][index % 5]
            actor = spawn_attack_vehicle(
                world,
                ego,
                prefix=attack_type,
                index=index,
                longitudinal=18.0 + 8.0 * index,
                lateral=lateral,
                speed=2.0,
                color="255,0,0",
            )
            if actor is not None:
                spawned.append(actor.id)
    elif attack_type == "fake_emergency":
        speed = max(0.0, float(command.get("speed", 22.0)))
        for index in range(count):
            lateral = -3.6 if index % 2 == 0 else 3.6
            actor = spawn_attack_vehicle(
                world,
                ego,
                prefix=attack_type,
                index=index,
                longitudinal=-28.0 - 10.0 * index,
                lateral=lateral,
                speed=speed,
                color="0,0,255",
            )
            if actor is not None:
                spawned.append(actor.id)

    attack = {
        "id": str(command.get("id") or uuid.uuid4()),
        "type": attack_type,
        "mode": mode,
        "start_time": sim_time,
        "end_time": sim_time + duration,
        "duration": duration,
        "data": {"spawned_actor_ids": spawned},
        "stats": {"spawned_actors": len(spawned), "requested_count": count},
    }
    print(
        f"[carla-sumo-mirror] attack started {attack_type} duration={duration:.1f}s "
        f"spawned={len(spawned)}/{count}",
        flush=True,
    )
    return attack


def start_universal_perturbation_attack(world, command, sim_time: float):
    duration = max(0.1, float(command.get("duration", 30.0)))
    epsilon = max(0.0, min(1.0, float(command.get("epsilon", 0.5))))
    velocity_scale = max(0.0, min(1.0, 1.0 - epsilon * float(command.get("scale_velocity", 0.3))))
    affected = []
    for actor in world.get_actors().filter("vehicle.*"):
        try:
            if is_ego_vehicle(actor):
                continue
            affected.append(actor.id)
        except RuntimeError:
            continue
    attack = {
        "id": str(command.get("id") or uuid.uuid4()),
        "type": "universal_perturbation",
        "mode": "fleet_slowdown",
        "start_time": sim_time,
        "end_time": sim_time + duration,
        "duration": duration,
        "data": {"affected_actor_ids": affected, "velocity_scale": velocity_scale},
        "stats": {"affected_actors": len(affected), "velocity_scale": round(velocity_scale, 3)},
    }
    print(
        f"[carla-sumo-mirror] attack started universal_perturbation duration={duration:.1f}s "
        f"affected={len(affected)} velocity_scale={velocity_scale:.2f}",
        flush=True,
    )
    return attack


def restore_attack(world, attack) -> None:
    attack_type = attack.get("type")
    if attack_type == "traffic_light_tampering":
        restore_traffic_light_attack(world, attack)
    for actor_id in attack.get("data", {}).get("spawned_actor_ids", []):
        try:
            actor = world.get_actor(int(actor_id))
            if actor is not None:
                actor.destroy()
        except RuntimeError:
            continue
    if attack_type == "universal_perturbation":
        for actor_id in attack.get("data", {}).get("affected_actor_ids", []):
            try:
                actor = world.get_actor(int(actor_id))
                if actor is not None:
                    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
            except RuntimeError:
                continue


def update_spawned_attack_actors(world, attack) -> None:
    held = 0
    for actor_id in attack.get("data", {}).get("spawned_actor_ids", []):
        try:
            actor = world.get_actor(int(actor_id))
            if actor is None:
                continue
            if attack.get("type") in {"fake_safety", "adversarial_sensor_spoofing"}:
                actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            elif attack.get("type") == "sybil":
                actor.apply_control(carla.VehicleControl(throttle=0.2, brake=0.0, hand_brake=False))
            elif attack.get("type") == "fake_emergency":
                actor.apply_control(carla.VehicleControl(throttle=0.55, brake=0.0, hand_brake=False))
            held += 1
        except RuntimeError:
            continue
    attack.setdefault("stats", {})["active_spawned_actors"] = held


def update_universal_perturbation(world, attack) -> None:
    velocity_scale = float(attack.get("data", {}).get("velocity_scale", 0.85))
    applied = 0
    for actor_id in attack.get("data", {}).get("affected_actor_ids", []):
        try:
            actor = world.get_actor(int(actor_id))
            if actor is None:
                continue
            velocity = actor.get_velocity()
            actor.set_target_velocity(
                carla.Vector3D(velocity.x * velocity_scale, velocity.y * velocity_scale, velocity.z)
            )
            if carla_speed(actor) > 1.0:
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.25, hand_brake=False))
            applied += 1
        except RuntimeError:
            continue
    attack.setdefault("stats", {})["applied_actors"] = applied


def start_traffic_light_attack(world, command, sim_time: float):
    duration = max(0.1, float(command.get("duration", 30.0)))
    carla_states = {}
    for light in world.get_actors().filter("traffic.traffic_light*"):
        try:
            carla_states[str(light.id)] = traffic_light_state_name(light)
        except RuntimeError:
            continue

    sumo_states = {}
    for tlid in traci.trafficlight.getIDList():
        try:
            sumo_states[tlid] = traci.trafficlight.getRedYellowGreenState(tlid)
        except traci.TraCIException:
            continue

    attack = {
        "id": str(command.get("id") or uuid.uuid4()),
        "type": "traffic_light_tampering",
        "mode": "all_red",
        "start_time": sim_time,
        "end_time": sim_time + duration,
        "duration": duration,
        "data": {
            "carla_states": carla_states,
            "sumo_states": sumo_states,
        },
        "stats": {},
    }
    apply_traffic_light_all_red(world, attack)
    print(
        "[carla-sumo-mirror] attack started traffic_light_tampering "
        f"duration={duration:.1f}s carla_lights={len(carla_states)} "
        f"sumo_lights={len(sumo_states)} "
        f"carla_stop_points={attack.get('stats', {}).get('carla_stop_points', 0)}",
        flush=True,
    )
    return attack


def clear_active_attacks(world, active_attacks) -> int:
    count = 0
    for attack in list(active_attacks):
        restore_attack(world, attack)
        active_attacks.remove(attack)
        count += 1
    return count


def clear_attacks_by_type(world, active_attacks, attack_type: str) -> int:
    count = 0
    for attack in list(active_attacks):
        if attack.get("type") != attack_type:
            continue
        restore_attack(world, attack)
        active_attacks.remove(attack)
        count += 1
    return count


def read_attack_commands(path: Path, cursor: dict):
    if not path:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return []
    size = path.stat().st_size
    offset = cursor.get("offset", 0)
    if size < offset:
        offset = 0
    commands = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                commands.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[carla-sumo-mirror] warning: bad attack command: {exc}", flush=True)
        cursor["offset"] = handle.tell()
    return commands


def process_attack_commands(world, active_attacks, commands, sim_time: float):
    for command in commands:
        command_type = str(command.get("type") or command.get("attack") or "").strip()
        if command_type in {"clear", "clear_attacks", "restore"}:
            cleared = clear_active_attacks(world, active_attacks)
            print(f"[carla-sumo-mirror] attack clear requested cleared={cleared}", flush=True)
            continue
        if command_type in {"traffic_light_tampering", "traffic_light_all_red", "all_red"}:
            clear_attacks_by_type(world, active_attacks, "traffic_light_tampering")
            active_attacks.append(start_traffic_light_attack(world, command, sim_time))
            continue
        if command_type in {"universal_perturbation", "universal_perturbation_attack"}:
            clear_attacks_by_type(world, active_attacks, "universal_perturbation")
            active_attacks.append(start_universal_perturbation_attack(world, command, sim_time))
            continue
        if command_type in {"sensor_spoofing", "adversarial_sensor_spoofing", "targeted_adversarial_sensor_spoofing"}:
            clear_attacks_by_type(world, active_attacks, "adversarial_sensor_spoofing")
            active_attacks.append(
                start_vehicle_injection_attack(world, command, sim_time, "adversarial_sensor_spoofing", "fake_obstacles")
            )
            continue
        if command_type in {"fake_safety", "fake_safety_message"}:
            clear_attacks_by_type(world, active_attacks, "fake_safety")
            active_attacks.append(start_vehicle_injection_attack(world, command, sim_time, "fake_safety", "static_obstacles"))
            continue
        if command_type in {"fake_emergency", "fake_emergency_vehicle"}:
            clear_attacks_by_type(world, active_attacks, "fake_emergency")
            active_attacks.append(start_vehicle_injection_attack(world, command, sim_time, "fake_emergency", "blue_fast_vehicle"))
            continue
        if command_type in {"sybil", "sybil_attack"}:
            clear_attacks_by_type(world, active_attacks, "sybil")
            active_attacks.append(start_vehicle_injection_attack(world, command, sim_time, "sybil", "slow_clone_vehicles"))
            continue
        print(f"[carla-sumo-mirror] warning: unsupported attack command type={command_type}", flush=True)


def update_active_attacks(world, active_attacks, sim_time: float, traffic_manager=None, red_stop_distance=42.0) -> None:
    for attack in list(active_attacks):
        if sim_time >= float(attack.get("end_time", 0.0)):
            restore_attack(world, attack)
            active_attacks.remove(attack)
            print(f"[carla-sumo-mirror] attack expired type={attack.get('type')}", flush=True)
            continue
        if attack.get("type") == "traffic_light_tampering":
            apply_traffic_light_all_red(world, attack)
            stopped = enforce_carla_red_light_stop(
                world,
                attack,
                traffic_manager=traffic_manager,
                max_distance=red_stop_distance,
            )
            attack.setdefault("stats", {})["carla_vehicles_red_stop_enforced"] = stopped
        elif attack.get("type") == "universal_perturbation":
            update_universal_perturbation(world, attack)
        else:
            update_spawned_attack_actors(world, attack)


def build_live_state(world, vehicles, active_attacks, frames, max_vehicles, tl_changed):
    speeds = []
    for actor in vehicles:
        try:
            speeds.append(carla_speed(actor))
        except RuntimeError:
            continue
    vehicle_count = len(speeds)
    avg_speed = sum(speeds) / vehicle_count if vehicle_count else 0.0
    stopped_count = sum(1 for speed in speeds if speed < 0.5)
    stopped_ratio = stopped_count / vehicle_count if vehicle_count else 0.0
    tls_under_attack = []
    for attack in active_attacks:
        if attack.get("type") == "traffic_light_tampering":
            tls_under_attack.extend(sorted(attack.get("data", {}).get("sumo_states", {}).keys()))
    return {
        "source": "vla-av-carla-sumo-mirror",
        "bridge_mode": "carla_owned_sumo_mirror",
        "step": frames,
        "simulation_time": traci.simulation.getTime(),
        "timestamp": time.time(),
        "vehicle_count": vehicle_count,
        "avg_speed": avg_speed,
        "stopped_count": stopped_count,
        "stopped_ratio": stopped_ratio,
        "max_vehicles": max_vehicles,
        "active_attack_count": len(active_attacks),
        "active_attack_types": sorted({a.get("type", "unknown") for a in active_attacks}),
        "active_attacks": [
            {
                "id": attack.get("id"),
                "type": attack.get("type"),
                "mode": attack.get("mode"),
                "remaining_s": max(0.0, float(attack.get("end_time", 0.0)) - traci.simulation.getTime()),
                "stats": attack.get("stats", {}),
            }
            for attack in active_attacks
        ],
        "tls_under_attack": tls_under_attack,
        "traffic_light_links_synced": tl_changed,
        "metrics": {
            "fuel_consumption": 0.0,
            "co2": 0.0,
            "noise": 0.0,
            "jam": float(stopped_count),
            "emergency_breaking": 0.0,
            "pm": 0.0,
            "nox": 0.0,
            "congestion": stopped_ratio,
            "collision": 0.0,
            "nvmoc": 0.0,
        },
    }


def is_fatal_traci_error(exc) -> bool:
    name = exc.__class__.__name__
    text = str(exc).lower()
    return name == "FatalTraCIError" or "connection closed" in text or "connection already closed" in text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--connect-retries", type=int, default=180)
    parser.add_argument("--connect-wait", type=float, default=1.0)
    parser.add_argument("--net-file", type=Path, required=True)
    parser.add_argument("--additional-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/sumo_mirror/runtime"))
    parser.add_argument("--step-length", type=float, default=0.05)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means run until interrupted")
    parser.add_argument("--wait-for-vehicles", type=int, default=1)
    parser.add_argument("--wait-for-vehicles-timeout", type=float, default=240.0)
    parser.add_argument("--attack-red-stop-distance", type=float, default=42.0)
    parser.add_argument("--sumo-gui", action="store_true")
    parser.add_argument("--sync-traffic-lights", action="store_true")
    parser.add_argument("--no-warnings", action="store_true")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--attack-command-file", type=Path)
    parser.add_argument("--live-state-file", type=Path)
    args = parser.parse_args()

    offset = parse_net_offset(args.net_file)
    cfg_file = write_mirror_files(args.output_dir, args.net_file, args.additional_file)
    _client, world = connect_carla(args.host, args.port, args.connect_retries, args.connect_wait, args.timeout)
    world = wait_for_vehicle_spawn(
        _client,
        world,
        args.wait_for_vehicles,
        args.wait_for_vehicles_timeout,
        args.poll_interval,
        args.host,
        args.port,
        args.timeout,
    )
    try:
        traffic_manager = _client.get_trafficmanager(args.tm_port)
    except RuntimeError:
        traffic_manager = None
    open_sumo(args, cfg_file)
    tl_mapping = build_tl_link_map() if args.sync_traffic_lights else {}

    stopping = {"value": False}

    def handle_signal(_signum, _frame):
        stopping["value"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    active_ids = set()
    active_ego_marker_ids = set()
    active_attacks = []
    command_cursor = {"offset": 0}
    if args.attack_command_file:
        args.attack_command_file.parent.mkdir(parents=True, exist_ok=True)
        args.attack_command_file.write_text("", encoding="utf-8")
    start = time.time()
    frames = 0
    max_vehicles = 0
    last_report = start

    try:
        while not stopping["value"]:
            if args.duration > 0 and time.time() - start >= args.duration:
                break

            try:
                vehicles = list(world.get_actors().filter("vehicle.*"))
            except RuntimeError:
                _client, world = connect_carla(args.host, args.port, 10, args.connect_wait, args.timeout)
                vehicles = list(world.get_actors().filter("vehicle.*"))

            seen = set()
            ego_seen = False
            fatal_sumo_error = None
            for actor in vehicles:
                try:
                    if is_ego_vehicle(actor):
                        ego_seen = True
                        update_ego_rectangle(
                            actor,
                            offset,
                            active_ego_marker_ids,
                            follow_in_gui=args.sumo_gui,
                        )
                        continue
                    seen.add(add_or_update_vehicle(actor, active_ids, offset))
                except Exception as exc:
                    if is_fatal_traci_error(exc):
                        fatal_sumo_error = exc
                        break
                    print(f"[carla-sumo-mirror] warning: vehicle mirror failed actor={actor.id}: {exc}", flush=True)
            if fatal_sumo_error is not None:
                print(f"[carla-sumo-mirror] SUMO connection closed: {fatal_sumo_error}", flush=True)
                break

            if not ego_seen and active_ego_marker_ids:
                remove_ego_rectangle(active_ego_marker_ids)

            for veh_id in sorted(active_ids - seen):
                try:
                    traci.vehicle.remove(veh_id)
                except Exception:
                    pass
                active_ids.discard(veh_id)

            try:
                sim_time = traci.simulation.getTime()
                process_attack_commands(
                    world,
                    active_attacks,
                    read_attack_commands(args.attack_command_file, command_cursor) if args.attack_command_file else [],
                    sim_time,
                )
                update_active_attacks(
                    world,
                    active_attacks,
                    sim_time,
                    traffic_manager=traffic_manager,
                    red_stop_distance=args.attack_red_stop_distance,
                )
            except Exception as exc:
                if is_fatal_traci_error(exc):
                    print(f"[carla-sumo-mirror] SUMO connection closed during attack update: {exc}", flush=True)
                    break
                raise
            try:
                tl_changed = sync_traffic_lights(world, tl_mapping) if args.sync_traffic_lights else 0
            except Exception as exc:
                if is_fatal_traci_error(exc):
                    print(f"[carla-sumo-mirror] SUMO connection closed during TLS sync: {exc}", flush=True)
                    break
                raise
            try:
                traci.simulationStep()
            except Exception as exc:
                if is_fatal_traci_error(exc):
                    print(f"[carla-sumo-mirror] SUMO connection closed during step: {exc}", flush=True)
                    break
                raise
            frames += 1
            max_vehicles = max(max_vehicles, len(seen))
            if args.live_state_file:
                atomic_write_json(
                    args.live_state_file,
                    build_live_state(world, vehicles, active_attacks, frames, max_vehicles, tl_changed),
                )

            now = time.time()
            if now - last_report >= 2.0:
                attack_text = ""
                if active_attacks:
                    attack_bits = []
                    for attack in active_attacks:
                        if attack.get("type") == "traffic_light_tampering":
                            stats = attack.get("stats", {})
                            attack_bits.append(
                                "all_red "
                                f"carla_stop={int(stats.get('carla_vehicles_red_stop_enforced', 0) or 0)} "
                                f"candidates={int(stats.get('carla_vehicles_red_stop_candidates', 0) or 0)}"
                            )
                    if attack_bits:
                        attack_text = " attacks=" + ";".join(attack_bits)
                print(
                    f"[carla-sumo-mirror] t={traci.simulation.getTime():.1f}s "
                    f"vehicles={len(seen)} tl_links={tl_changed}{attack_text}",
                    flush=True,
                )
                last_report = now
            time.sleep(max(0.0, args.poll_interval))
    finally:
        if active_attacks:
            clear_active_attacks(world, active_attacks)
        summary = {
            "net_file": str(args.net_file.resolve()),
            "additional_file": str(args.additional_file.resolve()) if args.additional_file else None,
            "cfg_file": str(cfg_file.resolve()),
            "offset": offset,
            "frames": frames,
            "max_vehicles": max_vehicles,
            "runtime_s": round(time.time() - start, 3),
            "traffic_light_mappings": len(tl_mapping),
            "attack_command_file": str(args.attack_command_file.resolve()) if args.attack_command_file else None,
            "live_state_file": str(args.live_state_file.resolve()) if args.live_state_file else None,
        }
        if args.summary:
            write_summary(args.summary, summary)
        print("[carla-sumo-mirror] summary=" + json.dumps(summary, sort_keys=True), flush=True)
        try:
            traci.close(False)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
