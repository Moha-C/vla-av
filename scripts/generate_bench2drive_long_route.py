#!/usr/bin/env python3
"""Generate a long Bench2Drive-compatible route XML from a live CARLA map."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
import xml.etree.ElementTree as ET
from typing import List, Sequence, Tuple

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


RoutePoint = Tuple[carla.Transform, object]


def connect_client(host: str, port: int, timeout: float, rpc_timeout: float) -> carla.Client:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            client = carla.Client(host, port)
            client.set_timeout(rpc_timeout)
            client.get_world()
            return client
        except Exception as exc:  # CARLA raises RuntimeError while booting.
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Could not connect to CARLA on {host}:{port}: {last_error}")


def load_town(client: carla.Client, town: str, timeout: float, rpc_timeout: float) -> carla.World:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            client.set_timeout(rpc_timeout)
            world = client.get_world()
            current_town = world.get_map().name.split("/")[-1]
            if current_town == town:
                return world
            world = client.load_world(town)
            for _ in range(20):
                time.sleep(0.5)
                world = client.get_world()
                if world.get_map().name.split("/")[-1] == town:
                    return world
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"CARLA did not load {town} within {timeout:.0f}s: {last_error}")


def distance_xy(a: carla.Location, b: carla.Location) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def choose_next_spawn(
    rng: random.Random,
    spawns: Sequence[carla.Transform],
    current: carla.Transform,
    recent_indices: Sequence[int],
    min_leg_distance: float,
) -> int:
    recent = set(recent_indices[-4:])
    scored = []
    for idx, transform in enumerate(spawns):
        if idx in recent:
            continue
        dist = distance_xy(current.location, transform.location)
        if dist > 20.0:
            scored.append((dist, idx))

    if not scored:
        raise RuntimeError("No usable spawn points found for route generation")

    scored.sort(reverse=True)
    far = [item for item in scored if item[0] >= min_leg_distance]
    candidates = far if far else scored[: max(4, min(20, len(scored)))]
    return rng.choice(candidates[: max(1, min(12, len(candidates)))])[1]


def append_trace(route: List[RoutePoint], trace: Sequence[RoutePoint]) -> None:
    for transform, road_option in trace:
        if route and distance_xy(route[-1][0].location, transform.location) < 0.5:
            continue
        route.append((transform, road_option))


def trace_long_route(
    world: carla.World,
    seed: int,
    segments: int,
    min_leg_distance: float,
    sampling_resolution: float,
    make_loop: bool,
) -> List[RoutePoint]:
    rng = random.Random(seed)
    carla_map = world.get_map()
    spawns = list(carla_map.get_spawn_points())
    if len(spawns) < 2:
        raise RuntimeError(f"Map {carla_map.name} has fewer than two spawn points")

    planner = GlobalRoutePlanner(carla_map, sampling_resolution)
    start_idx = rng.randrange(len(spawns))
    current_idx = start_idx
    current = spawns[current_idx]
    recent = [current_idx]
    route: List[RoutePoint] = []

    for _ in range(max(1, segments)):
        next_idx = choose_next_spawn(rng, spawns, current, recent, min_leg_distance)
        target = spawns[next_idx]
        trace = planner.trace_route(current.location, target.location)
        if not trace:
            recent.append(next_idx)
            current_idx = next_idx
            current = target
            continue
        append_trace(route, trace)
        recent.append(next_idx)
        current_idx = next_idx
        current = target

    if make_loop and current_idx != start_idx:
        trace = planner.trace_route(current.location, spawns[start_idx].location)
        append_trace(route, trace)

    if len(route) < 4:
        raise RuntimeError("Generated route is too short; try another seed or fewer restrictions")
    return route


def downsample_route(route: Sequence[RoutePoint], spacing: float, max_keypoints: int) -> List[carla.Location]:
    keypoints: List[carla.Location] = []
    previous_option = None
    distance_since_last = 0.0

    for idx, (transform, option) in enumerate(route):
        location = transform.location
        should_keep = False
        if idx == 0 or idx == len(route) - 1:
            should_keep = True
        elif previous_option is not None and option != previous_option:
            should_keep = True
        elif distance_since_last >= spacing:
            should_keep = True

        if should_keep:
            if not keypoints or distance_xy(keypoints[-1], location) > 0.5:
                keypoints.append(carla.Location(location.x, location.y, location.z))
            distance_since_last = 0.0
        elif idx > 0:
            distance_since_last += distance_xy(route[idx - 1][0].location, location)

        previous_option = option

    if len(keypoints) > max_keypoints:
        step = int(math.ceil(len(keypoints) / float(max_keypoints)))
        reduced = keypoints[::step]
        if reduced[-1] != keypoints[-1]:
            reduced.append(keypoints[-1])
        keypoints = reduced

    return keypoints


def route_length(route: Sequence[RoutePoint]) -> float:
    total = 0.0
    for previous, current in zip(route, route[1:]):
        total += distance_xy(previous[0].location, current[0].location)
    return total


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def write_route_xml(path: str, town: str, route_id: str, keypoints: Sequence[carla.Location]) -> None:
    routes = ET.Element("routes")
    route = ET.SubElement(routes, "route", {"id": str(route_id), "road_id": "vla_av_custom_long", "town": town})
    waypoints = ET.SubElement(route, "waypoints")
    for location in keypoints:
        ET.SubElement(
            waypoints,
            "position",
            {
                "x": f"{location.x:.6f}",
                "y": f"{location.y:.6f}",
                "z": f"{location.z:.6f}",
            },
        )

    ET.SubElement(route, "scenarios")
    weathers = ET.SubElement(route, "weathers")
    weather_attrs = {
        "cloudiness": "0.0",
        "precipitation": "0.0",
        "precipitation_deposits": "0.0",
        "wind_intensity": "0.0",
        "sun_azimuth_angle": "45.0",
        "sun_altitude_angle": "75.0",
        "fog_density": "0.0",
        "fog_distance": "0.0",
        "wetness": "0.0",
    }
    for percentage in ("0", "100"):
        attrs = dict(weather_attrs)
        attrs["route_percentage"] = percentage
        ET.SubElement(weathers, "weather", attrs)

    indent_xml(routes)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ET.ElementTree(routes).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town12")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--min-leg-distance", type=float, default=250.0)
    parser.add_argument("--sampling-resolution", type=float, default=2.0)
    parser.add_argument("--keypoint-spacing", type=float, default=25.0)
    parser.add_argument("--max-keypoints", type=int, default=420)
    parser.add_argument("--route-id", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--connect-timeout", type=float, default=120.0)
    parser.add_argument("--load-timeout", type=float, default=240.0)
    parser.add_argument("--rpc-timeout", type=float, default=120.0)
    args = parser.parse_args()

    client = connect_client(args.host, args.port, args.connect_timeout, args.rpc_timeout)
    world = load_town(client, args.town, args.load_timeout, args.rpc_timeout)
    full_route = trace_long_route(
        world=world,
        seed=args.seed,
        segments=args.segments,
        min_leg_distance=args.min_leg_distance,
        sampling_resolution=args.sampling_resolution,
        make_loop=args.loop,
    )
    keypoints = downsample_route(full_route, args.keypoint_spacing, args.max_keypoints)
    route_id = args.route_id or str(900000 + (args.seed % 99999))
    write_route_xml(args.output, args.town, route_id, keypoints)

    print(f"[route-generator] town={args.town}")
    print(f"[route-generator] seed={args.seed}")
    print(f"[route-generator] dense_points={len(full_route)}")
    print(f"[route-generator] keypoints={len(keypoints)}")
    print(f"[route-generator] approx_length_m={route_length(full_route):.1f}")
    print(f"[route-generator] output={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
