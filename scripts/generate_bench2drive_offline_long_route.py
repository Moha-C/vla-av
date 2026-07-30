#!/usr/bin/env python3
"""Generate a long Bench2Drive XML by stitching installed route keypoints."""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import xml.etree.ElementTree as ET
from typing import Iterable, List, Sequence, Tuple


Point = Tuple[float, float, float]


def distance_xy(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_route(path: str, town: str) -> Tuple[str, List[Point]]:
    root = ET.parse(path).getroot()
    route = root.find("route")
    if route is None or route.attrib.get("town") != town:
        return "", []
    waypoints = route.find("waypoints")
    if waypoints is None:
        return "", []
    points = []
    for elem in waypoints.iter("position"):
        points.append((float(elem.attrib["x"]), float(elem.attrib["y"]), float(elem.attrib["z"])))
    return route.attrib.get("id", os.path.basename(path)), points


def load_candidates(routes_root: str, town: str) -> List[Tuple[str, str, List[Point]]]:
    candidates = []
    for path in sorted(glob.glob(os.path.join(routes_root, "*.xml"))):
        route_id, points = parse_route(path, town)
        if len(points) >= 3:
            candidates.append((path, route_id, points))
    if not candidates:
        raise RuntimeError(f"No Bench2Drive route XML found for {town} in {routes_root}")
    return candidates


def route_length(points: Sequence[Point]) -> float:
    return sum(distance_xy(a, b) for a, b in zip(points, points[1:]))


def orient_route(points: Sequence[Point], current_end: Point) -> List[Point]:
    normal_start = distance_xy(current_end, points[0])
    reversed_start = distance_xy(current_end, points[-1])
    return list(points) if normal_start <= reversed_start else list(reversed(points))


def stitch_routes(
    candidates: Sequence[Tuple[str, str, List[Point]]],
    seed: int,
    segments: int,
    prefer_far_connections: bool,
) -> Tuple[List[Point], List[str]]:
    rng = random.Random(seed)
    remaining = list(candidates)
    start_idx = rng.randrange(len(remaining))
    path, route_id, points = remaining.pop(start_idx)
    stitched = list(points)
    used = [f"{os.path.basename(path)}#{route_id}"]

    for _ in range(max(1, segments) - 1):
        if not remaining:
            break
        current_end = stitched[-1]
        scored = []
        for idx, (candidate_path, candidate_route_id, candidate_points) in enumerate(remaining):
            oriented = orient_route(candidate_points, current_end)
            connection = distance_xy(current_end, oriented[0])
            internal = route_length(oriented)
            score = connection + 0.25 * internal if prefer_far_connections else internal - 0.15 * connection
            scored.append((score, idx, oriented, candidate_path, candidate_route_id))
        scored.sort(reverse=True)
        _, idx, oriented, candidate_path, candidate_route_id = rng.choice(scored[: max(1, min(8, len(scored)))])
        remaining.pop(idx)
        if distance_xy(stitched[-1], oriented[0]) < 0.5:
            stitched.extend(oriented[1:])
        else:
            stitched.extend(oriented)
        used.append(f"{os.path.basename(candidate_path)}#{candidate_route_id}")

    return stitched, used


def downsample(points: Sequence[Point], max_points: int) -> List[Point]:
    if len(points) <= max_points:
        return list(points)
    step = int(math.ceil(len(points) / float(max_points)))
    reduced = list(points[::step])
    if reduced[-1] != points[-1]:
        reduced.append(points[-1])
    return reduced


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


def write_xml(path: str, town: str, route_id: str, points: Iterable[Point]) -> None:
    routes = ET.Element("routes")
    route = ET.SubElement(routes, "route", {"id": str(route_id), "road_id": "vla_av_offline_long", "town": town})
    waypoints = ET.SubElement(route, "waypoints")
    for x, y, z in points:
        ET.SubElement(waypoints, "position", {"x": f"{x:.6f}", "y": f"{y:.6f}", "z": f"{z:.6f}"})
    ET.SubElement(route, "scenarios")
    weathers = ET.SubElement(route, "weathers")
    weather = {
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
        attrs = dict(weather)
        attrs["route_percentage"] = percentage
        ET.SubElement(weathers, "weather", attrs)
    indent_xml(routes)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ET.ElementTree(routes).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes-root", required=True)
    parser.add_argument("--town", default="Town12")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--segments", type=int, default=12)
    parser.add_argument("--max-keypoints", type=int, default=420)
    parser.add_argument("--route-id", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefer-local-connections", action="store_true")
    args = parser.parse_args()

    candidates = load_candidates(args.routes_root, args.town)
    stitched, used = stitch_routes(
        candidates,
        seed=args.seed,
        segments=args.segments,
        prefer_far_connections=not args.prefer_local_connections,
    )
    reduced = downsample(stitched, args.max_keypoints)
    route_id = args.route_id or str(800000 + (args.seed % 99999))
    write_xml(args.output, args.town, route_id, reduced)

    print(f"[offline-route] town={args.town}")
    print(f"[offline-route] seed={args.seed}")
    print(f"[offline-route] source_routes={len(used)}")
    print(f"[offline-route] source_keypoints={len(stitched)}")
    print(f"[offline-route] output_keypoints={len(reduced)}")
    print(f"[offline-route] approx_keypoint_polyline_m={route_length(stitched):.1f}")
    print(f"[offline-route] output={os.path.abspath(args.output)}")
    print("[offline-route] used=" + ",".join(used))


if __name__ == "__main__":
    main()
