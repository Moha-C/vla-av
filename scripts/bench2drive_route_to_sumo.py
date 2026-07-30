#!/usr/bin/env python3
"""Convert a Bench2Drive/SimLingo route XML into SUMO visual and route files."""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def ensure_sumo_tools() -> None:
    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    tools = Path(sumo_home) / "tools"
    if tools.exists() and str(tools) not in sys.path:
        sys.path.append(str(tools))


def parse_bench2drive(route_xml: Path):
    tree = ET.parse(route_xml)
    root = tree.getroot()
    route = root.find("route")
    if route is None:
        raise ValueError(f"No <route> tag in {route_xml}")

    points = []
    for position in route.findall("./waypoints/position"):
        points.append(
            (
                float(position.attrib["x"]),
                float(position.attrib["y"]),
                float(position.attrib.get("z", "0")),
            )
        )
    if not points:
        raise ValueError(f"No waypoint positions in {route_xml}")

    scenario = route.find("./scenarios/scenario")
    metadata = {
        "route_id": route.attrib.get("id", route_xml.stem),
        "town": route.attrib.get("town", "unknown"),
        "road_id": route.attrib.get("road_id", ""),
        "scenario_name": scenario.attrib.get("name", "") if scenario is not None else "",
        "scenario_type": scenario.attrib.get("type", "") if scenario is not None else "",
    }
    return metadata, points


def read_net_offset(net_file: Path):
    root = ET.parse(net_file).getroot()
    location = root.find("location")
    if location is None:
        return (0.0, 0.0)
    raw = location.attrib.get("netOffset", "0,0").split(",")
    return (float(raw[0]), float(raw[1]))


def carla_to_sumo(points, offset):
    ox, oy = offset
    return [(x + ox, -y + oy) for x, y, _z in points]


def write_additional(path: Path, sumo_points, metadata, stride: int) -> None:
    stride = max(1, stride)
    poly_shape = " ".join(f"{x:.2f},{y:.2f}" for x, y in sumo_points)
    root = ET.Element("additional")
    ET.SubElement(
        root,
        "poly",
        {
            "id": f"bench2drive_route_{metadata['route_id']}",
            "type": "bench2drive_route",
            "color": "0,80,255",
            "layer": "100",
            "lineWidth": "4",
            "shape": poly_shape,
        },
    )
    for index, (x, y) in enumerate(sumo_points[::stride]):
        ET.SubElement(
            root,
            "poi",
            {
                "id": f"wp_{index:04d}",
                "type": "bench2drive_wp",
                "color": "255,0,0",
                "layer": "101",
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
            },
        )
    indent(root)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def nearest_lane(net, x, y, initial_radius, max_radius):
    radius = initial_radius
    while radius <= max_radius:
        candidates = net.getNeighboringLanes(x, y, r=radius, includeJunctions=False, allowFallback=True)
        filtered = []
        for lane, dist in candidates:
            edge = lane.getEdge()
            if edge.getFunction():
                continue
            filtered.append((lane, dist))
        if filtered:
            filtered.sort(key=lambda item: item[1])
            return filtered[0]
        radius *= 2.0
    return None


def build_edge_sequence(net_file: Path, sumo_points, radius: float, max_radius: float):
    ensure_sumo_tools()
    import sumolib  # pylint: disable=import-error,import-outside-toplevel

    net = sumolib.net.readNet(str(net_file))
    raw_edges = []
    misses = 0
    max_distance = 0.0

    for x, y in sumo_points:
        result = nearest_lane(net, x, y, radius, max_radius)
        if result is None:
            misses += 1
            continue
        lane, dist = result
        max_distance = max(max_distance, dist)
        edge_id = lane.getEdge().getID()
        if not raw_edges or raw_edges[-1] != edge_id:
            raw_edges.append(edge_id)

    filled = []
    disconnected = []
    for edge_id in raw_edges:
        if not filled:
            filled.append(edge_id)
            continue
        if filled[-1] == edge_id:
            continue
        try:
            start = net.getEdge(filled[-1])
            end = net.getEdge(edge_id)
            path, _cost = net.getShortestPath(
                start,
                end,
                vClass="passenger",
                withInternal=False,
                includeFromToCost=True,
            )
        except Exception:
            path = None
        if path:
            for edge in path[1:]:
                eid = edge.getID()
                if not filled or filled[-1] != eid:
                    filled.append(eid)
        else:
            disconnected.append((filled[-1], edge_id))
            filled.append(edge_id)

    return {
        "edges": filled,
        "raw_edges": raw_edges,
        "misses": misses,
        "max_lane_distance_m": max_distance,
        "disconnected_pairs": disconnected,
    }


def write_route_file(path: Path, edge_info, vehicle_id: str) -> None:
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        {
            "id": "simlingo_route_probe",
            "vClass": "passenger",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.5",
            "length": "4.8",
            "maxSpeed": "13.89",
            "color": "0,80,255",
        },
    )
    ET.SubElement(
        root,
        "route",
        {"id": "bench2drive_edges", "edges": " ".join(edge_info["edges"])},
    )
    ET.SubElement(
        root,
        "vehicle",
        {
            "id": vehicle_id,
            "type": "simlingo_route_probe",
            "route": "bench2drive_edges",
            "depart": "0",
            "departLane": "best",
            "departSpeed": "0",
        },
    )
    indent(root)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def write_sumocfg(path: Path, net_file: Path, route_file: Path, additional_file: Path) -> None:
    root = ET.Element(
        "configuration",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/sumoConfiguration.xsd",
        },
    )
    input_node = ET.SubElement(root, "input")
    ET.SubElement(input_node, "net-file", {"value": str(net_file)})
    ET.SubElement(input_node, "route-files", {"value": str(route_file)})
    ET.SubElement(input_node, "additional-files", {"value": str(additional_file)})
    time_node = ET.SubElement(root, "time")
    ET.SubElement(time_node, "begin", {"value": "0"})
    ET.SubElement(time_node, "end", {"value": "300"})
    indent(root)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def indent(elem, level=0):
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-xml", type=Path, required=True)
    parser.add_argument("--net-file", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SUMO_ROUTE_OUTPUT_DIR", "generated_sumo_routes")),
    )
    parser.add_argument("--vehicle-id", default="simlingo_ego")
    parser.add_argument("--poi-stride", type=int, default=5)
    parser.add_argument("--nearest-radius", type=float, default=8.0)
    parser.add_argument("--max-radius", type=float, default=80.0)
    args = parser.parse_args()

    metadata, carla_points = parse_bench2drive(args.route_xml)
    offset = read_net_offset(args.net_file)
    sumo_points = carla_to_sumo(carla_points, offset)

    output_dir = args.output_dir / args.route_xml.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    additional_file = output_dir / f"{args.route_xml.stem}.add.xml"
    route_file = output_dir / f"{args.route_xml.stem}.rou.xml"
    sumocfg_file = output_dir / f"{args.route_xml.stem}.sumocfg"
    summary_file = output_dir / f"{args.route_xml.stem}.summary.json"

    write_additional(additional_file, sumo_points, metadata, args.poi_stride)
    edge_info = build_edge_sequence(args.net_file, sumo_points, args.nearest_radius, args.max_radius)
    if edge_info["edges"]:
        write_route_file(route_file, edge_info, args.vehicle_id)
    else:
        raise RuntimeError("Could not map any Bench2Drive waypoint to a SUMO edge.")
    write_sumocfg(sumocfg_file, args.net_file.resolve(), route_file.resolve(), additional_file.resolve())

    summary = {
        "metadata": metadata,
        "route_xml": str(args.route_xml.resolve()),
        "net_file": str(args.net_file.resolve()),
        "net_offset": offset,
        "carla_waypoints": len(carla_points),
        "sumo_edges": len(edge_info["edges"]),
        "raw_sumo_edges": len(edge_info["raw_edges"]),
        "missed_waypoints": edge_info["misses"],
        "max_lane_distance_m": round(edge_info["max_lane_distance_m"], 3),
        "disconnected_pairs": edge_info["disconnected_pairs"],
        "additional_file": str(additional_file.resolve()),
        "route_file": str(route_file.resolve()),
        "sumocfg_file": str(sumocfg_file.resolve()),
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
