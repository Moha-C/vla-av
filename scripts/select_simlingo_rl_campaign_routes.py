#!/usr/bin/env python3
"""Build a deterministic SimLingo route plan for Dreamer RL data collection."""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from scripts.simlingo_dashboard import route_catalog


Route = Dict[str, Any]


def _text(route: Route) -> str:
    return " ".join(
        str(route.get(key, ""))
        for key in ("id", "town", "scenario_type", "scenario_name", "file")
    ).lower()


def _is_blocked_lane(route: Route) -> bool:
    text = _text(route)
    return any(
        token in text
        for token in (
            "accident",
            "obstacle",
            "blocked",
            "hazard",
            "parked",
            "construction",
        )
    )


def _is_oncoming_overtake(route: Route) -> bool:
    text = _text(route)
    return any(
        token in text
        for token in (
            "twoways",
            "two_ways",
            "opposite",
            "invading",
            "priority",
            "accident",
        )
    )


def _is_pedestrian_crossing(route: Route) -> bool:
    text = _text(route)
    return route.get("vru") or "pedestrian" in text or "crossing" in text


def _is_red_light_intersection(route: Route) -> bool:
    text = _text(route)
    return bool(route.get("traffic_light") or route.get("junction") or "redlight" in text)


def _stable(route: Route, include_unstable: bool) -> bool:
    if include_unstable:
        return bool(route.get("compatible", True))
    return bool(route.get("compatible", True) and route.get("installed", True))


def _priority(route: Route) -> tuple:
    scenario = str(route.get("scenario_type", ""))
    route_id = str(route.get("id", "9999"))
    try:
        numeric_id = int(route_id)
    except ValueError:
        numeric_id = 9999
    weights = {
        "Accident": 0,
        "AccidentTwoWays": 1,
        "ConstructionObstacleTwoWays": 2,
        "ConstructionObstacle": 3,
        "ParkedObstacle": 4,
        "PedestrianCrossing": 5,
        "VehicleTurningRoutePedestrian": 6,
        "CrossingBicycleFlow": 7,
        "OppositeVehicleTakingPriority": 8,
        "OppositeVehicleRunningRedLight": 9,
        "VanillaSignalizedTurnEncounterRedLight": 10,
        "BlockedIntersection": 11,
        "EnterActorFlow": 12,
        "InterurbanActorFlow": 13,
        "InterurbanAdvancedActorFlow": 14,
    }
    return (weights.get(scenario, 50), numeric_id)


def _select(
    routes: Iterable[Route],
    predicate: Callable[[Route], bool],
    *,
    max_count: int,
    include_unstable: bool,
) -> List[Route]:
    candidates = [route for route in routes if _stable(route, include_unstable) and predicate(route)]
    candidates.sort(key=_priority)
    return candidates[:max_count]


def build_campaign_plan(
    *,
    max_per_bucket: int = 2,
    seed: int = 7302026,
    include_unstable: bool = False,
) -> Dict[str, Any]:
    routes = route_catalog()
    buckets: "OrderedDict[str, Callable[[Route], bool]]" = OrderedDict()
    buckets["town10hd_accidents"] = lambda r: r.get("town") == "Town10HD" and r.get("accident")
    buckets["town12_accidents"] = lambda r: r.get("town") == "Town12" and r.get("accident")
    buckets["town12_vru"] = lambda r: r.get("town") == "Town12" and _is_pedestrian_crossing(r)
    buckets["town12_traffic_flow"] = lambda r: r.get("town") == "Town12" and bool(
        r.get("actor_flow") or r.get("cut_in") or "traffic" in _text(r) or "flow" in _text(r)
    )
    buckets["town13_complex_urban"] = lambda r: r.get("town") == "Town13" and bool(
        r.get("accident")
        or r.get("vru")
        or r.get("traffic_light")
        or r.get("junction")
        or r.get("actor_flow")
        or r.get("cut_in")
        or "yield" in _text(r)
    )
    buckets["pedestrian_crossing"] = _is_pedestrian_crossing
    buckets["red_light_intersections"] = _is_red_light_intersection
    buckets["blocked_lane"] = _is_blocked_lane
    buckets["oncoming_vehicle_overtake"] = _is_oncoming_overtake

    selected: "OrderedDict[str, Route]" = OrderedDict()
    bucket_details: Dict[str, List[str]] = {}
    for bucket, predicate in buckets.items():
        chosen = _select(
            routes,
            predicate,
            max_count=max_per_bucket,
            include_unstable=include_unstable,
        )
        bucket_details[bucket] = [str(route.get("id")) for route in chosen]
        for route in chosen:
            key = str(route.get("id"))
            if key not in selected:
                selected[key] = {
                    "route_id": key,
                    "route_file": route.get("file"),
                    "town": route.get("town"),
                    "scenario_name": route.get("scenario_name"),
                    "scenario_type": route.get("scenario_type"),
                    "buckets": [],
                }
            selected[key]["buckets"].append(bucket)

    rng = random.Random(seed)
    runs = []
    for index, route in enumerate(selected.values(), start=1):
        run_seed = rng.randint(1, 999999)
        runs.append({
            "index": index,
            "seed": run_seed,
            **route,
        })

    return {
        "seed": seed,
        "max_per_bucket": max_per_bucket,
        "include_unstable": include_unstable,
        "available_routes": len(routes),
        "bucket_details": bucket_details,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-bucket", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7302026)
    parser.add_argument("--include-unstable", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = build_campaign_plan(
        max_per_bucket=max(1, args.max_per_bucket),
        seed=args.seed,
        include_unstable=args.include_unstable,
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
