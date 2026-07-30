#!/usr/bin/env python3
"""Print the local SimLingo/Bench2Drive route coverage used by the dashboard."""

import collections
import json
import sys

from simlingo_dashboard import route_catalog


def main() -> int:
    routes = route_catalog()
    by_town = collections.defaultdict(list)
    for route in routes:
        by_town[route["town"]].append(route)

    summary = []
    for town, town_routes in sorted(by_town.items()):
        compatible = [route for route in town_routes if route["compatible"]]
        summary.append({
            "town": town,
            "routes": len(town_routes),
            "dashboard_enabled": len(compatible),
            "stable": any(route["stable"] for route in town_routes),
            "installed": any(route["installed"] for route in town_routes),
            "vru": sum(1 for route in compatible if route["vru"]),
            "traffic_light": sum(1 for route in compatible if route["traffic_light"]),
            "stop": sum(1 for route in compatible if route["stop"]),
            "junction": sum(1 for route in compatible if route["junction"]),
        })

    print(json.dumps(summary, indent=2))
    enabled = sum(item["dashboard_enabled"] for item in summary)
    if enabled == 0:
        print("No dashboard-enabled routes found.", file=sys.stderr)
        return 1
    print(f"dashboard_enabled_routes={enabled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
