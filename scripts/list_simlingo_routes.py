#!/usr/bin/env python3
import glob
import os
import re
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMLINGO_ROOT = os.environ.get("SIMLINGO_ROOT", os.path.join(ROOT, "external", "simlingo"))
CARLA_ROOT = os.environ.get("CARLA_ROOT", os.path.expanduser("~/carla_simulator"))


def installed_towns():
    maps_dir = os.path.join(CARLA_ROOT, "CarlaUE4", "Content", "Carla", "Maps")
    towns = set()
    for path in glob.glob(os.path.join(maps_dir, "**", "Town*.umap"), recursive=True):
        town = os.path.splitext(os.path.basename(path))[0]
        if "_Tile_" not in town:
            towns.add(town)
    return towns


def route_towns():
    by_town = defaultdict(list)
    pattern = os.path.join(SIMLINGO_ROOT, "leaderboard", "data", "bench2drive_split", "*.xml")
    for path in sorted(glob.glob(pattern)):
        with open(path, errors="ignore") as f:
            text = f.read(4096)
        match = re.search(r'town="([^"]+)"', text)
        if match:
            by_town[match.group(1)].append(path)
    return by_town


def main():
    installed = installed_towns()
    by_town = route_towns()
    print("Installed CARLA towns:", ", ".join(sorted(installed)) or "none detected")
    print()
    print("Compatible SimLingo/Bench2Drive routes:")
    for town in sorted(by_town):
        status = "OK" if town in installed else "missing map"
        sample = os.path.basename(by_town[town][0])
        print(f"{town:10s} {len(by_town[town]):3d} routes | {status:11s} | sample={sample}")


if __name__ == "__main__":
    main()
