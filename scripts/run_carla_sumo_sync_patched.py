#!/usr/bin/env python3
"""Run CARLA's SUMO synchronization script with local compatibility patches."""

import argparse
import os
import runpy
import sys
from pathlib import Path


DEFAULT_COSIM_ROOT = Path("/home/mohm/carla_simulator/Co-Simulation/Sumo")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cosim-root",
        type=Path,
        default=Path(os.environ.get("CARLA_SUMO_COSIM_ROOT", str(DEFAULT_COSIM_ROOT))),
    )
    args, passthrough = parser.parse_known_args()

    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    os.environ["SUMO_HOME"] = sumo_home
    sumo_tools = Path(sumo_home) / "tools"
    if sumo_tools.exists() and str(sumo_tools) not in sys.path:
        sys.path.insert(0, str(sumo_tools))

    if str(args.cosim_root) not in sys.path:
        sys.path.insert(0, str(args.cosim_root))

    # CARLA 0.9.x co-sim expects traci.sumolib, but SUMO 1.18 does not expose it.
    import traci  # pylint: disable=import-error,import-outside-toplevel
    import sumolib  # pylint: disable=import-error,import-outside-toplevel

    if not hasattr(traci, "sumolib"):
        traci.sumolib = sumolib

    run_sync = args.cosim_root / "run_synchronization.py"
    if not run_sync.exists():
        raise FileNotFoundError(f"Missing CARLA/SUMO synchronization script: {run_sync}")

    os.chdir(str(args.cosim_root))
    sys.argv = [str(run_sync)] + passthrough
    runpy.run_path(str(run_sync), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
