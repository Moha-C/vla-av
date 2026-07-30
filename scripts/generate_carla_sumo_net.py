#!/usr/bin/env python3
"""Generate a SUMO network from an installed CARLA OpenDRIVE map."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_CARLA_ROOT = Path("/home/mohm/carla_simulator")
DEFAULT_COSIM_ROOT = DEFAULT_CARLA_ROOT / "Co-Simulation" / "Sumo"


def find_xodr(carla_root: Path, town: str) -> Path:
    candidates = [
        carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / f"{town}.xodr",
        carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / f"{town}_Opt.xodr",
        carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / town / "OpenDrive" / f"{town}.xodr",
        carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / town / "OpenDrive" / f"{town}_Opt.xodr",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find OpenDRIVE for {town}. Searched:\n{searched}")


def write_empty_routes(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<routes>
</routes>
""",
        encoding="utf-8",
    )


def write_sumocfg(path: Path, net_file: Path, route_file: Path) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
  <input>
    <net-file value="{net_file.name}"/>
    <route-files value="{route_file.name}"/>
  </input>
</configuration>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", default=os.environ.get("TOWN", "Town12"))
    parser.add_argument("--xodr", type=Path, help="Explicit OpenDRIVE file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SUMO_NET_OUTPUT_DIR", "generated_sumo_nets")),
    )
    parser.add_argument(
        "--carla-root",
        type=Path,
        default=Path(os.environ.get("CARLA_ROOT", str(DEFAULT_CARLA_ROOT))),
    )
    parser.add_argument(
        "--cosim-root",
        type=Path,
        default=Path(os.environ.get("CARLA_SUMO_COSIM_ROOT", str(DEFAULT_COSIM_ROOT))),
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    parser.add_argument("--guess-tls", action="store_true", help="Ask netconvert_carla to guess TLS")
    args = parser.parse_args()

    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    os.environ["SUMO_HOME"] = sumo_home

    town = args.town
    xodr = args.xodr if args.xodr else find_xodr(args.carla_root, town)
    netconvert_carla = args.cosim_root / "util" / "netconvert_carla.py"
    if not netconvert_carla.exists():
        raise FileNotFoundError(f"Missing CARLA/SUMO netconvert helper: {netconvert_carla}")

    output_dir = args.output_dir / town
    output_dir.mkdir(parents=True, exist_ok=True)
    net_file = output_dir / f"{town}.net.xml"
    routes_file = output_dir / "empty.rou.xml"
    sumocfg_file = output_dir / f"{town}.sumocfg"

    if args.force or not net_file.exists():
        command = [sys.executable, str(netconvert_carla), str(xodr), "--output", str(net_file)]
        if args.guess_tls:
            command.append("--guess-tls")
        print("[carla-sumo-net] " + " ".join(command))
        subprocess.run(command, check=True, cwd=str(args.cosim_root), env=os.environ.copy())
    else:
        print(f"[carla-sumo-net] reusing {net_file}")

    if not routes_file.exists():
        write_empty_routes(routes_file)
    write_sumocfg(sumocfg_file, net_file, routes_file)

    official_vtypes = args.cosim_root / "examples" / "carlavtypes.rou.xml"
    if official_vtypes.exists():
        shutil.copy2(official_vtypes, output_dir / "carlavtypes.rou.xml")

    print(f"[carla-sumo-net] town={town}")
    print(f"[carla-sumo-net] xodr={xodr}")
    print(f"[carla-sumo-net] net={net_file}")
    print(f"[carla-sumo-net] sumocfg={sumocfg_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
