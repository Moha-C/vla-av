#!/usr/bin/env python3
"""Run the active mirrored CarDreamer RSSM complement across Bench2Drive routes."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.select_simlingo_rl_campaign_routes import build_campaign_plan


LOG_ROOT = ROOT / "logs" / "cardreamer_rssm_campaign"
OFFICIAL_SHA256 = "123525828488d596e80dad0fad0681767cec937adcc04bf0d5aa8ee972aa8058"


def write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stop_stack() -> None:
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "stop_simlingo_dashboard.sh")],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    for sig, wait_seconds in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


def run_route(
    route: Dict[str, Any],
    run_dir: pathlib.Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    route_id = str(route["route_id"])
    seed = int(route["seed"])
    label = f"{int(route['index']):02d}_route_{route_id}_seed_{seed}"
    route_log = run_dir / f"{label}.log"
    env = os.environ.copy()
    env.update(
        {
            "ROUTE_FILE": str(route["route_file"]),
            "ROUTE_ID": route_id,
            "SEED": str(seed),
            "PORT": str(args.port),
            "TM_PORT": str(args.tm_port),
            "CARLA_QUALITY": args.quality,
            "SIMLINGO_VIEW_MODE": args.view_mode,
            "SIMLINGO_VIEW_WIDTH": str(args.width),
            "SIMLINGO_VIEW_HEIGHT": str(args.height),
            "SIMLINGO_VIEW_FPS": str(args.fps),
            "SIMLINGO_CARDREAMER_MODE": "residual",
            "SIMLINGO_CARDREAMER_LATERAL_ADAPTER": "mirror",
            "SIMLINGO_CARDREAMER_RESIDUAL_ALPHA": str(args.residual_alpha),
            "SIMLINGO_CARDREAMER_EXPECTED_SHA256": OFFICIAL_SHA256,
            "SIMLINGO_DREAMER_GUARD": "0",
            "SIMLINGO_DREAMER_RUNTIME": "",
            "SIMLINGO_DREAMER_RECOVERY": "0",
            "SIMLINGO_DREAMER_COLLISION_SHIELD": "0",
            "SIMLINGO_RECORD": "1" if args.record else "0",
            "SIMLINGO_PLAYBACK_AFTER": "0",
        }
    )
    started = time.time()
    print(
        f"[cardreamer-campaign] {route['index']}/{args.selected_count} "
        f"route={route_id} town={route.get('town')} scenario={route.get('scenario_type')} "
        f"seed={seed}",
        flush=True,
    )
    if args.dry_run:
        return {
            **route,
            "status": "dry_run",
            "command": "bash scripts/run_simlingo_with_pov.sh",
            "log": str(route_log),
        }

    stop_stack()
    with route_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            ["bash", str(ROOT / "scripts" / "run_simlingo_with_pov.sh")],
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            exit_code = process.wait(timeout=args.route_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_process_group(process)
            exit_code = process.returncode
    stop_stack()
    return {
        **route,
        "status": "timeout" if timed_out else ("finished" if exit_code == 0 else "failed"),
        "exit_code": exit_code,
        "elapsed_seconds": round(time.time() - started, 3),
        "log": str(route_log),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-bucket", type=int, default=1)
    parser.add_argument("--max-routes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17082026)
    parser.add_argument("--include-unstable", action="store_true")
    parser.add_argument("--route-timeout", type=int, default=900)
    parser.add_argument("--residual-alpha", type=float, default=0.35)
    parser.add_argument("--quality", choices=("Low", "Epic"), default="Low")
    parser.add_argument("--view-mode", choices=("chase", "wheel", "front", "top"), default="chase")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.10 <= args.residual_alpha <= 0.75:
        raise ValueError("--residual-alpha must be between 0.10 and 0.75")
    plan = build_campaign_plan(
        max_per_bucket=max(1, args.max_per_bucket),
        seed=args.seed,
        include_unstable=args.include_unstable,
    )
    routes: List[Dict[str, Any]] = plan["runs"]
    if args.max_routes >= 0:
        routes = routes[: args.max_routes]
    args.selected_count = len(routes)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plan["runs"] = routes
    plan.update(
        {
            "run_id": run_id,
            "mode": "cardreamer_rssm_mirror_residual",
            "checkpoint_sha256": OFFICIAL_SHA256,
            "residual_alpha": args.residual_alpha,
            "experimental_unvalidated": True,
            "privileged_information": True,
        }
    )
    write_json(run_dir / "campaign_plan.json", plan)
    (LOG_ROOT / "latest_campaign.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    results = []
    try:
        for route in routes:
            result = run_route(route, run_dir, args)
            results.append(result)
            write_json(run_dir / "summary.json", {**plan, "results": results})
    finally:
        if not args.dry_run:
            stop_stack()
    summary = {
        **plan,
        "results": results,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished": sum(item.get("status") == "finished" for item in results),
        "failed": sum(item.get("status") == "failed" for item in results),
        "timed_out": sum(item.get("status") == "timeout" for item in results),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"[cardreamer-campaign] summary={run_dir / 'summary.json'}", flush=True)
    return 0 if not any(item.get("status") == "failed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
