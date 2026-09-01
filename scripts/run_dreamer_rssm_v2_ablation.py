#!/usr/bin/env python3
"""Frozen Bench2Drive A/B evaluation for the experimental RSSM V2.

The runner compares native SimLingo, the protected no-guard PPO checkpoint and
the isolated RSSM V2 candidate on identical route/seed pairs. It never trains or
promotes a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import scripts.simlingo_dashboard as dashboard
from scripts.run_dreamer_curriculum_training import (
    aggregate,
    comparison_gate,
    fallback_metrics,
    first_collision,
)


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs" / "dreamer_rssm_v2_ablation"
PPO_CHECKPOINT = (
    ROOT
    / "external"
    / "simlingo"
    / "checkpoints"
    / "dreamer_ppo_rl_noguard"
    / "production_model.pt"
)
RSSM_CHECKPOINT = (
    ROOT
    / "external"
    / "simlingo"
    / "checkpoints"
    / "dreamer_ppo_rssm_v2"
    / "candidate_model.pt"
)

DEFAULT_SUITE: Tuple[Tuple[str, int], ...] = (
    ("148", 829101),  # Town10HD Accident
    ("32", 829102),   # Town12 Accident
    ("06", 829201),   # Town12 AccidentTwoWays
    ("70", 829202),   # Town13 AccidentTwoWays
    ("54", 829301),   # Town12 CrossingBicycleFlow
    ("93", 829302),   # Town13 EnterActorFlow
    ("113", 829401),  # Town12 PedestrianCrossing
    ("91", 829402),   # Town13 VehicleTurningRoutePedestrian
)

MODES: Dict[str, Dict[str, str]] = {
    "native": {
        "dreamer_mode": "off",
        "description": "SimLingo native",
    },
    "ppo": {
        "dreamer_mode": "dreamer_ppo_rl_noguard",
        "description": "SimLingo + protected PPO no-guard",
    },
    "rssm_v2": {
        "dreamer_mode": "dreamer_ppo_rssm_v2",
        "description": "SimLingo + experimental RSSM V2",
    },
}

ACTIVE_SIMULATION_PATTERN = (
    "[l]eaderboard_evaluator.py|[C]arlaUE4-Linux-Shipping|"
    "[r]un_simlingo_with_pov.sh|[r]un_simlingo_with_sumo_mirror.sh"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def active_simulation_processes() -> List[str]:
    result = subprocess.run(
        ["pgrep", "-af", ACTIVE_SIMULATION_PATTERN],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def checkpoint_for_mode(
    mode: str, rssm_checkpoint: Path = RSSM_CHECKPOINT
) -> Optional[Path]:
    if mode == "ppo":
        return PPO_CHECKPOINT
    if mode == "rssm_v2":
        return rssm_checkpoint
    return None


def suite_for_routes(route_ids: Iterable[str], repetitions: int) -> List[Tuple[str, int]]:
    requested = list(route_ids)
    seeds = dict(DEFAULT_SUITE)
    unknown = sorted(set(requested) - set(seeds))
    if unknown:
        raise ValueError(
            "Routes outside the frozen suite need an explicit reviewed seed: "
            + ", ".join(unknown)
        )
    rows: List[Tuple[str, int]] = []
    for repetition in range(repetitions):
        for route_id in requested:
            rows.append((route_id, seeds[route_id] + repetition * 10000))
    return rows


class FrozenAblation:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = LOG_ROOT / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "status.json"
        self.report_path = self.run_dir / "report.json"
        self.summary_path = self.run_dir / "summary.md"
        self.lock_path = LOG_ROOT / "ablation.lock"
        self.route_catalog = {row["id"]: row for row in dashboard.route_catalog()}
        self.modes = parse_csv(args.modes)
        invalid_modes = sorted(set(self.modes) - set(MODES))
        if invalid_modes:
            raise ValueError("Unknown modes: " + ", ".join(invalid_modes))
        if "native" not in self.modes:
            raise ValueError("The frozen comparison must include native SimLingo.")
        self.suite = suite_for_routes(parse_csv(args.routes), args.repetitions)
        self.rssm_checkpoint = args.rssm_checkpoint.expanduser().resolve()
        self.records: List[Dict[str, Any]] = []
        self.reused_records: List[Dict[str, Any]] = []
        self.started_at = time.time()
        self.owns_simulation = False

    def acquire_lock(self) -> None:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            try:
                old_pid = int(self.lock_path.read_text(encoding="utf-8").strip())
                os.kill(old_pid, 0)
            except (FileNotFoundError, OSError, ValueError):
                pass
            else:
                raise RuntimeError(f"another RSSM ablation is running with pid {old_pid}")
        self.lock_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    def release_lock(self) -> None:
        try:
            if self.lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                self.lock_path.unlink()
        except (FileNotFoundError, OSError):
            pass

    def validate(self) -> Dict[str, Any]:
        checkpoints = {}
        for mode in self.modes:
            checkpoint = checkpoint_for_mode(mode, self.rssm_checkpoint)
            if checkpoint is not None:
                if not checkpoint.exists():
                    raise FileNotFoundError(f"{mode} checkpoint not found: {checkpoint}")
                checkpoints[mode] = {
                    "path": str(checkpoint),
                    "sha256": sha256(checkpoint),
                }
        routes = []
        for route_id, seed in self.suite:
            route = self.route_catalog.get(route_id)
            if route is None:
                raise RuntimeError(f"route {route_id} is absent from the dashboard catalog")
            if not route.get("installed"):
                raise RuntimeError(f"CARLA map {route.get('town')} is not installed")
            routes.append({
                "route_id": route_id,
                "seed": seed,
                "town": route.get("town"),
                "scenario_type": route.get("scenario_type"),
                "route_file": route.get("file"),
            })
        reused_native_report = None
        if self.args.reuse_native_report is not None:
            source = self.args.reuse_native_report.expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(f"native baseline report not found: {source}")
            payload = json.loads(source.read_text(encoding="utf-8"))
            native_by_key = {
                (str(row.get("route_id")), int(row.get("seed", -1))): row
                for row in payload.get("records", [])
                if row.get("mode") == "native"
            }
            missing = [
                (route_id, seed)
                for route_id, seed in self.suite
                if (route_id, seed) not in native_by_key
            ]
            if missing:
                raise RuntimeError(
                    "reused native report is missing frozen route/seed pairs: "
                    + ", ".join(f"{route}/{seed}" for route, seed in missing)
                )
            self.reused_records = []
            for route_id, seed in self.suite:
                row = copy.deepcopy(native_by_key[(route_id, seed)])
                row["reused"] = True
                row["reused_from"] = str(source)
                self.reused_records.append(row)
            reused_native_report = {
                "path": str(source),
                "sha256": sha256(source),
                "records": len(self.reused_records),
            }
        return {
            "schema": "dreamer_rssm_v2_frozen_ablation_v1",
            "run_id": self.run_id,
            "dry_run": bool(self.args.dry_run),
            "modes": [
                {"id": mode, **MODES[mode]}
                for mode in self.modes
            ],
            "routes": routes,
            "checkpoints": checkpoints,
            "reused_native_report": reused_native_report,
            "expected_runs": len(self.modes) * len(self.suite),
        }

    def payload(self, mode: str, route: Dict[str, Any], seed: int, collision_path: Path) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "route_id": route["id"],
            "route_file": route["file"],
            "seed": seed,
            "quality": self.args.carla_quality,
            "camera": self.args.camera,
            "visual_weather": self.args.visual_weather,
            "video_quality": "preview",
            "record_video": "0",
            "playback_after": "0",
            "run_mode": "pov",
            "dreamer_mode": MODES[mode]["dreamer_mode"],
            "dreamer_rl_training": False,
            "dreamer_rl_deterministic_eval": "1",
            "dreamer_online_learning": "0",
            "dreamer_checkpoint_role": "production",
            "cot_mode": "off",
            "collision_events_path": str(collision_path),
            "view_fps": "20",
            "traffic_light_overlay": "0",
        }
        checkpoint = checkpoint_for_mode(mode, self.rssm_checkpoint)
        if checkpoint is not None:
            payload["dreamer_checkpoint_path"] = str(checkpoint)
            payload["dreamer_checkpoint_source"] = (
                f"frozen_ablation_{mode}_{sha256(checkpoint)[:12]}"
            )
        return payload

    def save_status(self, phase: str, **extra: Any) -> None:
        write_json(self.status_path, {
            "phase": phase,
            "run_id": self.run_id,
            "completed_runs": len(self.records),
            "total_runs": len(self.modes) * len(self.suite),
            "elapsed_seconds": round(time.time() - self.started_at, 3),
            **extra,
        })

    def run_episode(self, mode: str, route_id: str, seed: int) -> Dict[str, Any]:
        active = active_simulation_processes()
        if active:
            raise RuntimeError(
                "Refusing to interrupt an active CARLA/Bench2Drive simulation:\n"
                + "\n".join(active)
            )
        route = self.route_catalog[route_id]
        label = f"{mode}_route_{route_id}_seed_{seed}"
        episode_dir = self.run_dir / mode / f"route_{route_id}_seed_{seed}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        collision_path = episode_dir / "collision_events.jsonl"
        self.save_status(
            "running",
            current={
                "mode": mode,
                "route_id": route_id,
                "seed": seed,
                "town": route.get("town"),
                "scenario_type": route.get("scenario_type"),
            },
        )
        started = time.time()
        exit_code: Optional[int] = None
        timed_out = False
        try:
            self.owns_simulation = True
            dashboard.start_run(self.payload(mode, route, seed, collision_path))
            process = dashboard.STATE.get("process")
            if process is None:
                raise RuntimeError("dashboard did not start the simulation process")
            deadline = started + self.args.max_wall_seconds
            while process.poll() is None:
                if self.args.max_wall_seconds > 0 and time.time() >= deadline:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    break
                time.sleep(2.0)
            try:
                exit_code = process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                exit_code = process.wait(timeout=15)
        finally:
            dashboard.stop_current(kill_carla=True)
            self.owns_simulation = False

        result_path = dashboard.latest_result_after(started)
        metrics = (
            dashboard.parse_bench2drive_result(result_path)
            if result_path is not None
            else None
        )
        impact = first_collision(collision_path)
        if metrics is None:
            metrics = fallback_metrics(
                exit_code=exit_code,
                collision=impact,
                timed_out=timed_out,
            )
        record = {
            "label": label,
            "mode": mode,
            "description": MODES[mode]["description"],
            "route_id": route_id,
            "seed": seed,
            "town": route.get("town"),
            "scenario_type": route.get("scenario_type"),
            "checkpoint": (
                str(checkpoint_for_mode(mode, self.rssm_checkpoint))
                if checkpoint_for_mode(mode, self.rssm_checkpoint) else None
            ),
            "checkpoint_sha256": (
                sha256(checkpoint_for_mode(mode, self.rssm_checkpoint))
                if checkpoint_for_mode(mode, self.rssm_checkpoint) else None
            ),
            "metrics": metrics,
            "first_collision": impact,
            "result": str(result_path) if result_path is not None else None,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "wall_seconds": round(time.time() - started, 3),
        }
        write_json(episode_dir / "episode.json", record)
        self.records.append(record)
        self.save_status("running")
        time.sleep(max(0.0, self.args.cooldown_seconds))
        return record

    def report(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        by_mode = {
            mode: aggregate(row for row in self.records if row["mode"] == mode)
            for mode in self.modes
        }
        gates = {}
        native = by_mode.get("native", {})
        for mode in self.modes:
            if mode != "native":
                gates[mode] = comparison_gate(native, by_mode[mode], final=True)
        report = {
            **validation,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "records": self.records,
            "aggregate": by_mode,
            "promotion_gates": gates,
            "rssm_v2_approved": bool(gates.get("rssm_v2", {}).get("approved", False)),
        }
        write_json(self.report_path, report)
        lines = [
            "# Dreamer RSSM V2 frozen ablation",
            "",
            f"- Run: `{self.run_id}`",
            f"- Expected/completed: `{validation['expected_runs']}/{len(self.records)}`",
            f"- RSSM V2 approved: `{report['rssm_v2_approved']}`",
            "",
            "## Aggregate",
            "",
        ]
        for mode in self.modes:
            values = by_mode.get(mode, {})
            lines.append(
                f"- `{mode}`: runs={int(values.get('runs', 0))}, "
                f"route={values.get('mean_route', 0.0):.2f}, "
                f"driving={values.get('mean_driving', 0.0):.2f}, "
                f"collisions={values.get('collisions', 0.0):.0f}, "
                f"offroad={values.get('offroad', 0.0):.0f}, "
                f"blocked={values.get('blocked', 0.0):.0f}"
            )
        self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report

    def run(self) -> int:
        validation = self.validate()
        write_json(self.run_dir / "matrix.json", validation)
        if self.args.dry_run:
            self.save_status("dry_run_validated", matrix=str(self.run_dir / "matrix.json"))
            print(json.dumps(validation, indent=2, sort_keys=True))
            return 0
        active = active_simulation_processes()
        if active:
            raise RuntimeError(
                "Refusing to interrupt an active CARLA/Bench2Drive simulation:\n"
                + "\n".join(active)
            )
        self.acquire_lock()
        try:
            self.records.extend(copy.deepcopy(self.reused_records))
            for mode in self.modes:
                for route_id, seed in self.suite:
                    if any(
                        row.get("mode") == mode
                        and str(row.get("route_id")) == route_id
                        and int(row.get("seed", -1)) == seed
                        for row in self.records
                    ):
                        continue
                    self.run_episode(mode, route_id, seed)
            report = self.report(validation)
            self.save_status(
                "complete",
                report=str(self.report_path),
                rssm_v2_approved=report["rssm_v2_approved"],
            )
            return 0 if report["rssm_v2_approved"] else 2
        finally:
            if self.owns_simulation:
                dashboard.stop_current(kill_carla=True)
            self.release_lock()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--modes", default="native,ppo,rssm_v2")
    parser.add_argument(
        "--routes",
        default=",".join(route_id for route_id, _ in DEFAULT_SUITE),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    parser.add_argument("--cooldown-seconds", type=float, default=8.0)
    parser.add_argument("--carla-quality", choices=("Low", "Epic"), default="Low")
    parser.add_argument("--camera", default="chase")
    parser.add_argument("--visual-weather", default="day")
    parser.add_argument(
        "--rssm-checkpoint",
        type=Path,
        default=RSSM_CHECKPOINT,
        help="Isolated RSSM candidate to evaluate; the active checkpoint is untouched.",
    )
    parser.add_argument(
        "--reuse-native-report",
        type=Path,
        default=None,
        help=(
            "Reuse native rows from a prior frozen report after exact route/seed "
            "validation; only missing candidate runs are executed."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be >= 1")
    return FrozenAblation(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
