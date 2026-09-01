#!/usr/bin/env python3
"""Train and gate a no-guard Dreamer complement through a fixed curriculum.

The production checkpoint is never modified during training.  A candidate is
bootstrapped from clean SimLingo + guarded Dreamer-v1 demonstrations, then each
curriculum stage collects a frozen on-policy batch before one conservative PPO
update.  Stage rollbacks and final promotion use deterministic Bench2Drive
evaluations with fixed routes and seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import scripts.simlingo_dashboard as dashboard


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "external" / "simlingo" / "checkpoints" / "dreamer_ppo_rl_noguard"
PRODUCTION = CHECKPOINT_DIR / "production_model.pt"
CANDIDATE = CHECKPOINT_DIR / "candidate_model.pt"
MANAGER_MODULE = "scripts.manage_dreamer_rl_checkpoints"
UPDATER_MODULE = "scripts.dreamer_online_rl_batch_update"
PRETRAINER_MODULE = "scripts.pretrain_dreamer_rl_from_v1"
LOG_ROOT = ROOT / "logs" / "dreamer_curriculum"


CURRICULUM: List[Dict[str, Any]] = [
    {
        "name": "01_simple_accident",
        "description": "Single-direction blocked-lane accidents in Town10HD and Town12.",
        "train": [("148", 820101), ("148", 820102), ("32", 820103), ("33", 820104)],
        "evaluation": [("148", 829101), ("32", 829102)],
    },
    {
        "name": "02_oncoming_traffic",
        "description": "Two-way accident overtakes with closing traffic.",
        "train": [("06", 820201), ("06", 820202), ("70", 820203), ("70", 820204)],
        "evaluation": [("06", 829201), ("70", 829202)],
    },
    {
        "name": "03_dense_traffic",
        "description": "Dense urban/flow interactions before adding vulnerable road users.",
        "train": [("54", 820301), ("55", 820302), ("56", 820303), ("93", 820304)],
        "evaluation": [("54", 829301), ("93", 829302)],
    },
    {
        "name": "04_vru",
        "description": "Pedestrian and bicycle interactions after vehicle-only stages.",
        "train": [("113", 820401), ("91", 820402), ("109", 820403), ("18", 820404)],
        "evaluation": [("113", 829401), ("91", 829402)],
    },
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def first_collision(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    events: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "collision":
                events.append(event)
    if not events:
        return None
    return min(events, key=lambda event: float(event.get("wall_time", float("inf"))))


def fallback_metrics(
    *,
    exit_code: Optional[int],
    collision: Optional[Dict[str, Any]],
    timed_out: bool,
) -> Dict[str, float]:
    collision_kind = str((collision or {}).get("collision_kind", ""))
    crashed = exit_code not in (None, 0, 130, 143, -2, -15)
    impact = collision is not None
    return {
        "incomplete": 1.0,
        "route_score": 0.0,
        "driving_score": 0.0,
        "penalty": 0.0,
        "collisions": 1.0 if impact or crashed else 0.0,
        "pedestrian_collisions": 1.0 if collision_kind == "pedestrian" else 0.0,
        "vehicle_collisions": 1.0 if collision_kind == "vehicle" or (crashed and not impact) else 0.0,
        "layout_collisions": 1.0 if collision_kind == "static" else 0.0,
        "red_lights": 0.0,
        "stop_infractions": 0.0,
        "offroad": 1.0 if crashed and not impact else 0.0,
        "blocked": 0.0 if impact else 1.0,
        "scenario_timeouts": 0.0,
        "route_timeouts": 1.0 if timed_out else 0.0,
        "min_speed_infractions": 0.0,
        "success": 0.0,
    }


def aggregate(runs: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    records = list(runs)
    metric_rows = [record.get("metrics") or {} for record in records]
    if not metric_rows:
        return {"runs": 0}
    summed = {}
    for key in (
        "collisions", "pedestrian_collisions", "vehicle_collisions",
        "layout_collisions", "red_lights", "stop_infractions", "offroad",
        "blocked", "scenario_timeouts", "route_timeouts", "success",
    ):
        summed[key] = float(sum(float(row.get(key, 0.0) or 0.0) for row in metric_rows))
    summed.update({
        "runs": len(metric_rows),
        "mean_route": float(mean(float(row.get("route_score", 0.0) or 0.0) for row in metric_rows)),
        "mean_driving": float(mean(float(row.get("driving_score", 0.0) or 0.0) for row in metric_rows)),
    })
    summed["unsafe_events"] = float(sum(
        summed[key] for key in ("collisions", "red_lights", "stop_infractions", "offroad")
    ))
    return summed


def comparison_gate(
    native: Dict[str, float],
    candidate: Dict[str, float],
    *,
    final: bool,
) -> Dict[str, Any]:
    """Require SimLingo + Dreamer to improve strictly over native SimLingo."""
    reasons: List[str] = []
    expected_runs = int(native.get("runs", 0))
    if int(candidate.get("runs", 0)) != expected_runs or expected_runs < (6 if final else 2):
        reasons.append("incomplete frozen evaluation suite")
    if candidate.get("unsafe_events", 0.0) > native.get("unsafe_events", 0.0):
        reasons.append("more collisions/off-road/traffic-rule violations than native SimLingo")
    if candidate.get("collisions", 0.0) > native.get("collisions", 0.0):
        reasons.append("more collisions than native SimLingo")
    if candidate.get("blocked", 0.0) > native.get("blocked", 0.0):
        reasons.append("more blocked-agent outcomes than native SimLingo")
    if candidate.get("mean_route", 0.0) < native.get("mean_route", 0.0) - 3.0:
        reasons.append("mean route completion regressed by more than 3 points")
    if candidate.get("mean_driving", 0.0) < native.get("mean_driving", 0.0) - 5.0:
        reasons.append("mean driving score regressed by more than 5 points")
    if final and candidate.get("success", 0.0) < native.get("success", 0.0):
        reasons.append("fewer successful routes than native SimLingo")

    native_improvements = {
        "success": candidate.get("success", 0.0) > native.get("success", 0.0),
        "blocked": candidate.get("blocked", 0.0) < native.get("blocked", 0.0),
        "route": candidate.get("mean_route", 0.0) >= native.get("mean_route", 0.0) + 1.0,
        "driving": candidate.get("mean_driving", 0.0) >= native.get("mean_driving", 0.0) + 1.0,
    }
    if not any(native_improvements.values()):
        reasons.append("no strict measurable improvement over native SimLingo")
    return {
        "approved": not reasons,
        "reasons": reasons,
        "native_improvements": native_improvements,
        "comparison": "simlingo_plus_dreamer_vs_native_simlingo",
    }


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = LOG_ROOT / self.run_id
        self.status_path = self.run_dir / "status.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.md"
        self.lock_path = LOG_ROOT / "campaign.lock"
        self.routes = {route["id"]: route for route in dashboard.route_catalog()}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.completed_runs = 0
        self.run_durations: List[float] = []
        total_training = sum(len(stage["train"]) for stage in CURRICULUM)
        total_stage_evaluation = sum(len(stage["evaluation"]) for stage in CURRICULUM)
        total_frozen = len(self.final_suite())
        # Native SimLingo baseline + final SimLingo/Dreamer candidate.
        self.total_runs = total_training + total_stage_evaluation + 2 * total_frozen
        self.status: Dict[str, Any] = {
            "schema": "dreamer_curriculum_campaign_v1",
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "started_at": now(),
            "phase": "initializing",
            "completed_simulations": 0,
            "total_simulations": self.total_runs,
            "protected_checkpoint": str(PRODUCTION),
            "candidate_checkpoint": str(CANDIDATE),
            "no_guard": True,
            "complement_to_simlingo": True,
            "batch_before_update": True,
            "stages": [],
        }

    def final_suite(self) -> List[tuple[str, int]]:
        return [item for stage in CURRICULUM for item in stage["evaluation"]]

    def acquire_lock(self) -> None:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            try:
                old_pid = int(self.lock_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                old_pid = -1
            if old_pid > 0:
                try:
                    os.kill(old_pid, 0)
                except OSError:
                    pass
                else:
                    raise RuntimeError(f"another curriculum campaign is running with pid {old_pid}")
        self.lock_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    def release_lock(self) -> None:
        try:
            if self.lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                self.lock_path.unlink()
        except (FileNotFoundError, OSError):
            pass

    def event(self, event: str, **payload: Any) -> None:
        append_jsonl(self.events_path, {"event": event, "time": now(), **payload})

    def save(self) -> None:
        self.status["updated_at"] = now()
        self.status["completed_simulations"] = self.completed_runs
        if self.run_durations and self.completed_runs < self.total_runs:
            remaining = self.total_runs - self.completed_runs
            self.status["eta_seconds"] = round(mean(self.run_durations) * remaining, 1)
        else:
            self.status["eta_seconds"] = 0.0
        if CANDIDATE.exists():
            self.status["candidate_sha256"] = sha256(CANDIDATE)
        if PRODUCTION.exists():
            self.status["production_sha256"] = sha256(PRODUCTION)
        write_json(self.status_path, self.status)
        lines = [
            "# Dreamer PPO Curriculum Campaign",
            "",
            f"- Phase: `{self.status.get('phase')}`",
            f"- Simulations: `{self.completed_runs}/{self.total_runs}`",
            f"- Production: `{self.status.get('production_sha256', '-')}`",
            f"- Candidate: `{self.status.get('candidate_sha256', '-')}`",
            f"- No runtime guard: `{self.status.get('no_guard')}`",
            f"- SimLingo complement: `{self.status.get('complement_to_simlingo')}`",
            "",
            "## Stages",
        ]
        for stage in self.status.get("stages", []):
            gate = stage.get("gate") or {}
            lines.append(
                f"- `{stage.get('name')}`: `{stage.get('status')}`; "
                f"gate=`{gate.get('approved', '-')}`; episodes=`{len(stage.get('training_runs') or [])}`"
            )
        self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def route(self, route_id: str) -> Dict[str, Any]:
        route = self.routes.get(route_id)
        if not route:
            raise RuntimeError(f"Bench2Drive route {route_id} is unavailable")
        if not route.get("installed"):
            raise RuntimeError(f"CARLA map {route.get('town')} for route {route_id} is not installed")
        return route

    def command(self, command: List[str], log_path: Path) -> subprocess.CompletedProcess[str]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.event("command_start", command=command, log=str(log_path))
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_path.write_text(result.stdout, encoding="utf-8")
        self.event("command_done", returncode=result.returncode, log=str(log_path))
        return result

    def _run_episode_once(
        self,
        *,
        checkpoint: Optional[Path],
        route_id: str,
        seed: int,
        label: str,
        training: bool,
        native: bool = False,
    ) -> Dict[str, Any]:
        if training and (native or checkpoint is None):
            raise ValueError("training episodes require the candidate checkpoint")
        if not native and checkpoint is None:
            raise ValueError("Dreamer evaluation requires a checkpoint")
        route = self.route(route_id)
        episode_dir = self.run_dir / ("training" if training else "evaluation") / label
        episode_dir.mkdir(parents=True, exist_ok=True)
        trace_path = episode_dir / "trace.jsonl"
        collision_path = episode_dir / "collision_events.jsonl"
        result_path = ROOT / "logs" / "simlingo_eval" / f"results_bench2drive_{route_id}_seed_{seed}.json"
        payload: Dict[str, Any] = {
            "route_id": route_id,
            "route_file": route["file"],
            "seed": seed,
            "quality": self.args.carla_quality,
            "camera": self.args.camera,
            "visual_weather": self.args.visual_weather,
            "video_quality": "preview",
            "record_video": "0",
            "playback_after": "0",
            "run_mode": "action_dreaming" if training else "pov",
            "dreamer_mode": "off" if native else "dreamer_ppo_rl_noguard",
            "dreamer_rl_training": "1" if training else "0",
            "dreamer_rl_deterministic_eval": "0" if training else "1",
            "dreamer_online_learning": "0",
            "cot_mode": "off",
            "collision_events_path": str(collision_path),
            "view_fps": "20",
            "traffic_light_overlay": "0",
        }
        if checkpoint is not None:
            payload.update({
                "dreamer_checkpoint_path": str(checkpoint),
                "dreamer_checkpoint_source": (
                    "curriculum_candidate" if checkpoint == CANDIDATE else "frozen_production"
                ),
                "dreamer_checkpoint_role": "candidate" if checkpoint == CANDIDATE else "production",
            })
        if training:
            payload.update({
                "action_dreaming_sample_interval": str(self.args.sample_interval),
                "action_dreaming_out_dir": str(episode_dir),
                "action_dreaming_run_id": label,
                "action_dreaming_trace_path": str(trace_path),
            })
        self.status["phase"] = "collecting" if training else "evaluating"
        self.status["current"] = {
            "label": label,
            "route_id": route_id,
            "town": route.get("town"),
            "scenario": route.get("scenario_type"),
            "seed": seed,
            "training": training,
            "policy_mode": "simlingo_native" if native else "dreamer_ppo_rl_noguard",
            "checkpoint_sha256": sha256(checkpoint) if checkpoint is not None else None,
        }
        self.save()
        self.event("episode_start", **self.status["current"])

        started = time.time()
        exit_code: Optional[int] = None
        timed_out = False
        truncated_reason = ""
        try:
            dashboard.start_run(payload)
            proc = dashboard.STATE.get("process")
            if proc is None:
                raise RuntimeError("dashboard did not start the simulation process")
            deadline = started + self.args.max_wall_seconds
            while proc.poll() is None:
                if training:
                    impact = first_collision(collision_path)
                    if impact is not None:
                        truncated_reason = "first_real_collision"
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        break
                    status_file = ROOT / "logs" / "simlingo_eval" / "dreamer_guard_status.json"
                    try:
                        live = read_json(status_file)
                    except (FileNotFoundError, json.JSONDecodeError, OSError):
                        live = {}
                    if int(float(live.get("blocked_ticks", 0) or 0)) >= self.args.blocked_ticks:
                        truncated_reason = "irrecoverably_blocked_training_episode"
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        break
                if self.args.max_wall_seconds > 0 and time.time() >= deadline:
                    timed_out = True
                    truncated_reason = "wall_time_limit"
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    break
                time.sleep(2.0)
            try:
                exit_code = proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                exit_code = proc.wait(timeout=15)
        finally:
            dashboard.stop_current(kill_carla=True)

        elapsed = time.time() - started
        self.run_durations.append(elapsed)
        self.completed_runs += 1
        impact = first_collision(collision_path)
        metrics = None
        if result_path.exists() and result_path.stat().st_mtime >= started - 2.0:
            metrics = dashboard.parse_bench2drive_result(result_path)
        if metrics is None:
            latest = dashboard.latest_result_after(started)
            if latest is not None:
                metrics = dashboard.parse_bench2drive_result(latest)
                if metrics is not None:
                    result_path = latest
        if metrics is None:
            metrics = fallback_metrics(
                exit_code=exit_code,
                collision=impact,
                timed_out=timed_out,
            )
        metrics = {**metrics, "incomplete": float(metrics.get("incomplete", 0.0) or 0.0)}
        archived_result: Optional[Path] = None
        if result_path.exists():
            archived_result = episode_dir / "bench2drive_result.json"
            shutil.copy2(result_path, archived_result)
        record = {
            "label": label,
            "route_id": route_id,
            "route_file": route["file"],
            "town": route.get("town"),
            "scenario_type": route.get("scenario_type"),
            "seed": seed,
            "training": training,
            "policy_mode": "simlingo_native" if native else "dreamer_ppo_rl_noguard",
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "checkpoint_sha256": sha256(checkpoint) if checkpoint is not None else None,
            "trace": str(trace_path) if training else None,
            "trace_rows": count_jsonl(trace_path) if training else 0,
            "collision_events": str(collision_path),
            "first_collision": impact,
            "result": str(archived_result) if archived_result is not None else None,
            "metrics": metrics,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "truncated_reason": truncated_reason,
            "wall_seconds": round(elapsed, 3),
        }
        write_json(episode_dir / "episode.json", record)
        self.event("episode_done", **record)
        self.status.pop("current", None)
        self.save()
        time.sleep(max(0.0, self.args.cooldown_seconds))
        return record

    def run_episode(
        self,
        *,
        checkpoint: Optional[Path],
        route_id: str,
        seed: int,
        label: str,
        training: bool,
        native: bool = False,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(self.args.route_retries + 1):
            attempt_label = label if attempt == 0 else f"{label}_retry_{attempt}"
            try:
                return self._run_episode_once(
                    checkpoint=checkpoint,
                    route_id=route_id,
                    seed=seed,
                    label=attempt_label,
                    training=training,
                    native=native,
                )
            except Exception as exc:
                last_error = exc
                self.event(
                    "episode_retry",
                    label=label,
                    route_id=route_id,
                    seed=seed,
                    attempt=attempt,
                    error=str(exc),
                )
                dashboard.stop_current(kill_carla=True)
                if attempt < self.args.route_retries:
                    time.sleep(max(5.0, self.args.cooldown_seconds))
        assert last_error is not None
        raise last_error

    def evaluate(
        self,
        checkpoint: Optional[Path],
        suite: List[tuple[str, int]],
        label: str,
        *,
        native: bool = False,
    ) -> Dict[str, Any]:
        runs = []
        for index, (route_id, seed) in enumerate(suite, start=1):
            runs.append(self.run_episode(
                checkpoint=checkpoint,
                route_id=route_id,
                seed=seed,
                label=f"{label}_{index:02d}_route_{route_id}_seed_{seed}",
                training=False,
                native=native,
            ))
        result = {
            "label": label,
            "policy_mode": "simlingo_native" if native else "dreamer_ppo_rl_noguard",
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "sha256": sha256(checkpoint) if checkpoint is not None else None,
            "runs": runs,
        }
        result["aggregate"] = aggregate(runs)
        write_json(self.run_dir / "evaluations" / f"{label}.json", result)
        return result

    def pretrain(self) -> Dict[str, Any]:
        self.status["phase"] = "offline_teacher_pretraining"
        self.save()
        backup = self.run_dir / "checkpoints" / "candidate_before_pretraining.pt"
        copy_checkpoint(CANDIDATE, backup)
        summary = self.run_dir / "pretraining" / "summary.json"
        result = self.command([
            sys.executable,
            "-m", PRETRAINER_MODULE,
            "--checkpoint", str(PRODUCTION),
            "--output-checkpoint", str(CANDIDATE),
            "--summary", str(summary),
            "--epochs", str(self.args.pretrain_epochs),
            "--device", self.args.device,
        ], self.run_dir / "pretraining" / "pretraining.log")
        if result.returncode != 0 or not summary.exists():
            copy_checkpoint(backup, CANDIDATE)
            raise RuntimeError("clean Dreamer-v1 teacher pretraining failed; candidate restored")
        payload = read_json(summary)
        payload["candidate_sha256"] = sha256(CANDIDATE)
        self.status["pretraining"] = payload
        self.event("pretraining_done", **payload)
        self.save()
        return payload

    def train_stage(
        self,
        stage: Dict[str, Any],
        native_evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        stage_dir = self.run_dir / "stages" / stage["name"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        before = stage_dir / "candidate_before_stage.pt"
        copy_checkpoint(CANDIDATE, before)
        frozen_hash = sha256(CANDIDATE)
        stage_record: Dict[str, Any] = {
            "name": stage["name"],
            "description": stage["description"],
            "status": "collecting",
            "frozen_checkpoint_sha256": frozen_hash,
            "training_runs": [],
        }
        self.status["stages"].append(stage_record)
        self.save()

        for index, (route_id, seed) in enumerate(stage["train"], start=1):
            if sha256(CANDIDATE) != frozen_hash:
                raise RuntimeError("candidate changed inside a frozen collection batch")
            record = self.run_episode(
                checkpoint=CANDIDATE,
                route_id=route_id,
                seed=seed,
                label=f"{stage['name']}_{index:02d}_route_{route_id}_seed_{seed}",
                training=True,
            )
            stage_record["training_runs"].append(record)
            self.save()

        manifest = {
            "schema": "dreamer_frozen_on_policy_batch_v1",
            "stage": stage["name"],
            "created_at": now(),
            "frozen_checkpoint": str(CANDIDATE),
            "frozen_checkpoint_sha256": frozen_hash,
            "episodes": [
                {
                    "trace": run["trace"],
                    "collision_events": run["collision_events"],
                    "checkpoint_sha256": run["checkpoint_sha256"],
                    "route_id": run["route_id"],
                    "seed": run["seed"],
                    "stage": stage["name"],
                    "metrics": run["metrics"],
                }
                for run in stage_record["training_runs"]
            ],
        }
        manifest_path = stage_dir / "batch_manifest.json"
        write_json(manifest_path, manifest)
        update_summary = stage_dir / "batch_update.json"
        stage_record["status"] = "updating"
        self.status["phase"] = "batch_ppo_update"
        self.save()
        result = self.command([
            sys.executable,
            "-m", UPDATER_MODULE,
            "--manifest", str(manifest_path),
            "--checkpoint", str(CANDIDATE),
            "--output-checkpoint", str(CANDIDATE),
            "--summary", str(update_summary),
            "--device", self.args.device,
            "--epochs", str(self.args.update_epochs),
            "--batch-size", str(self.args.batch_size),
            "--imagination-horizon", str(self.args.imagination_horizon),
            "--imagination-starts", str(self.args.imagination_starts),
        ], stage_dir / "batch_update.log")
        if result.returncode != 0 or not update_summary.exists():
            copy_checkpoint(before, CANDIDATE)
            stage_record.update({
                "status": "update_failed_rolled_back",
                "update_error": result.stdout[-3000:],
            })
            self.save()
            return stage_record
        stage_record["update"] = read_json(update_summary)
        stage_record["candidate_after_update_sha256"] = sha256(CANDIDATE)

        candidate_eval = self.evaluate(CANDIDATE, stage["evaluation"], f"candidate_{stage['name']}")
        native_subset = [
            run for run in native_evaluation["runs"]
            if (run["route_id"], int(run["seed"])) in set(stage["evaluation"])
        ]
        native_aggregate = aggregate(native_subset)
        gate = comparison_gate(
            native_aggregate,
            candidate_eval["aggregate"],
            final=False,
        )
        stage_record["evaluation"] = candidate_eval
        stage_record["native_aggregate"] = native_aggregate
        stage_record["gate"] = gate
        if gate["approved"]:
            stage_record["status"] = "accepted"
            copy_checkpoint(CANDIDATE, stage_dir / "candidate_accepted.pt")
        else:
            stage_record["status"] = "rejected_rolled_back"
            copy_checkpoint(CANDIDATE, stage_dir / "candidate_rejected.pt")
            copy_checkpoint(before, CANDIDATE)
        write_json(stage_dir / "stage_summary.json", stage_record)
        self.save()
        return stage_record

    def run(self) -> None:
        self.acquire_lock()
        try:
            if not PRODUCTION.exists() or not CANDIDATE.exists():
                raise RuntimeError("checkpoint roles are not initialized")
            for stage in CURRICULUM:
                for route_id, _ in stage["train"] + stage["evaluation"]:
                    self.route(route_id)
            (LOG_ROOT / "latest_campaign.txt").write_text(str(self.run_dir) + "\n", encoding="utf-8")
            write_json(self.run_dir / "curriculum.json", CURRICULUM)
            copy_checkpoint(PRODUCTION, self.run_dir / "checkpoints" / "production_at_start.pt")
            copy_checkpoint(CANDIDATE, self.run_dir / "checkpoints" / "candidate_at_start.pt")
            self.status["production_sha256_at_start"] = sha256(PRODUCTION)
            self.status["candidate_sha256_at_start"] = sha256(CANDIDATE)
            self.save()

            if not self.args.skip_pretrain:
                self.pretrain()

            final_suite = self.final_suite()
            self.status["phase"] = "frozen_native_baseline"
            self.save()
            native_baseline = self.evaluate(
                None,
                final_suite,
                "native_baseline",
                native=True,
            )
            self.status["native_baseline"] = native_baseline
            self.save()

            for stage in CURRICULUM:
                self.train_stage(stage, native_baseline)

            self.status["phase"] = "final_frozen_evaluation"
            self.save()
            candidate_evaluation = self.evaluate(CANDIDATE, final_suite, "candidate_final")
            gate = comparison_gate(
                native_baseline["aggregate"],
                candidate_evaluation["aggregate"],
                final=True,
            )
            report = {
                "schema": "dreamer_frozen_promotion_evaluation_v1",
                "created_at": now(),
                "candidate": candidate_evaluation,
                "native": native_baseline,
                "initial_checkpoint": {
                    "path": str(PRODUCTION),
                    "sha256": sha256(PRODUCTION),
                    "role": "protected initialization and rollback only",
                },
                "candidate_sha256": sha256(CANDIDATE),
                "production_sha256": sha256(PRODUCTION),
                "promotion_approved": bool(gate["approved"]),
                "gate": gate,
                "routes_and_seeds": final_suite,
                "deterministic_policy": True,
                "no_guard": True,
                "complement_to_simlingo": True,
            }
            evaluation_path = self.run_dir / "frozen_promotion_evaluation.json"
            write_json(evaluation_path, report)
            self.status["final_evaluation"] = report
            if gate["approved"]:
                promotion = self.command([
                    sys.executable,
                    "-m", MANAGER_MODULE,
                    "promote",
                    "--kind", "ppo",
                    "--evaluation", str(evaluation_path),
                ], self.run_dir / "promotion.log")
                if promotion.returncode != 0:
                    raise RuntimeError("promotion gate passed but atomic promotion failed")
                self.status["promotion"] = json.loads(promotion.stdout)
                self.status["phase"] = "completed_promoted"
            else:
                self.status["phase"] = "completed_candidate_not_promoted"
                self.status["promotion"] = {"status": "rejected", "reasons": gate["reasons"]}
            self.status["finished_at"] = now()
            self.save()
            self.event("campaign_done", phase=self.status["phase"], promotion=self.status["promotion"])
        finally:
            try:
                dashboard.stop_current(kill_carla=True)
            finally:
                self.release_lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("DREAMER_CURRICULUM_RUN_ID"))
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=int(os.environ.get("DREAMER_CURRICULUM_PRETRAIN_EPOCHS", "360")))
    parser.add_argument("--update-epochs", type=int, default=int(os.environ.get("DREAMER_CURRICULUM_UPDATE_EPOCHS", "4")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DREAMER_CURRICULUM_BATCH_SIZE", "192")))
    parser.add_argument("--imagination-horizon", type=int, default=int(os.environ.get("DREAMER_CURRICULUM_IMAGINATION_HORIZON", "3")))
    parser.add_argument("--imagination-starts", type=int, default=int(os.environ.get("DREAMER_CURRICULUM_IMAGINATION_STARTS", "128")))
    parser.add_argument("--sample-interval", type=float, default=float(os.environ.get("DREAMER_CURRICULUM_SAMPLE_INTERVAL", "0.10")))
    parser.add_argument("--max-wall-seconds", type=float, default=float(os.environ.get("DREAMER_CURRICULUM_MAX_WALL_SECONDS", "720")))
    parser.add_argument("--blocked-ticks", type=int, default=int(os.environ.get("DREAMER_CURRICULUM_BLOCKED_TICKS", "800")))
    parser.add_argument("--route-retries", type=int, default=int(os.environ.get("DREAMER_CURRICULUM_ROUTE_RETRIES", "1")))
    parser.add_argument("--cooldown-seconds", type=float, default=float(os.environ.get("DREAMER_CURRICULUM_COOLDOWN_SECONDS", "5")))
    parser.add_argument("--carla-quality", default=os.environ.get("DREAMER_CURRICULUM_CARLA_QUALITY", "Low"))
    parser.add_argument("--camera", default=os.environ.get("SIMLINGO_VIEW_MODE", "chase"))
    parser.add_argument("--visual-weather", default=os.environ.get("SIMLINGO_VISUAL_WEATHER", "day"))
    parser.add_argument("--device", default=os.environ.get("DREAMER_CURRICULUM_DEVICE", "auto"))
    return parser.parse_args()


def main() -> int:
    campaign = Campaign(parse_args())
    try:
        campaign.run()
        return 0
    except KeyboardInterrupt:
        campaign.status["phase"] = "interrupted"
        campaign.status["error"] = "keyboard interrupt"
        campaign.save()
        return 130
    except Exception as exc:
        campaign.status["phase"] = "failed"
        campaign.status["error"] = str(exc)
        campaign.status["failed_at"] = now()
        campaign.save()
        campaign.event("campaign_failed", error=str(exc))
        print(f"[dreamer-curriculum] failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
