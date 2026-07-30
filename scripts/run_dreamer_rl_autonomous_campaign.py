#!/usr/bin/env python3
"""Autonomous overnight campaign for SimLingo + Dreamer RL preparation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.select_simlingo_rl_campaign_routes import build_campaign_plan
import scripts.simlingo_dashboard as dashboard


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl_count(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = row.get("status") or {}
            kind = str(status.get("chosen_kind") or "unknown")
            counts[kind] += 1
            if status.get("collision_shield_active"):
                counts["collision_shield_active"] += 1
            if status.get("gap_recovery_sides"):
                counts["gap_recovery_sides"] += 1
    return counts


def latest_result_after(start_time: float) -> Optional[Path]:
    result_dir = ROOT / "logs" / "simlingo_eval"
    candidates = [
        path for path in result_dir.glob("results_*.json")
        if path.stat().st_mtime >= start_time - 2.0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def classify_run(metrics: Optional[Dict[str, Any]], exit_code: Optional[int], trace_path: Path) -> Dict[str, Any]:
    good: List[str] = []
    bad: List[str] = []
    if trace_path.exists() and trace_path.stat().st_size > 0:
        good.append("Dreamer trace collected")
    else:
        bad.append("No Dreamer trace collected")

    if exit_code == 0:
        good.append("Process exited cleanly")
    else:
        bad.append(f"Process exit code {exit_code}")

    if not metrics:
        bad.append("No Bench2Drive metrics found")
        return {"score": "bad", "good": good, "bad": bad}

    route_score = float(metrics.get("route_score") or 0.0)
    driving_score = float(metrics.get("driving_score") or 0.0)
    collisions = float(metrics.get("collisions") or 0.0)
    offroad = float(metrics.get("offroad") or 0.0)
    red_lights = float(metrics.get("red_lights") or 0.0)
    blocked = float(metrics.get("blocked") or 0.0)
    status = str(metrics.get("status") or "")

    if route_score >= 95.0:
        good.append(f"High route completion ({route_score:.1f}%)")
    else:
        bad.append(f"Low route completion ({route_score:.1f}%)")
    if driving_score >= 80.0:
        good.append(f"Good driving score ({driving_score:.1f})")
    else:
        bad.append(f"Weak driving score ({driving_score:.1f})")
    if collisions == 0:
        good.append("No collision")
    else:
        bad.append(f"Collisions: {collisions:g}")
    if offroad == 0:
        good.append("No off-road infraction")
    else:
        bad.append(f"Off-road infractions: {offroad:g}")
    if red_lights == 0:
        good.append("No red-light infraction")
    else:
        bad.append(f"Red-light infractions: {red_lights:g}")
    if blocked == 0:
        good.append("No blocked-agent infraction")
    else:
        bad.append(f"Blocked-agent infractions: {blocked:g}")
    if status:
        good.append(f"Bench2Drive status: {status}")

    score = "good" if route_score >= 95.0 and collisions == 0 and offroad == 0 and red_lights == 0 else "bad"
    return {"score": score, "good": good, "bad": bad}


def render_summary(status: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Dreamer RL Autonomous Campaign")
    lines.append("")
    lines.append(f"- run_id: `{status['run_id']}`")
    lines.append(f"- phase: `{status.get('phase', '-')}`")
    lines.append(f"- started_at: `{status.get('started_at', '-')}`")
    lines.append(f"- updated_at: `{status.get('updated_at', '-')}`")
    lines.append(f"- run_dir: `{status['run_dir']}`")
    lines.append(f"- progress: {status.get('completed_runs', 0)}/{status.get('total_runs', 0)} route runs")
    lines.append("")
    lines.append("## Scenario Coverage")
    for bucket, route_ids in status.get("bucket_details", {}).items():
        routes = ", ".join(route_ids) if route_ids else "none"
        lines.append(f"- {bucket}: {routes}")
    lines.append("")
    lines.append("## Route Runs")
    for run in status.get("runs", []):
        metrics = run.get("metrics") or {}
        lines.append(
            f"- [{run.get('quality', '-')}] route `{run.get('route_id')}` "
            f"{run.get('town')} / {run.get('scenario_type')} / seed `{run.get('seed')}`"
        )
        if metrics:
            lines.append(
                f"  route={metrics.get('route_score')} score={metrics.get('driving_score')} "
                f"coll={metrics.get('collisions')} offroad={metrics.get('offroad')} "
                f"red={metrics.get('red_lights')} blocked={metrics.get('blocked')}"
            )
        for point in run.get("good", [])[:8]:
            lines.append(f"  + {point}")
        for point in run.get("bad", [])[:8]:
            lines.append(f"  - {point}")
    if status.get("dataset"):
        lines.append("")
        lines.append("## Dataset")
        lines.append(f"- dataset: `{status['dataset'].get('path')}`")
        lines.append(f"- audit: `{status['dataset'].get('audit')}`")
    if status.get("warmstarts"):
        lines.append("")
        lines.append("## Warm Starts")
        for kind, info in status["warmstarts"].items():
            lines.append(f"- {kind}: `{info.get('checkpoint')}`")
    if status.get("rl_runs"):
        lines.append("")
        lines.append("## RL Runs")
        for kind, info in status["rl_runs"].items():
            lines.append(f"- {kind}: `{info.get('run_dir')}` status={info.get('status')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = ROOT / "logs" / "dreamer_rl_campaign" / self.run_id
        self.trace_dir = self.run_dir / "traces"
        self.summary_path = self.run_dir / "summary.md"
        self.status_path = self.run_dir / "status.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.status: Dict[str, Any] = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "started_at": now(),
            "updated_at": now(),
            "phase": "initializing",
            "runs": [],
            "completed_runs": 0,
            "total_runs": 0,
            "bucket_details": {},
        }

    def save(self) -> None:
        self.status["updated_at"] = now()
        write_json(self.status_path, self.status)
        render_summary(self.status, self.summary_path)

    def event(self, **payload: Any) -> None:
        payload.setdefault("time", now())
        append_jsonl(self.events_path, payload)

    def build_plan(self) -> Dict[str, Any]:
        self.status["phase"] = "selecting_routes"
        self.save()
        plan = build_campaign_plan(
            max_per_bucket=self.args.max_per_bucket,
            seed=self.args.seed,
            include_unstable=self.args.include_unstable,
        )
        write_json(self.run_dir / "campaign_plan.json", plan)
        self.status["total_runs"] = len(plan["runs"])
        self.status["bucket_details"] = plan.get("bucket_details", {})
        self.status["phase"] = "collecting"
        self.save()
        return plan

    def collect_run_attempt(self, run: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        suffix = "" if attempt == 1 else f"_retry{attempt}"
        trace_path = self.trace_dir / (
            f"{run['index']:02d}_route_{run['route_id']}_seed_{run['seed']}{suffix}.jsonl"
        )
        payload = {
            "route_id": run["route_id"],
            "route_file": run["route_file"],
            "seed": run["seed"],
            "quality": os.environ.get("CARLA_QUALITY", self.args.carla_quality),
            "camera": self.args.camera,
            "visual_weather": self.args.visual_weather,
            "video_quality": self.args.video_quality,
            "playback_after": "0",
            "playback_speed": "1",
            "run_mode": "action_dreaming",
            "dreamer_mode": self.args.teacher_dreamer_mode,
            "cot_mode": "off",
            "action_dreaming_sample_interval": str(self.args.sample_interval),
            "action_dreaming_out_dir": str(self.trace_dir),
            "action_dreaming_run_id": f"{self.run_id}_{run['index']:02d}",
            "action_dreaming_trace_path": str(trace_path),
            "traffic_light_overlay": "1",
            "view_fps": "30",
        }

        self.status["current_run"] = {
            "index": run["index"],
            "attempt": attempt,
            "route_id": run["route_id"],
            "town": run["town"],
            "scenario_type": run["scenario_type"],
            "seed": run["seed"],
            "trace": str(trace_path),
        }
        self.save()
        self.event(event="collect_start", **self.status["current_run"])

        start_time = time.time()
        exit_code: Optional[int] = None
        error: Optional[str] = None
        timed_out = False
        try:
            dashboard.start_run(payload)
            proc = dashboard.STATE.get("process")
            if proc is None:
                raise RuntimeError("dashboard.start_run did not create a process")
            if self.args.collect_max_wall_seconds > 0:
                deadline = start_time + self.args.collect_max_wall_seconds
                while True:
                    exit_code = proc.poll()
                    if exit_code is not None:
                        break
                    if time.time() >= deadline:
                        timed_out = True
                        error = f"collection wall-time limit reached ({self.args.collect_max_wall_seconds:.0f}s)"
                        try:
                            dashboard.stop_current(kill_carla=True)
                        except Exception:
                            pass
                        try:
                            exit_code = proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            exit_code = proc.wait(timeout=30)
                        break
                    time.sleep(5)
            else:
                exit_code = proc.wait()
        except Exception as exc:  # keep the campaign alive after one bad route
            error = str(exc)
            exit_code = -1
        finally:
            try:
                dashboard.stop_current(kill_carla=True)
            except Exception:
                pass

        wall_seconds = time.time() - start_time
        result_path = latest_result_after(start_time)
        metrics = dashboard.parse_bench2drive_result(result_path) if result_path else None
        trace_counts = read_jsonl_count(trace_path)
        classification = classify_run(metrics, exit_code, trace_path)
        summary = {
            **run,
            "attempt": attempt,
            "trace": str(trace_path),
            "trace_size_bytes": trace_path.stat().st_size if trace_path.exists() else 0,
            "trace_counts": dict(trace_counts),
            "result": str(result_path) if result_path else None,
            "exit_code": exit_code,
            "error": error,
            "timed_out": timed_out,
            "wall_seconds": round(wall_seconds, 3),
            "metrics": metrics,
            "quality": classification["score"],
            "good": classification["good"],
            "bad": classification["bad"],
            "ended_at": now(),
        }
        self.event(
            event="collect_attempt_done",
            route_id=run["route_id"],
            attempt=attempt,
            quality=summary["quality"],
            exit_code=exit_code,
            timed_out=timed_out,
            trace_size_bytes=summary["trace_size_bytes"],
            wall_seconds=summary["wall_seconds"],
        )
        return summary

    def should_retry_collect(self, summary: Dict[str, Any]) -> bool:
        if summary.get("metrics"):
            return False
        if int(summary.get("trace_size_bytes") or 0) > 1024:
            return False
        if float(summary.get("wall_seconds") or 0.0) >= self.args.retry_if_shorter_than:
            return False
        return int(summary.get("attempt") or 1) <= self.args.collect_retries

    def collect_run(self, run: Dict[str, Any]) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {}
        total_attempts = 1 + max(0, self.args.collect_retries)
        for attempt in range(1, total_attempts + 1):
            summary = self.collect_run_attempt(run, attempt)
            attempts.append({key: value for key, value in summary.items() if key != "attempts"})
            if not self.should_retry_collect(summary):
                break
            self.event(
                event="collect_retry",
                route_id=run["route_id"],
                attempt=attempt,
                reason="short run without metrics/trace",
            )
            time.sleep(self.args.route_cooldown_seconds)

        summary["attempts"] = attempts
        self.status["runs"].append(summary)
        self.status["completed_runs"] = len(self.status["runs"])
        self.status.pop("current_run", None)
        self.save()
        self.event(
            event="collect_done",
            route_id=run["route_id"],
            quality=summary["quality"],
            exit_code=summary.get("exit_code"),
        )
        time.sleep(self.args.route_cooldown_seconds)
        return summary

    def build_dataset(self) -> Optional[Path]:
        good_traces = [
            Path(run["trace"]) for run in self.status["runs"]
            if run.get("quality") == "good" and Path(run.get("trace", "")).exists()
        ]
        fallback_traces = [
            Path(run["trace"]) for run in self.status["runs"]
            if Path(run.get("trace", "")).exists()
        ]
        traces = good_traces if len(good_traces) >= self.args.min_good_traces else fallback_traces
        if not traces:
            self.status["phase"] = "dataset_failed"
            self.status["dataset_error"] = "No trace files were collected"
            self.save()
            return None

        self.status["phase"] = "building_dataset"
        self.status["dataset_trace_count"] = len(traces)
        self.save()
        out_dir = self.run_dir / "dataset"
        env = os.environ.copy()
        env.update({
            "DREAMER_RL_DATASET_RUN_ID": self.run_id,
            "DREAMER_RL_DATASET_OUT_DIR": str(out_dir),
            "DREAMER_RL_MIN_TRANSITIONS": str(self.args.min_transitions),
            "DREAMER_RL_MIN_RUNS": str(self.args.min_runs),
            "DREAMER_RL_MIN_ROUTES": str(self.args.min_routes),
            "DREAMER_RL_MAX_STATIONARY_FRACTION": str(self.args.max_stationary_fraction),
            "DREAMER_RL_MIN_RECOVERY_FRACTION": str(self.args.min_recovery_fraction),
        })
        try:
            subprocess.run(
                ["bash", str(ROOT / "scripts" / "build_dreamer_rl_dataset.sh"), *[str(p) for p in traces]],
                cwd=str(ROOT),
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            self.status["phase"] = "dataset_failed"
            self.status["dataset_error"] = f"dataset build failed: {exc}"
            self.save()
            return None

        dataset = out_dir / "dreamer_rl_dataset.npz"
        audit = out_dir / "audit.json"
        self.status["dataset"] = {"path": str(dataset), "audit": str(audit)}
        self.save()
        return dataset

    def train_warmstart(self, kind: str, dataset: Path) -> Optional[Path]:
        self.status["phase"] = f"warmstart_{kind}"
        self.save()
        env = os.environ.copy()
        env.update({
            "DREAMER_RL_KIND": kind,
            "DREAMER_RL_DATASET": str(dataset),
            "DREAMER_RL_WM_RUN_ID": self.run_id,
            "DREAMER_RL_WM_EPOCHS": str(self.args.warmstart_epochs),
            "DREAMER_RL_WM_BATCH_SIZE": str(self.args.warmstart_batch_size),
            "DREAMER_RL_DEVICE": self.args.device,
            "DREAMER_RL_SET_AS_INIT": "0",
        })
        try:
            subprocess.run(
                ["bash", str(ROOT / "scripts" / "train_dreamer_rl_world_model_warmstart.sh")],
                cwd=str(ROOT),
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            self.status.setdefault("warmstarts", {})[kind] = {"status": "failed", "error": str(exc)}
            self.save()
            return None

        checkpoint = ROOT / "logs" / "dreamer_rl_warmstart" / kind / self.run_id / "best_world_model.pt"
        self.status.setdefault("warmstarts", {})[kind] = {
            "status": "done",
            "checkpoint": str(checkpoint),
        }
        self.save()
        return checkpoint if checkpoint.exists() else None

    def wait_rl_run(self, kind: str, run_id: str, run_dir: Path) -> str:
        done = run_dir / "training.done"
        failed = run_dir / "training.failed"
        pid_file = run_dir / "training.pid"
        while True:
            if done.exists():
                return "done"
            if failed.exists():
                return f"failed exit={failed.read_text(encoding='utf-8').strip()}"
            pid = pid_file.read_text(encoding="utf-8").strip() if pid_file.exists() else ""
            if pid:
                alive = subprocess.run(["ps", "-p", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
                if not alive:
                    return "stopped"
            self.status["phase"] = f"rl_{kind}_running"
            self.status.setdefault("rl_runs", {}).setdefault(kind, {})
            self.status["rl_runs"][kind].update({"run_id": run_id, "run_dir": str(run_dir), "status": "running"})
            self.save()
            time.sleep(self.args.rl_poll_seconds)

    def train_rl(self, kind: str, warmstart: Path) -> None:
        self.status["phase"] = f"rl_{kind}_starting"
        self.save()
        run_id = f"{self.run_id}_{kind}_rl"
        env = os.environ.copy()
        env.update({
            "DREAMER_RL_KIND": kind,
            "DREAMER_RL_RUN_ID": run_id,
            "DREAMER_RL_INIT_WORLD_MODEL": str(warmstart),
            "DREAMER_RL_EPISODES": str(self.args.rl_episodes),
            "DREAMER_RL_MAX_EPISODE_STEPS": str(self.args.rl_max_episode_steps),
            "DREAMER_RL_ROLLOUT_SIZE": str(self.args.rl_rollout_size),
            "DREAMER_RL_EVAL_INTERVAL": str(self.args.rl_eval_interval),
            "DREAMER_RL_DEVICE": self.args.device,
            "DREAMER_RL_INSTALL_LATEST": "1",
            "RESTART_EXISTING": "1",
            "CARLA_QUALITY": self.args.carla_quality,
        })
        try:
            subprocess.run(
                ["bash", str(ROOT / "scripts" / "start_dreamer_rl_noguard_training.sh")],
                cwd=str(ROOT),
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            self.status.setdefault("rl_runs", {})[kind] = {"run_id": run_id, "status": "failed_to_start", "error": str(exc)}
            self.save()
            return

        run_dir = ROOT / "logs" / "dreamer_rl_noguard" / kind / run_id
        status = self.wait_rl_run(kind, run_id, run_dir)
        self.status.setdefault("rl_runs", {})[kind] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": status,
        }
        self.save()

    def run(self) -> None:
        self.event(event="campaign_start", run_id=self.run_id)
        plan = self.build_plan()
        for run in plan["runs"]:
            self.collect_run(run)

        dataset = self.build_dataset()
        if dataset is None:
            self.event(event="campaign_dataset_failed")
            return

        warmstarts: Dict[str, Path] = {}
        for kind in ("ppo", "sdbs"):
            checkpoint = self.train_warmstart(kind, dataset)
            if checkpoint:
                warmstarts[kind] = checkpoint

        if self.args.skip_live_rl:
            self.status["phase"] = "done_skip_live_rl"
            self.save()
            return

        for kind, checkpoint in warmstarts.items():
            self.train_rl(kind, checkpoint)

        self.status["phase"] = "done"
        self.status["finished_at"] = now()
        self.save()
        self.event(event="campaign_done", run_id=self.run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("DREAMER_RL_CAMPAIGN_ID"))
    parser.add_argument("--max-per-bucket", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_MAX_ROUTES_PER_BUCKET", "2")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_SEED", "7302026")))
    parser.add_argument("--include-unstable", action="store_true", default=os.environ.get("DREAMER_RL_CAMPAIGN_INCLUDE_UNSTABLE", "0") == "1")
    parser.add_argument("--teacher-dreamer-mode", default=os.environ.get("DREAMER_RL_CAMPAIGN_TEACHER_MODE", "dreamer_ppo"))
    parser.add_argument("--camera", default=os.environ.get("SIMLINGO_VIEW_MODE", "chase"))
    parser.add_argument("--visual-weather", default=os.environ.get("SIMLINGO_VISUAL_WEATHER", "day"))
    parser.add_argument("--video-quality", default=os.environ.get("DREAMER_RL_CAMPAIGN_VIDEO_QUALITY", "preview"))
    parser.add_argument("--carla-quality", default=os.environ.get("CARLA_QUALITY", "Low"))
    parser.add_argument("--sample-interval", type=float, default=float(os.environ.get("DREAMER_RL_CAMPAIGN_SAMPLE_INTERVAL", "0.20")))
    parser.add_argument("--min-good-traces", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_MIN_GOOD_TRACES", "2")))
    parser.add_argument("--min-transitions", type=int, default=int(os.environ.get("DREAMER_RL_MIN_TRANSITIONS", "1500")))
    parser.add_argument("--min-runs", type=int, default=int(os.environ.get("DREAMER_RL_MIN_RUNS", "2")))
    parser.add_argument("--min-routes", type=int, default=int(os.environ.get("DREAMER_RL_MIN_ROUTES", "2")))
    parser.add_argument("--max-stationary-fraction", default=os.environ.get("DREAMER_RL_MAX_STATIONARY_FRACTION", "0.65"))
    parser.add_argument("--min-recovery-fraction", default=os.environ.get("DREAMER_RL_MIN_RECOVERY_FRACTION", "0.02"))
    parser.add_argument("--warmstart-epochs", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_WM_EPOCHS", "80")))
    parser.add_argument("--warmstart-batch-size", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_WM_BATCH_SIZE", "128")))
    parser.add_argument("--device", default=os.environ.get("DREAMER_RL_DEVICE", "auto"))
    parser.add_argument("--rl-episodes", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_RL_EPISODES", "80")))
    parser.add_argument("--rl-max-episode-steps", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_RL_MAX_EPISODE_STEPS", "700")))
    parser.add_argument("--rl-rollout-size", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_RL_ROLLOUT_SIZE", "1024")))
    parser.add_argument("--rl-eval-interval", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_RL_EVAL_INTERVAL", "20")))
    parser.add_argument("--rl-poll-seconds", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_RL_POLL_SECONDS", "30")))
    parser.add_argument("--skip-live-rl", action="store_true", default=os.environ.get("DREAMER_RL_CAMPAIGN_SKIP_LIVE_RL", "0") == "1")
    parser.add_argument("--collect-retries", type=int, default=int(os.environ.get("DREAMER_RL_CAMPAIGN_COLLECT_RETRIES", "1")))
    parser.add_argument("--retry-if-shorter-than", type=float, default=float(os.environ.get("DREAMER_RL_CAMPAIGN_RETRY_IF_SHORTER_THAN", "90")))
    parser.add_argument("--collect-max-wall-seconds", type=float, default=float(os.environ.get("DREAMER_RL_CAMPAIGN_COLLECT_MAX_WALL_SECONDS", "900")))
    parser.add_argument("--route-cooldown-seconds", type=float, default=float(os.environ.get("DREAMER_RL_CAMPAIGN_ROUTE_COOLDOWN_SECONDS", "12")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = Campaign(args)
    try:
        campaign.run()
        return 0
    except KeyboardInterrupt:
        campaign.status["phase"] = "interrupted"
        campaign.status["interrupted_at"] = now()
        campaign.save()
        return 130
    except Exception as exc:
        campaign.status["phase"] = "failed"
        campaign.status["error"] = str(exc)
        campaign.save()
        campaign.event(event="campaign_failed", error=str(exc))
        print(f"[campaign] failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
