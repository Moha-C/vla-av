#!/usr/bin/env python3
"""Run true online RL episodes for SimLingo + Dreamer no-guard variants.

The loop is:
  Bench2Drive/SimLingo episode -> no-guard Dreamer policy action trace ->
  reward computation -> PPO/SDBS checkpoint update -> next episode.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
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
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


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


def checkpoint_for_kind(kind: str) -> Path:
    if kind == "ppo":
        return ROOT / "external" / "simlingo" / "checkpoints" / "dreamer_ppo_rl_noguard" / "latest_rl_model.pt"
    if kind == "sdbs":
        return ROOT / "external" / "simlingo" / "checkpoints" / "dreamer_sdbs_rl_noguard" / "latest_rl_model.pt"
    raise ValueError(f"unknown kind: {kind}")


def dreamer_mode_for_kind(kind: str) -> str:
    return "dreamer_ppo_rl_noguard" if kind == "ppo" else "dreamer_sdbs_rl_noguard"


def trace_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def rl_trace_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                status = (json.loads(line).get("status") or {})
            except Exception:
                continue
            if status.get("mode") == "rl_noguard":
                count += 1
    return count


class OnlineTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = ROOT / "logs" / "dreamer_online_rl" / self.run_id
        self.trace_dir = self.run_dir / "traces"
        self.update_dir = self.run_dir / "updates"
        self.status_path = self.run_dir / "status.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.md"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.update_dir.mkdir(parents=True, exist_ok=True)
        self.status: Dict[str, Any] = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "started_at": now(),
            "updated_at": now(),
            "phase": "initializing",
            "completed_episodes": 0,
            "total_episodes": 0,
            "episodes": [],
            "checkpoints": {},
            "no_guard": True,
            "complement_to_simlingo": True,
        }

    def save(self) -> None:
        self.status["updated_at"] = now()
        write_json(self.status_path, self.status)
        lines = [
            "# Dreamer Online RL No-Guard Training",
            "",
            f"- run_id: `{self.run_id}`",
            f"- phase: `{self.status.get('phase')}`",
            f"- progress: {self.status.get('completed_episodes', 0)}/{self.status.get('total_episodes', 0)} episodes",
            f"- no_guard: `{self.status.get('no_guard')}`",
            f"- complement_to_simlingo: `{self.status.get('complement_to_simlingo')}`",
            "",
            "## Episodes",
        ]
        for ep in self.status.get("episodes", []):
            metrics = ep.get("metrics") or {}
            update = ep.get("update") or {}
            lines.append(
                f"- {ep.get('kind')} route `{ep.get('route_id')}` seed `{ep.get('seed')}` "
                f"trace_rows={ep.get('trace_rows')} rl_rows={ep.get('rl_trace_rows')} "
                f"update={update.get('status', '-')}"
            )
            if metrics:
                lines.append(
                    f"  route={metrics.get('route_score')} score={metrics.get('driving_score')} "
                    f"coll={metrics.get('collisions')} offroad={metrics.get('offroad')} "
                    f"red={metrics.get('red_lights')} blocked={metrics.get('blocked')}"
                )
            if update:
                lines.append(
                    f"  transitions={update.get('transitions')} reward_sum={update.get('reward_sum')} "
                    f"policy_loss={update.get('policy_loss')} wm_loss={update.get('world_model_loss')}"
                )
        self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def event(self, **payload: Any) -> None:
        payload.setdefault("time", now())
        append_jsonl(self.events_path, payload)

    def backup_checkpoints(self, kinds: List[str]) -> None:
        backup_dir = self.run_dir / "checkpoint_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for kind in kinds:
            ckpt = checkpoint_for_kind(kind)
            if not ckpt.exists():
                raise RuntimeError(f"missing RL checkpoint for {kind}: {ckpt}")
            dest = backup_dir / f"{kind}_latest_rl_model_before_online.pt"
            shutil.copy2(ckpt, dest)
            self.status["checkpoints"][kind] = {
                "active": str(ckpt),
                "backup_before_online": str(dest),
            }
        self.save()

    def run_episode(self, kind: str, route: Dict[str, Any], episode_index: int) -> Dict[str, Any]:
        trace_path = self.trace_dir / f"{episode_index:04d}_{kind}_route_{route['route_id']}_seed_{route['seed']}.jsonl"
        payload = {
            "route_id": route["route_id"],
            "route_file": route["route_file"],
            "seed": route["seed"],
            "quality": self.args.carla_quality,
            "camera": self.args.camera,
            "visual_weather": self.args.visual_weather,
            "video_quality": self.args.video_quality,
            "playback_after": "0",
            "playback_speed": "1",
            "run_mode": "action_dreaming",
            "dreamer_mode": dreamer_mode_for_kind(kind),
            "dreamer_rl_training": "1",
            "dreamer_rl_action_space": self.args.action_space,
            "cot_mode": "off",
            "action_dreaming_sample_interval": str(self.args.sample_interval),
            "action_dreaming_out_dir": str(self.trace_dir),
            "action_dreaming_run_id": f"{self.run_id}_{episode_index:04d}_{kind}",
            "action_dreaming_trace_path": str(trace_path),
            "traffic_light_overlay": "1",
            "view_fps": "30",
        }
        current = {
            "kind": kind,
            "route_id": route["route_id"],
            "town": route.get("town"),
            "scenario_type": route.get("scenario_type"),
            "seed": route["seed"],
            "trace": str(trace_path),
            "episode_index": episode_index,
        }
        self.status["phase"] = "running_episode"
        self.status["current_episode"] = current
        self.save()
        self.event(event="episode_start", **current)

        start_time = time.time()
        exit_code: Optional[int] = None
        error: Optional[str] = None
        timed_out = False
        try:
            dashboard.start_run(payload)
            proc = dashboard.STATE.get("process")
            if proc is None:
                raise RuntimeError("dashboard.start_run did not create a process")
            deadline = start_time + self.args.max_wall_seconds
            while True:
                exit_code = proc.poll()
                if exit_code is not None:
                    break
                if self.args.max_wall_seconds > 0 and time.time() >= deadline:
                    timed_out = True
                    error = f"episode wall-time limit reached ({self.args.max_wall_seconds:.0f}s)"
                    dashboard.stop_current(kill_carla=True)
                    try:
                        exit_code = proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        exit_code = proc.wait(timeout=30)
                    break
                time.sleep(5)
        except Exception as exc:
            error = str(exc)
            exit_code = -1
        finally:
            try:
                dashboard.stop_current(kill_carla=True)
            except Exception:
                pass

        result_path = latest_result_after(start_time)
        metrics = dashboard.parse_bench2drive_result(result_path) if result_path else {}
        if not metrics:
            metrics = {
                "path": str(result_path) if result_path else "",
                "status": "incomplete_or_ineligible",
                "incomplete": 1.0,
                "route_score": 0.0,
                "driving_score": 0.0,
                "penalty": 0.0,
                "collisions": 1.0 if exit_code not in (None, 0, 130, 143, -2, -15) else 0.0,
                "pedestrian_collisions": 0.0,
                "vehicle_collisions": 1.0 if exit_code not in (None, 0, 130, 143, -2, -15) else 0.0,
                "layout_collisions": 0.0,
                "red_lights": 0.0,
                "stop_infractions": 0.0,
                "offroad": 1.0 if exit_code not in (None, 0, 130, 143, -2, -15) else 0.0,
                "blocked": 1.0,
                "scenario_timeouts": 0.0,
                "route_timeouts": 0.0,
                "min_speed_infractions": 0.0,
                "success": 0.0,
            }
        summary = {
            **current,
            "exit_code": exit_code,
            "error": error,
            "timed_out": timed_out,
            "wall_seconds": round(time.time() - start_time, 3),
            "trace_rows": trace_count(trace_path),
            "rl_trace_rows": rl_trace_count(trace_path),
            "result": str(result_path) if result_path else None,
            "metrics": metrics,
        }
        self.event(event="episode_done", **summary)
        return summary

    def update_checkpoint(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        kind = episode["kind"]
        ckpt = checkpoint_for_kind(kind)
        summary_path = self.update_dir / f"{episode['episode_index']:04d}_{kind}_update.json"
        cmd = [
            "conda",
            "run",
            "-n",
            self.args.conda_env,
            "python",
            str(ROOT / "scripts" / "dreamer_online_rl_update.py"),
            "--trace",
            str(episode["trace"]),
            "--checkpoint",
            str(ckpt),
            "--output-checkpoint",
            str(ckpt),
            "--metrics-json",
            json.dumps(episode.get("metrics") or {}),
            "--summary",
            str(summary_path),
            "--device",
            self.args.device,
            "--epochs",
            str(self.args.update_epochs),
            "--batch-size",
            str(self.args.batch_size),
            "--min-transitions",
            str(self.args.min_transitions),
            "--min-save-reward-sum",
            str(self.args.min_save_reward_sum),
            "--max-save-unsafe-side-loss",
            str(self.args.max_save_unsafe_side_loss),
            "--max-save-stuck-loss",
            str(self.args.max_save_stuck_loss),
        ]
        self.status["phase"] = "updating_checkpoint"
        self.save()
        self.event(event="update_start", kind=kind, checkpoint=str(ckpt), trace=str(episode["trace"]))
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        update: Dict[str, Any] = {
            "status": "failed" if proc.returncode not in (0, 2) else ("skipped" if proc.returncode == 2 else "updated"),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "summary": str(summary_path),
        }
        if summary_path.exists():
            try:
                update.update(json.loads(summary_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        self.event(event="update_done", kind=kind, status=update.get("status"), returncode=proc.returncode)
        return update

    def run(self) -> None:
        kinds = ["ppo", "sdbs"] if self.args.kind == "both" else [self.args.kind]
        self.backup_checkpoints(kinds)
        plan = build_campaign_plan(
            max_per_bucket=self.args.max_per_bucket,
            seed=self.args.seed,
            include_unstable=self.args.include_unstable,
        )
        write_json(self.run_dir / "route_plan.json", plan)
        routes = plan["runs"]
        if self.args.max_routes >= 0:
            routes = routes[: self.args.max_routes]
        schedule: List[Dict[str, Any]] = []
        for kind in kinds:
            for route in routes:
                schedule.append({"kind": kind, "route": route})
        self.status["total_episodes"] = len(schedule)
        self.status["phase"] = "scheduled"
        self.save()

        for index, item in enumerate(schedule, start=1):
            episode = self.run_episode(item["kind"], item["route"], index)
            update = self.update_checkpoint(episode)
            episode["update"] = update
            self.status["episodes"].append(episode)
            self.status["completed_episodes"] = len(self.status["episodes"])
            self.status.pop("current_episode", None)
            self.status["phase"] = "cooldown"
            self.save()
            time.sleep(max(0.0, self.args.cooldown_seconds))

        self.status["phase"] = "done"
        self.status["finished_at"] = now()
        self.save()
        self.event(event="online_rl_done", run_id=self.run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("DREAMER_ONLINE_RL_RUN_ID"))
    parser.add_argument("--kind", choices=("ppo", "sdbs", "both"), default=os.environ.get("DREAMER_ONLINE_RL_KIND", "both"))
    parser.add_argument("--max-per-bucket", type=int, default=int(os.environ.get("DREAMER_ONLINE_RL_MAX_ROUTES_PER_BUCKET", "1")))
    parser.add_argument("--max-routes", type=int, default=int(os.environ.get("DREAMER_ONLINE_RL_MAX_ROUTES", "6")), help="Number of selected routes to run; use -1 for all selected routes.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DREAMER_ONLINE_RL_SEED", "7312026")))
    parser.add_argument("--include-unstable", action="store_true", default=os.environ.get("DREAMER_ONLINE_RL_INCLUDE_UNSTABLE", "0") == "1")
    parser.add_argument("--camera", default=os.environ.get("SIMLINGO_VIEW_MODE", "chase"))
    parser.add_argument("--visual-weather", default=os.environ.get("SIMLINGO_VISUAL_WEATHER", "day"))
    parser.add_argument("--video-quality", default=os.environ.get("DREAMER_ONLINE_RL_VIDEO_QUALITY", "preview"))
    parser.add_argument("--carla-quality", default=os.environ.get("CARLA_QUALITY", "Low"))
    parser.add_argument("--sample-interval", type=float, default=float(os.environ.get("DREAMER_ONLINE_RL_SAMPLE_INTERVAL", "0.10")))
    parser.add_argument("--max-wall-seconds", type=float, default=float(os.environ.get("DREAMER_ONLINE_RL_MAX_WALL_SECONDS", "900")))
    parser.add_argument("--cooldown-seconds", type=float, default=float(os.environ.get("DREAMER_ONLINE_RL_COOLDOWN_SECONDS", "8")))
    parser.add_argument("--conda-env", default=os.environ.get("DREAMER_ONLINE_RL_CONDA_ENV", "simlingo"))
    parser.add_argument("--device", default=os.environ.get("DREAMER_ONLINE_RL_DEVICE", "auto"))
    parser.add_argument("--update-epochs", type=int, default=int(os.environ.get("DREAMER_ONLINE_RL_UPDATE_EPOCHS", "4")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DREAMER_ONLINE_RL_BATCH_SIZE", "128")))
    parser.add_argument("--min-transitions", type=int, default=int(os.environ.get("DREAMER_ONLINE_RL_MIN_TRANSITIONS", "64")))
    parser.add_argument("--min-save-reward-sum", type=float, default=float(os.environ.get("DREAMER_ONLINE_RL_MIN_SAVE_REWARD_SUM", "-250")))
    parser.add_argument("--max-save-unsafe-side-loss", type=float, default=float(os.environ.get("DREAMER_ONLINE_RL_MAX_SAVE_UNSAFE_SIDE_LOSS", "-100")))
    parser.add_argument("--max-save-stuck-loss", type=float, default=float(os.environ.get("DREAMER_ONLINE_RL_MAX_SAVE_STUCK_LOSS", "-60")))
    parser.add_argument("--action-space", choices=("residual", "absolute"), default=os.environ.get("DREAMER_ONLINE_RL_ACTION_SPACE", "residual"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trainer = OnlineTrainer(args)
    try:
        trainer.run()
        return 0
    except KeyboardInterrupt:
        trainer.status["phase"] = "interrupted"
        trainer.status["interrupted_at"] = now()
        trainer.save()
        return 130
    except Exception as exc:
        trainer.status["phase"] = "failed"
        trainer.status["error"] = str(exc)
        trainer.save()
        trainer.event(event="online_rl_failed", error=str(exc))
        print(f"[dreamer-online-rl] failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
