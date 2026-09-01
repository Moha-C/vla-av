#!/usr/bin/env python3
"""Evaluate the unmodified CarDreamer overtake checkpoint deterministically.

This harness intentionally reuses the upstream CarDreamer task, observation
wrappers, DreamerV3 agent, and checkpoint loader.  It only replaces the
upstream infinite evaluation supervisor with a finite, machine-readable run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3
import embodied
from embodied.envs import from_gym
from dreamerv3.eval import wrap_env


ACCELERATIONS = (-2.0, 0.0, 2.0)
STEERING = (-0.6, -0.2, 0.0, 0.2, 0.6)


class NullMonitor:
    """Disable only CarDreamer's optional Flask visualization thread."""

    def __init__(self, config):
        del config

    def render(self, obs, info):
        del obs, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--carla-port", type=int, default=2100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    return parser.parse_args()


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(values: Dict[str, np.ndarray], key: str, reduction: str = "sum") -> float:
    value = values.get(key)
    if value is None or np.asarray(value).size == 0:
        return 0.0
    array = np.asarray(value, dtype=np.float64)
    if reduction == "max":
        return float(np.nanmax(array))
    if reduction == "min":
        return float(np.nanmin(array))
    if reduction == "mean":
        return float(np.nanmean(array))
    if reduction == "last":
        return float(array.reshape(-1)[-1])
    return float(np.nansum(array))


def compress_actions(indices: Iterable[int]) -> List[Dict[str, int]]:
    runs: List[Dict[str, int]] = []
    for raw in indices:
        index = int(raw)
        if runs and runs[-1]["action"] == index:
            runs[-1]["steps"] += 1
        else:
            runs.append({"action": index, "steps": 1})
    return runs


def summarize_episode(ep: Dict[str, np.ndarray], info: Dict[str, np.ndarray], index: int) -> Dict[str, Any]:
    action_vectors = np.asarray(ep.get("action", []))
    if action_vectors.ndim == 2 and action_vectors.shape[-1] == len(ACCELERATIONS) * len(STEERING):
        action_indices = np.argmax(action_vectors, axis=-1).astype(int)
    else:
        action_indices = np.asarray([], dtype=int)

    counts = Counter(int(value) for value in action_indices)
    action_histogram = []
    for action_index in sorted(counts):
        action_histogram.append(
            {
                "action": action_index,
                "acceleration": ACCELERATIONS[action_index // len(STEERING)],
                "steering": STEERING[action_index % len(STEERING)],
                "steps": counts[action_index],
            }
        )

    collision = as_float(info, "is_collision", "max") > 0.0
    out_of_lane = as_float(info, "out_of_lane", "max") > 0.0
    time_exceeded = as_float(info, "time_exceeded", "max") > 0.0
    destination_reached = as_float(info, "destination_reached", "max") > 0.0
    exceeding = as_float(info, "r_exceeding") > 0.0
    overtake = as_float(info, "r_overtake") > 0.0

    termination = "unknown"
    for name, active in (
        ("collision", collision),
        ("out_of_lane", out_of_lane),
        ("destination_reached", destination_reached),
        ("time_exceeded", time_exceeded),
    ):
        if active:
            termination = name
            break

    return {
        "episode": index,
        "length_steps": max(0, len(ep.get("reward", [])) - 1),
        "return": as_float(ep, "reward"),
        "overtake_started": exceeding,
        "overtake_completed": overtake,
        "clean_overtake": bool(overtake and not collision and not out_of_lane),
        "destination_reached": destination_reached,
        "collision": collision,
        "out_of_lane": out_of_lane,
        "time_exceeded": time_exceeded,
        "termination": termination,
        "travel_distance": as_float(info, "travel_distance", "max"),
        "mean_speed_mps": as_float(info, "speed_norm", "mean"),
        "min_ttc_seconds": as_float(info, "ttc", "min"),
        "max_waypoint_distance": as_float(info, "wpt_dis", "max"),
        "reward_components": {
            "waypoints": as_float(info, "r_waypoints"),
            "speed": as_float(info, "r_speed"),
            "collision": as_float(info, "r_collision"),
            "stay_same_lane": as_float(info, "p_stay_same_lane"),
            "early_lane_change": as_float(info, "p_early_lane_change"),
            "exceeding": as_float(info, "r_exceeding"),
            "overtake": as_float(info, "r_overtake"),
        },
        "action_histogram": action_histogram,
        "action_runs": compress_actions(action_indices),
    }


def build_env_and_agent(args: argparse.Namespace):
    # EnvMonitorOpenCV starts a non-daemon Flask thread even with
    # display.enable=False.  Replacing that optional monitor keeps finite
    # command-line evaluations able to exit; task dynamics stay untouched.
    import car_dreamer.carla_base_env as carla_base_env

    carla_base_env.EnvMonitorOpenCV = NullMonitor
    upstream = Path(car_dreamer.__file__).resolve().parent.parent
    config_file = upstream / "dreamerv3" / "dreamerv3.yaml"
    model_configs = yaml.YAML(typ="safe").load(config_file.read_text())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    env_argv = [
        "--env.world.carla_port",
        str(args.carla_port),
        "--env.world.traffic.tm_seed",
        str(args.seed),
        "--env.display.enable",
        "False",
        "--env.eval",
        "False",
    ]
    gym_env, task_config = car_dreamer.create_task("carla_overtake", env_argv)
    config = config.update(task_config)
    config = config.update(
        {
            "dreamerv3.seed": args.seed,
            "dreamerv3.logdir": str(args.logdir),
            "dreamerv3.run.from_checkpoint": str(args.checkpoint),
            "dreamerv3.jax.platform": "gpu",
            "dreamerv3.jax.policy_devices": (0,),
            "dreamerv3.jax.train_devices": (0,),
            "dreamerv3.jax.prealloc": False,
        }
    )

    wrapped = from_gym.FromGym(gym_env)
    wrapped = wrap_env(wrapped, config.dreamerv3)
    env = embodied.BatchEnv([wrapped], parallel=False)
    step = embodied.Counter()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config.dreamerv3)
    return env, gym_env, agent, step, upstream, config


def cleanup_carla_task(gym_env) -> None:
    """Release task-owned sensors/actors and synchronous CARLA state."""

    try:
        raw_env = gym_env.unwrapped
    except Exception:
        raw_env = gym_env
    try:
        raw_env._observer.destroy()
    except Exception as exc:
        print(f"CARDREAMER_CLEANUP observer warning: {exc}", file=sys.stderr)
    try:
        world_manager = raw_env._world
        world_manager._set_synchronous_mode(False)
        actor_ids = list(world_manager.actor_dict)
        if actor_ids:
            import carla

            world_manager._client.apply_batch_sync(
                [carla.command.DestroyActor(actor_id) for actor_id in actor_ids],
                False,
            )
        world_manager.actor_dict = {}
    except Exception as exc:
        print(f"CARDREAMER_CLEANUP world warning: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.logdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    np.random.seed(args.seed)

    started = time.time()
    env = None
    gym_env = None
    episodes: List[Dict[str, Any]] = []
    try:
        env, gym_env, agent, step, upstream, config = build_env_and_agent(args)
        checkpoint = embodied.Checkpoint()
        checkpoint.agent = agent
        checkpoint.load(str(args.checkpoint), keys=["agent"])

        driver = embodied.Driver(env)

        def on_episode(ep, ep_info, worker):
            del worker
            result = summarize_episode(ep, ep_info, len(episodes))
            episodes.append(result)
            print(
                "CARDREAMER_EPISODE "
                f"seed={args.seed} episode={result['episode']} "
                f"clean_overtake={int(result['clean_overtake'])} "
                f"collision={int(result['collision'])} "
                f"out_of_lane={int(result['out_of_lane'])} "
                f"return={result['return']:.3f}",
                flush=True,
            )

        driver.on_episode(on_episode)
        policy = lambda *policy_args: agent.policy(*policy_args, mode="eval")
        driver(policy, episodes=args.episodes)

        total = len(episodes)
        clean = sum(item["clean_overtake"] for item in episodes)
        completed = sum(item["overtake_completed"] for item in episodes)
        collisions = sum(item["collision"] for item in episodes)
        offroad = sum(item["out_of_lane"] for item in episodes)
        payload = {
            "schema_version": 1,
            "protocol": "cardreamer_official_overtake_unchanged_checkpoint",
            "task": "carla_overtake",
            "seed": args.seed,
            "requested_episodes": args.episodes,
            "completed_episodes": total,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "upstream_commit": git_revision(upstream),
            "python": platform.python_version(),
            "elapsed_seconds": round(time.time() - started, 3),
            "observation_contract": {
                "inputs": ["collision", "birdeye_wpt"],
                "birdeye_shape": [128, 128, 3],
                "observability": "full privileged BEV",
            },
            "action_contract": {
                "type": "discrete",
                "count": len(ACCELERATIONS) * len(STEERING),
                "accelerations": list(ACCELERATIONS),
                "steering": list(STEERING),
            },
            "summary": {
                "clean_overtake_rate": clean / total if total else 0.0,
                "overtake_completion_rate": completed / total if total else 0.0,
                "collision_rate": collisions / total if total else 0.0,
                "out_of_lane_rate": offroad / total if total else 0.0,
                "mean_return": float(np.mean([item["return"] for item in episodes])) if total else 0.0,
            },
            "episodes": episodes,
        }
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
        print(f"CARDREAMER_RESULT {args.output}", flush=True)
        return 0
    finally:
        if gym_env is not None:
            cleanup_carla_task(gym_env)
        if env is not None:
            env.close()


if __name__ == "__main__":
    sys.exit(main())
