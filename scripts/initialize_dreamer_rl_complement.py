#!/usr/bin/env python3
"""Migrate an RL checkpoint to the learned SimLingo-complement action semantics."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


ACTION_SEMANTICS = "simlingo_residual_with_learned_gate_v1"


def default_policy_state_scale(dim: int) -> np.ndarray:
    scale = np.ones(max(dim, 28), dtype=np.float32)
    scale[0] = 1000.0
    scale[1] = 1000.0
    scale[2] = 15.0
    scale[3] = math.pi
    scale[4] = 8.0
    scale[6] = 8.0
    scale[8] = math.pi
    scale[10] = 2.0
    scale[11] = 50.0
    for idx in (13, 16, 18, 21, 23, 26):
        scale[idx] = 80.0
    for idx in (14, 19, 20, 24, 25):
        scale[idx] = 20.0
    return scale[:dim]


def atomic_save(path: Path, payload) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), delete=False) as handle:
        torch.save(payload, handle.name)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--gate-bias", type=float, default=-1.4)
    parser.add_argument("--exploration-log-std", type=float, default=-0.2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("policy_action_semantics") == ACTION_SEMANTICS and not args.force:
        print(json.dumps({"status": "already_migrated", "checkpoint": str(checkpoint)}))
        return 0

    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise KeyError("checkpoint has no PPO policy state_dict")
    required = ("actor_mean.weight", "actor_mean.bias", "log_std", "critic.weight", "critic.bias")
    missing = [key for key in required if key not in policy]
    if missing:
        raise KeyError(f"policy is missing: {', '.join(missing)}")
    if int(policy["actor_mean.bias"].shape[0]) < 4:
        raise ValueError("the complement policy requires four actor outputs")

    backup_dir = (args.backup_dir or checkpoint.parent / "rollback_backups").expanduser().resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = backup_dir / f"{checkpoint.stem}_before_learned_blend_{stamp}{checkpoint.suffix}"
    shutil.copy2(checkpoint, backup)

    parent_episode = int(payload.get("episode", 0))
    parent_online_rl = payload.get("online_rl")
    migrated_policy = {key: value.clone() if torch.is_tensor(value) else value for key, value in policy.items()}
    migrated_policy["actor_mean.weight"].zero_()
    migrated_policy["actor_mean.bias"].zero_()
    migrated_policy["actor_mean.bias"][3] = float(args.gate_bias)
    migrated_policy["log_std"].fill_(float(args.exploration_log_std))
    migrated_policy["critic.weight"].zero_()
    migrated_policy["critic.bias"].zero_()

    state_dim = int(migrated_policy["trunk.0.weight"].shape[1])
    payload["policy"] = migrated_policy
    payload.pop("optimizer_pi", None)
    payload["episode"] = 0
    payload["policy_state_mean"] = np.zeros(state_dim, dtype=np.float32)
    payload["policy_state_std"] = default_policy_state_scale(state_dim)
    payload["policy_action_semantics"] = ACTION_SEMANTICS
    payload["policy_role"] = "online_rl_complement_to_simlingo"
    payload["online_rl_update_count"] = 0
    payload["online_rl"] = {
        "status": "initialized_learned_complement",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "no_guard": True,
        "complement_to_simlingo": True,
        "action_semantics": ACTION_SEMANTICS,
        "initial_dreamer_weight": float(torch.sigmoid(torch.tensor(args.gate_bias))),
        "parent_episode": parent_episode,
        "parent_online_rl": parent_online_rl,
        "rollback_checkpoint": str(backup),
    }
    atomic_save(checkpoint, payload)
    print(json.dumps({
        "status": "migrated",
        "checkpoint": str(checkpoint),
        "backup": str(backup),
        "action_semantics": ACTION_SEMANTICS,
        "initial_dreamer_weight": payload["online_rl"]["initial_dreamer_weight"],
        "state_dim": state_dim,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
