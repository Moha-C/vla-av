#!/usr/bin/env python3
"""Update one frozen Dreamer candidate from a balanced on-policy batch.

Every episode in the manifest must have been collected with the exact same
checkpoint hash.  Successful passes, justified waits, and failures contribute
at episode level, so long stationary traces cannot dominate the PPO objective.
The world model is first fitted on real transitions, then used for short,
differentiable imagined rollouts with a deliberately small actor coefficient.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from scripts import dreamer_online_rl_update as core


ROOT = Path(__file__).resolve().parents[1]
REWARD_SCHEMA = "simlingo_complement_balanced_bounded_v6"
CATEGORY_WEIGHTS = {
    "clean_success": 1.00,
    "justified_wait": 0.90,
    "collision": 1.15,
    "unsafe_failure": 1.05,
    "partial_progress": 0.75,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True, default=str)
        tmp.write("\n")
        temporary = Path(tmp.name)
    temporary.replace(path)


def bounded_rewards(
    rewards: np.ndarray,
    category: str,
    metrics: Dict[str, float],
) -> np.ndarray:
    """Map raw shaped rewards to a stable, interpretable optimization scale."""
    raw = np.asarray(rewards, dtype=np.float32)
    if raw.size == 0:
        return raw
    # Legacy shaping contains several dense penalties whose sums can reach the
    # thousands on long routes. Preserve their within-episode ordering, but
    # remove route-length bias before applying a bounded terminal outcome.
    dense = 2.0 * np.tanh(raw / 4.0)
    reference = dense[:-1] if dense.size > 1 else dense
    center = float(np.median(reference))
    mad = float(np.median(np.abs(reference - center)))
    scale = max(0.50, 1.4826 * mad)
    bounded = np.clip((dense - center) / scale, -2.0, 2.0)
    if bounded.size > 1:
        body = bounded[:-1] - float(np.mean(bounded[:-1]))
        peak = float(np.max(np.abs(body))) if body.size else 0.0
        if peak > 2.0:
            body *= 2.0 / peak
        # Scaling preserves the zero mean; the second subtraction only removes
        # floating-point residue so every episode sum is its terminal outcome.
        body -= float(np.mean(body))
        bounded[:-1] = body

    # Keep terminal outcomes decisive without making one episode numerically
    # larger than an entire balanced batch.
    if category == "collision":
        bounded[-1] = -10.0
    elif category == "unsafe_failure":
        bounded[-1] = -6.0
    elif category == "clean_success":
        bounded[-1] = 8.0
    elif category == "justified_wait":
        bounded[-1] = 2.0
    route = core.as_float(metrics.get("route_score"), 0.0)
    if route >= 99.0 and category not in ("clean_success", "collision", "unsafe_failure"):
        bounded[-1] += 2.0
    elif route >= 50.0 and category == "partial_progress":
        bounded[-1] += 1.0
    return np.clip(bounded, -10.0, 8.0).astype(np.float32)


def waiting_is_justified(rows: List[Dict[str, Any]]) -> bool:
    waiting = 0
    justified = 0
    for row in rows:
        status = row.get("status") or {}
        state = status.get("state_vector") or []
        speed = core.as_float(state[2], 0.0) if len(state) > 2 else 0.0
        front = core.as_float(status.get("front_vehicle_m"), 80.0)
        blocked = core.as_float(status.get("blocked_ticks"), 0.0)
        if speed > 0.35 or (front > 18.0 and blocked < 20.0):
            continue
        waiting += 1
        nearest_vru = min(
            core.as_float(status.get("nearest_walker_m"), 80.0),
            core.as_float(status.get("nearest_bike_m"), 80.0),
        )
        light = str(status.get("traffic_light", "none")).lower()
        if light in ("red", "yellow") or nearest_vru <= 12.0:
            justified += 1
        elif not core.overtake_escape_available(status):
            justified += 1
    return waiting >= 8 and justified / max(1, waiting) >= 0.65


def episode_category(
    metrics: Dict[str, float],
    rows: List[Dict[str, Any]],
    rollout: Dict[str, Any],
    collision_event: Optional[Dict[str, Any]],
) -> str:
    collisions = core.as_float(metrics.get("collisions"), 0.0)
    if collision_event is not None or collisions > 0.0:
        return "collision"
    unsafe = sum(
        core.as_float(metrics.get(key), 0.0)
        for key in ("offroad", "red_lights", "stop_infractions")
    )
    if unsafe > 0.0:
        return "unsafe_failure"
    route = core.as_float(metrics.get("route_score"), 0.0)
    blocked = core.as_float(metrics.get("blocked"), 0.0)
    incomplete = core.as_float(metrics.get("incomplete"), 0.0)
    clean_pass = core.as_float((rollout.get("reward_parts") or {}).get("clean_pass"), 0.0)
    if route >= 95.0 and blocked == 0.0 and incomplete == 0.0:
        return "clean_success"
    if waiting_is_justified(rows):
        return "justified_wait"
    return "partial_progress"


def sample_indices(
    rollout: Dict[str, np.ndarray],
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(rollout["states"].shape[0])
    if maximum <= 0 or n <= maximum:
        return np.arange(n, dtype=np.int64)
    risk = np.asarray(rollout["risk_targets"], dtype=np.float32)
    intervention = np.asarray(rollout["intervention_strengths"], dtype=np.float32)
    important = np.flatnonzero((risk >= 0.35) | (intervention >= 0.20))
    # Always retain the terminal transition and a representative temporal grid.
    important = np.unique(np.concatenate([important, np.asarray([n - 1])]))
    if important.size >= maximum:
        selected = np.linspace(0, important.size - 1, maximum, dtype=int)
        return np.sort(important[selected])
    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), important, assume_unique=False)
    needed = maximum - important.size
    if remaining.size > needed:
        selected = np.linspace(0, remaining.size - 1, needed, dtype=int)
        remaining = remaining[selected]
    result = np.unique(np.concatenate([important, remaining]))
    if result.size > maximum:
        result = np.sort(rng.choice(result, size=maximum, replace=False))
    return np.sort(result)


def load_episode(
    entry: Dict[str, Any],
    policy_state_dim: int,
    world_state_dim: int,
    policy_semantics: str,
) -> Dict[str, Any]:
    trace = Path(entry["trace"]).expanduser().resolve()
    if not trace.exists():
        raise FileNotFoundError(trace)
    metrics = {
        str(key): core.as_float(value)
        for key, value in (entry.get("metrics") or {}).items()
        if isinstance(value, (int, float, str, bool))
    }
    rows = core.read_jsonl(trace)
    original_rows = len(rows)
    core.enrich_current_oncoming(rows)
    events_path = Path(entry["collision_events"]).expanduser().resolve() if entry.get("collision_events") else None
    collision_events = core.read_collision_events(events_path)
    rows, collision_event = core.truncate_rows_at_first_collision(rows, collision_events, metrics)
    rollout = core.build_rollout(
        rows,
        policy_state_dim,
        world_state_dim,
        metrics,
        policy_input_semantics=policy_semantics,
        collision_event=collision_event,
    )
    category = episode_category(metrics, rows, rollout, collision_event)
    rollout["raw_rewards"] = np.asarray(rollout["rewards"], dtype=np.float32).copy()
    rollout["rewards"] = bounded_rewards(rollout["rewards"], category, metrics)
    return {
        "entry": entry,
        "trace": str(trace),
        "rows": rows,
        "original_rows": original_rows,
        "collision_event": collision_event,
        "metrics": metrics,
        "category": category,
        "rollout": rollout,
    }


def decode_policy_mean(
    mean: torch.Tensor,
    base_action: torch.Tensor,
    semantics: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    steering = torch.tanh(mean[..., 0:1])
    if semantics == "simlingo_signed_longitudinal_target_with_learned_gate_v3":
        signed = torch.tanh(mean[..., 1:2] - mean[..., 2:3])
        throttle = torch.relu(signed)
        brake = torch.relu(-signed)
    else:
        throttle = torch.sigmoid(mean[..., 1:2])
        brake = torch.sigmoid(mean[..., 2:3])
    gate = torch.sigmoid(mean[..., 3:4])
    target = torch.cat([steering, throttle, brake, gate], dim=-1)
    base_longitudinal = base_action[..., 1:2] - base_action[..., 2:3]
    target_longitudinal = throttle - brake
    blended_longitudinal = base_longitudinal + gate * (target_longitudinal - base_longitudinal)
    chosen = torch.cat(
        [
            base_action[..., 0:1] + gate * (steering - base_action[..., 0:1]),
            torch.relu(blended_longitudinal),
            torch.relu(-blended_longitudinal),
            gate,
        ],
        dim=-1,
    )
    return target, chosen


def imagined_actor_loss(
    policy: core.ActorCritic,
    reference_policy: core.ActorCritic,
    world_model: core.WorldModel,
    raw_policy_states: torch.Tensor,
    raw_world_states: torch.Tensor,
    policy_mean: torch.Tensor,
    policy_std: torch.Tensor,
    wm_state_mean: torch.Tensor,
    wm_state_std: torch.Tensor,
    wm_action_mean: torch.Tensor,
    wm_action_std: torch.Tensor,
    wm_progress_mean: torch.Tensor,
    wm_progress_std: torch.Tensor,
    semantics: str,
    horizon: int,
    starts: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    count = int(raw_policy_states.shape[0])
    if count == 0 or horizon <= 0 or starts <= 0:
        zero = next(policy.parameters()).sum() * 0.0
        return zero, {"starts": 0, "horizon": 0, "risk": 0.0, "progress": 0.0}
    chosen_indices = rng.choice(count, size=min(starts, count), replace=False)
    idx = torch.as_tensor(chosen_indices, dtype=torch.long, device=raw_policy_states.device)
    observation = raw_policy_states[idx].clone()
    world_state = raw_world_states[idx].clone()
    minimum = raw_world_states.amin(dim=0) - 0.10 * raw_world_states.std(dim=0).clamp_min(1.0)
    maximum = raw_world_states.amax(dim=0) + 0.10 * raw_world_states.std(dim=0).clamp_min(1.0)
    previous_target: Optional[torch.Tensor] = None
    total = next(policy.parameters()).sum() * 0.0
    risk_values: List[float] = []
    progress_values: List[float] = []

    wm_requires_grad = [parameter.requires_grad for parameter in world_model.parameters()]
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)
    try:
        for step in range(horizon):
            normalized = (observation - policy_mean.reshape(1, -1)) / policy_std.reshape(1, -1)
            mean, _, _ = policy(normalized)
            with torch.no_grad():
                reference_mean, _, _ = reference_policy(normalized)
            base_offset = world_model.state_dim
            base = observation[:, base_offset:base_offset + 3].clamp(-1.0, 1.0)
            target, chosen = decode_policy_mean(mean, base, semantics)
            normalized_world = (
                world_state - wm_state_mean.reshape(1, -1)
            ) / wm_state_std.reshape(1, -1)
            normalized_action = (
                chosen - wm_action_mean.reshape(1, -1)
            ) / wm_action_std.reshape(1, -1)
            next_state_normalized, risk, progress_normalized = world_model(
                normalized_world, normalized_action
            )
            next_state = (
                next_state_normalized * wm_state_std.reshape(1, -1)
                + wm_state_mean.reshape(1, -1)
            )
            progress = (
                progress_normalized * wm_progress_std.reshape(1, -1)
                + wm_progress_mean.reshape(1, -1)
            )
            next_state = torch.maximum(torch.minimum(next_state, maximum), minimum)
            risk = torch.nan_to_num(risk, nan=1.0, posinf=1.0, neginf=0.0)
            progress = torch.nan_to_num(progress, nan=0.0, posinf=0.0, neginf=0.0)
            smooth = torch.zeros_like(risk)
            if previous_target is not None:
                smooth = (target[:, :3] - previous_target[:, :3]).square().mean(dim=-1, keepdim=True)
            trust = (mean - reference_mean).square().mean(dim=-1, keepdim=True)
            gate_cost = target[:, 3:4]
            imagined_cost = (
                2.40 * risk
                - 0.55 * progress.clamp(0.0, 3.0)
                + 0.08 * smooth
                + 0.025 * gate_cost
                + 0.10 * trust
            ).mean()
            total = total + (0.92 ** step) * imagined_cost
            risk_values.append(float(risk.detach().mean().item()))
            progress_values.append(float(progress.detach().mean().item()))

            observation = observation.clone()
            observation[:, :world_model.state_dim] = next_state
            if semantics in (
                core.MAP_INVARIANT_POLICY_INPUT_SEMANTICS,
                core.MAP_INVARIANT_CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS,
            ):
                for column in core.MAP_INVARIANT_WORLD_STATE_INDICES:
                    if column < world_model.state_dim:
                        observation[:, column] = 0.0
            if observation.shape[1] >= world_model.state_dim + 18:
                observation[:, -4:] = target
            world_state = next_state
            previous_target = target
    finally:
        for parameter, enabled in zip(world_model.parameters(), wm_requires_grad):
            parameter.requires_grad_(enabled)
    return total / max(1, horizon), {
        "starts": int(len(chosen_indices)),
        "horizon": int(horizon),
        "risk": float(np.mean(risk_values)) if risk_values else 0.0,
        "progress": float(np.mean(progress_values)) if progress_values else 0.0,
    }


def validate_world_model(
    world_model: core.WorldModel,
    tensors: Dict[str, torch.Tensor],
    holdout_episode: int,
    wm_state_mean: torch.Tensor,
    wm_state_std: torch.Tensor,
    wm_action_mean: torch.Tensor,
    wm_action_std: torch.Tensor,
    wm_progress_mean: torch.Tensor,
    wm_progress_std: torch.Tensor,
    *,
    max_state_mae: float,
    max_risk_mae: float,
    max_progress_mae: float,
) -> Dict[str, Any]:
    """Validate imagined dynamics on one episode excluded from WM fitting."""
    holdout = tensors["episode_ids"] == int(holdout_episode)
    training = ~holdout
    if not bool(holdout.any()) or not bool(training.any()):
        return {
            "reliable": False,
            "reason": "world-model holdout split is empty",
            "holdout_episode": int(holdout_episode),
        }

    with torch.no_grad():
        predicted_state_normalized, predicted_risk, predicted_progress_normalized = world_model(
            tensors["wm_states_normalized"][holdout],
            tensors["wm_actions_normalized"][holdout],
        )
        predicted_state = (
            predicted_state_normalized * wm_state_std.reshape(1, -1)
            + wm_state_mean.reshape(1, -1)
        )
        predicted_progress = (
            predicted_progress_normalized * wm_progress_std.reshape(1, -1)
            + wm_progress_mean.reshape(1, -1)
        )
        state_mae = (
            predicted_state - tensors["wm_next_states"][holdout]
        ).abs().mean()
        persistence_mae = (
            tensors["wm_states"][holdout] - tensors["wm_next_states"][holdout]
        ).abs().mean().clamp_min(0.05)
        state_error_ratio = state_mae / persistence_mae
        risk_mae = (
            predicted_risk.squeeze(-1) - tensors["risk_targets"][holdout]
        ).abs().mean()
        progress_mae = (
            predicted_progress.squeeze(-1) - tensors["progress_targets"][holdout]
        ).abs().mean()
        zero_progress_mae = tensors["progress_targets"][holdout].abs().mean().clamp_min(0.05)
        progress_error_ratio = progress_mae / zero_progress_mae

    finite = all(
        math.isfinite(float(value.item()))
        for value in (state_error_ratio, risk_mae, progress_error_ratio)
    )
    reliable = bool(
        finite
        and float(state_mae.item()) <= max_state_mae
        and float(risk_mae.item()) <= max_risk_mae
        and float(progress_mae.item()) <= max_progress_mae
    )
    return {
        "reliable": reliable,
        "reason": "validated" if reliable else "held-out prediction error exceeds threshold",
        "holdout_episode": int(holdout_episode),
        "holdout_transitions": int(holdout.sum().item()),
        "training_transitions": int(training.sum().item()),
        "state_mae": float(state_mae.item()),
        "persistence_state_mae": float(persistence_mae.item()),
        "state_error_ratio": float(state_error_ratio.item()),
        "risk_mae": float(risk_mae.item()),
        "progress_mae": float(progress_mae.item()),
        "zero_progress_mae": float(zero_progress_mae.item()),
        "progress_error_ratio": float(progress_error_ratio.item()),
        "thresholds": {
            "state_mae": float(max_state_mae),
            "risk_mae": float(max_risk_mae),
            "progress_mae": float(max_progress_mae),
        },
    }


def infer_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-transitions-per-episode", type=int, default=768)
    parser.add_argument("--min-transitions-per-episode", type=int, default=48)
    parser.add_argument("--lr-policy", type=float, default=5e-5)
    parser.add_argument("--lr-world-model", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--value-coef", type=float, default=0.35)
    parser.add_argument("--trust-coef", type=float, default=0.025)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--imagination-coef", type=float, default=0.035)
    parser.add_argument("--imagination-horizon", type=int, default=3)
    parser.add_argument("--imagination-starts", type=int, default=128)
    parser.add_argument("--wm-max-state-mae", type=float, default=0.25)
    parser.add_argument("--wm-max-risk-mae", type=float, default=0.30)
    parser.add_argument("--wm-max-progress-mae", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = (args.output_checkpoint or args.checkpoint).expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    manifest = load_json(manifest_path)
    expected_hash = str(manifest.get("frozen_checkpoint_sha256", ""))
    actual_hash = sha256(checkpoint_path)
    if not expected_hash or expected_hash != actual_hash:
        raise RuntimeError(
            f"frozen checkpoint mismatch: manifest={expected_hash or '<missing>'}, actual={actual_hash}"
        )
    entries = list(manifest.get("episodes") or [])
    if len(entries) < 2:
        raise RuntimeError("a PPO batch requires at least two episodes collected before the update")
    mismatched = [
        entry for entry in entries
        if str(entry.get("checkpoint_sha256", expected_hash)) != expected_hash
    ]
    if mismatched:
        raise RuntimeError("batch contains episodes collected by different policy versions")

    device = infer_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint, migration = core.upgrade_policy_observation_checkpoint(checkpoint)
    policy, world_model = core.infer_models(checkpoint, device)
    reference_policy = copy.deepcopy(policy).eval()
    for parameter in reference_policy.parameters():
        parameter.requires_grad_(False)
    policy_mean, policy_std = core.policy_normalizer_from_checkpoint(
        checkpoint, policy.state_dim, device
    )
    policy_semantics = str(checkpoint.get("policy_input_semantics", "world_state_v1"))
    action_semantics = str(
        checkpoint.get(
            "policy_action_semantics",
            "simlingo_signed_longitudinal_target_with_learned_gate_v3",
        )
    )

    episodes: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for entry in entries:
        episode = load_episode(
            entry,
            policy.state_dim,
            world_model.state_dim,
            policy_semantics,
        )
        transitions = int(episode["rollout"]["states"].shape[0])
        if transitions < args.min_transitions_per_episode:
            rejected.append({
                "trace": episode["trace"],
                "reason": f"{transitions} transitions < {args.min_transitions_per_episode}",
            })
            continue
        episodes.append(episode)
    if len(episodes) < 2:
        raise RuntimeError(f"only {len(episodes)} usable episodes remain after validation")

    assembled: Dict[str, List[np.ndarray]] = {
        key: [] for key in (
            "states", "next_states", "policy_states", "raw_actions",
            "chosen_actions", "old_log_probs", "advantages", "returns",
            "risk_targets", "progress_targets", "weights",
        )
    }
    episode_id_rows: List[np.ndarray] = []
    category_counts = Counter(episode["category"] for episode in episodes)
    episode_summaries: List[Dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes):
        rollout = episode["rollout"]
        values = np.clip(np.asarray(rollout["values"], dtype=np.float32), -12.0, 12.0)
        advantages, returns = core.compute_gae(
            rollout["rewards"], values, args.gamma, args.lam
        )
        returns = np.clip(returns, -20.0, 20.0).astype(np.float32)
        indices = sample_indices(rollout, args.max_transitions_per_episode, rng)
        category = episode["category"]
        # Each observed outcome family has comparable total mass, irrespective
        # of how many episodes of that family happened to enter this batch.
        contribution = CATEGORY_WEIGHTS[category] / max(1, category_counts[category])
        weights = np.full(indices.shape[0], contribution / max(1, indices.shape[0]), dtype=np.float32)
        for source, target in (
            ("states", "states"),
            ("next_states", "next_states"),
            ("policy_states", "policy_states"),
            ("raw_actions", "raw_actions"),
            ("chosen_actions", "chosen_actions"),
            ("log_probs", "old_log_probs"),
            ("risk_targets", "risk_targets"),
            ("progress_targets", "progress_targets"),
        ):
            assembled[target].append(np.asarray(rollout[source])[indices])
        assembled["advantages"].append(advantages[indices])
        assembled["returns"].append(returns[indices])
        assembled["weights"].append(weights)
        episode_id_rows.append(
            np.full(indices.shape[0], episode_index, dtype=np.int64)
        )
        episode_summaries.append({
            "trace": episode["trace"],
            "route_id": episode["entry"].get("route_id"),
            "seed": episode["entry"].get("seed"),
            "stage": episode["entry"].get("stage"),
            "category": category,
            "transitions": int(rollout["states"].shape[0]),
            "sampled_transitions": int(indices.shape[0]),
            "raw_reward_sum": float(np.sum(rollout["raw_rewards"])),
            "bounded_reward_sum": float(np.sum(rollout["rewards"])),
            "reward_parts": rollout.get("reward_parts") or {},
            "metrics": episode["metrics"],
            "collision_event": episode["collision_event"],
            "original_rows": episode["original_rows"],
            "used_rows": len(episode["rows"]),
        })

    arrays = {key: np.concatenate(value, axis=0) for key, value in assembled.items()}
    arrays["wm_states"] = arrays["states"].copy()
    arrays["wm_next_states"] = arrays["next_states"].copy()
    for column in core.MAP_INVARIANT_WORLD_STATE_INDICES:
        if column < arrays["wm_states"].shape[1]:
            arrays["wm_states"][:, column] = 0.0
            arrays["wm_next_states"][:, column] = 0.0
    arrays["weights"] = arrays["weights"] / max(float(arrays["weights"].mean()), 1e-8)
    arrays["advantages"] = (
        arrays["advantages"] - arrays["advantages"].mean()
    ) / (arrays["advantages"].std() + 1e-8)
    tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in arrays.items()
    }
    tensors["episode_ids"] = torch.as_tensor(
        np.concatenate(episode_id_rows, axis=0), dtype=torch.long, device=device
    )
    normalizer_is_initialized = bool(
        checkpoint.get("world_model_state_semantics")
        == "map_invariant_normalized_one_step_v2"
    )
    if normalizer_is_initialized:
        state_mean_np = np.asarray(checkpoint.get("state_mean"), dtype=np.float32)
        state_std_np = np.asarray(checkpoint.get("state_std"), dtype=np.float32)
        action_mean_np = np.asarray(checkpoint.get("action_mean"), dtype=np.float32)
        action_std_np = np.asarray(checkpoint.get("action_std"), dtype=np.float32)
        progress_mean_np = np.asarray(checkpoint.get("progress_mean"), dtype=np.float32).reshape(-1)
        progress_std_np = np.asarray(checkpoint.get("progress_std"), dtype=np.float32).reshape(-1)
    else:
        state_mean_np = arrays["wm_states"].mean(axis=0).astype(np.float32)
        state_std_np = arrays["wm_states"].std(axis=0).clip(0.25).astype(np.float32)
        action_mean_np = arrays["chosen_actions"].mean(axis=0).astype(np.float32)
        action_std_np = arrays["chosen_actions"].std(axis=0).clip(0.10).astype(np.float32)
        progress_mean_np = np.asarray([arrays["progress_targets"].mean()], dtype=np.float32)
        progress_std_np = np.asarray([
            max(0.05, float(arrays["progress_targets"].std()))
        ], dtype=np.float32)
        for column in core.MAP_INVARIANT_WORLD_STATE_INDICES:
            if column < state_mean_np.shape[0]:
                state_mean_np[column] = 0.0
                state_std_np[column] = 1.0

    wm_state_mean = torch.as_tensor(state_mean_np, dtype=torch.float32, device=device)
    wm_state_std = torch.as_tensor(state_std_np, dtype=torch.float32, device=device).clamp_min(1e-6)
    wm_action_mean = torch.as_tensor(action_mean_np, dtype=torch.float32, device=device)
    wm_action_std = torch.as_tensor(action_std_np, dtype=torch.float32, device=device).clamp_min(1e-6)
    wm_progress_mean = torch.as_tensor(progress_mean_np, dtype=torch.float32, device=device)
    wm_progress_std = torch.as_tensor(progress_std_np, dtype=torch.float32, device=device).clamp_min(1e-6)
    tensors["wm_states_normalized"] = (
        tensors["wm_states"] - wm_state_mean.reshape(1, -1)
    ) / wm_state_std.reshape(1, -1)
    tensors["wm_next_states_normalized"] = (
        tensors["wm_next_states"] - wm_state_mean.reshape(1, -1)
    ) / wm_state_std.reshape(1, -1)
    tensors["wm_actions_normalized"] = (
        tensors["chosen_actions"] - wm_action_mean.reshape(1, -1)
    ) / wm_action_std.reshape(1, -1)
    tensors["wm_progress_targets_normalized"] = (
        tensors["progress_targets"] - wm_progress_mean.reshape(1)
    ) / wm_progress_std.reshape(1)
    tensors["normalized_policy_states"] = (
        tensors["policy_states"] - policy_mean.reshape(1, -1)
    ) / policy_std.reshape(1, -1)

    optimizer_pi = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    optimizer_wm = torch.optim.Adam(world_model.parameters(), lr=args.lr_world_model)
    policy_losses: List[float] = []
    value_losses: List[float] = []
    world_losses: List[float] = []
    trust_losses: List[float] = []
    entropies: List[float] = []
    n = int(tensors["states"].shape[0])
    wm_holdout_episode = int(args.seed % len(episodes))
    for _ in range(max(1, args.epochs)):
        for indices in core.minibatches(n, max(1, args.batch_size), rng):
            idx = torch.as_tensor(indices, dtype=torch.long, device=device)
            weights = tensors["weights"][idx]
            weights = weights / weights.mean().clamp_min(1e-8)
            log_prob, entropy, value = policy.evaluate(
                tensors["normalized_policy_states"][idx], tensors["raw_actions"][idx]
            )
            ratio = torch.exp(
                torch.clamp(log_prob - tensors["old_log_probs"][idx], -8.0, 8.0)
            )
            surrogate = torch.minimum(
                ratio * tensors["advantages"][idx],
                ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps)
                * tensors["advantages"][idx],
            )
            actor_loss = -(weights * surrogate).mean()
            value_per_row = nn.functional.smooth_l1_loss(
                value, tensors["returns"][idx], reduction="none"
            )
            value_loss = (weights * value_per_row).mean()
            mean, _, _ = policy(tensors["normalized_policy_states"][idx])
            with torch.no_grad():
                reference_mean, _, _ = reference_policy(tensors["normalized_policy_states"][idx])
            trust_loss = nn.functional.smooth_l1_loss(mean, reference_mean)
            loss = (
                actor_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy.mean()
                + args.trust_coef * trust_loss
            )
            optimizer_pi.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer_pi.step()

            wm_idx = idx[tensors["episode_ids"][idx] != wm_holdout_episode]
            if wm_idx.numel() == 0:
                continue
            predicted_state, predicted_risk, predicted_progress = world_model(
                tensors["wm_states_normalized"][wm_idx],
                tensors["wm_actions_normalized"][wm_idx],
            )
            wm_state = nn.functional.smooth_l1_loss(
                predicted_state, tensors["wm_next_states_normalized"][wm_idx]
            )
            wm_risk = nn.functional.binary_cross_entropy(
                predicted_risk.clamp(1e-5, 1.0 - 1e-5),
                tensors["risk_targets"][wm_idx].unsqueeze(-1),
            )
            wm_progress = nn.functional.smooth_l1_loss(
                predicted_progress,
                tensors["wm_progress_targets_normalized"][wm_idx].unsqueeze(-1),
            )
            wm_loss = wm_state + 1.5 * wm_risk + 0.75 * wm_progress
            optimizer_wm.zero_grad()
            wm_loss.backward()
            nn.utils.clip_grad_norm_(world_model.parameters(), args.max_grad_norm)
            optimizer_wm.step()

            policy_losses.append(float(actor_loss.item()))
            value_losses.append(float(value_loss.item()))
            world_losses.append(float(wm_loss.item()))
            trust_losses.append(float(trust_loss.item()))
            entropies.append(float(entropy.mean().item()))

    world_model_validation = validate_world_model(
        world_model,
        tensors,
        wm_holdout_episode,
        wm_state_mean,
        wm_state_std,
        wm_action_mean,
        wm_action_std,
        wm_progress_mean,
        wm_progress_std,
        max_state_mae=args.wm_max_state_mae,
        max_risk_mae=args.wm_max_risk_mae,
        max_progress_mae=args.wm_max_progress_mae,
    )
    if world_model_validation["reliable"]:
        imagination_loss, imagination = imagined_actor_loss(
            policy,
            reference_policy,
            world_model,
            tensors["policy_states"],
            tensors["wm_states"],
            policy_mean,
            policy_std,
            wm_state_mean,
            wm_state_std,
            wm_action_mean,
            wm_action_std,
            wm_progress_mean,
            wm_progress_std,
            action_semantics,
            args.imagination_horizon,
            args.imagination_starts,
            rng,
        )
        if torch.isfinite(imagination_loss) and args.imagination_coef > 0.0:
            optimizer_pi.zero_grad()
            (args.imagination_coef * imagination_loss).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer_pi.step()
            imagination["loss"] = float(imagination_loss.detach().item())
            imagination["coefficient"] = float(args.imagination_coef)
            imagination["applied"] = True
        else:
            imagination.update({
                "loss": None,
                "coefficient": float(args.imagination_coef),
                "applied": False,
                "reason": "non-finite loss or zero coefficient",
            })
    else:
        imagination = {
            "starts": 0,
            "horizon": int(args.imagination_horizon),
            "loss": None,
            "coefficient": float(args.imagination_coef),
            "applied": False,
            "reason": world_model_validation["reason"],
        }

    saved = {
        **{
            key: value for key, value in checkpoint.items()
            if key not in ("policy", "world_model", "optimizer_pi", "optimizer_wm")
        },
        "policy": policy.state_dict(),
        "world_model": world_model.state_dict(),
        "optimizer_pi": optimizer_pi.state_dict(),
        "optimizer_wm": optimizer_wm.state_dict(),
        "policy_state_mean": policy_mean.detach().cpu().numpy(),
        "policy_state_std": policy_std.detach().cpu().numpy(),
        "state_mean": wm_state_mean.detach().cpu().numpy(),
        "state_std": wm_state_std.detach().cpu().numpy(),
        "action_mean": wm_action_mean.detach().cpu().numpy(),
        "action_std": wm_action_std.detach().cpu().numpy(),
        "progress_mean": wm_progress_mean.detach().cpu().numpy(),
        "progress_std": wm_progress_std.detach().cpu().numpy(),
        "world_model_state_semantics": "map_invariant_normalized_one_step_v2",
        "episode": int(checkpoint.get("episode", 0)) + len(episodes),
        "online_rl_update_count": int(checkpoint.get("online_rl_update_count", 0)) + 1,
        "policy_role": "candidate_online_rl_complement_to_simlingo",
        "reward_schema": REWARD_SCHEMA,
        "batch_training": {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": str(manifest_path),
            "frozen_parent_sha256": actual_hash,
            "episodes": len(episodes),
            "category_counts": dict(category_counts),
            "transitions": n,
            "reward_schema": REWARD_SCHEMA,
            "no_guard": True,
            "complement_to_simlingo": True,
            "policy_observation_migration": migration,
            "world_model_validation": world_model_validation,
            "imagination": imagination,
        },
    }
    atomic_torch_save(output_path, saved)
    output_hash = sha256(output_path)
    summary = {
        "status": "updated_candidate",
        "checkpoint": str(output_path),
        "input_sha256": actual_hash,
        "output_sha256": output_hash,
        "device": str(device),
        "episodes": episode_summaries,
        "rejected": rejected,
        "category_counts": dict(category_counts),
        "transitions": n,
        "reward_schema": REWARD_SCHEMA,
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "world_model_loss": float(np.mean(world_losses)) if world_losses else 0.0,
        "world_model_validation": world_model_validation,
        "trust_loss": float(np.mean(trust_losses)) if trust_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "imagination": imagination,
        "no_guard": True,
        "complement_to_simlingo": True,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
