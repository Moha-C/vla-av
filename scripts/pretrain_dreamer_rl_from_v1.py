#!/usr/bin/env python3
"""Bootstrap the no-guard SimLingo complement from clean Dreamer-v1 runs.

The guarded controller is used only as an offline teacher. The exported policy
receives observations and SimLingo's proposed control, then predicts a target
control plus a learned intervention gate. No guard code is enabled at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from scripts.dreamer_online_rl_update import (
    ActorCritic,
    MAP_INVARIANT_POLICY_INPUT_SEMANTICS,
    action_dict,
    default_policy_state_scale,
    policy_state_from_status,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_INPUT_SEMANTICS = MAP_INVARIANT_POLICY_INPUT_SEMANTICS
POLICY_ACTION_SEMANTICS = "simlingo_signed_longitudinal_target_with_learned_gate_v3"
WORLD_STATE_DIM = 28
COMPACT_STATE_DIM = 42
POLICY_STATE_DIM = 46
CRITICAL_INFRACTIONS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
)


def atomic_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    temporary.replace(path)


def clean_result(result_path: Path) -> Tuple[bool, str]:
    if not result_path.exists():
        return False, "missing Bench2Drive result"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid result: {exc}"
    records = ((payload.get("_checkpoint") or {}).get("records") or [])
    if payload.get("entry_status") != "Finished" or not payload.get("eligible") or len(records) != 1:
        return False, "run is incomplete or ineligible"
    record = records[0]
    scores = record.get("scores") or {}
    if record.get("status") != "Completed" or float(scores.get("score_route", 0.0)) < 99.99:
        return False, "route was not completed"
    if float(scores.get("score_penalty", 0.0)) < 0.999:
        return False, "route contains a driving penalty"
    infractions = record.get("infractions") or {}
    bad = [name for name in CRITICAL_INFRACTIONS if infractions.get(name)]
    if bad:
        return False, "critical infractions: " + ", ".join(bad)
    return True, "clean completed route"


def sample_class(status: Dict[str, Any]) -> str:
    if not bool(status.get("applied")):
        return "defer"
    kind = str(status.get("chosen_kind", ""))
    if kind.startswith("recovery_"):
        return "recovery"
    if kind in ("collision_shield_hold", "hazard_hold", "hazard_strong_hold", "model_cautious"):
        return "safety"
    return "intervention"


def prior_validated_traces() -> Dict[Tuple[str, str, str], str]:
    validated: Dict[Tuple[str, str, str], str] = {}
    for summary_path in ROOT.glob("logs/dreamer_rl_distillation/*/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("status") != "saved":
            continue
        for run in summary.get("runs") or []:
            if not run.get("accepted") or run.get("reason") != "clean completed route":
                continue
            trace = Path(str(run.get("trace", ""))).expanduser()
            try:
                trace_key = str(trace.resolve())
            except Exception:
                continue
            key = (trace_key, str(run.get("route_id", "")), str(run.get("seed", "")))
            validated[key] = str(summary_path.resolve())
    return validated


def load_demonstrations(paths: List[Path], result_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    samples: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    seen = set()
    prior_audits = prior_validated_traces()
    for path in paths:
        route_id = ""
        seed = ""
        rows: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                status = row.get("status") or {}
                route_id = str(row.get("route_id") or route_id)
                seed = str(row.get("seed") or seed)
                if "dreamer_guard_v1" not in str(status.get("variant", "")):
                    continue
                if abs(float(row.get("collector_time", 0.0)) - float(status.get("timestamp", 0.0))) > 5.0:
                    continue
                timestamp = float(status.get("timestamp", 0.0))
                dedupe_key = (path, timestamp)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(row)
        result_path = result_dir / f"results_bench2drive_{route_id}_seed_{seed}.json"
        accepted, reason = clean_result(result_path)
        audit_source = ""
        if not accepted:
            audit_source = prior_audits.get((str(path.resolve()), route_id, seed), "")
            if audit_source:
                accepted = True
                reason = "clean completed route (immutable prior audit)"
        runs.append({
            "trace": str(path),
            "route_id": route_id,
            "seed": seed,
            "result": str(result_path),
            "accepted": accepted,
            "reason": reason,
            "audit_source": audit_source,
            "candidate_rows": len(rows),
        })
        if not accepted:
            continue
        previous_target: Optional[np.ndarray] = None
        for row in rows:
            status = row.get("status") or {}
            observation = policy_state_from_status(
                status,
                COMPACT_STATE_DIM,
                WORLD_STATE_DIM,
                policy_input_semantics=POLICY_INPUT_SEMANTICS,
            )
            state = status.get("state_vector")
            if observation is None or not isinstance(state, list) or len(state) < WORLD_STATE_DIM:
                continue
            base = action_dict(status, "base_action")
            chosen = action_dict(status, "chosen_action")
            applied = bool(status.get("applied"))
            target_control = chosen[:3] if applied else base[:3]
            target_gate = 0.995 if applied else 0.005
            sequence_start = previous_target is None
            if sequence_start:
                previous_target = np.asarray([*base[:3], 0.0], dtype=np.float32)
            observation = np.concatenate([observation, previous_target]).astype(np.float32)
            current_target = np.asarray([*target_control, target_gate], dtype=np.float32)
            category = sample_class(status)
            weight = {
                "recovery": 8.0,
                "safety": 1.0,
                "intervention": 1.5,
                "defer": 1.0,
            }[category]
            samples.append({
                "observation": observation,
                "target": current_target,
                "category": category,
                "weight": weight,
                "route_id": route_id,
                "sequence_start": sequence_start,
            })
            previous_target = current_target
    return samples, runs


def split_indices(
    samples: List[Dict[str, Any]],
    seed: int,
    validation_route: str = "",
) -> Tuple[List[int], List[int], str]:
    rng = random.Random(seed)
    by_route: Dict[str, List[int]] = {}
    for index, sample in enumerate(samples):
        by_route.setdefault(str(sample["route_id"]), []).append(index)
    if len(by_route) < 2:
        raise RuntimeError("route-held-out validation requires at least two clean routes")
    if validation_route:
        if validation_route not in by_route:
            raise RuntimeError(
                f"requested validation route {validation_route!r} is unavailable; "
                f"choices: {sorted(by_route)}"
            )
        held_out_route = validation_route
    else:
        held_out_route = min(by_route, key=lambda route: (len(by_route[route]), route))
    validation = list(by_route[held_out_route])
    train = [
        index
        for route, indices in by_route.items()
        if route != held_out_route
        for index in indices
    ]
    rng.shuffle(train)
    return train, validation, held_out_route


def expanded_policy(checkpoint: Dict[str, Any], device: torch.device) -> ActorCritic:
    old_state = checkpoint["policy"]
    hidden = int(old_state["trunk.0.weight"].shape[0])
    action_dim = int(old_state["log_std"].shape[0])
    policy = ActorCritic(POLICY_STATE_DIM, action_dim, hidden).to(device)
    new_state = policy.state_dict()
    for key, value in old_state.items():
        if key == "trunk.0.weight":
            columns = min(int(value.shape[1]), POLICY_STATE_DIM)
            new_state[key][:, :columns].copy_(value[:, :columns])
        elif key in new_state and tuple(new_state[key].shape) == tuple(value.shape):
            new_state[key].copy_(value)
    policy.load_state_dict(new_state)
    return policy


def predict_bounded(policy: ActorCritic, observations: torch.Tensor) -> torch.Tensor:
    mean, _, _ = policy(observations)
    steering = torch.tanh(mean[:, 0:1])
    longitudinal = torch.tanh(mean[:, 1:2] - mean[:, 2:3])
    throttle = torch.relu(longitudinal)
    brake = torch.relu(-longitudinal)
    gate = torch.sigmoid(mean[:, 3:4])
    return torch.cat([steering, throttle, brake, gate], dim=1)


@torch.no_grad()
def evaluate(
    policy: ActorCritic,
    observations: torch.Tensor,
    targets: torch.Tensor,
    categories: List[str],
) -> Dict[str, float]:
    policy.eval()
    predicted = predict_bounded(policy, observations)
    gate_pred = predicted[:, 3]
    gate_truth = targets[:, 3]
    applied_truth = gate_truth >= 0.5
    applied_pred = gate_pred >= 0.5
    recovery_mask = torch.as_tensor([name == "recovery" for name in categories], device=observations.device)
    defer_mask = torch.as_tensor([name == "defer" for name in categories], device=observations.device)
    active_mask = applied_truth
    result = {
        "samples": int(observations.shape[0]),
        "gate_accuracy": float((applied_pred == applied_truth).float().mean().item()),
        "control_mae_active": float(torch.abs(predicted[active_mask, :3] - targets[active_mask, :3]).mean().item()) if active_mask.any() else 0.0,
        "recovery_gate_recall": float(applied_pred[recovery_mask].float().mean().item()) if recovery_mask.any() else 0.0,
        "defer_specificity": float((~applied_pred[defer_mask]).float().mean().item()) if defer_mask.any() else 0.0,
        "mean_gate": float(gate_pred.mean().item()),
    }
    policy.train()
    return result


@torch.no_grad()
def evaluate_autoregressive(
    policy: ActorCritic,
    observations: torch.Tensor,
    targets: torch.Tensor,
    categories: List[str],
    sequence_starts: List[bool],
) -> Dict[str, float]:
    policy.eval()
    predicted_rows: List[torch.Tensor] = []
    previous: Optional[torch.Tensor] = None
    for index, sequence_start in enumerate(sequence_starts):
        observation = observations[index:index + 1].clone()
        if sequence_start:
            previous = None
        if previous is not None:
            observation[:, -4:] = previous.reshape(1, 4)
        predicted = predict_bounded(policy, observation)
        predicted_rows.append(predicted)
        previous = predicted[0].detach()
    predicted = torch.cat(predicted_rows, dim=0)
    gate_pred = predicted[:, 3]
    gate_truth = targets[:, 3]
    applied_truth = gate_truth >= 0.5
    applied_pred = gate_pred >= 0.5
    recovery_mask = torch.as_tensor(
        [name == "recovery" for name in categories], device=observations.device
    )
    defer_mask = torch.as_tensor(
        [name == "defer" for name in categories], device=observations.device
    )
    result = {
        "samples": int(observations.shape[0]),
        "gate_accuracy": float((applied_pred == applied_truth).float().mean().item()),
        "control_mae_active": float(
            torch.abs(predicted[applied_truth, :3] - targets[applied_truth, :3]).mean().item()
        ) if applied_truth.any() else 0.0,
        "recovery_gate_recall": float(applied_pred[recovery_mask].float().mean().item())
        if recovery_mask.any() else 0.0,
        "defer_specificity": float((~applied_pred[defer_mask]).float().mean().item())
        if defer_mask.any() else 0.0,
        "mean_gate": float(gate_pred.mean().item()),
    }
    policy.train()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--trace-glob", default="logs/action_dreaming_collect/*.jsonl")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "logs" / "simlingo_eval")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--validation-route",
        default="",
        help="Hold this complete route out of model selection (default: smallest clean route).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exploration-log-std", type=float, default=-1.8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = (args.output_checkpoint or checkpoint_path).expanduser().resolve()
    trace_paths = [path.expanduser().resolve() for path in args.trace]
    if not trace_paths:
        trace_paths = sorted(ROOT.glob(args.trace_glob))
    samples, runs = load_demonstrations(trace_paths, args.result_dir.expanduser().resolve())
    if len(samples) < 128:
        raise RuntimeError(f"not enough validated teacher samples: {len(samples)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy = expanded_policy(checkpoint, device)
    normalizer_mean = np.zeros(POLICY_STATE_DIM, dtype=np.float32)
    normalizer_std = default_policy_state_scale(POLICY_STATE_DIM)
    observations = np.stack([sample["observation"] for sample in samples]).astype(np.float32)
    observations = (observations - normalizer_mean) / np.maximum(normalizer_std, 1e-6)
    targets = np.stack([sample["target"] for sample in samples]).astype(np.float32)
    weights = np.asarray([sample["weight"] for sample in samples], dtype=np.float32)
    train_indices, validation_indices, held_out_route = split_indices(
        samples,
        args.seed,
        validation_route=str(args.validation_route),
    )

    observations_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    targets_t = torch.as_tensor(targets, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    rng = np.random.default_rng(args.seed)
    best_state = None
    best_score = -math.inf
    best_epoch = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        policy.train()
        shuffled = np.asarray(train_indices, dtype=np.int64)
        rng.shuffle(shuffled)
        epoch_losses: List[float] = []
        for start in range(0, len(shuffled), max(1, args.batch_size)):
            idx = torch.as_tensor(shuffled[start:start + args.batch_size], dtype=torch.long, device=device)
            predicted = predict_bounded(policy, observations_t[idx])
            target = targets_t[idx]
            sample_weight = weights_t[idx]
            active_weight = torch.where(target[:, 3] >= 0.5, sample_weight, sample_weight * 0.20)
            steering_loss = nn.functional.smooth_l1_loss(
                predicted[:, 0], target[:, 0], reduction="none"
            )
            longitudinal_loss = nn.functional.smooth_l1_loss(
                predicted[:, 1] - predicted[:, 2],
                target[:, 1] - target[:, 2],
                reduction="none",
            )
            control_per_sample = 0.75 * steering_loss + 1.25 * longitudinal_loss
            control_loss = (control_per_sample * active_weight).sum() / active_weight.sum().clamp_min(1e-6)
            gate_per_sample = nn.functional.binary_cross_entropy(
                predicted[:, 3].clamp(1e-5, 1.0 - 1e-5),
                target[:, 3],
                reduction="none",
            )
            gate_loss = (gate_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
            loss = 2.0 * control_loss + gate_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        validation = evaluate(
            policy,
            observations_t[validation_indices],
            targets_t[validation_indices],
            [samples[index]["category"] for index in validation_indices],
        )
        score = (
            validation["gate_accuracy"]
            + validation["recovery_gate_recall"]
            + validation["defer_specificity"]
            - 2.0 * validation["control_mae_active"]
        )
        history.append({"epoch": epoch, "loss": float(np.mean(epoch_losses)), **validation})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}

    if best_state is None:
        raise RuntimeError("distillation produced no policy")
    policy.load_state_dict(best_state)
    policy.log_std.data.fill_(float(args.exploration_log_std))
    validation = evaluate(
        policy,
        observations_t[validation_indices],
        targets_t[validation_indices],
        [samples[index]["category"] for index in validation_indices],
    )
    training = evaluate(
        policy,
        observations_t[train_indices],
        targets_t[train_indices],
        [samples[index]["category"] for index in train_indices],
    )
    heldout_autoregressive = evaluate_autoregressive(
        policy,
        observations_t[validation_indices],
        targets_t[validation_indices],
        [samples[index]["category"] for index in validation_indices],
        [bool(samples[index]["sequence_start"]) for index in validation_indices],
    )
    autoregressive = evaluate_autoregressive(
        policy,
        observations_t,
        targets_t,
        [sample["category"] for sample in samples],
        [bool(sample["sequence_start"]) for sample in samples],
    )
    accepted = (
        validation["gate_accuracy"] >= 0.80
        and validation["recovery_gate_recall"] >= 0.85
        and validation["defer_specificity"] >= 0.75
        and validation["control_mae_active"] <= 0.18
        and heldout_autoregressive["recovery_gate_recall"] >= 0.85
        and heldout_autoregressive["control_mae_active"] <= 0.20
    )
    if not accepted:
        raise RuntimeError(
            "distilled policy failed route-held-out validation gates: "
            f"route={held_out_route} frame={validation} temporal={heldout_autoregressive}"
        )

    # Model selection never sees the held-out route. Once its epoch budget is
    # validated, fit the production policy on every clean route for that fixed
    # number of epochs; no frame from the final fit influences model selection.
    policy = expanded_policy(checkpoint, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    final_rng = np.random.default_rng(args.seed + 1)
    all_indices = np.arange(len(samples), dtype=np.int64)
    for _ in range(best_epoch):
        final_rng.shuffle(all_indices)
        for start in range(0, len(all_indices), max(1, args.batch_size)):
            idx = torch.as_tensor(
                all_indices[start:start + args.batch_size],
                dtype=torch.long,
                device=device,
            )
            predicted = predict_bounded(policy, observations_t[idx])
            target = targets_t[idx]
            sample_weight = weights_t[idx]
            active_weight = torch.where(
                target[:, 3] >= 0.5,
                sample_weight,
                sample_weight * 0.20,
            )
            steering_loss = nn.functional.smooth_l1_loss(
                predicted[:, 0], target[:, 0], reduction="none"
            )
            longitudinal_loss = nn.functional.smooth_l1_loss(
                predicted[:, 1] - predicted[:, 2],
                target[:, 1] - target[:, 2],
                reduction="none",
            )
            control_per_sample = 0.75 * steering_loss + 1.25 * longitudinal_loss
            control_loss = (
                (control_per_sample * active_weight).sum()
                / active_weight.sum().clamp_min(1e-6)
            )
            gate_per_sample = nn.functional.binary_cross_entropy(
                predicted[:, 3].clamp(1e-5, 1.0 - 1e-5),
                target[:, 3],
                reduction="none",
            )
            gate_loss = (
                (gate_per_sample * sample_weight).sum()
                / sample_weight.sum().clamp_min(1e-6)
            )
            loss = 2.0 * control_loss + gate_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

    policy.log_std.data.fill_(float(args.exploration_log_std))
    production = evaluate(
        policy,
        observations_t,
        targets_t,
        [sample["category"] for sample in samples],
    )
    production_autoregressive = evaluate_autoregressive(
        policy,
        observations_t,
        targets_t,
        [sample["category"] for sample in samples],
        [bool(sample["sequence_start"]) for sample in samples],
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = output_path.parent / "rollback_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.copy2(output_path, backup_dir / f"{output_path.stem}_before_teacher_bootstrap_{stamp}{output_path.suffix}")
    exported = {
        **{key: value for key, value in checkpoint.items() if key not in ("policy", "optimizer_pi")},
        "episode": 0,
        "policy": policy.state_dict(),
        "policy_state_mean": normalizer_mean,
        "policy_state_std": normalizer_std,
        "policy_input_semantics": POLICY_INPUT_SEMANTICS,
        "policy_action_semantics": POLICY_ACTION_SEMANTICS,
        "policy_role": "online_rl_no_guard_temporal_complement_bootstrapped_from_clean_v1",
        "online_rl_update_count": 0,
        "distillation": {
            "status": "validated",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "teacher": "Dreamer v1 guarded clean runs",
            "temporal_memory": "previous learned target control and intervention gate",
            "runtime_guard": False,
            "samples": len(samples),
            "class_counts": dict(Counter(sample["category"] for sample in samples)),
            "runs": runs,
            "validation_protocol": "complete_route_holdout_then_fixed_epoch_all_route_fit",
            "held_out_route": held_out_route,
            "selected_epoch": best_epoch,
            "selection_training": training,
            "validation": validation,
            "heldout_autoregressive": heldout_autoregressive,
            "selection_all_autoregressive": autoregressive,
            "production_all_data": production,
            "production_autoregressive": production_autoregressive,
            "exploration_log_std": float(args.exploration_log_std),
        },
        "online_rl": {
            "status": "bootstrapped_from_clean_v1_teacher",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "no_guard": True,
            "complement_to_simlingo": True,
            "policy_input_semantics": POLICY_INPUT_SEMANTICS,
            "action_semantics": POLICY_ACTION_SEMANTICS,
            "samples": len(samples),
        },
    }
    atomic_save(output_path, exported)
    summary = {
        "status": "saved",
        "checkpoint": str(output_path),
        "device": str(device),
        "samples": len(samples),
        "class_counts": dict(Counter(sample["category"] for sample in samples)),
        "runs": runs,
        "validation_protocol": "complete_route_holdout_then_fixed_epoch_all_route_fit",
        "held_out_route": held_out_route,
        "selected_epoch": best_epoch,
        "selection_training": training,
        "validation": validation,
        "heldout_autoregressive": heldout_autoregressive,
        "selection_all_autoregressive": autoregressive,
        "production_all_data": production,
        "production_autoregressive": production_autoregressive,
        "history_tail": history[-5:],
        "policy_input_semantics": POLICY_INPUT_SEMANTICS,
        "policy_action_semantics": POLICY_ACTION_SEMANTICS,
        "runtime_guard": False,
    }
    summary_path = args.summary or (
        ROOT / "logs" / "dreamer_rl_distillation" / stamp / "summary.json"
    )
    summary_path = summary_path.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
