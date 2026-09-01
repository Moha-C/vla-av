#!/usr/bin/env python3
"""Train an isolated, collision-aware RSSM V4 world-model candidate.

V4 fixes the two invalid assumptions found in the first RSSM experiment:

* impacts are aligned to their exact Bench2Drive sensor timestamp;
* validation is seed-held-out and must include real collision windows.

This script only produces ``world_model_candidate.pt``.  A separately validated
RSSM-conditioned actor must be fitted before closed-loop testing, and production
checkpoints are never overwritten here.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from external.simlingo.team_code.dreamer_world_models import (
    RSSMConfig,
    RSSMState,
    TemporalRSSMWorldModel,
    symlog,
    symexp,
)
from scripts import dreamer_online_rl_update as core
from scripts import dreamer_rssm_data_v4 as data_v4
from scripts import train_dreamer_rssm_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "external/simlingo/checkpoints/dreamer_ppo_rl_noguard/production_model.pt"
)
DEFAULT_OUTPUT = ROOT / "external/simlingo/checkpoints/dreamer_ppo_rssm_v4"


class SafetySequenceWindows(v2.SequenceWindows):
    """Balance clean motion, blocked traffic, and exact pre-impact windows."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        balanced: List[float] = []
        for episode_index, start in self.windows:
            episode = self.episodes[episode_index]
            stop = start + self.sequence_length
            collision = float(episode.events[start:stop, 0].max(initial=0.0))
            offroad = float(episode.events[start:stop, 1].max(initial=0.0))
            blocked = float(episode.events[start:stop, 4].max(initial=0.0))
            risk = float(episode.risks[start:stop].max(initial=0.0))
            weight = 1.0 + 4.0 * risk + 12.0 * collision + 7.0 * offroad
            weight += 2.0 * blocked
            if episode.source == "validated_guard_teacher":
                weight += 2.0
            elif episode.source == "curriculum_success":
                weight += 1.0
            balanced.append(weight)
        self.weights = balanced


def event_positive_weights(episodes: Sequence[v2.Episode]) -> torch.Tensor:
    labels = np.concatenate([episode.events for episode in episodes], axis=0)
    positive = labels.sum(axis=0)
    negative = labels.shape[0] - positive
    weights = negative / np.maximum(positive, 1.0)
    # Rare terminal events need attention, but an unbounded class weight makes
    # every imagined future look catastrophic.
    return torch.from_numpy(np.clip(weights, 1.0, 40.0).astype(np.float32))


def world_model_loss(
    model: TemporalRSSMWorldModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    event_pos_weight: torch.Tensor,
    overshoot_horizon: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)
    predictions = model.observe_sequence(
        observations, actions, deterministic=True
    )
    target_delta = observations[:, 1:] - observations[:, :-1]
    reward_target = symlog(batch["rewards"].to(device))
    progress_target = symlog(batch["progress"].to(device))
    continuation_target = batch["continuation"].to(device)
    risk_target = batch["risks"].to(device)
    event_target = batch["events"].to(device)

    observation_mask = model.observation_delta_mask.reshape(1, 1, -1)
    observation_loss = v2._weighted_observation_loss(
        predictions["observation_delta"], target_delta, observation_mask
    )
    rollout_loss = v2.overshooting_loss(
        model,
        predictions,
        observations,
        actions,
        overshoot_horizon,
        deterministic=True,
    )
    reward_loss = F.smooth_l1_loss(predictions["reward_symlog"], reward_target)
    continuation_loss = F.binary_cross_entropy_with_logits(
        predictions["continuation_logit"], continuation_target
    )
    risk_elements = F.binary_cross_entropy_with_logits(
        predictions["risk_logit"], risk_target, reduction="none"
    )
    risk_weights = 1.0 + 4.0 * risk_target + 4.0 * event_target[..., 0]
    risk_loss = (risk_elements * risk_weights).sum() / risk_weights.sum().clamp_min(1.0)
    progress_loss = F.smooth_l1_loss(
        predictions["progress_symlog"], progress_target
    )
    event_loss = F.binary_cross_entropy_with_logits(
        predictions["event_logits"],
        event_target,
        pos_weight=event_pos_weight.to(device),
    )
    kl = model.kl_loss(
        predictions["posterior_logits"], predictions["prior_logits"]
    )
    total = (
        observation_loss
        + 0.65 * rollout_loss
        + 0.45 * reward_loss
        + 0.15 * continuation_loss
        + 1.40 * risk_loss
        + 0.30 * progress_loss
        + 1.00 * event_loss
        + kl["loss"]
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "observation": float(observation_loss.detach().cpu()),
        "overshooting": float(rollout_loss.detach().cpu()),
        "reward": float(reward_loss.detach().cpu()),
        "continuation": float(continuation_loss.detach().cpu()),
        "risk": float(risk_loss.detach().cpu()),
        "progress": float(progress_loss.detach().cpu()),
        "events": float(event_loss.detach().cpu()),
        "kl_dynamic": float(kl["dynamic"].detach().cpu()),
        "kl_representation": float(kl["representation"].detach().cpu()),
    }


@torch.no_grad()
def dataset_loss(
    model: TemporalRSSMWorldModel,
    loader: DataLoader,
    device: torch.device,
    event_pos_weight: torch.Tensor,
    overshoot_horizon: int,
) -> Dict[str, float]:
    model.eval()
    rows: List[Dict[str, float]] = []
    for batch in loader:
        _, metrics = world_model_loss(
            model, batch, device, event_pos_weight, overshoot_horizon
        )
        rows.append(metrics)
    if not rows:
        return {"loss": math.inf}
    return {
        key: float(np.mean([row[key] for row in rows])) for key in rows[0]
    }


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives <= 0:
        return math.nan
    order = np.argsort(-scores)
    ordered = labels[order]
    true_positives = np.cumsum(ordered)
    ranks = np.arange(1, len(ordered) + 1)
    precision = true_positives / ranks
    return float((precision * ordered).sum() / positives)


@torch.no_grad()
def collision_horizon_metrics(
    model: TemporalRSSMWorldModel,
    episodes: Sequence[v2.Episode],
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    horizon: int = 10,
) -> Dict[str, float]:
    model.eval()
    labels: List[float] = []
    scores: List[float] = []
    risk_scores: List[float] = []
    for episode in episodes:
        observations, actions = v2.normalize_episode(
            episode,
            observation_mean,
            observation_std,
            action_mean,
            action_std,
        )
        observations = observations.to(device)
        actions = actions.to(device)
        posterior = model.observe_initial(observations[0:1], deterministic=True)
        states = [posterior.detach()]
        for step in range(episode.transitions):
            posterior, _ = model.obs_step(
                posterior,
                actions[step:step + 1],
                observations[step + 1:step + 2],
                deterministic=True,
            )
            states.append(posterior.detach())
        for start in range(episode.transitions):
            stop = min(episode.transitions, start + max(1, horizon))
            latent = states[start]
            event_probability: List[torch.Tensor] = []
            risk_probability: List[torch.Tensor] = []
            for offset in range(start, stop):
                latent, heads = model.imagine_step(
                    latent, actions[offset:offset + 1], deterministic=True
                )
                event_probability.append(torch.sigmoid(heads["event_logits"])[0, 0])
                risk_probability.append(torch.sigmoid(heads["risk_logit"])[0])
            labels.append(float(episode.events[start:stop, 0].max(initial=0.0)))
            scores.append(float(torch.stack(event_probability).max().cpu()))
            risk_scores.append(float(torch.stack(risk_probability).max().cpu()))
    label_array = np.asarray(labels, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32)
    risk_array = np.asarray(risk_scores, dtype=np.float32)
    threshold = 0.35
    predicted = score_array >= threshold
    positive = label_array >= 0.5
    tp = int(np.logical_and(predicted, positive).sum())
    fp = int(np.logical_and(predicted, ~positive).sum())
    fn = int(np.logical_and(~predicted, positive).sum())
    return {
        "horizon": int(horizon),
        "samples": int(len(labels)),
        "positive_windows": int(positive.sum()),
        "average_precision": average_precision(label_array, score_array),
        "brier": float(np.mean((score_array - label_array) ** 2)),
        "threshold": threshold,
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "positive_event_score_mean": (
            float(score_array[positive].mean()) if bool(positive.any()) else math.nan
        ),
        "negative_event_score_mean": (
            float(score_array[~positive].mean()) if bool((~positive).any()) else math.nan
        ),
        "positive_risk_mean": (
            float(risk_array[positive].mean()) if bool(positive.any()) else math.nan
        ),
        "negative_risk_mean": (
            float(risk_array[~positive].mean()) if bool((~positive).any()) else math.nan
        ),
    }


def quality_gate(
    validation: Dict[str, Any],
    collision: Dict[str, float],
) -> Tuple[bool, Dict[str, Any]]:
    h1 = validation.get("1") or {}
    h5 = validation.get("5") or {}
    h1_ego = (h1.get("families") or {}).get("ego") or {}
    h5_decision = (h5.get("families") or {}).get("decision") or {}
    checks = {
        "h1_ego_beats_persistence": float(
            h1_ego.get("persistence_ratio", math.inf)
        ) <= 1.10,
        "h5_decision_near_or_better_than_persistence": float(
            h5_decision.get("persistence_ratio", math.inf)
        ) <= 1.25,
        "h5_changed_decision_near_or_better_than_persistence": float(
            h5.get("changed_decision_persistence_ratio", math.inf)
        ) <= 1.30,
        "h5_risk_mae": float(h5.get("risk_mae", math.inf)) <= 0.20,
        "h5_event_brier": float(h5.get("event_brier", math.inf)) <= 0.10,
        "collision_validation_present": int(
            collision.get("positive_windows", 0)
        ) >= 2,
        "collision_average_precision": float(
            collision.get("average_precision", -math.inf)
        ) >= 0.10,
        "collision_recall": float(collision.get("recall", 0.0)) >= 0.40,
        "collision_risk_separation": float(
            collision.get("positive_risk_mean", -math.inf)
        ) > float(collision.get("negative_risk_mean", math.inf)),
    }
    return all(checks.values()), {
        "checks": checks,
        "thresholds": {
            "h1_ego_persistence_ratio_max": 1.10,
            "h5_decision_persistence_ratio_max": 1.25,
            "h5_changed_decision_persistence_ratio_max": 1.30,
            "h5_risk_mae_max": 0.20,
            "h5_event_brier_max": 0.10,
            "collision_average_precision_min": 0.10,
            "collision_recall_at_0_35_min": 0.40,
        },
    }


def arbitration(validation: Dict[str, Any]) -> Dict[str, Any]:
    h5 = validation.get("5") or validation.get("1") or {}
    risk_error = float(h5.get("risk_mae", 0.15))
    progress_error = float(h5.get("progress_mae_m", 0.15))
    temperature = float(np.clip(progress_error + 2.0 * risk_error, 0.12, 1.20))
    return {
        "objective": "collision_aware_closed_loop_rssm_complement_v4",
        "progress_weight": 1.0,
        "risk_weight": 2.5,
        "risk_curvature": float(np.clip(1.5 + 10.0 * risk_error, 2.0, 4.0)),
        "action_penalty": float(np.clip(0.10 + progress_error, 0.15, 0.35)),
        "candidate_commit_horizon": 1,
        "rollout_strategy": "first_candidate_then_closed_loop_actor_replanning_v1",
        "closed_loop_actor_rollout": True,
        "actor_sigma_shooting": True,
        "authority_mapping": "one_minus_exp_negative_positive_margin_over_temperature_v1",
        "authority_temperature": temperature,
        "actor_gate_role": "upper_bound_scaled_by_model_confidence",
        "hard_thresholds": False,
        "calibration_basis": {
            "validation_horizon": 5,
            "risk_mae": risk_error,
            "progress_mae_m": progress_error,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--overshoot-horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--trace-pattern", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    patterns = args.trace_pattern or [
        "logs/dreamer_curriculum/20260811_155716/training/*/trace.jsonl",
        "logs/action_dreaming_collect/*.jsonl",
        "logs/dreamer_online_rl/webapp_*/trace.jsonl",
        "logs/dreamer_online_rl/*/traces/*.jsonl",
        "logs/dreamer_rl_campaign/*/traces/*.jsonl",
    ]
    paths = v2.discover_traces(patterns)
    episodes, trace_audit = data_v4.load_episodes(paths, args.sequence_length)
    if not episodes:
        raise RuntimeError("no collision-aware ordered traces were found")
    training, validation_episodes, validation_groups = (
        data_v4.split_route_seed_stratified(episodes, args.seed)
    )

    source_path = args.source_checkpoint.expanduser().resolve()
    source = torch.load(source_path, map_location="cpu")
    source, observation_migration = core.upgrade_policy_observation_checkpoint(
        copy.deepcopy(source)
    )
    policy_mean = np.asarray(source["policy_state_mean"], dtype=np.float32)[
        :v2.OBSERVATION_DIM
    ]
    policy_std = np.maximum(
        np.asarray(source["policy_state_std"], dtype=np.float32)[
            :v2.OBSERVATION_DIM
        ],
        1e-6,
    )
    observation_mean, observation_std = v2.compute_world_observation_normalizer(
        training, policy_std
    )
    training_actions = np.concatenate(
        [episode.actions for episode in training], axis=0
    )
    action_mean = training_actions.mean(axis=0).astype(np.float32)
    action_std = np.maximum(training_actions.std(axis=0), 0.05).astype(np.float32)

    training_dataset = SafetySequenceWindows(
        training,
        args.sequence_length,
        args.stride,
        observation_mean,
        observation_std,
        action_mean,
        action_std,
    )
    validation_dataset = v2.SequenceWindows(
        validation_episodes,
        args.sequence_length,
        max(args.stride, 16),
        observation_mean,
        observation_std,
        action_mean,
        action_std,
    )
    summary = {
        "status": "inspected" if args.inspect_only else "training",
        "source_checkpoint": str(source_path),
        "source_sha256": v2.sha256(source_path),
        "device": str(device),
        "traces_discovered": len(paths),
        "dataset": data_v4.dataset_summary(episodes),
        "training": data_v4.dataset_summary(training),
        "validation": data_v4.dataset_summary(validation_episodes),
        "validation_groups": validation_groups,
        "training_windows": len(training_dataset),
        "validation_windows": len(validation_dataset),
        "observation_migration": observation_migration,
        "trace_audit": trace_audit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v2.atomic_json(args.output_dir / "dataset_audit.json", summary)
    print(
        f"[rssm-v4] traces={len(paths)} episodes={len(episodes)} "
        f"transitions={summary['dataset']['transitions']} "
        f"collisions_train={summary['training']['collision_transitions']} "
        f"collisions_val={summary['validation']['collision_transitions']} "
        f"windows={len(training_dataset)} device={device}",
        flush=True,
    )
    if args.inspect_only:
        return 0
    if not training_dataset or not validation_dataset:
        raise RuntimeError("RSSM V4 sequence windows are empty")

    config = RSSMConfig(
        observation_dim=v2.OBSERVATION_DIM,
        action_dim=v2.ACTION_DIM,
        encoder_dim=192,
        hidden_dim=384,
        deter_dim=192,
        stoch_dim=16,
        classes=16,
        free_nats=0.10,
        dyn_scale=1.0,
        rep_scale=0.1,
        deterministic_state_mode="probabilities",
    )
    model = TemporalRSSMWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    sampler = WeightedRandomSampler(
        training_dataset.weights,
        num_samples=len(training_dataset),
        replacement=True,
    )
    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    pos_weight = event_positive_weights(training)
    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = math.inf
    started = time.time()
    for epoch in range(max(1, args.epochs)):
        model.train()
        train_rows: List[Dict[str, float]] = []
        for batch in train_loader:
            loss, metrics = world_model_loss(
                model,
                batch,
                device,
                pos_weight,
                args.overshoot_horizon,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite V4 loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
            optimizer.step()
            train_rows.append(metrics)
        train_metrics = {
            key: float(np.mean([row[key] for row in train_rows]))
            for key in train_rows[0]
        }
        validation_loss = dataset_loss(
            model,
            validation_loader,
            device,
            pos_weight,
            args.overshoot_horizon,
        )
        if validation_loss["loss"] < best_validation_loss:
            best_validation_loss = validation_loss["loss"]
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_loss.items()},
        }
        history.append(row)
        print(
            f"[rssm-v4] epoch={epoch + 1}/{args.epochs} "
            f"train={train_metrics['loss']:.4f} "
            f"val={validation_loss['loss']:.4f} "
            f"risk={validation_loss.get('risk', math.inf):.4f} "
            f"events={validation_loss.get('events', math.inf):.4f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    model.eval()

    validation = v2.evaluate_horizons(
        model,
        validation_episodes,
        observation_mean,
        observation_std,
        action_mean,
        action_std,
        device,
    )
    collision_validation = collision_horizon_metrics(
        model,
        validation_episodes,
        observation_mean,
        observation_std,
        action_mean,
        action_std,
        device,
    )
    accepted, gate = quality_gate(validation, collision_validation)
    checkpoint, _ = v2.migrate_source_checkpoint(
        source_path,
        config,
        model.cpu(),
        policy_mean,
        policy_std,
        observation_mean,
        observation_std,
        action_mean,
        action_std,
    )
    model.to(device)
    arbitration_config = arbitration(validation)
    checkpoint["rssm_v2"] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generation": "v4_collision_aware",
        "source_checkpoint": str(source_path),
        "source_sha256": v2.sha256(source_path),
        "runtime_guard": False,
        "complementary_to_simlingo": True,
        "model_based_arbitration": True,
        "planning_horizon": 8,
        "planning_discount": 0.95,
        "arbitration": arbitration_config,
        "hard_safety_thresholds": False,
        "validation": validation,
        "collision_validation": collision_validation,
        "quality_gate_passed": accepted,
        "validation_groups": validation_groups,
        "transitions": summary["dataset"]["transitions"],
        "training_seconds": time.time() - started,
    }
    checkpoint["rssm_v4"] = {
        "world_model_quality_passed": accepted,
        "actor_fitted": False,
        "closed_loop_promoted": False,
        "collision_aware": True,
        "post_impact_rows_excluded": True,
        "quality_gate": gate,
    }
    checkpoint = {
        key: (
            {
                inner: value.detach().cpu()
                for inner, value in item.items()
            }
            if isinstance(item, dict)
            and item
            and all(torch.is_tensor(value) for value in item.values())
            else item
        )
        for key, item in checkpoint.items()
    }
    attempt_path = args.output_dir / "last_world_model_attempt.pt"
    v2.atomic_torch_save(attempt_path, checkpoint)
    candidate_path = args.output_dir / "world_model_candidate.pt"
    if accepted:
        v2.atomic_torch_save(candidate_path, checkpoint)
    report = {
        **summary,
        "status": "world_model_candidate_saved" if accepted else "quality_gate_rejected",
        "accepted": accepted,
        "event_positive_weights": pos_weight.tolist(),
        "validation_metrics": validation,
        "collision_validation": collision_validation,
        "quality_gate": gate,
        "arbitration": arbitration_config,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "last_attempt": str(attempt_path),
        "candidate": str(candidate_path) if accepted else "",
        "elapsed_seconds": time.time() - started,
    }
    v2.atomic_json(args.output_dir / "world_model_training_report.json", report)
    h5 = validation.get("5") or {}
    h5_decision = (h5.get("families") or {}).get("decision") or {}
    print(
        f"[rssm-v4] quality_gate={'PASS' if accepted else 'REJECT'} "
        f"h5_decision_ratio={float(h5_decision.get('persistence_ratio', math.inf)):.3f} "
        f"collision_ap={float(collision_validation.get('average_precision', math.nan)):.3f} "
        f"collision_recall={float(collision_validation.get('recall', 0.0)):.3f}",
        flush=True,
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
