#!/usr/bin/env python3
"""Train an isolated learned risk/progress utility for RSSM arbitration.

The recurrent world model, its physical risk/progress heads, the PPO actor and
SimLingo stay frozen byte-for-byte. A compact pairwise calibrator learns only a
continuous residual score between SimLingo's imagined future and a Dreamer
proposal. Geometry is used to curate offline labels, never as a runtime guard.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from external.simlingo.team_code.dreamer_guard import rssm_authority_confidence
from external.simlingo.team_code.dreamer_world_models import (
    UTILITY_MODEL_TYPE,
    UTILITY_CONTEXT_OBSERVATION_INDICES,
    PairwiseUtilityCalibrator,
    RSSMConfig,
    TemporalRSSMWorldModel,
    discounted_feature_pool,
)
from scripts.finetune_dreamer_rssm_decision_utility import (
    DEFAULT_NEGATIVE_TRAIN,
    DEFAULT_NEGATIVE_VALIDATION,
    DEFAULT_POSITIVE_TRAIN,
    DEFAULT_POSITIVE_VALIDATION,
    UtilityPairs,
    concatenate_pairs,
    make_pairs,
    resolve_paths,
    utility_scores,
)
from scripts.train_dreamer_rssm_v2 import (
    ROOT,
    Episode,
    actor_from_checkpoint,
    atomic_json,
    atomic_torch_save,
    load_episodes,
    sha256,
)


DEFAULT_CHECKPOINT = (
    ROOT / "external/simlingo/checkpoints/dreamer_ppo_rssm_v2/candidate_model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--promote-to", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--output-scale", type=float, default=1.5)
    parser.add_argument("--blend-step", type=float, default=0.025)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--positive-stride", type=int, default=1)
    parser.add_argument("--negative-stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--positive-train-trace", action="append", default=[])
    parser.add_argument("--positive-validation-trace", action="append", default=[])
    parser.add_argument("--negative-train-trace", action="append", default=[])
    parser.add_argument("--negative-validation-trace", action="append", default=[])
    return parser.parse_args()


def load_group(
    paths: Iterable[Path], sequence_length: int
) -> Tuple[List[Episode], List[Dict[str, Any]]]:
    episodes, audit = load_episodes(list(paths), sequence_length)
    if not episodes:
        raise RuntimeError("trace group produced no usable episodes")
    return episodes, audit


def pair_tensors(
    model: TemporalRSSMWorldModel,
    pairs: UtilityPairs,
    checkpoint: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    values = utility_scores(model, pairs, checkpoint)
    discount = float((checkpoint.get("rssm_v2") or {}).get("planning_discount", 0.95))
    return {
        **values,
        "base_feature": discounted_feature_pool(
            pairs.base_features, pairs.base_continuation, discount
        ),
        "candidate_feature": discounted_feature_pool(
            pairs.candidate_features, pairs.candidate_continuation, discount
        ),
    }


def residual_for(
    calibrator: PairwiseUtilityCalibrator,
    pairs: UtilityPairs,
    tensors: Dict[str, torch.Tensor],
) -> torch.Tensor:
    observation_indices = tuple(
        index
        for index in UTILITY_CONTEXT_OBSERVATION_INDICES
        if index < pairs.current_observation.shape[-1]
    )
    current_observation = pairs.current_observation[:, observation_indices]
    return calibrator(
        tensors["base_feature"],
        tensors["candidate_feature"],
        tensors["candidate_progress"] - tensors["base_progress"],
        tensors["candidate_risk"] - tensors["base_risk"],
        pairs.control_delta,
        current_observation,
    )


@torch.no_grad()
def pair_metrics(
    calibrator: PairwiseUtilityCalibrator,
    pairs: UtilityPairs,
    tensors: Dict[str, torch.Tensor],
    checkpoint: Dict[str, Any],
    blend: float,
) -> Dict[str, Any]:
    if not pairs.samples:
        return {"samples": 0, "accuracy": 0.0}
    residual = residual_for(calibrator, pairs, tensors) * float(blend)
    margin = tensors["margin"] + residual
    positive = pairs.target_margin > 0.0
    correct = torch.where(positive, margin > 0.0, margin < 0.0)
    temperature = float(
        ((checkpoint.get("rssm_v2") or {}).get("arbitration") or {}).get(
            "authority_temperature", 0.35
        )
    )
    confidence = [
        rssm_authority_confidence(float(value), temperature)
        for value in margin[positive].tolist()
    ]
    return {
        "samples": pairs.samples,
        "positive_samples": int(positive.sum()),
        "negative_samples": int((~positive).sum()),
        "accuracy": float(correct.float().mean()),
        "positive_accuracy": (
            float((margin[positive] > 0.0).float().mean())
            if bool(positive.any()) else 0.0
        ),
        "negative_accuracy": (
            float((margin[~positive] < 0.0).float().mean())
            if bool((~positive).any()) else 0.0
        ),
        "mean_margin": float(margin.mean()),
        "mean_positive_margin": (
            float(margin[positive].mean()) if bool(positive.any()) else 0.0
        ),
        "mean_negative_margin": (
            float(margin[~positive].mean()) if bool((~positive).any()) else 0.0
        ),
        "mean_residual": float(residual.mean()),
        "mean_abs_residual": float(residual.abs().mean()),
        "max_abs_residual": float(residual.abs().max()),
        "target_margin_mae": float((margin - pairs.target_margin).abs().mean()),
        "mean_positive_authority_confidence": (
            float(np.mean(confidence)) if confidence else 0.0
        ),
        "p10_positive_authority_confidence": (
            float(np.percentile(confidence, 10)) if confidence else 0.0
        ),
        "risk_increase_selected": int(
            (
                (margin > 0.0)
                & (
                    tensors["candidate_risk"]
                    > tensors["base_risk"] + 0.01
                )
            ).sum()
        ),
    }


def validation_gate(
    before_positive: Dict[str, Any],
    after_positive: Dict[str, Any],
    before_negative: Dict[str, Any],
    after_negative: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    positive_floor = max(
        0.70, float(before_positive["positive_accuracy"]) + 0.05
    )
    negative_floor = max(
        0.96, float(before_negative["negative_accuracy"]) - 0.02
    )
    margin_gain = (
        float(after_positive["mean_positive_margin"])
        - float(before_positive["mean_positive_margin"])
    )
    checks = {
        "positive_accuracy": bool(
            after_positive["positive_accuracy"] >= positive_floor
        ),
        "positive_margin_gain": bool(margin_gain >= 0.10),
        "positive_authority": bool(
            after_positive["mean_positive_authority_confidence"] >= 0.30
        ),
        "negative_accuracy": bool(
            after_negative["negative_accuracy"] >= negative_floor
        ),
        "negative_margin": bool(
            after_negative["mean_negative_margin"] <= -0.05
        ),
        "negative_residual_bound": bool(
            after_negative["max_abs_residual"] <= 1.25
        ),
    }
    return bool(all(checks.values())), {
        "checks": checks,
        "positive_accuracy_floor": positive_floor,
        "negative_accuracy_floor": negative_floor,
        "positive_margin_gain": margin_gain,
        "positive_margin_gain_floor": 0.10,
        "positive_authority_floor": 0.30,
        "negative_margin_ceiling": -0.05,
        "negative_residual_absolute_ceiling": 1.25,
    }


def audit_blends(
    calibrator: PairwiseUtilityCalibrator,
    pair_groups: Dict[str, UtilityPairs],
    tensors: Dict[str, Dict[str, torch.Tensor]],
    checkpoint: Dict[str, Any],
    before_positive: Dict[str, Any],
    before_negative: Dict[str, Any],
    blend_step: float,
) -> Tuple[List[Dict[str, Any]], Optional[float], Dict[str, Any]]:
    """Return the smallest continuous blend satisfying every held-out gate."""
    audit: List[Dict[str, Any]] = []
    selected_blend: Optional[float] = None
    selected_gate: Dict[str, Any] = {}
    for raw in np.arange(blend_step, 1.0 + 0.5 * blend_step, blend_step):
        blend = float(min(1.0, raw))
        positive = pair_metrics(
            calibrator,
            pair_groups["positive_validation"],
            tensors["positive_validation"],
            checkpoint,
            blend,
        )
        negative = pair_metrics(
            calibrator,
            pair_groups["negative_validation"],
            tensors["negative_validation"],
            checkpoint,
            blend,
        )
        passed, gate = validation_gate(
            before_positive, positive, before_negative, negative
        )
        audit.append({
            "blend": blend,
            "passed": passed,
            "positive": positive,
            "negative": negative,
            "gate": gate,
        })
        if passed and selected_blend is None:
            selected_blend = blend
            selected_gate = gate
    return audit, selected_blend, selected_gate


def state_dict_equal(
    left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else checkpoint_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RSSMConfig.from_dict(checkpoint.get("world_model_config"))
    model = TemporalRSSMWorldModel(config)
    model.load_state_dict(checkpoint["world_model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    actor = actor_from_checkpoint(checkpoint, torch.device("cpu"))
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    raw_groups = {
        "positive_train": args.positive_train_trace or DEFAULT_POSITIVE_TRAIN,
        "positive_validation": (
            args.positive_validation_trace or DEFAULT_POSITIVE_VALIDATION
        ),
        "negative_train": args.negative_train_trace or DEFAULT_NEGATIVE_TRAIN,
        "negative_validation": (
            args.negative_validation_trace or DEFAULT_NEGATIVE_VALIDATION
        ),
    }
    path_groups = {name: resolve_paths(patterns) for name, patterns in raw_groups.items()}
    missing = [name for name, paths in path_groups.items() if not paths]
    if missing:
        raise RuntimeError("missing trace group(s): " + ", ".join(missing))

    episode_groups: Dict[str, List[Episode]] = {}
    audits: Dict[str, Any] = {}
    for name, paths in path_groups.items():
        episode_groups[name], audits[name] = load_group(
            paths, args.sequence_length
        )

    pair_groups = {
        "positive_train": make_pairs(
            model, actor, episode_groups["positive_train"], checkpoint,
            "positive", args.positive_stride,
        ),
        "positive_validation": make_pairs(
            model, actor, episode_groups["positive_validation"], checkpoint,
            "positive", args.positive_stride,
        ),
        "negative_train": make_pairs(
            model, actor, episode_groups["negative_train"], checkpoint,
            "negative", args.negative_stride,
        ),
        "negative_validation": make_pairs(
            model, actor, episode_groups["negative_validation"], checkpoint,
            "negative", args.negative_stride,
        ),
    }
    if min(group.samples for group in pair_groups.values()) < 20:
        raise RuntimeError("each independent utility pair split needs >=20 samples")
    training_pairs = concatenate_pairs(
        [pair_groups["positive_train"], pair_groups["negative_train"]]
    )
    tensors = {
        name: pair_tensors(model, pairs, checkpoint)
        for name, pairs in pair_groups.items()
    }
    training_tensors = pair_tensors(model, training_pairs, checkpoint)

    calibrator = PairwiseUtilityCalibrator(
        config.feature_dim,
        observation_dim=len(tuple(
            index
            for index in UTILITY_CONTEXT_OBSERVATION_INDICES
            if index < config.observation_dim
        )),
        hidden_dim=args.hidden_dim,
        output_scale=args.output_scale,
    )
    optimizer = torch.optim.AdamW(
        calibrator.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    positive_mask = training_pairs.target_margin > 0.0
    positive_count = max(1, int(positive_mask.sum()))
    negative_count = max(1, training_pairs.samples - positive_count)
    history: List[Dict[str, Any]] = []
    calibrator.eval()
    before_positive = pair_metrics(
        calibrator,
        pair_groups["positive_validation"],
        tensors["positive_validation"],
        checkpoint,
        0.0,
    )
    before_negative = pair_metrics(
        calibrator,
        pair_groups["negative_validation"],
        tensors["negative_validation"],
        checkpoint,
        0.0,
    )
    blend_step = float(np.clip(args.blend_step, 0.005, 1.0))
    selected_epoch: Optional[int] = None
    selected_blend: Optional[float] = None
    selected_gate: Dict[str, Any] = {}
    selected_state: Optional[Dict[str, torch.Tensor]] = None
    selected_audit: List[Dict[str, Any]] = []

    for epoch in range(max(1, args.epochs)):
        calibrator.train()
        residual = residual_for(calibrator, training_pairs, training_tensors)
        margin = training_tensors["margin"].detach() + residual
        weights = torch.where(
            positive_mask,
            torch.full_like(margin, training_pairs.samples / (2.0 * positive_count)),
            torch.full_like(margin, training_pairs.samples / (2.0 * negative_count)),
        )
        regression = F.smooth_l1_loss(
            margin,
            training_pairs.target_margin,
            reduction="none",
            beta=0.10,
        )
        signed_margin = torch.where(positive_mask, margin, -margin)
        ranking = F.softplus(-signed_margin / 0.10)
        # Keep the residual economical. The physical RSSM score remains the
        # default unless held-out data supports a learned correction.
        residual_regularization = residual.square().mean()
        loss = (
            weights * (2.0 * regression + 0.30 * ranking)
        ).mean() + 0.015 * residual_regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 2.0)
        optimizer.step()

        calibrator.eval()
        epoch_audit, epoch_blend, epoch_gate = audit_blends(
            calibrator,
            pair_groups,
            tensors,
            checkpoint,
            before_positive,
            before_negative,
            blend_step,
        )
        positive = pair_metrics(
            calibrator,
            pair_groups["positive_validation"],
            tensors["positive_validation"],
            checkpoint,
            1.0,
        )
        negative = pair_metrics(
            calibrator,
            pair_groups["negative_validation"],
            tensors["negative_validation"],
            checkpoint,
            1.0,
        )
        should_log = (
            epoch == 0
            or (epoch + 1) % 25 == 0
            or epoch + 1 == args.epochs
            or epoch_blend is not None
        )
        if should_log:
            row = {
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "positive_validation": positive,
                "negative_validation": negative,
                "first_passing_blend": epoch_blend,
            }
            history.append(row)
            print(
                f"[rssm-utility-calibrator] epoch={epoch + 1}/{args.epochs} "
                f"loss={row['loss']:.4f} "
                f"positive_acc={positive['positive_accuracy']:.3f} "
                f"positive_margin={positive['mean_positive_margin']:+.3f} "
                f"negative_acc={negative['negative_accuracy']:.3f} "
                f"negative_margin={negative['mean_negative_margin']:+.3f}",
                flush=True,
            )

        # Stop at the first model that clears every independent behavior gate.
        # Continuing past this point previously overfit the scarce positive
        # examples and inverted the held-out oncoming-traffic decisions.
        if epoch_blend is not None:
            selected_epoch = epoch + 1
            selected_blend = epoch_blend
            selected_gate = epoch_gate
            selected_audit = epoch_audit
            selected_state = {
                key: value.detach().cpu().clone()
                for key, value in calibrator.state_dict().items()
            }
            print(
                "[rssm-utility-calibrator] first safe held-out epoch="
                f"{selected_epoch} blend={selected_blend:.3f}; stopping early",
                flush=True,
            )
            break

    calibrator.eval()
    if selected_state is not None:
        calibrator.load_state_dict(selected_state)
        blend_audit = selected_audit
    else:
        blend_audit, selected_blend, selected_gate = audit_blends(
            calibrator,
            pair_groups,
            tensors,
            checkpoint,
            before_positive,
            before_negative,
            blend_step,
        )

    effective_blend = selected_blend if selected_blend is not None else 0.0
    after_positive = pair_metrics(
        calibrator,
        pair_groups["positive_validation"],
        tensors["positive_validation"],
        checkpoint,
        effective_blend,
    )
    after_negative = pair_metrics(
        calibrator,
        pair_groups["negative_validation"],
        tensors["negative_validation"],
        checkpoint,
        effective_blend,
    )
    passed, gate = validation_gate(
        before_positive, after_positive, before_negative, after_negative
    )
    passed = bool(selected_blend is not None and passed)

    candidate = copy.deepcopy(checkpoint)
    candidate["utility_model_type"] = UTILITY_MODEL_TYPE
    candidate["utility_calibrator"] = {
        key: value.detach().cpu()
        for key, value in calibrator.state_dict().items()
    }
    metadata = copy.deepcopy(candidate.get("rssm_v2") or {})
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata["utility_calibrator"] = {
        "model_type": UTILITY_MODEL_TYPE,
        "observation_dim": int(calibrator.observation_dim),
        "observation_indices": [
            int(index)
            for index in UTILITY_CONTEXT_OBSERVATION_INDICES
            if index < config.observation_dim
        ],
        "hidden_dim": int(args.hidden_dim),
        "output_scale": float(args.output_scale),
        "blend": float(effective_blend),
        "trained_at": created_at,
        "parent_sha256": sha256(checkpoint_path),
        "runtime_inputs": (
            "pooled_rssm_latent_delta_plus_learned_risk_progress_delta_"
            "plus_action_delta"
        ),
        "offline_geometry_labels_only": True,
        "runtime_geometry_thresholds": False,
        "runtime_guard": False,
    }
    metadata.update({
        "runtime_guard": False,
        "hard_safety_thresholds": False,
        "complementary_to_simlingo": True,
    })
    candidate["rssm_v2"] = metadata

    world_unchanged = state_dict_equal(
        checkpoint["world_model"], candidate["world_model"]
    )
    actor_unchanged = state_dict_equal(
        checkpoint["policy"], candidate["policy"]
    )
    passed = bool(passed and world_unchanged and actor_unchanged)
    attempt_path = output_dir / "utility_calibrator_last_attempt.pt"
    atomic_torch_save(attempt_path, candidate)

    report: Dict[str, Any] = {
        "status": "validated" if passed else "rejected",
        "created_at": created_at,
        "parent_checkpoint": str(checkpoint_path),
        "parent_sha256": sha256(checkpoint_path),
        "paths": {
            name: [str(path) for path in paths]
            for name, paths in path_groups.items()
        },
        "audits": audits,
        "pair_counts": {
            name: pairs.samples for name, pairs in pair_groups.items()
        },
        "before_positive_validation": before_positive,
        "after_positive_validation": after_positive,
        "before_negative_validation": before_negative,
        "after_negative_validation": after_negative,
        "selected_blend": selected_blend,
        "selected_epoch": selected_epoch,
        "selection_policy": "first_epoch_passing_all_held_out_gates",
        "validation_gate": gate or selected_gate,
        "blend_audit": blend_audit,
        "history": history,
        "world_model_unchanged": world_unchanged,
        "physical_risk_progress_heads_unchanged": world_unchanged,
        "actor_unchanged": actor_unchanged,
        "simlingo_unchanged": True,
        "runtime_guard": False,
        "hard_safety_thresholds": False,
        "runtime_geometry_thresholds": False,
        "offline_geometry_labels_only": True,
        "attempt": str(attempt_path),
        "attempt_sha256": sha256(attempt_path),
        "promoted": False,
    }
    if args.promote and passed:
        promote_to = args.promote_to.expanduser().resolve()
        backup = output_dir / (
            "candidate_model_before_utility_calibrator_"
            + time.strftime("%Y%m%d_%H%M%S")
            + ".pt"
        )
        if promote_to.exists():
            shutil.copy2(promote_to, backup)
        atomic_torch_save(promote_to, candidate)
        report.update({
            "promoted": True,
            "promoted_to": str(promote_to),
            "promoted_sha256": sha256(promote_to),
            "backup": str(backup),
            "backup_sha256": sha256(backup),
        })

    report_path = output_dir / "utility_calibrator_training_report.json"
    atomic_json(report_path, report)
    print(
        f"[rssm-utility-calibrator] gate={'PASS' if passed else 'REJECT'} "
        f"blend={selected_blend} promoted={int(report['promoted'])}",
        flush=True,
    )
    print(f"[rssm-utility-calibrator] report={report_path}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
