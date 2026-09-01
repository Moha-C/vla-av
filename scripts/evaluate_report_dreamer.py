#!/usr/bin/env python3
"""Frozen test-split evaluation for report-aligned RSSM prediction heads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_report_dreamer import (
    DEFAULT_TRACE_GLOBS,
    batch_to_device,
    loader_for,
    prepare_splits,
    world_model_metrics,
)
from src.world_model.agent import SimLingoDreamerAgent
from src.world_model.config import PredictionLossConfig, load_config
from src.world_model.dataset import discover_traces


def action_sensitivity_metrics(
    model: Any,
    loader: Any,
    device: torch.device,
    max_states: int = 256,
    loss_config: Optional[PredictionLossConfig] = None,
) -> Dict[str, Any]:
    """Measure whether imagined outcomes actually depend on physical control."""

    thresholds = loss_config or PredictionLossConfig()

    transition_spreads: List[float] = []
    output_spreads: List[float] = []
    progress_spreads: List[float] = []
    risk_spreads: List[float] = []
    collision_spreads: List[float] = []
    hazard_brake_risk_advantage: List[float] = []
    hazard_brake_collision_advantage: List[float] = []
    hazard_throttle_progress_advantage: List[float] = []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = batch_to_device(raw, device)
            state, _, _ = model.observe_sequence(
                batch["observations"],
                batch["actions"],
                deterministic=True,
            )
            observation = batch["observations"][:, -1]
            native = observation[:, 2:5]
            hard_brake = native.clone()
            hard_brake[:, 1] = 0.0
            hard_brake[:, 2] = 1.0
            full_throttle = native.clone()
            full_throttle[:, 1] = 1.0
            full_throttle[:, 2] = 0.0
            steer_left = native.clone()
            steer_left[:, 0] = (steer_left[:, 0] - 0.35).clamp(-1.0, 1.0)
            steer_right = native.clone()
            steer_right[:, 0] = (steer_right[:, 0] + 0.35).clamp(-1.0, 1.0)
            probes = torch.stack(
                (native, hard_brake, full_throttle, steer_left, steer_right),
                dim=1,
            )
            probe_count = probes.shape[1]
            imagined = model.imagine_step(
                state.repeat_interleave(probe_count),
                probes.reshape(-1, 3),
                deterministic=True,
            )
            prediction = model.heads(imagined)
            batch_size = native.shape[0]
            deterministic = imagined.deterministic.reshape(
                batch_size, probe_count, -1
            )
            progress = prediction.progress.reshape(batch_size, probe_count)
            risk = prediction.risk.reshape(batch_size, probe_count)
            collision = prediction.collision.reshape(batch_size, probe_count)
            offroad = prediction.offroad.reshape(batch_size, probe_count)
            outputs = torch.stack((progress, risk, collision, offroad), dim=-1)

            transition = torch.linalg.vector_norm(
                deterministic[:, 1:] - deterministic[:, :1], dim=-1
            ).amax(dim=1)
            output = torch.linalg.vector_norm(
                outputs[:, 1:] - outputs[:, :1], dim=-1
            ).amax(dim=1)
            transition_spreads.extend(transition.cpu().tolist())
            output_spreads.extend(output.cpu().tolist())
            progress_spreads.extend(
                (progress.amax(dim=1) - progress.amin(dim=1)).cpu().tolist()
            )
            risk_spreads.extend(
                (risk.amax(dim=1) - risk.amin(dim=1)).cpu().tolist()
            )
            collision_spreads.extend(
                (collision.amax(dim=1) - collision.amin(dim=1)).cpu().tolist()
            )

            hazard = (
                (
                    observation[:, 15]
                    <= float(thresholds.hazard_front_clearance)
                )
                | (
                    observation[:, 19]
                    <= float(thresholds.hazard_oncoming_ttc)
                )
                | (
                    observation[:, 21]
                    <= float(thresholds.hazard_oncoming_ttc)
                )
                | (
                    observation[:, 23]
                    <= float(thresholds.hazard_oncoming_ttc)
                )
                | (
                    observation[:, 26]
                    <= float(thresholds.hazard_vru_distance)
                )
            )
            if hazard.any():
                hazard_brake_risk_advantage.extend(
                    (risk[hazard, 2] - risk[hazard, 1]).cpu().tolist()
                )
                hazard_brake_collision_advantage.extend(
                    (collision[hazard, 2] - collision[hazard, 1]).cpu().tolist()
                )
                hazard_throttle_progress_advantage.extend(
                    (progress[hazard, 2] - progress[hazard, 1]).cpu().tolist()
                )
            if len(transition_spreads) >= max_states:
                break

    def values(items: List[float]) -> np.ndarray:
        return np.asarray(items[:max_states], dtype=np.float64)

    transition_array = values(transition_spreads)
    output_array = values(output_spreads)
    progress_array = values(progress_spreads)
    risk_array = values(risk_spreads)
    collision_array = values(collision_spreads)
    hazard_risk = values(hazard_brake_risk_advantage)
    hazard_collision = values(hazard_brake_collision_advantage)
    hazard_progress = values(hazard_throttle_progress_advantage)

    def mean(array: np.ndarray) -> float:
        return float(array.mean()) if array.size else 0.0

    return {
        "schema_version": "report_action_sensitivity_v1",
        "states": int(transition_array.size),
        "probe_actions": [
            "native",
            "hard_brake",
            "full_throttle",
            "steer_left",
            "steer_right",
        ],
        "mean_transition_spread": mean(transition_array),
        "mean_output_spread": mean(output_array),
        "mean_progress_spread": mean(progress_array),
        "mean_risk_spread": mean(risk_array),
        "mean_collision_spread": mean(collision_array),
        "collapsed_output_fraction_1e-4": (
            float(np.mean(output_array < 1.0e-4)) if output_array.size else 1.0
        ),
        "hazard_states": int(hazard_risk.size),
        "hazard_brake_risk_advantage": mean(hazard_risk),
        "hazard_brake_collision_advantage": mean(hazard_collision),
        "hazard_throttle_progress_advantage": mean(hazard_progress),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="",
        help="Optional compatibility check; the frozen manifest remains authoritative.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trace-glob", action="append", dest="trace_globs")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-windows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest.get("config"), dict):
        raise RuntimeError("frozen manifest does not contain its training config")
    config = load_config(overrides=manifest["config"])
    if args.config:
        requested = load_config(args.config)
        if requested.to_dict() != config.to_dict():
            raise RuntimeError(
                "--config differs from the frozen training manifest; refusing test evaluation"
            )
    accepted_paths = [
        row["path"]
        for row in manifest.get("audit", [])
        if row.get("accepted") and row.get("path")
    ]
    if not accepted_paths:
        raise RuntimeError("frozen manifest has no accepted trace paths")
    if args.trace_globs:
        requested_paths = {
            str(path) for path in discover_traces(args.trace_globs)
        }
        if requested_paths != set(accepted_paths):
            raise RuntimeError(
                "--trace-glob does not resolve to the exact frozen training dataset"
            )
    missing = [path for path in accepted_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("frozen trace is missing: %s" % missing[0])
    splits, _ = prepare_splits(config, accepted_paths)
    if splits.seed_sets != manifest.get("seed_sets"):
        raise RuntimeError(
            "dataset seed split differs from the frozen manifest; refusing test evaluation"
        )
    agent = SimLingoDreamerAgent.load(args.checkpoint, device=args.device)
    if agent._architecture_signature(agent.config) != agent._architecture_signature(config):
        raise RuntimeError("checkpoint architecture differs from the frozen dataset config")
    device = torch.device(args.device)
    test_loader = loader_for(splits.test, config, False, args.max_windows)
    aggregate = world_model_metrics(
        agent.world_model, test_loader, config, device
    )
    sensitivity = action_sensitivity_metrics(
        agent.world_model,
        test_loader,
        device,
        loss_config=config.prediction_loss,
    )
    per_seed: Dict[str, Dict[str, float]] = {}
    for seed in splits.seed_sets["test"]:
        episodes = [episode for episode in splits.test if episode.seed == seed]
        loader = loader_for(episodes, config, False, args.max_windows)
        per_seed[seed] = world_model_metrics(
            agent.world_model, loader, config, device
        )
    keys = sorted(aggregate)
    dispersion = {}
    for key in keys:
        values = [row[key] for row in per_seed.values() if key in row]
        dispersion[key] = {
            "mean_across_seeds": float(np.mean(values)) if values else None,
            "std_across_seeds": float(np.std(values)) if values else None,
            "seed_count": len(values),
        }
    result: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "test_seed_count": len(splits.seed_sets["test"]),
        "test_seeds": splits.seed_sets["test"],
        "aggregate_prediction_losses": aggregate,
        "per_seed": per_seed,
        "dispersion": dispersion,
        "action_sensitivity": sensitivity,
        "closed_loop_claim": False,
        "note": (
            "These are frozen test-split prediction losses. They do not imply "
            "closed-loop driving improvement; CARLA A/B runs are evaluated separately."
        ),
    }
    destination = Path(args.output) if args.output else Path(args.checkpoint).with_name("test_prediction_metrics.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
