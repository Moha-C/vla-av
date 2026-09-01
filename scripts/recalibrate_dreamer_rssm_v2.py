#!/usr/bin/env python3
"""Recalibrate a validated RSSM checkpoint without retraining its weights.

The procedure reevaluates the existing world model after trace-representation
repairs, derives a continuous risk/progress utility from held-out errors, and
only promotes the metadata update when both the regular and forced safety
validation gates pass. No hard safety guard or TTC/clearance veto is added.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from external.simlingo.team_code.dreamer_world_models import (
    WORLD_MODEL_TYPE,
    RSSMConfig,
    TemporalRSSMWorldModel,
)
from scripts.train_dreamer_rssm_v2 import (
    ROOT,
    atomic_json,
    atomic_torch_save,
    calibrated_arbitration,
    discover_traces,
    evaluate_horizons,
    load_episodes,
    rssm_quality_gate,
    sha256,
    split_routes,
)


DEFAULT_CHECKPOINT = (
    ROOT / "external/simlingo/checkpoints/dreamer_ppo_rssm_v2/candidate_model.pt"
)
DEFAULT_PATTERNS = (
    "logs/dreamer_online_rl/webapp_*/trace.jsonl",
    "logs/dreamer_online_rl/*/traces/*.jsonl",
    "logs/dreamer_rl_campaign/*/traces/*.jsonl",
    "logs/action_dreaming_collect/*.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--trace-pattern", action="append", default=[])
    parser.add_argument(
        "--validation-trace-pattern",
        action="append",
        default=[],
        help="Trace excluded from fitting and reserved for safety validation.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--backup", type=Path, default=None)
    return parser.parse_args()


def _evaluate(
    model: TemporalRSSMWorldModel,
    episodes: Sequence[Any],
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    if not episodes:
        return {}
    return evaluate_horizons(
        model,
        episodes,
        np.asarray(checkpoint["world_observation_mean"], dtype=np.float32),
        np.asarray(checkpoint["world_observation_std"], dtype=np.float32),
        np.asarray(checkpoint["action_mean"], dtype=np.float32),
        np.asarray(checkpoint["action_std"], dtype=np.float32),
        torch.device("cpu"),
    )


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    patterns = args.trace_pattern or list(DEFAULT_PATTERNS)
    all_paths = discover_traces(patterns, 0)
    forced_paths = discover_traces(args.validation_trace_pattern, 0)
    if not forced_paths:
        raise RuntimeError(
            "at least one --validation-trace-pattern is required for promotion"
        )
    forced_set = {path.resolve() for path in forced_paths}
    pool_episodes, _ = load_episodes(
        [path for path in all_paths if path.resolve() not in forced_set],
        args.sequence_length,
    )
    _, regular_validation, regular_routes = split_routes(pool_episodes, args.seed)
    forced_validation_episodes, _ = load_episodes(
        forced_paths,
        args.sequence_length,
    )
    if not forced_validation_episodes:
        raise RuntimeError("forced validation traces contain no usable episodes")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("world_model_type") != WORLD_MODEL_TYPE:
        raise ValueError("checkpoint is not a temporal RSSM V2 world model")
    config = RSSMConfig.from_dict(checkpoint.get("world_model_config"))
    model = TemporalRSSMWorldModel(config)
    model.load_state_dict(checkpoint["world_model"])
    model.eval()

    regular_validation_report = _evaluate(
        model,
        regular_validation,
        checkpoint,
    )
    forced_validation_report = _evaluate(
        model,
        forced_validation_episodes,
        checkpoint,
    )
    combined_validation_report = _evaluate(
        model,
        list(regular_validation) + list(forced_validation_episodes),
        checkpoint,
    )
    quality_passed, quality_gate = rssm_quality_gate(
        combined_validation_report,
        forced_validation_report,
    )
    arbitration = calibrated_arbitration(combined_validation_report)
    parent_sha = sha256(checkpoint_path)
    report_path = (
        args.report.expanduser().resolve()
        if args.report else checkpoint_path.parent / "recalibration_report.json"
    )
    backup_path = (
        args.backup.expanduser().resolve()
        if args.backup else checkpoint_path.parent
        / "candidate_model_before_stationary_oncoming_fix_20260812.pt"
    )
    report: Dict[str, Any] = {
        "status": "validated" if quality_passed else "quality_gate_rejected",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": str(checkpoint_path),
        "parent_sha256": parent_sha,
        "weights_retrained": False,
        "representation": {
            "name": "opposing_heading_geometry_v2",
            "stationary_opposing_vehicle_is_oncoming": True,
        },
        "arbitration": arbitration,
        "regular_validation_routes": regular_routes,
        "forced_validation_traces": [str(path) for path in forced_paths],
        "regular_validation": regular_validation_report,
        "forced_validation": forced_validation_report,
        "combined_validation": combined_validation_report,
        "quality_gate": quality_gate,
        "quality_gate_passed": quality_passed,
        "promoted": False,
        "backup": "",
    }

    if args.promote and quality_passed:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(checkpoint_path, backup_path)
        calibrated = copy.deepcopy(checkpoint)
        metadata = copy.deepcopy(calibrated.get("rssm_v2") or {})
        metadata.update({
            "recalibrated_at": report["created_at"],
            "recalibration_parent_sha256": parent_sha,
            "weights_retrained_for_recalibration": False,
            "oncoming_representation": report["representation"],
            "planning_horizon": 5,
            "planning_discount": 0.95,
            "arbitration": arbitration,
            "hard_safety_thresholds": False,
            "runtime_guard": False,
            "complementary_to_simlingo": True,
            "model_based_arbitration": True,
            "recalibration_validation": combined_validation_report,
            "recalibration_forced_validation": forced_validation_report,
            "recalibration_quality_gate_passed": True,
        })
        calibrated["rssm_v2"] = metadata
        atomic_torch_save(checkpoint_path, calibrated)
        report["promoted"] = True
        report["backup"] = str(backup_path)
        report["backup_sha256"] = sha256(backup_path)
        report["promoted_sha256"] = sha256(checkpoint_path)

    atomic_json(report_path, report)
    print(
        f"[rssm-recalibrate] gate={'PASS' if quality_passed else 'REJECT'} "
        f"promoted={int(report['promoted'])} "
        f"risk_curvature={arbitration['risk_curvature']:.3f} "
        f"action_penalty={arbitration['action_penalty']:.3f} "
        f"authority_temperature={arbitration['authority_temperature']:.3f}",
        flush=True,
    )
    print(f"[rssm-recalibrate] report={report_path}", flush=True)
    return 0 if quality_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
