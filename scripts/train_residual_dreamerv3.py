#!/usr/bin/env python3
"""Reproducible candidate-only pipeline for residual DreamerV3."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.residual_dreamerv3.baselines import (  # noqa: E402
    Normalization,
    PersistenceBaseline,
    RidgeDynamicsBaseline,
    evaluate_baseline,
    write_json,
)
from src.residual_dreamerv3.config import ResidualDreamerConfig, load_config  # noqa: E402
from src.residual_dreamerv3.data import (  # noqa: E402
    Episode,
    build_episode,
    discover_traces,
    split_manifest,
    stratified_seed_split,
)
from src.residual_dreamerv3.training import (  # noqa: E402
    load_world_model,
    promote_checkpoint,
    resolve_device,
    train_actor_critic,
    train_world_model,
    validate_world_model,
)


DEFAULT_TRACE = "data/report_dreamer/native/runs/native_report12_v1/**/trace.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dataset(
    patterns: Sequence[str],
    config: ResidualDreamerConfig,
    output: Path,
) -> Tuple[List[Episode], Any, Dict[str, Any]]:
    resolved_patterns = [str((ROOT / value).resolve()) if not Path(value).is_absolute() else value for value in patterns]
    traces = discover_traces(resolved_patterns)
    episodes: List[Episode] = []
    rejected: List[Dict[str, str]] = []
    for path in traces:
        try:
            episode = build_episode(path, config)
            if episode is not None:
                episodes.append(episode)
        except Exception as exc:
            rejected.append({"path": str(path), "reason": "%s: %s" % (type(exc).__name__, exc)})
    if not episodes:
        raise RuntimeError("no admissible native Bench2Drive trace was found")
    splits = stratified_seed_split(episodes, config)
    manifest = split_manifest(splits, config)
    manifest.update(
        {
            "created_at": utc_now(),
            "trace_patterns": resolved_patterns,
            "discovered_traces": len(traces),
            "accepted_traces": len(episodes),
            "rejected_traces": rejected,
        }
    )
    write_json(output / "dataset_manifest.json", manifest)
    return episodes, splits, manifest


def print_manifest(manifest: Dict[str, Any]) -> None:
    print("[residual-dreamerv3] accepted=%d rejected=%d" % (
        manifest["accepted_traces"], len(manifest["rejected_traces"])
    ))
    for name in ("train", "validation", "test"):
        item = manifest[name]
        print(
            "  %-10s episodes=%d transitions=%d seeds=%d towns=%s"
            % (name, item["episodes"], item["transitions"], len(item["seeds"]), ",".join(item["towns"]))
        )


def baseline_reports(splits: Any, config: ResidualDreamerConfig, output: Path) -> Dict[str, Any]:
    normalization = Normalization.fit(splits.train)
    persistence = PersistenceBaseline(splits.train)
    ridge = RidgeDynamicsBaseline().fit(splits.train)
    ridge.save(output / "action_conditioned_ridge.npz")
    report: Dict[str, Any] = {
        "schema_version": "residual_dreamerv3_frozen_baselines_v1",
        "normalization": normalization.to_dict(),
        "splits": {},
    }
    for name in ("validation", "test"):
        episodes = getattr(splits, name)
        report["splits"][name] = [
            evaluate_baseline(persistence, episodes, normalization, config.gate.horizons),
            evaluate_baseline(ridge, episodes, normalization, config.gate.horizons),
        ]
    write_json(output / "frozen_baselines.json", report)
    return report


def apply_training_overrides(config: ResidualDreamerConfig, args: argparse.Namespace) -> None:
    if args.device:
        config.training.device = args.device
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.max_windows is not None:
        config.training.maximum_windows = args.max_windows
    config.validate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("inspect", "baselines", "world-model", "actor", "all", "evaluate", "promote"),
    )
    parser.add_argument("--config", default=str(ROOT / "configs/residual_dreamerv3.yaml"))
    parser.add_argument("--trace", action="append", default=[])
    parser.add_argument("--output", default=str(ROOT / "checkpoints/residual_dreamerv3/candidate"))
    parser.add_argument("--world-checkpoint")
    parser.add_argument("--actor-checkpoint")
    parser.add_argument("--closed-loop-report")
    parser.add_argument("--promoted-output")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--world-epochs", type=int)
    parser.add_argument("--actor-epochs", type=int)
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help="diagnostic only: save reports, but never create a controlling checkpoint",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.phase == "promote":
        if not args.actor_checkpoint or not args.closed_loop_report:
            parser.error("promote requires --actor-checkpoint and --closed-loop-report")
        promoted = Path(args.promoted_output or (output / "residual_dreamerv3_promoted.pt"))
        promote_checkpoint(Path(args.actor_checkpoint), Path(args.closed_loop_report), promoted)
        print("[residual-dreamerv3] promoted checkpoint: %s" % promoted)
        return 0

    config = load_config(args.config)
    apply_training_overrides(config, args)
    _, splits, manifest = load_dataset(args.trace or [DEFAULT_TRACE], config, output)
    print_manifest(manifest)
    if args.phase == "inspect":
        return 0
    if args.phase == "baselines":
        baseline_reports(splits, config, output)
        print("[residual-dreamerv3] frozen baselines written to %s" % output)
        return 0

    device = resolve_device(config.training.device)
    print("[residual-dreamerv3] device=%s" % device)
    world_checkpoint: Path
    if args.phase in ("world-model", "all"):
        baseline_reports(splits, config, output)
        world_checkpoint, _ = train_world_model(
            splits, config, output, device, args.world_epochs, args.max_windows
        )
    elif args.world_checkpoint:
        world_checkpoint = Path(args.world_checkpoint).resolve()
    else:
        world_checkpoint = output / "world_model_candidate.pt"
    if not world_checkpoint.exists():
        raise FileNotFoundError("world-model checkpoint not found: %s" % world_checkpoint)
    model = load_world_model(world_checkpoint, config, device)
    validation_gate = validate_world_model(
        model, splits, config, output, device, "validation"
    )
    print(
        "[residual-dreamerv3] validation gate=%s observation=%+.2f%% reward=%+.2f%% risk=%+.2f%%"
        % (
            "PASS" if validation_gate["passed"] else "FAIL",
            100.0 * validation_gate["improvement"]["observation"],
            100.0 * validation_gate["improvement"]["reward"],
            100.0 * validation_gate["improvement"]["risk"],
        )
    )
    if args.phase == "world-model":
        return 0 if validation_gate["passed"] else 2
    if not validation_gate["passed"] and not args.allow_gate_failure:
        print("[residual-dreamerv3] actor training refused: world model did not beat frozen baselines")
        return 2
    test_gate = validate_world_model(model, splits, config, output, device, "test")
    combined_gate = {
        "schema_version": "residual_dreamerv3_combined_world_gate_v1",
        "passed": bool(validation_gate["passed"] and test_gate["passed"]),
        "validation": validation_gate,
        "test": test_gate,
    }
    write_json(output / "world_model_gate_combined.json", combined_gate)
    if args.phase == "evaluate":
        print("[residual-dreamerv3] combined gate=%s" % ("PASS" if combined_gate["passed"] else "FAIL"))
        return 0 if combined_gate["passed"] else 2
    if not combined_gate["passed"]:
        print("[residual-dreamerv3] test gate failed; no actor checkpoint may be produced")
        return 2
    actor_checkpoint, _ = train_actor_critic(
        model,
        splits,
        config,
        output,
        device,
        combined_gate,
        args.actor_epochs,
        args.max_windows,
    )
    print("[residual-dreamerv3] non-controlling actor candidate: %s" % actor_checkpoint)
    print("[residual-dreamerv3] status=candidate; a fixed multi-seed closed-loop report is required for promotion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
