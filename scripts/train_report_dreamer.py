#!/usr/bin/env python3
"""Train the report-aligned SimLingo Dreamer in explicit scientific phases."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.world_model.agent import SimLingoDreamerAgent
from src.world_model.config import DreamerConfig, load_config
from src.world_model.dataset import (
    DatasetSplits,
    SequenceDataset,
    build_episode,
    dataset_summary,
    discover_traces,
    read_jsonl,
    split_by_seed,
    split_manifest,
)
from src.world_model.pairwise import PairwiseCalibrator
from src.world_model.policy import LatentCritic, ResidualActor
from src.world_model.rssm import CompactRSSM
from src.world_model.training import imagine_actor_critic


DEFAULT_TRACE_GLOBS = (
    "data/report_dreamer/native/runs/**/trace.jsonl",
)

DIAGNOSTIC_MIXED_TRACE_GLOBS = (
    "logs/dreamer_curriculum/**/trace.jsonl",
    "logs/dreamer_online_rl/**/trace.jsonl",
    "logs/dreamer_rl_campaign/**/traces/*.jsonl",
    "logs/action_dreaming_collect/**/*.jsonl",
)


class DatasetAuditError(RuntimeError):
    """Dataset eligibility/split failure carrying the complete trace audit."""

    def __init__(self, message: str, audit: Sequence[Dict[str, Any]]):
        super().__init__(message)
        self.audit = list(audit)


def json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_splits(
    config: DreamerConfig,
    patterns: Sequence[str],
    max_traces: int = 0,
) -> Tuple[DatasetSplits, List[Dict[str, Any]]]:
    expanded = [str((ROOT / pattern) if not Path(pattern).is_absolute() else Path(pattern)) for pattern in patterns]
    paths = discover_traces(expanded)
    if max_traces > 0:
        paths = paths[:max_traces]
    episodes = []
    audit = []
    for path in paths:
        try:
            episode = build_episode(path, config)
        except Exception as exc:
            audit.append({"path": str(path), "accepted": False, "reason": str(exc)})
            continue
        if episode is None:
            audit.append({"path": str(path), "accepted": False, "reason": "too_short"})
            continue
        if episode.transitions < config.training.sequence_length:
            audit.append(
                {
                    "path": str(path),
                    "accepted": False,
                    "reason": "shorter_than_sequence_length",
                    "transitions": episode.transitions,
                }
            )
            continue
        source = str(episode.metadata.get("policy_source", "unknown"))
        if (
            config.training.source_policy == "simlingo_native"
            and source != "simlingo_native"
        ):
            audit.append(
                {
                    "path": str(path),
                    "accepted": False,
                    "reason": "policy_source_not_native_simlingo",
                    "policy_source": source,
                }
            )
            continue
        if (
            config.training.require_event_ground_truth
            and not bool(episode.metadata.get("event_ground_truth", False))
        ):
            audit.append(
                {
                    "path": str(path),
                    "accepted": False,
                    "reason": "missing_bench2drive_event_ground_truth",
                    "policy_source": source,
                }
            )
            continue
        episodes.append(episode)
        audit.append(
            {
                "path": str(path),
                "accepted": True,
                "transitions": episode.transitions,
                "seed": episode.seed,
                "route": episode.metadata.get("route_id"),
                "town": episode.metadata.get("town"),
                "scenario": episode.metadata.get("scenario"),
                "policy_source": source,
                "event_ground_truth": bool(
                    episode.metadata.get("event_ground_truth", False)
                ),
            }
        )
    if not episodes:
        raise DatasetAuditError(
            "no eligible ordered Dreamer traces were found", audit
        )
    try:
        splits = split_by_seed(episodes, config)
    except RuntimeError as exc:
        raise DatasetAuditError(str(exc), audit) from exc
    return splits, audit


def loader_for(
    episodes: Sequence[Any],
    config: DreamerConfig,
    shuffle: bool,
    max_windows: int = 0,
) -> DataLoader:
    dataset = SequenceDataset(episodes, config.training.sequence_length)
    if max_windows > 0 and len(dataset.indices) > max_windows:
        rng = random.Random(config.training.split_seed + (1 if shuffle else 2))
        rng.shuffle(dataset.indices)
        dataset.indices = dataset.indices[:max_windows]
    if not len(dataset):
        raise RuntimeError("split has no temporal windows")
    return DataLoader(
        dataset,
        batch_size=min(config.training.batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=torch.float32) for key, value in batch.items()}


def world_model_metrics(
    model: CompactRSSM,
    loader: DataLoader,
    config: DreamerConfig,
    device: torch.device,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = batch_to_device(raw, device)
            targets = {key: batch[key] for key in ("progress", "risk", "continuation", "value", "collision", "offroad")}
            _, losses = model.loss(
                batch["observations"], batch["actions"], targets, config.prediction_loss
            )
            batch_size = int(batch["observations"].shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
    return {key: value / max(1, count) for key, value in totals.items()}


def train_world_model(
    splits: DatasetSplits,
    config: DreamerConfig,
    output: Path,
    device: torch.device,
    epochs: Optional[int],
    max_windows: int,
) -> Path:
    training = loader_for(splits.train, config, True, max_windows)
    validation = loader_for(splits.validation, config, False, max_windows)
    model = CompactRSSM(config.rssm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    best = float("inf")
    checkpoint = output / "world_model_candidate.pt"
    history = []
    for epoch in range(1, int(epochs or config.training.world_model_epochs) + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for raw in training:
            batch = batch_to_device(raw, device)
            targets = {key: batch[key] for key in ("progress", "risk", "continuation", "value", "collision", "offroad")}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(
                batch["observations"], batch["actions"], targets, config.prediction_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            size = int(batch["observations"].shape[0])
            train_total += float(loss.detach().cpu()) * size
            train_count += size
        metrics = world_model_metrics(model, validation, config, device)
        row = {"epoch": epoch, "train_total": train_total / max(1, train_count), "validation": metrics}
        history.append(row)
        print("[world-model] epoch=%d train=%.5f val=%.5f" % (epoch, row["train_total"], metrics["total"]), flush=True)
        if metrics["total"] < best:
            best = metrics["total"]
            torch.save(
                {
                    "kind": "report_aligned_world_model",
                    "config": config.to_dict(),
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation": metrics,
                    "seed_sets": splits.seed_sets,
                },
                str(checkpoint),
            )
    json_dump(output / "world_model_history.json", {"history": history, "best_validation_total": best})
    return checkpoint


def load_world_model(path: Path, config: DreamerConfig, device: torch.device) -> CompactRSSM:
    payload = torch.load(str(path), map_location=device)
    model = CompactRSSM(config.rssm).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def start_latent(
    model: CompactRSSM,
    batch: Dict[str, torch.Tensor],
    config: DreamerConfig,
) -> Tuple[Any, torch.Tensor]:
    with torch.no_grad():
        state, _, _ = model.observe_sequence(
            batch["observations"], batch["actions"], deterministic=True
        )
    state = state.detach()
    native = batch["observations"][:, -1, 2:5].detach()
    return state, native


def policy_metrics(
    model: CompactRSSM,
    actor: ResidualActor,
    critic: LatentCritic,
    loader: DataLoader,
    config: DreamerConfig,
    device: torch.device,
) -> Dict[str, float]:
    totals: Dict[str, float] = {"objective": 0.0, "risk": 0.0, "collision": 0.0, "offroad": 0.0, "alpha": 0.0}
    count = 0
    actor.eval()
    critic.eval()
    for raw in loader:
        batch = batch_to_device(raw, device)
        start, native = start_latent(model, batch, config)
        imagined = imagine_actor_critic(model, actor, critic, start, native, config, deterministic=True)
        size = int(native.shape[0])
        count += size
        totals["objective"] += float(imagined.objective.detach().cpu()) * size
        totals["risk"] += float(imagined.mean_risk.detach().cpu()) * size
        totals["collision"] += float(imagined.mean_collision.detach().cpu()) * size
        totals["offroad"] += float(imagined.mean_offroad.detach().cpu()) * size
        totals["alpha"] += float(imagined.mean_alpha.detach().cpu()) * size
    return {key: value / max(1, count) for key, value in totals.items()}


def train_policy(
    splits: DatasetSplits,
    config: DreamerConfig,
    output: Path,
    device: torch.device,
    world_checkpoint: Path,
    epochs: Optional[int],
    max_windows: int,
) -> Path:
    model = load_world_model(world_checkpoint, config, device)
    actor = ResidualActor(model.feature_dim, config.policy).to(device)
    critic = LatentCritic(model.feature_dim, config.policy).to(device)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=config.training.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=config.training.critic_lr)
    training = loader_for(splits.train, config, True, max_windows)
    validation = loader_for(splits.validation, config, False, max_windows)
    checkpoint = output / "actor_critic_candidate.pt"
    initial_metrics = policy_metrics(
        model, actor, critic, validation, config, device
    )
    best = initial_metrics["objective"]
    history = [
        {
            "epoch": 0,
            "actor_loss": None,
            "critic_loss": None,
            "validation": initial_metrics,
            "selection_baseline": True,
        }
    ]
    torch.save(
        {
            "kind": "report_aligned_imagined_actor_critic",
            "config": config.to_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "epoch": 0,
            "validation": initial_metrics,
            "seed_sets": splits.seed_sets,
        },
        str(checkpoint),
    )
    print(
        "[policy] epoch=0 initialized val_objective=%.5f"
        % initial_metrics["objective"],
        flush=True,
    )
    for epoch in range(1, int(epochs or config.training.policy_epochs) + 1):
        actor.train()
        critic.train()
        actor_total = 0.0
        critic_total = 0.0
        count = 0
        for raw in training:
            batch = batch_to_device(raw, device)
            start, native = start_latent(model, batch, config)
            imagined = imagine_actor_critic(model, actor, critic, start, native, config)
            actor_optimizer.zero_grad(set_to_none=True)
            imagined.actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.training.gradient_clip)
            actor_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)
            imagined.critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.training.gradient_clip)
            critic_optimizer.step()
            size = int(native.shape[0])
            count += size
            actor_total += float(imagined.actor_loss.detach().cpu()) * size
            critic_total += float(imagined.critic_loss.detach().cpu()) * size
        metrics = policy_metrics(model, actor, critic, validation, config, device)
        row = {
            "epoch": epoch,
            "actor_loss": actor_total / max(1, count),
            "critic_loss": critic_total / max(1, count),
            "validation": metrics,
        }
        history.append(row)
        print("[policy] epoch=%d actor=%.5f critic=%.5f val_objective=%.5f" % (epoch, row["actor_loss"], row["critic_loss"], metrics["objective"]), flush=True)
        if metrics["objective"] > best:
            best = metrics["objective"]
            torch.save(
                {
                    "kind": "report_aligned_imagined_actor_critic",
                    "config": config.to_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "epoch": epoch,
                    "validation": metrics,
                    "seed_sets": splits.seed_sets,
                },
                str(checkpoint),
            )
    json_dump(output / "policy_history.json", {"history": history, "best_validation_objective": best})
    return checkpoint


def load_pairwise_rows(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    required = ("seed", "candidate_a_features", "candidate_b_features", "label")
    valid = []
    for row in rows:
        if all(key in row for key in required):
            a = np.asarray(row["candidate_a_features"], dtype=np.float32)
            b = np.asarray(row["candidate_b_features"], dtype=np.float32)
            label = row.get("label")
            if (
                a.shape == (5,)
                and b.shape == (5,)
                and np.isfinite(a).all()
                and np.isfinite(b).all()
                and label in (0, 1, 0.0, 1.0)
            ):
                valid.append(row)
    return valid


def pairwise_seed_split(rows: Sequence[Dict[str, Any]], config: DreamerConfig) -> Dict[str, List[Dict[str, Any]]]:
    seeds = sorted({str(row["seed"]) for row in rows})
    minimum_train = int(config.training.minimum_train_seeds)
    minimum_validation = int(config.training.minimum_validation_seeds)
    minimum_test = int(config.training.minimum_test_seeds)
    minimum_total = minimum_train + minimum_validation + minimum_test
    if len(seeds) < minimum_total:
        raise RuntimeError(
            "pairwise training needs at least %d seed groups" % minimum_total
        )
    random.Random(config.training.split_seed).shuffle(seeds)
    test_count = max(
        minimum_test,
        int(round(len(seeds) * config.training.test_ratio)),
    )
    validation_count = max(
        minimum_validation,
        int(round(len(seeds) * config.training.validation_ratio)),
    )
    while validation_count + test_count > len(seeds) - minimum_train:
        if validation_count >= test_count and validation_count > minimum_validation:
            validation_count -= 1
        elif test_count > minimum_test:
            test_count -= 1
        else:
            raise RuntimeError("pairwise seed split cannot satisfy configured minimums")
    test = set(seeds[:test_count])
    validation = set(seeds[test_count : test_count + validation_count])
    train = set(seeds) - test - validation
    if not train:
        raise RuntimeError("empty pairwise training seed set")
    return {
        "train": [row for row in rows if str(row["seed"]) in train],
        "validation": [row for row in rows if str(row["seed"]) in validation],
        "test": [row for row in rows if str(row["seed"]) in test],
    }


def pairwise_tensors(rows: Sequence[Dict[str, Any]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = torch.as_tensor([row["candidate_a_features"] for row in rows], dtype=torch.float32, device=device)
    second = torch.as_tensor([row["candidate_b_features"] for row in rows], dtype=torch.float32, device=device)
    label = torch.as_tensor([float(row["label"]) for row in rows], dtype=torch.float32, device=device)
    return first, second, label


def train_pairwise(
    path: Path,
    config: DreamerConfig,
    output: Path,
    device: torch.device,
    epochs: Optional[int],
) -> Path:
    rows = load_pairwise_rows(path)
    if not rows:
        raise RuntimeError(
            "pairwise dataset has no valid finite 5D comparison with a binary label"
        )
    splits = pairwise_seed_split(rows, config)
    model = PairwiseCalibrator(5, config.pairwise.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.pairwise_lr)
    train_a, train_b, train_y = pairwise_tensors(splits["train"], device)
    val_a, val_b, val_y = pairwise_tensors(splits["validation"], device)
    best = float("inf")
    checkpoint = output / "pairwise_candidate.pt"
    history = []
    for epoch in range(1, int(epochs or config.training.pairwise_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model.logits(train_a, train_b), train_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = F.binary_cross_entropy_with_logits(model.logits(val_a, val_b), val_y)
        row = {"epoch": epoch, "train_loss": float(loss.detach().cpu()), "validation_loss": float(validation_loss.detach().cpu())}
        history.append(row)
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            torch.save({"kind": "report_aligned_pairwise", "config": config.to_dict(), "model": model.state_dict()}, str(checkpoint))
    manifest = {
        "seed_sets": {key: sorted({str(row["seed"]) for row in values}) for key, values in splits.items()},
        "counts": {key: len(values) for key, values in splits.items()},
        "history": history,
    }
    json_dump(output / "pairwise_history.json", manifest)
    # The held-out split is opened once, after checkpoint selection.  Its
    # metrics are descriptive only and never influence training/checkpoint
    # choice.
    selected = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(selected["model"])
    test_a, test_b, test_y = pairwise_tensors(splits["test"], device)
    model.eval()
    with torch.no_grad():
        test_logits = model.logits(test_a, test_b)
        test_loss = F.binary_cross_entropy_with_logits(test_logits, test_y)
        test_accuracy = (
            (torch.sigmoid(test_logits) >= 0.5) == (test_y >= 0.5)
        ).float().mean()
    json_dump(
        output / "pairwise_test_metrics.json",
        {
            "loss": float(test_loss.cpu()),
            "accuracy": float(test_accuracy.cpu()),
            "examples": len(splits["test"]),
            "seeds": manifest["seed_sets"]["test"],
            "used_for_selection": False,
        },
    )
    return checkpoint


def compose_checkpoint(
    config: DreamerConfig,
    output: Path,
    world_checkpoint: Path,
    policy_checkpoint: Path,
    pairwise_checkpoint: Optional[Path],
    device: torch.device,
    manifest: Dict[str, Any],
) -> Path:
    world = torch.load(str(world_checkpoint), map_location=device)
    policy = torch.load(str(policy_checkpoint), map_location=device)
    checkpoint_config = copy.deepcopy(config)
    if pairwise_checkpoint is not None:
        checkpoint_config.runtime.ablation = "E"
        checkpoint_config.pairwise.enabled = True
    agent = SimLingoDreamerAgent(checkpoint_config, device=str(device))
    agent.world_model.load_state_dict(world["model"])
    agent.actor.load_state_dict(policy["actor"])
    agent.critic.load_state_dict(policy["critic"])
    if pairwise_checkpoint is not None and agent.pairwise is not None:
        pairwise = torch.load(str(pairwise_checkpoint), map_location=device)
        agent.pairwise.load_state_dict(pairwise["model"])
    destination = output / "report_dreamer_candidate.pt"
    agent.save(
        str(destination),
        metadata={
            "status": "candidate_not_promoted",
            "world_model_validation": world.get("validation"),
            "policy_validation": policy.get("validation"),
            "test_evaluated": False,
            "seed_sets": manifest.get("seed_sets"),
            "dataset_manifest": "dataset_manifest.json",
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("inspect", "world-model", "policy", "pairwise", "all"))
    parser.add_argument("--config", default=str(ROOT / "configs/dreamer_report_aligned.yaml"))
    parser.add_argument("--trace-glob", action="append", dest="trace_globs")
    parser.add_argument("--output", default=str(ROOT / "checkpoints/report_aligned_dreamer/candidate"))
    parser.add_argument("--world-checkpoint", default="")
    parser.add_argument("--policy-checkpoint", default="")
    parser.add_argument("--pairwise-data", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--source-policy",
        choices=("simlingo_native", "any"),
        default="",
        help=(
            "Production default is simlingo_native. 'any' is reserved for "
            "explicit diagnostics/smoke tests with historical traces."
        ),
    )
    parser.add_argument(
        "--allow-missing-event-ground-truth",
        action="store_true",
        help="Diagnostic only: permit traces without finalized Bench2Drive labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.sequence_length:
        config.training.sequence_length = args.sequence_length
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.source_policy:
        config.training.source_policy = args.source_policy
    if args.allow_missing_event_ground_truth:
        config.training.require_event_ground_truth = False
    config.validate()
    set_seed(config.training.split_seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    patterns = tuple(args.trace_globs or DEFAULT_TRACE_GLOBS)
    splits: Optional[DatasetSplits] = None
    manifest: Dict[str, Any] = {}
    if args.phase != "pairwise":
        try:
            splits, audit = prepare_splits(config, patterns, args.max_traces)
        except DatasetAuditError as exc:
            audit_failure = {
                "error": str(exc),
                "accepted": int(sum(bool(row.get("accepted")) for row in exc.audit)),
                "rejected": int(sum(not bool(row.get("accepted")) for row in exc.audit)),
                "audit": exc.audit,
                "config": config.to_dict(),
            }
            json_dump(output / "dataset_audit.json", audit_failure)
            print(json.dumps(audit_failure, indent=2), flush=True)
            raise
        json_dump(
            output / "dataset_audit.json",
            {
                "error": None,
                "accepted": int(sum(bool(row.get("accepted")) for row in audit)),
                "rejected": int(sum(not bool(row.get("accepted")) for row in audit)),
                "audit": audit,
                "config": config.to_dict(),
            },
        )
        manifest = split_manifest(splits)
        manifest["audit"] = audit
        manifest["config"] = config.to_dict()
        json_dump(output / "dataset_manifest.json", manifest)
        print(json.dumps({key: manifest[key] for key in ("train", "validation", "test")}, indent=2), flush=True)
    if args.phase == "inspect":
        return
    world_checkpoint = Path(args.world_checkpoint) if args.world_checkpoint else output / "world_model_candidate.pt"
    policy_checkpoint = Path(args.policy_checkpoint) if args.policy_checkpoint else output / "actor_critic_candidate.pt"
    pairwise_checkpoint: Optional[Path] = None
    epochs = args.epochs or None
    if args.phase in ("world-model", "all"):
        assert splits is not None
        world_checkpoint = train_world_model(splits, config, output, device, epochs, args.max_windows)
    if args.phase in ("policy", "all"):
        assert splits is not None
        if not world_checkpoint.exists():
            raise FileNotFoundError("world-model checkpoint not found: %s" % world_checkpoint)
        policy_checkpoint = train_policy(splits, config, output, device, world_checkpoint, epochs, args.max_windows)
    if args.phase in ("pairwise", "all") and args.pairwise_data:
        pairwise_checkpoint = train_pairwise(Path(args.pairwise_data), config, output, device, epochs)
    elif args.phase == "pairwise":
        raise ValueError("--pairwise-data is required; pairwise labels are never fabricated")
    if args.phase == "all":
        candidate = compose_checkpoint(
            config,
            output,
            world_checkpoint,
            policy_checkpoint,
            pairwise_checkpoint,
            device,
            manifest,
        )
        print("[candidate] %s" % candidate, flush=True)


if __name__ == "__main__":
    main()
