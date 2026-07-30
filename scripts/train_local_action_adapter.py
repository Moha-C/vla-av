#!/usr/bin/env python
"""Train a small local action adapter on a CARLA/Cosmos Transfer manifest."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.local_action_adapter import (
    LocalActionAdapter,
    LocalActionAdapterConfig,
    action_target_from_record,
    base_action_from_trajectory,
    build_feature_vector,
    checkpoint_payload,
    state_from_record,
    trajectory_xy_from_record,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/alpamayo_transfer_dataset_local_hq/manifest.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/local_action_adapter")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-points", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-speed-kmh", type=float, default=12.0)
    parser.add_argument("--max-throttle", type=float, default=0.30)
    parser.add_argument("--max-brake", type=float, default=0.70)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--require-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return records


def build_arrays(
    records: list[dict[str, Any]],
    *,
    manifest_path: Path,
    config: LocalActionAdapterConfig,
    require_images: bool,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    dataset_root = manifest_path.parent
    skipped_missing_images = 0
    skipped_invalid_records = 0

    for record in records:
        try:
            if require_images:
                image_path = dataset_root / str(record.get("photoreal_frame_path") or record.get("image_path") or "")
                if not image_path.exists():
                    skipped_missing_images += 1
                    continue
            trajectory = trajectory_xy_from_record(record)
            state = state_from_record(record)
            base_action = base_action_from_trajectory(
                trajectory,
                speed_kmh=float(state.get("speed_kmh", 0.0)),
                config=config,
            )
            feature = build_feature_vector(
                trajectory,
                state=state,
                base_action=base_action,
                config=config,
            )
            target = action_target_from_record(record)
        except (TypeError, ValueError, OverflowError):
            skipped_invalid_records += 1
            continue
        if not (np.isfinite(feature).all() and np.isfinite(target).all()):
            skipped_invalid_records += 1
            continue
        features.append(feature)
        targets.append(target)

    if skipped_missing_images:
        LOGGER.warning("Skipped %s records with missing images.", skipped_missing_images)
    if skipped_invalid_records:
        LOGGER.warning("Skipped %s records with invalid non-finite data.", skipped_invalid_records)
    if not features:
        raise RuntimeError("No usable records found in manifest.")

    return np.stack(features).astype(np.float32), np.stack(targets).astype(np.float32)


def split_indices(count: int, *, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(count * max(0.0, min(0.5, val_ratio)))))
    if count <= 2:
        val_count = 1
    return indices[val_count:], indices[:val_count]


def evaluate(
    model: LocalActionAdapter,
    loader: DataLoader,
    device: torch.device,
    weights: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    errors: list[torch.Tensor] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = (((predictions - targets) ** 2) * weights).mean()
            losses.append(float(loss.detach().cpu()))
            errors.append((predictions - targets).abs().detach().cpu())
    if not losses:
        return {"loss": float("nan"), "mae_steer": float("nan"), "mae_throttle": float("nan"), "mae_brake": float("nan")}
    mae = torch.cat(errors, dim=0).mean(dim=0)
    return {
        "loss": float(np.mean(losses)),
        "mae_steer": float(mae[0]),
        "mae_throttle": float(mae[1]),
        "mae_brake": float(mae[2]),
    }


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    logging.basicConfig(
        level=getattr(logging, args.log_level) if rank == 0 else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    requested_device = args.device
    if distributed:
        backend = "nccl" if requested_device != "cpu" and torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if backend == "nccl":
            torch.cuda.set_device(local_rank)

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    config = LocalActionAdapterConfig(
        feature_points=int(args.feature_points),
        hidden_dim=int(args.hidden_dim),
        rank=int(args.rank),
        dropout=float(args.dropout),
        target_speed_kmh=float(args.target_speed_kmh),
        max_throttle=float(args.max_throttle),
        max_brake=float(args.max_brake),
    )
    records = read_manifest(manifest_path)
    features, targets = build_arrays(
        records,
        manifest_path=manifest_path,
        config=config,
        require_images=bool(args.require_images),
    )
    train_idx, val_idx = split_indices(len(features), val_ratio=float(args.val_ratio), seed=int(args.seed))
    if not train_idx:
        train_idx = val_idx

    device_name = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if device_name == "auto":
        device_name = "cpu"
    if distributed and device_name == "cuda":
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)

    train_ds = TensorDataset(
        torch.from_numpy(features[train_idx]),
        torch.from_numpy(targets[train_idx]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(features[val_idx]),
        torch.from_numpy(targets[val_idx]),
    )
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=int(args.seed))
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)

    model = LocalActionAdapter(input_dim=int(features.shape[1]), config=config).to(device)
    raw_model = model
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    weights = torch.tensor([2.0, 1.0, 2.0], dtype=torch.float32, device=device)

    if rank == 0:
        LOGGER.info(
            "Training local action adapter: records=%s train=%s val=%s input_dim=%s device=%s world_size=%s",
            len(features),
            len(train_ds),
            len(val_ds),
            features.shape[1],
            device,
            world_size,
        )

    best_loss = float("inf")
    best_metrics: dict[str, float] = {}
    epoch_iter = range(1, int(args.epochs) + 1)
    progress = (
        tqdm(
            epoch_iter,
            desc="training",
            unit="epoch",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
        )
        if rank == 0
        else epoch_iter
    )
    for epoch in progress:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            loss = (((predictions - batch_targets) ** 2) * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        if distributed:
            dist.barrier()
        metrics = evaluate(raw_model, val_loader, device, weights) if rank == 0 else {}
        if distributed:
            dist.barrier()
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        if rank == 0:
            progress.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{metrics['loss']:.4f}",
                steer=f"{metrics['mae_steer']:.3f}",
                throttle=f"{metrics['mae_throttle']:.3f}",
                brake=f"{metrics['mae_brake']:.3f}",
            )
        if rank == 0 and metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_metrics = {"epoch": float(epoch), "train_loss": train_loss, **metrics}
            torch.save(
                checkpoint_payload(raw_model, config, metrics=best_metrics),
                output_dir / "best.pt",
            )

    if distributed:
        dist.barrier()
    if rank == 0:
        final_metrics = evaluate(raw_model, val_loader, device, weights)
        torch.save(
            checkpoint_payload(
                raw_model,
                config,
                metrics={"final": final_metrics, "best": best_metrics},
            ),
            output_dir / "last.pt",
        )
        summary = {
            "manifest": str(manifest_path),
            "records": int(len(features)),
            "train_records": int(len(train_ds)),
            "val_records": int(len(val_ds)),
            "input_dim": int(features.shape[1]),
            "world_size": int(world_size),
            "best": best_metrics,
            "final": final_metrics,
            "checkpoint": str(output_dir / "best.pt"),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("Best checkpoint: %s", output_dir / "best.pt")
        LOGGER.info("Summary: %s", output_dir / "summary.json")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
