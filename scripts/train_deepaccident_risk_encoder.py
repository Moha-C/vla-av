#!/usr/bin/env python3
"""Pretrain the isolated DeepAccident temporal risk representation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deepaccident.data import DeepAccidentClipDataset
from src.deepaccident.risk_model import DeepAccidentRiskEncoder, RiskEncoderConfig
from src.deepaccident.training import binary_metrics, promotion_gate, risk_ttc_loss


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _progress(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _evaluate(
    model: DeepAccidentRiskEncoder,
    loader: DataLoader,
    device: torch.device,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    targets: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    ttc_errors: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            output = model(frames)
            target = batch["risk"].numpy()
            targets.append(target)
            probabilities.append(output["risk"].cpu().numpy())
            mask = batch["ttc_mask"].numpy() > 0.5
            if mask.any():
                error = np.abs(output["ttc_s"].cpu().numpy()[mask] - batch["ttc_s"].numpy()[mask])
                ttc_errors.append(error)
    target_array = np.concatenate(targets)
    probability_array = np.concatenate(probabilities)
    metrics = binary_metrics(target_array, probability_array, threshold)
    metrics["ttc_mae_s"] = float(np.concatenate(ttc_errors).mean()) if ttc_errors else float("nan")
    metrics["examples"] = float(target_array.size)
    return metrics, target_array, probability_array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "deepaccident" / "processed" / "mini",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "checkpoints" / "deepaccident" / "risk_encoder_mini",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--temporal-dim", type=int, default=192)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=230401168)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    audit_path = args.processed_dir / "audit.json"
    manifest_path = args.processed_dir / "frames.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    dataset_root = (args.dataset_root or Path(audit["dataset_root"])).resolve()
    prediction_horizon_s = float(audit["target_semantics"]["prediction_horizon_s"])
    _set_seed(args.seed)
    device = _device(args.device)
    pin_memory = device.type == "cuda"

    common = dict(
        dataset_root=dataset_root,
        manifest=manifest_path,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        image_height=args.image_height,
        image_width=args.image_width,
        seed=args.seed,
        max_clips=args.max_clips,
    )
    training = DeepAccidentClipDataset(split="train", augment=True, **common)
    validation = DeepAccidentClipDataset(split="validation", augment=False, **common)
    testing = DeepAccidentClipDataset(split="test", augment=False, **common)
    train_loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        testing,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    targets = training.targets()
    positive = float(targets.sum())
    negative = float(targets.numel() - positive)
    if positive <= 0.0 or negative <= 0.0:
        raise RuntimeError("training clips need both positive and negative risk targets")
    positive_weight = torch.tensor(min(20.0, negative / positive), device=device)

    model_config = RiskEncoderConfig(
        embedding_dim=args.embedding_dim,
        temporal_dim=args.temporal_dim,
        pretrained_backbone=not args.no_pretrained_backbone,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        prediction_horizon_s=prediction_horizon_s,
    )
    model = DeepAccidentRiskEncoder(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_risk_encoder.pt"
    history = []
    best_average_precision = -math.inf
    epochs_without_improvement = 0
    start_time = time.time()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    print(
        "[deepaccident] device=%s train=%d validation=%d test=%d pos_weight=%.3f"
        % (device, len(training), len(validation), len(testing), float(positive_weight.cpu())),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        training.set_epoch(epoch)
        frozen = epoch <= args.freeze_backbone_epochs
        model.freeze_backbone(frozen)
        model.train()
        running = 0.0
        examples = 0
        for batch_index, batch in enumerate(train_loader, 1):
            frames = batch["frames"].to(device, non_blocking=True)
            risk = batch["risk"].to(device, non_blocking=True)
            ttc_s = batch["ttc_s"].to(device, non_blocking=True)
            ttc_mask = batch["ttc_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(frames)
                loss, _ = risk_ttc_loss(
                    output,
                    risk,
                    ttc_s,
                    ttc_mask,
                    positive_weight,
                    prediction_horizon_s,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            size = int(frames.shape[0])
            running += float(loss.detach().cpu()) * size
            examples += size
            if batch_index % 25 == 0 or batch_index == len(train_loader):
                elapsed = time.time() - start_time
                completed = (epoch - 1) * len(train_loader) + batch_index
                total = args.epochs * len(train_loader)
                eta = elapsed / max(1, completed) * max(0, total - completed)
                _progress(
                    args.output_dir / "training_progress.jsonl",
                    {
                        "event": "batch",
                        "epoch": epoch,
                        "epochs_total": args.epochs,
                        "batch": batch_index,
                        "batches_total": len(train_loader),
                        "progress_percent": round(100.0 * completed / total, 3),
                        "loss": float(loss.detach().cpu()),
                        "elapsed_seconds": round(elapsed, 2),
                        "eta_seconds": round(eta, 2),
                    },
                )

        validation_metrics, _, _ = _evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "backbone_frozen": frozen,
            "train_loss": running / max(1, examples),
            "validation": validation_metrics,
        }
        history.append(row)
        print(
            "[deepaccident] epoch=%d loss=%.5f val_ap=%.4f val_auc=%.4f val_f1=%.4f"
            % (
                epoch,
                row["train_loss"],
                validation_metrics["average_precision"],
                validation_metrics["roc_auc"],
                validation_metrics["f1"],
            ),
            flush=True,
        )
        score = validation_metrics["average_precision"]
        if score > best_average_precision + 1.0e-6:
            best_average_precision = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "kind": "deepaccident_temporal_risk_encoder",
                    "schema_version": 1,
                    "model_config": model_config.to_dict(),
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "decision_threshold": validation_metrics["threshold"],
                    "manifest_sha256": manifest_sha256,
                    "target_semantics": audit["target_semantics"],
                    "control_policy": None,
                    "provenance": {
                        "dataset": "DeepAccident",
                        "project": "https://deepaccident.github.io/",
                        "camera": "ego_vehicle/Camera_Front",
                    },
                },
                str(checkpoint_path),
            )
        else:
            epochs_without_improvement += 1
        _json_dump(
            args.output_dir / "history.json",
            {"history": history, "best_validation_average_precision": best_average_precision},
        )
        if epochs_without_improvement >= args.patience:
            print("[deepaccident] early stopping", flush=True)
            break

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, _, _ = _evaluate(
        model, test_loader, device, float(checkpoint["decision_threshold"])
    )
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "validation": checkpoint["validation"],
        "test": test_metrics,
        "dataset_audit": audit,
        "dataset_clips": {
            "train": len(training),
            "validation": len(validation),
            "test": len(testing),
        },
        "device": str(device),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "runtime_integration": "not_promoted",
    }
    summary["promotion_gate"] = promotion_gate(summary)
    _json_dump(args.output_dir / "evaluation.json", summary)
    _json_dump(args.output_dir / "promotion_decision.json", summary["promotion_gate"])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
