#!/usr/bin/env python3
"""Evaluate an exported DeepAccident risk encoder on a frozen split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deepaccident.data import DeepAccidentClipDataset
from src.deepaccident.risk_model import load_risk_encoder
from src.deepaccident.training import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clip-length", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=384)
    args = parser.parse_args()
    audit = json.loads((args.processed_dir / "audit.json").read_text(encoding="utf-8"))
    dataset_root = (args.dataset_root or Path(audit["dataset_root"])).resolve()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_risk_encoder(args.checkpoint, device)
    dataset = DeepAccidentClipDataset(
        dataset_root=dataset_root,
        manifest=args.processed_dir / "frames.jsonl",
        split=args.split,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    targets: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch["frames"].to(device))
            targets.append(batch["risk"].numpy())
            probabilities.append(output["risk"].cpu().numpy())
    metrics = binary_metrics(
        np.concatenate(targets),
        np.concatenate(probabilities),
        float(raw_checkpoint["decision_threshold"]),
    )
    print(json.dumps({"split": args.split, "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
