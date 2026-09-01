#!/usr/bin/env python3
"""Index and audit an extracted DeepAccident dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deepaccident.index import DeepAccidentIndexConfig, build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "deepaccident" / "processed" / "mini",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--prediction-horizon-s", type=float, default=2.0)
    parser.add_argument("--split-seed", type=int, default=230401168)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--allow-missing-labels", action="store_true")
    args = parser.parse_args()
    audit = build_index(
        args.dataset_root,
        args.output_dir,
        DeepAccidentIndexConfig(
            fps=args.fps,
            prediction_horizon_s=args.prediction_horizon_s,
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            require_labels=not args.allow_missing_labels,
        ),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
