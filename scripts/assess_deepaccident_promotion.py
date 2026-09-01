#!/usr/bin/env python3
"""Apply the frozen DeepAccident offline promotion gate to an evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deepaccident.training import promotion_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-source-groups", type=int, default=30)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    decision = promotion_gate(evaluation, min_source_groups=args.min_source_groups)
    output = args.output or args.evaluation.with_name("promotion_decision.json")
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
