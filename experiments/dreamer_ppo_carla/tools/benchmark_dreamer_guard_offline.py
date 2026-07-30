#!/usr/bin/env python3
"""Offline benchmark for a conservative SimLingo + Dreamer guard.

Policy:
  - keep SimLingo's current action_star by default;
  - allow Dreamer to override only if another candidate has lower predicted
    risk by at least `risk_margin`;
  - and its original SimLingo candidate score is not much worse than action_star.

This is the safer first integration path: Dreamer acts as a risk-aware guard,
not as a replacement driver.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from models.world_model import WorldModel
from tools.benchmark_simlingo_vs_dreamer_offline import (
    candidate_meta_progress,
    candidate_meta_risk,
    candidate_meta_score,
    dreamer_scores,
)


def choose_guard(scored, max_meta_drop, risk_margin):
    sim = scored[0]
    eligible = []
    for row in scored[1:]:
        risk_drop = sim["dreamer_risk"] - row["dreamer_risk"]
        meta_drop = sim["meta_score"] - row["meta_score"]
        if risk_drop >= risk_margin and meta_drop <= max_meta_drop:
            eligible.append((risk_drop, row["dreamer_score"], row))
    if not eligible:
        return sim, False
    # Prefer strongest risk reduction, then Dreamer score.
    eligible.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return eligible[0][2], True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/simlingo_dreamer_guard")
    parser.add_argument("--w-progress", type=float, default=1.0)
    parser.add_argument("--w-risk", type=float, default=2.0)
    parser.add_argument("--max-meta-drop", type=float, default=0.05)
    parser.add_argument("--risk-margin", type=float, default=0.05)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = WorldModel(state_dim=28, action_dim=4, hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    totals = Counter()
    scenario = defaultdict(Counter)
    numeric = defaultdict(list)

    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            if not sample.get("candidate_actions"):
                continue
            scored = dreamer_scores(model, sample, ckpt, args.w_progress, args.w_risk)
            if not scored:
                continue
            chosen, override = choose_guard(scored, args.max_meta_drop, args.risk_margin)
            sim = scored[0]
            scen = sample.get("scenario_type", "unknown")

            totals["samples"] += 1
            totals["overrides"] += int(override)
            totals[f"chosen_idx_{chosen['candidate_index']}"] += 1
            scenario[scen]["samples"] += 1
            scenario[scen]["overrides"] += int(override)

            numeric["dreamer_risk_delta_vs_simlingo"].append(chosen["dreamer_risk"] - sim["dreamer_risk"])
            numeric["dreamer_progress_delta_vs_simlingo"].append(chosen["dreamer_progress"] - sim["dreamer_progress"])
            numeric["meta_score_delta_vs_simlingo"].append(chosen["meta_score"] - sim["meta_score"])
            numeric["meta_risk_delta_vs_simlingo"].append(chosen["meta_risk"] - sim["meta_risk"])
            numeric["meta_progress_delta_vs_simlingo"].append(chosen["meta_progress"] - sim["meta_progress"])

    def stats(values):
        arr = np.asarray(values, dtype=np.float32)
        return {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
            "min": float(arr.min()) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
        }

    samples = max(totals["samples"], 1)
    summary = {
        "jsonl": args.jsonl,
        "checkpoint": args.checkpoint,
        "policy": {
            "w_progress": args.w_progress,
            "w_risk": args.w_risk,
            "max_meta_drop": args.max_meta_drop,
            "risk_margin": args.risk_margin,
        },
        "samples": int(totals["samples"]),
        "override_rate": totals["overrides"] / samples,
        "overrides": int(totals["overrides"]),
        "chosen_candidate_index_counts": {
            key.replace("chosen_idx_", ""): int(value)
            for key, value in totals.items()
            if key.startswith("chosen_idx_")
        },
        "metrics": {key: stats(value) for key, value in numeric.items()},
        "scenario_summary": {
            scen: {
                "samples": int(c["samples"]),
                "override_rate": c["overrides"] / max(c["samples"], 1),
                "overrides": int(c["overrides"]),
            }
            for scen, c in sorted(scenario.items())
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps({
        "samples": summary["samples"],
        "overrides": summary["overrides"],
        "override_rate": summary["override_rate"],
        "chosen_candidate_index_counts": summary["chosen_candidate_index_counts"],
        "metrics": summary["metrics"],
        "summary": str(summary_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
