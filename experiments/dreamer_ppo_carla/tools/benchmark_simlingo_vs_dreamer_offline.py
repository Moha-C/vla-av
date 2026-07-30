#!/usr/bin/env python3
"""Offline benchmark: SimLingo Action Dreaming heuristic vs trained Dreamer WM.

This does not claim closed-loop driving improvement. It answers a narrower
question: given the same SimLingo candidate actions, how often does the trained
Dreamer world model choose a different action, and how does that choice compare
against the original candidate metadata?
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from models.world_model import WorldModel
from tools.simlingo_jsonl_to_dreamer_npz import build_action, build_state


def normalize(x, mean, std):
    return (x - mean.reshape(-1)) / std.reshape(-1)


def candidate_meta_score(candidate):
    score = candidate.get("score") or {}
    return float(score.get("score", 0.0))


def candidate_meta_risk(candidate):
    score = candidate.get("score") or {}
    return float(score.get("risk", 0.0))


def candidate_meta_progress(candidate):
    score = candidate.get("score") or {}
    return float(score.get("progress", 0.0))


@torch.no_grad()
def dreamer_scores(model, sample, ckpt, w_progress, w_risk):
    state_mean = ckpt["state_mean"].reshape(-1)
    state_std = ckpt["state_std"].reshape(-1)
    action_mean = ckpt["action_mean"].reshape(-1)
    action_std = ckpt["action_std"].reshape(-1)
    progress_mean = float(ckpt["progress_mean"].reshape(-1)[0])
    progress_std = float(ckpt["progress_std"].reshape(-1)[0])

    state = build_state(sample)
    state_n = torch.as_tensor(
        normalize(state, state_mean, state_std),
        dtype=torch.float32,
    ).unsqueeze(0)

    rows = []
    for idx, cand in enumerate(sample.get("candidate_actions") or []):
        action = cand.get("action") or {}
        action_vec = build_action({"action_star": action}, "action_star")
        action_n = torch.as_tensor(
            normalize(action_vec, action_mean, action_std),
            dtype=torch.float32,
        ).unsqueeze(0)
        _ns, risk_hat, progress_hat = model(state_n, action_n)
        risk = float(risk_hat.item())
        progress = float(progress_hat.item() * progress_std + progress_mean)
        score = w_progress * progress - w_risk * risk
        rows.append({
            "candidate_index": idx,
            "dreamer_score": score,
            "dreamer_risk": risk,
            "dreamer_progress": progress,
            "meta_score": candidate_meta_score(cand),
            "meta_risk": candidate_meta_risk(cand),
            "meta_progress": candidate_meta_progress(cand),
            "action": action_vec.tolist(),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/simlingo_vs_dreamer_benchmark")
    parser.add_argument("--w-progress", type=float, default=1.0)
    parser.add_argument("--w-risk", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = WorldModel(state_dim=28, action_dim=4, hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_sample.csv"
    summary_path = out_dir / "summary.json"

    totals = Counter()
    scenario = defaultdict(Counter)
    numeric = defaultdict(list)
    sample_rows = []

    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if args.limit and totals["samples"] >= args.limit:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            candidates = sample.get("candidate_actions") or []
            if not candidates:
                totals["no_candidates"] += 1
                continue

            scored = dreamer_scores(model, sample, ckpt, args.w_progress, args.w_risk)
            if not scored:
                totals["no_scored_candidates"] += 1
                continue

            dreamer_best = max(scored, key=lambda r: r["dreamer_score"])
            meta_best = max(scored, key=lambda r: r["meta_score"])
            simlingo_idx = 0
            dreamer_idx = dreamer_best["candidate_index"]
            meta_idx = meta_best["candidate_index"]
            scen = sample.get("scenario_type", "unknown")

            totals["samples"] += 1
            totals["dreamer_same_as_simlingo"] += int(dreamer_idx == simlingo_idx)
            totals["dreamer_same_as_meta_best"] += int(dreamer_idx == meta_idx)
            totals[f"dreamer_idx_{dreamer_idx}"] += 1
            scenario[scen]["samples"] += 1
            scenario[scen]["dreamer_same_as_simlingo"] += int(dreamer_idx == simlingo_idx)
            scenario[scen]["dreamer_same_as_meta_best"] += int(dreamer_idx == meta_idx)

            simlingo_row = scored[simlingo_idx]
            numeric["dreamer_score_delta_vs_simlingo"].append(
                dreamer_best["dreamer_score"] - simlingo_row["dreamer_score"]
            )
            numeric["meta_score_delta_vs_simlingo"].append(
                dreamer_best["meta_score"] - simlingo_row["meta_score"]
            )
            numeric["meta_risk_delta_vs_simlingo"].append(
                dreamer_best["meta_risk"] - simlingo_row["meta_risk"]
            )
            numeric["meta_progress_delta_vs_simlingo"].append(
                dreamer_best["meta_progress"] - simlingo_row["meta_progress"]
            )
            numeric["dreamer_best_risk"].append(dreamer_best["dreamer_risk"])
            numeric["dreamer_best_progress"].append(dreamer_best["dreamer_progress"])

            sample_rows.append({
                "run_id": sample.get("run_id", ""),
                "sample_index": sample.get("sample_index", ""),
                "town": sample.get("town", ""),
                "scenario": scen,
                "simlingo_idx": simlingo_idx,
                "dreamer_idx": dreamer_idx,
                "meta_best_idx": meta_idx,
                "dreamer_same_as_simlingo": int(dreamer_idx == simlingo_idx),
                "dreamer_same_as_meta_best": int(dreamer_idx == meta_idx),
                "dreamer_score_delta_vs_simlingo": numeric["dreamer_score_delta_vs_simlingo"][-1],
                "meta_score_delta_vs_simlingo": numeric["meta_score_delta_vs_simlingo"][-1],
                "meta_risk_delta_vs_simlingo": numeric["meta_risk_delta_vs_simlingo"][-1],
                "meta_progress_delta_vs_simlingo": numeric["meta_progress_delta_vs_simlingo"][-1],
                "dreamer_best_risk": dreamer_best["dreamer_risk"],
                "dreamer_best_progress": dreamer_best["dreamer_progress"],
            })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    def stats(values):
        arr = np.asarray(values, dtype=np.float32)
        return {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
            "min": float(arr.min()) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
        }

    samples = max(totals["samples"], 1)
    scenario_summary = {}
    for scen, c in sorted(scenario.items()):
        n = max(c["samples"], 1)
        scenario_summary[scen] = {
            "samples": int(c["samples"]),
            "dreamer_same_as_simlingo_rate": c["dreamer_same_as_simlingo"] / n,
            "dreamer_same_as_meta_best_rate": c["dreamer_same_as_meta_best"] / n,
        }

    summary = {
        "jsonl": args.jsonl,
        "checkpoint": args.checkpoint,
        "weights": {"w_progress": args.w_progress, "w_risk": args.w_risk},
        "samples": int(totals["samples"]),
        "dreamer_same_as_simlingo_rate": totals["dreamer_same_as_simlingo"] / samples,
        "dreamer_same_as_meta_best_rate": totals["dreamer_same_as_meta_best"] / samples,
        "dreamer_candidate_index_counts": {
            key.replace("dreamer_idx_", ""): int(value)
            for key, value in totals.items()
            if key.startswith("dreamer_idx_")
        },
        "metrics": {key: stats(value) for key, value in numeric.items()},
        "scenario_summary": scenario_summary,
        "csv": str(csv_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps({
        "samples": summary["samples"],
        "dreamer_same_as_simlingo_rate": summary["dreamer_same_as_simlingo_rate"],
        "dreamer_same_as_meta_best_rate": summary["dreamer_same_as_meta_best_rate"],
        "candidate_counts": summary["dreamer_candidate_index_counts"],
        "metrics": summary["metrics"],
        "summary": str(summary_path),
        "csv": str(csv_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
