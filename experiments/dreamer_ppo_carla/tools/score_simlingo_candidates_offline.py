#!/usr/bin/env python3
"""Use the trained world model to re-score SimLingo candidate actions offline."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from models.world_model import WorldModel
from tools.simlingo_jsonl_to_dreamer_npz import build_action, build_state


def normalize(x, mean, std):
    return (x - mean.reshape(-1)) / std.reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--w-progress", type=float, default=1.0)
    parser.add_argument("--w-risk", type=float, default=2.0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = WorldModel(state_dim=28, action_dim=4, hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    state_mean = ckpt["state_mean"].reshape(-1)
    state_std = ckpt["state_std"].reshape(-1)
    action_mean = ckpt["action_mean"].reshape(-1)
    action_std = ckpt["action_std"].reshape(-1)
    progress_mean = float(ckpt["progress_mean"].reshape(-1)[0])
    progress_std = float(ckpt["progress_std"].reshape(-1)[0])

    total = 0
    with_candidates = 0
    same_as_first = 0
    selected = Counter()
    scenario_counts = Counter()
    score_gaps = []
    pred_risks = []
    pred_progresses = []

    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if args.limit and total >= args.limit:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            total += 1
            candidates = sample.get("candidate_actions") or []
            if not candidates:
                continue
            with_candidates += 1
            scenario_counts[sample.get("scenario_type", "unknown")] += 1

            state = build_state(sample)
            state_n = torch.as_tensor(normalize(state, state_mean, state_std), dtype=torch.float32).unsqueeze(0)
            scores = []
            risks = []
            progresses = []
            for cand in candidates:
                action = cand.get("action") or {}
                tmp = {"action_star": action}
                action_vec = build_action(tmp, "action_star")
                action_n = torch.as_tensor(normalize(action_vec, action_mean, action_std), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _ns, risk_hat, progress_hat = model(state_n, action_n)
                risk = float(risk_hat.item())
                progress = float(progress_hat.item() * progress_std + progress_mean)
                score = args.w_progress * progress - args.w_risk * risk
                scores.append(score)
                risks.append(risk)
                progresses.append(progress)

            best = int(np.argmax(scores))
            selected[best] += 1
            same_as_first += int(best == 0)
            pred_risks.append(risks[best])
            pred_progresses.append(progresses[best])
            if len(scores) > 1:
                ordered = sorted(scores, reverse=True)
                score_gaps.append(ordered[0] - ordered[1])

    print(f"total_rows={total} with_candidates={with_candidates}")
    print(f"same_as_original_action_star={same_as_first}/{with_candidates} ({same_as_first / max(with_candidates, 1):.3f})")
    print(f"selected_candidate_index={dict(selected)}")
    print(f"avg_best_pred_risk={np.mean(pred_risks):.4f}")
    print(f"avg_best_pred_progress={np.mean(pred_progresses):.6f}")
    print(f"avg_top1_top2_score_gap={np.mean(score_gaps):.6f}")
    print(f"scenario_counts={dict(scenario_counts)}")


if __name__ == "__main__":
    main()
