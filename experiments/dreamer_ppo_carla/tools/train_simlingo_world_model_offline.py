#!/usr/bin/env python3
"""Offline world-model test on converted SimLingo Action Dreaming data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.world_model import WorldModel


def standardize(train, *arrays, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + eps
    return mean.astype(np.float32), std.astype(np.float32), [
        ((arr - mean) / std).astype(np.float32) for arr in arrays
    ]


def make_loader(data, indices, batch_size, shuffle=True):
    idx = np.array(indices, dtype=np.int64)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield {k: torch.as_tensor(v[batch_idx]) for k, v in data.items()}


@torch.no_grad()
def evaluate(model, data, indices, batch_size, device):
    model.eval()
    losses = []
    state_mae = []
    risk_mae = []
    progress_mae = []
    for batch in make_loader(data, indices, batch_size, shuffle=False):
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        next_states = batch["next_states"].to(device)
        risks = batch["risk_targets"].to(device)
        progresses = batch["progress_targets"].to(device)
        ns_hat, risk_hat, prog_hat = model(states, actions)
        loss_state = F.mse_loss(ns_hat, next_states)
        loss_risk = F.mse_loss(risk_hat.squeeze(-1), risks)
        loss_progress = F.mse_loss(prog_hat.squeeze(-1), progresses)
        losses.append(float((loss_state + loss_risk + loss_progress).item()))
        state_mae.append(float((ns_hat - next_states).abs().mean().item()))
        risk_mae.append(float((risk_hat.squeeze(-1) - risks).abs().mean().item()))
        progress_mae.append(float((prog_hat.squeeze(-1) - progresses).abs().mean().item()))
    return {
        "loss": float(np.mean(losses)),
        "state_mae_norm": float(np.mean(state_mae)),
        "risk_mae": float(np.mean(risk_mae)),
        "progress_mae_norm": float(np.mean(progress_mae)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/simlingo_world_model")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    raw = np.load(args.data, allow_pickle=True)
    n = raw["states"].shape[0]
    idx = np.random.permutation(n)
    n_train = int(n * 0.8)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    state_mean, state_std, norm_states = standardize(
        raw["states"][train_idx], raw["states"], raw["next_states"]
    )
    action_mean, action_std, norm_actions = standardize(
        raw["actions"][train_idx], raw["actions"]
    )
    progress_mean, progress_std, norm_progress = standardize(
        raw["progress_targets"][train_idx, None], raw["progress_targets"][:, None]
    )

    data = {
        "states": norm_states[0],
        "actions": norm_actions[0],
        "next_states": norm_states[1],
        "risk_targets": raw["risk_targets"].astype(np.float32),
        "progress_targets": norm_progress[0].reshape(-1).astype(np.float32),
    }

    model = WorldModel(state_dim=data["states"].shape[1],
                       action_dim=data["actions"].shape[1],
                       hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = None
    history = []

    print(f"device={device} transitions={n} train={len(train_idx)} val={len(val_idx)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for batch in make_loader(data, train_idx, args.batch_size, shuffle=True):
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            next_states = batch["next_states"].to(device)
            risks = batch["risk_targets"].to(device)
            progresses = batch["progress_targets"].to(device)
            ns_hat, risk_hat, prog_hat = model(states, actions)
            loss_state = F.mse_loss(ns_hat, next_states)
            loss_risk = F.mse_loss(risk_hat.squeeze(-1), risks)
            loss_progress = F.mse_loss(prog_hat.squeeze(-1), progresses)
            loss = loss_state + loss_risk + loss_progress
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss.item()))

        val = evaluate(model, data, val_idx, args.batch_size, device)
        rec = {"epoch": epoch, "train_loss": float(np.mean(batch_losses)), **val}
        history.append(rec)
        if best is None or val["loss"] < best["loss"]:
            best = rec
            torch.save({
                "model": model.state_dict(),
                "state_mean": state_mean,
                "state_std": state_std,
                "action_mean": action_mean,
                "action_std": action_std,
                "progress_mean": progress_mean,
                "progress_std": progress_std,
                "config": vars(args),
                "best": best,
            }, out_dir / "best_world_model.pt")

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} train={rec['train_loss']:.4f} "
                f"val={rec['loss']:.4f} state_mae={rec['state_mae_norm']:.4f} "
                f"risk_mae={rec['risk_mae']:.4f} progress_mae={rec['progress_mae_norm']:.4f}"
            )

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"best": best, "transitions": int(n), "data": args.data}, f, indent=2)
    print(f"best={best}")
    print(f"saved={out_dir / 'best_world_model.pt'}")


if __name__ == "__main__":
    main()
