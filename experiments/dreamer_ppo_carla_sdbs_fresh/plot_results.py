"""Plot training curves from a CSV produced by training.logger.Logger.

Usage:
    python plot_results.py --log logs/training_log.csv

Produces PNGs in logs/plots/:
    return.png            episode return (raw + smoothed, window=10)
    vru_collisions.png    VRU collisions per episode
    ppo_loss.png          PPO loss over time
    vru_risk.png          VRU risk reward over time      (if column present)
    progress.png          progress reward over time      (if column present)
    collision_rate.png    collision rate over time       (if column present)
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


def _read_csv(path):
    """Read the log CSV into a dict of column-name -> list of floats."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No data rows in {path}")
    columns = {}
    for key in rows[0]:
        columns[key] = [float(r[key]) for r in rows]
    return columns


def _smooth(values, window=10):
    """Trailing moving average; output has the same length as `values`."""
    if window <= 1 or len(values) < 2:
        return list(values)
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _line_plot(x, series, title, ylabel, out_path):
    plt.figure(figsize=(8, 4.5))
    for label, y, style in series:
        plt.plot(x, y, style, label=label)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series) > 1:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument("--log", default="logs/training_log.csv",
                        help="path to the training CSV log")
    parser.add_argument("--window", type=int, default=10,
                        help="smoothing window for the return curve")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f"Log file not found: {args.log}")

    data = _read_csv(args.log)
    episodes = data["episode"]
    out_dir = os.path.join(os.path.dirname(args.log) or ".", "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Episode return (raw + smoothed).
    returns = data.get("return", [])
    _line_plot(
        episodes,
        [
            ("return", returns, "-"),
            (f"smoothed (w={args.window})", _smooth(returns, args.window), "-"),
        ],
        "Episode return over time", "Return",
        os.path.join(out_dir, "return.png"),
    )

    # VRU collisions per episode.
    _line_plot(
        episodes,
        [("vru_collisions", data.get("vru_collisions", []), "-")],
        "VRU collisions per episode", "Collisions",
        os.path.join(out_dir, "vru_collisions.png"),
    )

    # PPO loss over time.
    _line_plot(
        episodes,
        [("ppo_loss", data.get("ppo_loss", []), "-")],
        "PPO loss over time", "Loss",
        os.path.join(out_dir, "ppo_loss.png"),
    )

    # Optional reward-component plots (present in newer CSV logs).
    optional = [
        ("r_vru", "VRU risk reward over time", "VRU risk reward", "vru_risk.png"),
        ("r_progress", "Progress reward over time", "Progress reward", "progress.png"),
        ("vru_collisions", "Collision rate over time", "Collisions / episode",
         "collision_rate.png"),
        ("loss_wm", "World model loss over episodes", "WM total loss", "wm_loss.png"),
        ("wm_state_err", "State prediction error over episodes",
         "State MAE", "wm_state_err.png"),
        ("wm_risk_err", "Risk prediction error over episodes",
         "Risk MAE", "wm_risk_err.png"),
    ]
    for column, title, ylabel, fname in optional:
        if column in data:
            _line_plot(
                episodes,
                [(column, data[column], "-")],
                title, ylabel, os.path.join(out_dir, fname),
            )

    # Tier-3 safety-focused plots (only when the safety columns are present).
    _safety_plots(data, episodes, out_dir, args.window)

    print(f"\nDone. Plots written to {out_dir}/")


def _safety_plots(data, episodes, out_dir, window):
    """VRU vs vehicle safety, lane-change safety, and TTC progression."""
    has = lambda *cols: all(c in data for c in cols)

    # Plot 1: VRU vs vehicle safety comparison.
    if has("vru_collisions", "vru_near_misses",
           "vehicle_collisions", "vehicle_near_misses"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(episodes, data["vru_collisions"], color="red", linewidth=2,
                 label="VRU Collisions")
        ax1.plot(episodes, data["vru_near_misses"], color="orange", linewidth=1,
                 label="VRU Near-Misses")
        ax1.set_title("VRU Safety (Primary)")
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Count")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(episodes, data["vehicle_collisions"], color="darkred",
                 linewidth=2, label="Vehicle Collisions")
        ax2.plot(episodes, data["vehicle_near_misses"], color="darkorange",
                 linewidth=1, label="Vehicle Near-Misses")
        if "rear_incidents" in data:
            ax2.plot(episodes, data["rear_incidents"], color="purple",
                     linewidth=1, label="Rear Incidents")
        ax2.set_title("Vehicle Safety (Secondary)")
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Count")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, "safety_comparison.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"saved {path}")

    # Plot 2: lane-change safety (stacked).
    if has("lane_changes_safe", "lane_changes_unsafe_prevented"):
        safe = data["lane_changes_safe"]
        blocked = data["lane_changes_unsafe_prevented"]
        plt.figure(figsize=(10, 6))
        plt.bar(episodes, safe, color="green", alpha=0.7, label="Safe")
        plt.bar(episodes, blocked, bottom=safe, color="orange", alpha=0.7,
                label="Blocked by Mandate")
        plt.xlabel("Episode"); plt.ylabel("Lane Change Count")
        plt.title("Lane Change Safety Over Time")
        plt.legend(); plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, "lane_change_safety.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"saved {path}")

    # Plot 3: time-to-collision progression (critical safety margin).
    if has("min_ttc_vru", "min_ttc_vehicle"):
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, _smooth(data["min_ttc_vru"], window), color="red",
                 linewidth=2, label="Min TTC (VRU)")
        plt.plot(episodes, _smooth(data["min_ttc_vehicle"], window),
                 color="darkred", linewidth=2, label="Min TTC (Vehicle)")
        plt.axhline(2.0, color="red", ls="--", alpha=0.5, label="Critical (2s)")
        plt.axhline(3.0, color="orange", ls="--", alpha=0.5, label="Warning (3s)")
        plt.xlabel("Episode"); plt.ylabel("Time-to-Collision (s)")
        plt.title("Safety Margin Over Training")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, "ttc_progression.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"saved {path}")


if __name__ == "__main__":
    main()
