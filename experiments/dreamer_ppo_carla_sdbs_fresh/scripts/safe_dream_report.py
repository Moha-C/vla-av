"""SAFE-DREAM comparison report across variants.

Reads one training-log CSV per variant, builds a :class:`SafeDreamEvaluator`
for each, and prints a Table-2-style comparison (Reactive / Standard VLA /
Proposed <-> baseline / dreamer / sdbs) followed by the SDBS safety gains and
the explicitly out-of-scope SAFE-DREAM dimensions.

Usage:
    python -m scripts.safe_dream_report baseline=logs/baseline.csv \\
        dreamer=logs/dreamer.csv sdbs=logs/sdbs.csv [--plot logs/plots/safe_dream.png]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from evaluation.safe_dream_metrics import SafeDreamEvaluator, NOT_APPLICABLE

# Paper Table 2 column labels for each variant key.
VARIANT_LABELS = {
    "baseline": "Reactive",
    "dreamer": "Standard VLA",
    "sdbs": "Proposed",
}

# (row label, extractor from a report dict).
ROWS = [
    ("Collision Rate (/Mkm)", lambda r: r["collision_rate"]),
    ("Near-Collision Rate (/Mkm)", lambda r: r["near_collision_rate"]),
    ("Min TTC (s)", lambda r: r["min_ttc"]["min"]),
    ("P5 TTC (s)", lambda r: r["min_ttc"]["p5"]),
    ("Min Safety Distance (m)", lambda r: r["min_safety_distance"]),
    ("Safety Envelope Violations", lambda r: r["safety_envelope_violations"]),
    ("Disengagement Rate (/Mkm)", lambda r: r["disengagement_rate"]),
    ("Traffic Rule Compliance (%)", lambda r: r["traffic_rule_compliance"]),
    ("Mean Jerk (m/s^3)", lambda r: r["driving_comfort"]["mean_jerk"]),
    ("Counterfactual Coverage", lambda r: r["counterfactual_coverage"]),
    ("Future Diversity", lambda r: r["future_diversity"]),
    ("Unsafe Maneuver Rej. Rate (%)", lambda r: r["unsafe_maneuver_rejection_rate"]),
    ("Risk Prediction Accuracy", lambda r: r["risk_prediction_accuracy"]),
    ("Hazard Anticipation Horizon", lambda r: r["hazard_anticipation_horizon"]),
]


def _parse_variant_args(items):
    """Turn ['baseline=path', 'sdbs=path'] into an ordered {variant: path}."""
    order = ["baseline", "dreamer", "sdbs"]
    mapping = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Expected variant=path, got '{item}'")
        key, path = item.split("=", 1)
        key = key.strip().lower()
        mapping[key] = path.strip()
    # Keep a stable Reactive -> Standard VLA -> Proposed ordering.
    ordered = {k: mapping[k] for k in order if k in mapping}
    for k in mapping:                       # any non-standard variant names last
        ordered.setdefault(k, mapping[k])
    return ordered


def _fmt(value):
    """Format a metric cell; None -> 'N/A'."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:                  # NaN
            return "N/A"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.3f}"
    return str(value)


def _print_table(reports):
    variants = list(reports.keys())
    label_row = [VARIANT_LABELS.get(v, v) for v in variants]

    col_w = 30
    var_w = 16
    header = f"{'Metric':<{col_w}}" + "".join(f"{lbl:>{var_w}}" for lbl in label_row)
    sub = f"{'':<{col_w}}" + "".join(f"{('('+v+')'):>{var_w}}" for v in variants)
    line = "=" * len(header)

    print("\n" + line)
    print("SAFE-DREAM SAFETY EVALUATION — Table 2 (variant comparison)")
    print(line)
    print(header)
    print(sub)
    print("-" * len(header))
    for label, extract in ROWS:
        cells = "".join(f"{_fmt(extract(reports[v])):>{var_w}}" for v in variants)
        print(f"{label:<{col_w}}{cells}")
    print(line)


def _print_safety_gains(evaluators):
    print("\nSafety Gain (positive = safer than the compared variant):")
    if "sdbs" in evaluators and "baseline" in evaluators:
        sg = evaluators["sdbs"].safety_gain(evaluators["baseline"])
        print(f"  Safety Gain (SDBS vs Baseline): {sg:,.3f}")
    if "sdbs" in evaluators and "dreamer" in evaluators:
        sg = evaluators["sdbs"].safety_gain(evaluators["dreamer"])
        print(f"  Safety Gain (SDBS vs Dreamer):  {sg:,.3f}")


def _print_out_of_scope():
    print("\nExplicitly out of scope for this system (documented, not skipped):")
    for item in NOT_APPLICABLE:
        print(f"  - {item}")


def _plot(reports, evaluators, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    variants = list(reports.keys())
    labels = [VARIANT_LABELS.get(v, v) for v in variants]
    cr = [reports[v]["collision_rate"] for v in variants]
    ncr = [reports[v]["near_collision_rate"] for v in variants]
    # Safety gain relative to baseline (0 for baseline itself).
    if "baseline" in evaluators:
        sg = [evaluators[v].safety_gain(evaluators["baseline"]) for v in variants]
    else:
        sg = [0.0 for _ in variants]

    x = np.arange(len(variants))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(x - width, cr, width, color="firebrick", label="Collision Rate")
    ax1.bar(x, ncr, width, color="darkorange", label="Near-Collision Rate")
    ax1.set_ylabel("Rate (per million km)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{lbl}\n({v})" for lbl, v in zip(labels, variants)])
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, sg, "o-", color="seagreen", label="Safety Gain (vs Baseline)")
    ax2.set_ylabel("Safety Gain (vs Baseline)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("SAFE-DREAM: collision / near-collision rate and safety gain")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nsaved plot {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SAFE-DREAM comparison report across variants")
    parser.add_argument("variants", nargs="+",
                        help="variant=path pairs, e.g. baseline=logs/baseline.csv")
    parser.add_argument("--plot", default=None,
                        help="optional path to save a comparison bar chart")
    args = parser.parse_args(argv)

    paths = _parse_variant_args(args.variants)
    evaluators, reports = {}, {}
    for variant, path in paths.items():
        if not os.path.exists(path):
            raise SystemExit(f"CSV not found for '{variant}': {path}")
        ev = SafeDreamEvaluator(path, variant)
        evaluators[variant] = ev
        reports[variant] = ev.generate_report()

    _print_table(reports)
    _print_safety_gains(evaluators)
    _print_out_of_scope()

    if args.plot:
        _plot(reports, evaluators, args.plot)


if __name__ == "__main__":
    main()
