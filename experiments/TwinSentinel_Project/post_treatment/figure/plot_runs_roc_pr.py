import os
import json
import glob
import math
import numpy as np
import matplotlib.pyplot as plt

# Switch to non-GUI backend if no display is available
if not os.environ.get("DISPLAY"):
    plt.switch_backend("Agg")

RUNS_DIR = "/home/mehdi/VANET_Project/Docker_files/runs"
BASELINES_DIR = "/home/mehdi/VANET_Project/Docker_files/baselines"
OUTPUT_DIR = "/home/mehdi/VANET_Project/Docker_files/post_treatment/figure"

def get_point_time(pt):
    t = pt.get("simulation_time", pt.get("step", 0))
    try:
        return float(t)
    except:
        return 0.0

def downsample_to_seconds(history):
    if not history:
        return []
    sampled = []
    for i, pt in enumerate(history):
        t = get_point_time(pt)
        sec = int(np.floor(t))
        
        next_sec = -1
        if i + 1 < len(history):
            next_sec = int(np.floor(get_point_time(history[i + 1])))
            
        if next_sec != sec:
            sampled.append(pt)
    return sampled

def cliffs_delta(x, y):
    n = len(x)
    m = len(y)
    diff = y[:, None] - x[None, :]
    delta = np.sum(np.sign(diff)) / (n * m)
    abs_d = abs(delta)
    if abs_d < 0.147:
        size = "negligible"
    elif abs_d < 0.33:
        size = "small"
    elif abs_d < 0.474:
        size = "medium"
    else:
        size = "large"
    return delta, size

def standard_normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def mann_whitney_u(x, y):
    n1 = len(x)
    n2 = len(y)
    combined = np.concatenate([x, y])
    temp = combined.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(combined)) + 1
    
    for val in np.unique(combined):
        mask = (combined == val)
        if np.sum(mask) > 1:
            ranks[mask] = np.mean(ranks[mask])
            
    R1 = np.sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    
    mu = n1 * n2 / 2.0
    unique_vals, counts = np.unique(combined, return_counts=True)
    tie_sum = np.sum(counts**3 - counts)
    N = n1 + n2
    if tie_sum > 0:
        sig = np.sqrt((n1 * n2 / (N * (N - 1))) * (((N**3 - N) - tie_sum) / 12.0))
    else:
        sig = np.sqrt(n1 * n2 * (N + 1) / 12.0)
        
    if sig == 0:
        return U, 1.0
        
    z = (U - mu) / sig
    p = 2.0 * (1.0 - standard_normal_cdf(abs(z)))
    return U, p

def holm_bonferroni_correct(p_values):
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    adjusted_p = np.zeros(m)
    running_max = 0.0
    for i in range(m):
        val = sorted_p[i] * (m - i)
        running_max = max(running_max, val)
        adjusted_p[i] = min(running_max, 1.0)
        
    final_p = np.zeros(m)
    final_p[sorted_indices] = adjusted_p
    return final_p

def get_window_features(sampled, num_windows):
    features = []
    for w in range(num_windows):
        window_points = sampled[w * 10 : (w + 1) * 10]
        if not window_points:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
            
        sum_stopped = sum(pt.get("stopped_ratio", 0.0) for pt in window_points)
        sum_speed = sum(pt.get("avg_speed", 0.0) for pt in window_points)
        sum_eb = sum(pt.get("metrics", {}).get("emergency_breaking", 0.0) for pt in window_points)
        sum_fuel = sum(pt.get("metrics", {}).get("fuel_consumption", 0.0) for pt in window_points)
        sum_col = sum(pt.get("metrics", {}).get("collision", 0.0) for pt in window_points)
        
        n = len(window_points)
        features.append([
            sum_stopped / n,
            sum_speed / n,
            sum_eb / n,
            sum_fuel / n,
            sum_col / n
        ])
    return np.array(features)

def process_run(live_history, baseline_history):
    baseline_sampled = downsample_to_seconds(baseline_history)
    live_sampled = downsample_to_seconds(live_history)
    
    baseline_sampled = [pt for pt in baseline_sampled if get_point_time(pt) <= 300]
    live_sampled = [pt for pt in live_sampled if get_point_time(pt) <= 300]
    
    if len(baseline_sampled) < 150 or len(live_sampled) < 150:
        print(f"  [WARN] Insufficient data (baseline: {len(baseline_sampled)}s, live: {len(live_sampled)}s). Skipping.")
        return None
        
    max_elapsed = min(300, int(np.floor(len(live_sampled))))
    num_windows = max_elapsed // 10
    
    baseline_filtered = [pt for pt in baseline_sampled if get_point_time(pt) <= max_elapsed]
    live_filtered = [pt for pt in live_sampled if get_point_time(pt) <= max_elapsed]
    
    X_baseline = get_window_features(baseline_filtered, num_windows)
    X_live = get_window_features(live_filtered, num_windows)
    
    # Step 1 & 2: Standardization according to Equation 1
    mu_base = np.mean(X_baseline, axis=0)
    std_base = np.std(X_baseline, axis=0)
    std_base[std_base < 1e-4] = 1e-4  # Avoid division by zero
    
    Z_baseline = (X_baseline - mu_base) / std_base
    Z_live = (X_live - mu_base) / std_base
    
    # Step 3: Covariance matrix and Mahalanobis distance according to Equation 2
    cov = np.cov(Z_baseline, rowvar=False)
    cov += np.eye(5) * 1e-4  # Regularization
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(5) * 1e-2
        cov_inv = np.linalg.inv(cov)
        
    distances = []
    for z in Z_live:
        dist = np.sqrt(z.dot(cov_inv).dot(z))
        distances.append(dist)

    baseline_distances = []
    for z in Z_baseline:
        dist = np.sqrt(z.dot(cov_inv).dot(z))
        baseline_distances.append(dist)
        
    # Step 4: CUSUM according to Equation 4
    g = []
    g_prev = 0.0
    k = 0.5  # Slack
    for z in Z_live:
        z_bar = np.mean(z)
        g_curr = max(0.0, g_prev + z_bar - k)
        g.append(g_curr)
        g_prev = g_curr
        
    # Step 5: Statistical Analysis (Mann-Whitney U, Holm-Bonferroni, Cliff's delta)
    p_values = []
    deltas = []
    
    for i in range(5):
        # Mann-Whitney U test
        _, p_val = mann_whitney_u(X_baseline[:, i], X_live[:, i])
        p_values.append(p_val)
        
        # Cliff's delta
        delta, size = cliffs_delta(X_baseline[:, i], X_live[:, i])
        deltas.append((delta, size))
        
    adjusted_p = holm_bonferroni_correct(np.array(p_values))
    
    # Ground truth: time-matched macro metrics +/- 10%
    y_baseline = [0] * len(X_baseline)
    y_live = []
    for w in range(num_windows):
        x_live_stopped = X_live[w, 0]
        x_live_speed = X_live[w, 1]
        
        x_base_stopped = X_baseline[w, 0]
        x_base_speed = X_baseline[w, 1]
        
        base_speed_ref = max(x_base_speed, 0.5)
        base_stopped_ref = max(x_base_stopped, 0.05)
        
        speed_ok = (0.9 * base_speed_ref <= x_live_speed <= 1.1 * base_speed_ref)
        stopped_ok = (abs(x_live_stopped - x_base_stopped) <= 0.05) or (0.9 * base_stopped_ref <= x_live_stopped <= 1.1 * base_stopped_ref)
        
        if speed_ok and stopped_ok:
            y_live.append(0)
        else:
            y_live.append(1)
            
    # Print beautiful ASCII statistical table
    print("  +---------------------+-------------------+-------------------+----------+----------+-----------------+")
    print("  | KPI                 | Baseline Mean     | Attack Mean       | p-val    | p-adj    | Cliff's delta   |")
    print("  +---------------------+-------------------+-------------------+----------+----------+-----------------+")
    KPI_LABELS = ["Congestion Ratio", "Average Speed", "Emergency Braking", "Fuel Consumption", "Collisions"]
    for i in range(5):
        b_mean = np.mean(X_baseline[:, i])
        a_mean = np.mean(X_live[:, i])
        delta_val, delta_size = deltas[i]
        p_val = p_values[i]
        p_adj = adjusted_p[i]
        print(f"  | {KPI_LABELS[i]:<19} | {b_mean:<17.4f} | {a_mean:<17.4f} | {p_val:<8.4f} | {p_adj:<8.4f} | {delta_val:>+6.3f} ({delta_size:<9}) |")
    print("  +---------------------+-------------------+-------------------+----------+----------+-----------------+")
    print(f"  -> Maximum CUSUM value (g_t): {max(g):.4f}")
    
    return distances, baseline_distances, y_live, y_baseline, g

def compute_curves(distances, y_true):
    thresholds = np.linspace(0.0, 40.0, 201)
    fprs = []
    tprs = []
    precisions = []
    recalls = []
    
    for tau in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for dist, y in zip(distances, y_true):
            pred = 1 if dist > tau else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            elif pred == 0 and y == 1:
                fn += 1
                
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tpr
        
        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)
        
    return fprs, tprs, recalls, precisions

def compute_auc(x, y):
    x = np.array(x)
    y = np.array(y)
    idx = np.argsort(x)
    xs = x[idx]
    ys = y[idx]
    
    auc = 0.0
    for i in range(len(xs) - 1):
        dx = xs[i+1] - xs[i]
        mean_y = (ys[i+1] + ys[i]) / 2.0
        auc += dx * mean_y
    return min(auc, 1.0)

def compute_metrics_at_threshold(distances, y_true, tau):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for dist, y in zip(distances, y_true):
        pred = 1 if dist > tau else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        elif pred == 0 and y == 1:
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision * 100.0, recall * 100.0, f1 * 100.0

def main():
    print("==================================================")
    print("   VANET Comparative ROC & PR Curves Generator   ")
    print("==================================================")
    
    # 1. Scan for run files
    run_pattern = os.path.join(RUNS_DIR, "run_*.json")
    run_files = sorted(glob.glob(run_pattern))
    
    if not run_files:
        print(f"No run files found in directory: {RUNS_DIR}")
        print("Please export some simulation runs from the web dashboard first.")
        return
        
    print(f"Found {len(run_files)} run file(s) to process.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
    colors = ["#2563eb", "#e11d48", "#16a34a", "#9333ea", "#ca8a04", "#0891b2", "#ea580c"]
    color_idx = 0
    processed_count = 0
    
    all_runs_data = []
    
    for filepath in run_files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing file: {filename}")
        
        # Parse filename to identify map and attack
        # Format: run_${map_name}_${attack_type}_${timestamp}.json
        parts = filename.replace(".json", "").split("_")
        if len(parts) < 3:
            print(f"  [ERROR] Invalid filename format: {filename}. Skipping.")
            continue
            
        map_name = parts[1]
        # Gather attack type (check if last part is a timestamp)
        last_part = parts[-1]
        if last_part.isdigit() or len(last_part) > 10:
            attack_type = "_".join(parts[2:-1])
        else:
            attack_type = "_".join(parts[2:])
        
        # Load corresponding baseline file
        baseline_filename = f"baseline_{map_name}.json"
        baseline_path = os.path.join(BASELINES_DIR, baseline_filename)
        if not os.path.exists(baseline_path):
            # Try general baseline name fallback
            baseline_path = os.path.join(BASELINES_DIR, "baseline_paris.json")
            print(f"  [WARN] Baseline {baseline_filename} not found. Falling back to: baseline_paris.json")
            if not os.path.exists(baseline_path):
                print("  [ERROR] No baseline file found. Skipping run.")
                continue
                
        # Load JSON data
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                live_history = json.load(f)
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline_history = json.load(f)
        except Exception as e:
            print(f"  [ERROR] Failed to load JSON data: {e}. Skipping.")
            continue
            
        res_data = process_run(live_history, baseline_history)
        if res_data is None:
            continue
            
        distances, baseline_distances, y_live, _, g = res_data
        
        # Compute curves and metrics using only the windows of the live run
        fprs, tprs, recalls, precisions = compute_curves(distances, y_live)
        auc_roc = compute_auc(fprs, tprs)
        auc_pr = compute_auc(recalls, precisions)
        
        # Calculate thresholds based on baseline quantiles (99%, 95%, 90%)
        tau_99 = np.percentile(baseline_distances, 99)
        tau_95 = np.percentile(baseline_distances, 95)
        tau_90 = np.percentile(baseline_distances, 90)
        
        p99, r99, f99 = compute_metrics_at_threshold(distances, y_live, tau_99)
        p95, r95, f95 = compute_metrics_at_threshold(distances, y_live, tau_95)
        p90, r90, f90 = compute_metrics_at_threshold(distances, y_live, tau_90)
        
        print(f"  -> Detected Attack Label: '{attack_type.upper()}' on Map: '{map_name.upper()}'")
        print(f"  -> Simulation Duration: {len(downsample_to_seconds(live_history))}s")
        print(f"  -> AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")
        
        color = colors[color_idx % len(colors)]
        label_text = f"{attack_type.replace('_', ' ').title()} ({map_name.title()})"
        
        # ROC Plot
        ax1.plot(fprs, tprs, color=color, linewidth=2.5,
                 label=f"{label_text} (AUC = {auc_roc:.3f})")
                 
        all_runs_data.append({
            "label": label_text,
            "color": color,
            "metrics": [p99, r99, f99, p95, r95, f95, p90, r90, f90]
        })
                 
        color_idx += 1
        processed_count += 1
        
    if processed_count == 0:
        print("\nNo runs were successfully processed. Comparative curves could not be generated.")
        return
        
    # Styling ROC plot
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#64748b", alpha=0.7)
    ax1.set_title("Comparative ROC Curves", fontsize=13, fontweight="bold", fontfamily="sans-serif")
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    
    # Styling Grouped Bar Chart (ax2)
    x_labels = [
        'Prec @ 0.99', 'Recall @ 0.99', 'F1 @ 0.99',
        'Prec @ 0.95', 'Recall @ 0.95', 'F1 @ 0.95',
        'Prec @ 0.90', 'Recall @ 0.90', 'F1 @ 0.90'
    ]
    x = np.arange(len(x_labels))
    
    num_runs = len(all_runs_data)
    width = 0.8 / num_runs  # Total group width is 0.8
    
    for idx, run_data in enumerate(all_runs_data):
        offset = (idx - (num_runs - 1) / 2.0) * width
        rects = ax2.bar(x + offset, run_data["metrics"], width, 
                        label=run_data["label"], color=run_data["color"], edgecolor="black", linewidth=0.5)
        # Add labels on top of the bars (rotated 90 for spacing)
        ax2.bar_label(rects, padding=3, fmt='%.1f', fontsize=6, rotation=90)
        
    ax2.set_title("Detection Metrics at Significance Thresholds", fontsize=13, fontweight="bold", fontfamily="sans-serif")
    ax2.set_xlabel("Evaluation Metrics", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Performance (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=8)
    
    # Set y limits dynamically to zoom in if all metrics are high
    all_vals = []
    for r in all_runs_data:
        all_vals.extend(r["metrics"])
    min_val = min(all_vals) if all_vals else 0.0
    if min_val > 80.0:
        ymin = int(np.floor(min_val - 2.0))
        ax2.set_ylim([ymin, 105.0])
    else:
        ax2.set_ylim([0, 115])
        
    ax2.grid(True, axis='y', linestyle=":", alpha=0.6)
    # Put legend below the bar chart
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, framealpha=0.9, fontsize=8)
    
    plt.tight_layout()
    
    # Save files
    png_path = os.path.join(OUTPUT_DIR, "/home/mehdi/VANET_Project/Docker_files/post_treatment/figure/runs_roc_pr_comparison.png")
    pdf_path = os.path.join(OUTPUT_DIR, "/home/mehdi/VANET_Project/Docker_files/post_treatment/figure/runs_roc_pr_comparison.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    
    print("\n==================================================")
    print("✓ Success! Comparison graphs generated:")
    print(f"  - PNG format: {png_path}")
    print(f"  - PDF format: {pdf_path}")
    print("==================================================")
    
    if os.environ.get("DISPLAY"):
        plt.show()

if __name__ == "__main__":
    main()
