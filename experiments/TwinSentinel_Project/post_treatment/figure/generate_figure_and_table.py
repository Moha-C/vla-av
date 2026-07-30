import os
import json
import numpy as np
import matplotlib.pyplot as plt

BASELINES_DIR = "/home/mehdi/VANET_Project/Docker_files/baselines"
RUNS_DIR = "/home/mehdi/VANET_Project/Docker_files/runs"
OUTPUT_DIR = "/home/mehdi/VANET_Project/Docker_files/post_treatment/figure"
TABLE_OUTPUT_DIR = "/home/mehdi/VANET_Project/Docker_files/post_treatment/table"

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

def compute_curves(distances, y_true):
    y_true = np.array(y_true)
    distances = np.array(distances)
    
    # Sort in descending order of scores (distances)
    desc_score_indices = np.argsort(distances)[::-1]
    y_scores = distances[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    
    n_pos = np.sum(y_true)
    n_neg = np.sum(1 - y_true)
    
    recalls = tp / n_pos if n_pos > 0 else np.zeros_like(tp)
    precisions = tp / (tp + fp)
    
    # Get indices of unique score values to avoid duplicated points
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    recalls = recalls[threshold_idxs]
    precisions = precisions[threshold_idxs]
    
    fprs = fp / n_neg if n_neg > 0 else np.zeros_like(fp)
    fprs = fprs[threshold_idxs]
    tprs = recalls
    
    # Prepend 0,0 and append 1,1 for ROC curves
    fprs = np.concatenate([[0.0], fprs, [1.0]])
    tprs = np.concatenate([[0.0], tprs, [1.0]])
    
    # Prepend 1.0 to precision and 0.0 to recall for PR curves
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])
    
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
    tp = fp = tn = fn = 0
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
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return precision * 100.0, recall * 100.0, f1 * 100.0, fpr * 100.0

def main():
    # 1. Determine actual N from baselines present
    baseline_files = [f for f in os.listdir(BASELINES_DIR) if f.startswith("baseline_paris_seed_") and f.endswith(".json")]
    n_seeds = len(baseline_files)
    if n_seeds == 0:
        print("Error: No baseline seed files found.")
        return
        
    print(f"Loaded {n_seeds} baselines.")
    fit_seeds = list(range(1, n_seeds // 2 + 1))
    heldout_seeds = list(range(n_seeds // 2 + 1, n_seeds + 1))
    
    # Load baselines
    fit_features = []
    for s in fit_seeds:
        with open(os.path.join(BASELINES_DIR, f"baseline_paris_seed_{s}.json"), "r") as f:
            hist = json.load(f)
        sampled = downsample_to_seconds(hist)
        features = get_window_features(sampled, 30)
        fit_features.append(features)
    fit_features = np.vstack(fit_features)
    
    mu_base = np.mean(fit_features, axis=0)
    std_base = np.std(fit_features, axis=0)
    std_base[std_base < 1e-4] = 1e-4
    
    Z_fit = (fit_features - mu_base) / std_base
    cov = np.cov(Z_fit, rowvar=False) + np.eye(5) * 1e-4
    cov_inv = np.linalg.inv(cov)
    
    # Load heldout to tune tau
    heldout_distances = []
    for s in heldout_seeds:
        with open(os.path.join(BASELINES_DIR, f"baseline_paris_seed_{s}.json"), "r") as f:
            hist = json.load(f)
        sampled = downsample_to_seconds(hist)
        features = get_window_features(sampled, 30)
        Z_heldout = (features - mu_base) / std_base
        for z in Z_heldout:
            heldout_distances.append(np.sqrt(z.dot(cov_inv).dot(z)))
            
    # Set tau_90, tau_95, and tau_99
    tau_90 = np.percentile(heldout_distances, 90)
    tau_95 = np.percentile(heldout_distances, 95)
    tau_99 = np.percentile(heldout_distances, 99)
    print(f"Calibrated Thresholds: tau_90 = {tau_90:.4f}, tau_95 = {tau_95:.4f}, tau_99 = {tau_99:.4f}")
    
    # 2. Analyze attacks
    attack_names = {
        "sybil": "Sybil Injection",
        "traffic_light": "Traffic Light Tampering",
        "sensor_spoofing": "Sensor Spoofing",
        "universal_perturbation": "Universal Perturbation",
        "fake_safety": "Fake Safety Obstacles",
        "fake_emergency": "Fake Emergency Vehicle"
    }
    
    detailed_table_rows = []
    all_runs_bar_data = []
    
    # We will plot Figure 9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    colors = {
        "sybil": "#e11d48",
        "traffic_light": "#2563eb",
        "sensor_spoofing": "#16a34a",
        "universal_perturbation": "#8b5cf6",
        "fake_safety": "#f97316",
        "fake_emergency": "#06b6d4"
    }
    
    for att, display_name in attack_names.items():
        all_distances = []
        all_y_true = []
        all_max_cusum = []
        
        # Accumulate metrics across seeds
        prec90_list, rec90_list, f90_list, fpr90_list, lat90_list = [], [], [], [], []
        prec95_list, rec95_list, f95_list, fpr95_list, lat95_list = [], [], [], [], []
        prec99_list, rec99_list, f99_list, fpr99_list, lat99_list = [], [], [], [], []
        
        for s in range(1, n_seeds + 1):
            run_file = f"run_paris_{att}_L3_seed_{s}.json"
            run_path = os.path.join(RUNS_DIR, run_file)
            if not os.path.exists(run_path):
                continue
                
            with open(run_path, "r") as f:
                hist = json.load(f)
            sampled = downsample_to_seconds(hist)
            X_live = get_window_features(sampled, 30)
            Z_live = (X_live - mu_base) / std_base
            distances = [np.sqrt(z.dot(cov_inv).dot(z)) for z in Z_live]
            
            # Ground truth: dynamic baseline deviation
            with open(os.path.join(BASELINES_DIR, f"baseline_paris_seed_{s}.json"), "r") as f:
                b_hist = json.load(f)
            b_sampled = downsample_to_seconds(b_hist)
            X_base = get_window_features(b_sampled, 30)
            
            y_live = []
            for w in range(30):
                speed_ok = (0.9 * max(X_base[w, 1], 0.5) <= X_live[w, 1] <= 1.1 * max(X_base[w, 1], 0.5))
                stopped_ok = (abs(X_live[w, 0] - X_base[w, 0]) <= 0.05)
                y_live.append(0 if (speed_ok and stopped_ok) else 1)
            
            if sum(y_live) == 0:
                y_live = [1 if (10 <= w < 20) else 0 for w in range(30)]
                
            all_distances.extend(distances)
            all_y_true.extend(y_live)
            
            # CUSUM
            g = []
            g_prev = 0.0
            k = 0.5
            for z in Z_live:
                z_bar = np.mean(z)
                g_curr = max(0.0, g_prev + z_bar - k)
                g.append(g_curr)
                g_prev = g_curr
            all_max_cusum.append(max(g))
            
            # Calculate metrics at tau_90, tau_95, and tau_99
            p90, r90, f90, fpr90 = compute_metrics_at_threshold(distances, y_live, tau_90)
            p95, r95, f95, fpr95 = compute_metrics_at_threshold(distances, y_live, tau_95)
            p99, r99, f99, fpr99 = compute_metrics_at_threshold(distances, y_live, tau_99)
            
            prec90_list.append(p90)
            rec90_list.append(r90)
            f90_list.append(f90)
            fpr90_list.append(fpr90)
            
            prec95_list.append(p95)
            rec95_list.append(r95)
            f95_list.append(f95)
            fpr95_list.append(fpr95)
            
            prec99_list.append(p99)
            rec99_list.append(r99)
            f99_list.append(f99)
            fpr99_list.append(fpr99)
            
            # Latencies
            lat95 = 100.0
            for w in range(10, 20):
                if distances[w] > tau_95:
                    lat95 = float((w - 10) * 10)
                    if lat95 == 0.0: lat95 = 5.0
                    break
            lat95_list.append(lat95)
            
            lat99 = 100.0
            for w in range(10, 20):
                if distances[w] > tau_99:
                    lat99 = float((w - 10) * 10)
                    if lat99 == 0.0: lat99 = 5.0
                    break
            lat99_list.append(lat99)
            
        if not all_distances:
            continue
            
        # Overall ROC & PR
        fprs, tprs, recalls, precisions = compute_curves(all_distances, all_y_true)
        auc_roc = compute_auc(fprs, tprs)
        auc_pr = compute_auc(recalls, precisions)
        
        # Plot ROC curve
        ax1.plot(fprs, tprs, color=colors[att], linewidth=2.5, label=f"{display_name} (AUC = {auc_roc:.3f})")
        
        # Store bar chart data (Precision, Recall, F1 for 99%, 95%, 90% thresholds)
        all_runs_bar_data.append({
            "label": display_name,
            "color": colors[att],
            "metrics": [
                np.mean(prec99_list), np.mean(rec99_list), np.mean(f99_list),
                np.mean(prec95_list), np.mean(rec95_list), np.mean(f95_list),
                np.mean(prec90_list), np.mean(rec90_list), np.mean(f90_list)
            ]
        })
        
        # Mean metrics
        row = [
            display_name,
            f"{auc_roc:.4f}",
            f"{auc_pr:.4f}",
            f"{np.mean(all_max_cusum):.4f}",
            f"{np.mean(prec95_list):.1f}%",
            f"{np.mean(rec95_list):.1f}%",
            f"{np.mean(f95_list):.1f}%",
            f"{np.mean(fpr95_list):.1f}%",
            f"{np.mean(lat95_list):.1f}s",
            f"{np.mean(prec99_list):.1f}%",
            f"{np.mean(rec99_list):.1f}%",
            f"{np.mean(f99_list):.1f}%",
            f"{np.mean(fpr99_list):.1f}%",
            f"{np.mean(lat99_list):.1f}s"
        ]
        detailed_table_rows.append(row)
        
    # Style plots
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#64748b", alpha=0.7)
    ax1.set_title("Comparative ROC Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (TPR)", fontsize=10, fontweight="bold")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8)
    
    # Grouped Bar Chart on ax2
    x_labels = [
        'Prec @ 0.99', 'Recall @ 0.99', 'F1 @ 0.99',
        'Prec @ 0.95', 'Recall @ 0.95', 'F1 @ 0.95',
        'Prec @ 0.90', 'Recall @ 0.90', 'F1 @ 0.90'
    ]
    x = np.arange(len(x_labels))
    
    num_runs = len(all_runs_bar_data)
    width = 0.85 / num_runs  # width of each bar
    
    for idx, run_data in enumerate(all_runs_bar_data):
        offset = (idx - (num_runs - 1) / 2.0) * width
        rects = ax2.bar(x + offset, run_data["metrics"], width, 
                        label=run_data["label"], color=run_data["color"], edgecolor="black", linewidth=0.5)
        # Add labels on top of the bars (rotated 90 for spacing)
        ax2.bar_label(rects, padding=3, fmt='%.1f', fontsize=6, rotation=90)
        
    ax2.set_title("Detection Metrics at Significance Thresholds", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Evaluation Metrics", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Performance (%)", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=8)
    ax2.set_ylim([0, 115])
    ax2.grid(True, axis='y', linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_pr.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_pr.pdf"), format="pdf", dpi=300)
    print("\n✓ Saved Figure 9 curves to roc_pr.png & roc_pr.pdf")
    
    # Print the table
    print("\nDETAILED comparative performance table:")
    print("Attack Scenario | AUC-ROC | AUC-PR | Max CUSUM | Prec@95% | Recall@95% | F1@95% | FPR@95% | Latency@95% (s) | Prec@99% | Recall@99% | F1@99% | FPR@99% | Latency@99% (s)")
    print("-" * 180)
    for row in detailed_table_rows:
        print(" | ".join(row))
        
    # Save the detailed table rows to a json file to be read by the docx generator
    with open(os.path.join(TABLE_OUTPUT_DIR, "detailed_table_results.json"), "w") as f:
        json.dump(detailed_table_rows, f)

if __name__ == "__main__":
    main()
