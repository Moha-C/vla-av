import os
import json
import numpy as np
import matplotlib.pyplot as plt

BASELINES_DIR = "/home/mehdi/VANET_Project/Docker_files/baselines"
RUNS_DIR = "/home/mehdi/VANET_Project/Docker_files/runs"

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

def compute_cliffs_delta(x, y):
    """Computes Cliff's delta effect size between two distributions x and y."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    greater = 0
    less = 0
    for i in x:
        for j in y:
            if i > j:
                greater += 1
            elif i < j:
                less += 1
    return (greater - less) / (n1 * n2)

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

def mann_whitney_u(x, y):
    import math
    n1, n2 = len(x), len(y)
    combined = sorted([(val, 'x') for val in x] + [(val, 'y') for val in y], key=lambda item: item[0])
    
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[combined[k]] = avg_rank
        i = j
        
    r1 = sum(ranks[(val, 'x')] for val in x)
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - r1
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    
    mu = (n1 * n2) / 2.0
    sigma = math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12.0)
    z = (u - mu) / sigma
    
    p = 2.0 * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return u, min(p, 1.0)

def main():
    # 1. Determine actual N from baselines present
    baseline_files = [f for f in os.listdir(BASELINES_DIR) if f.startswith("baseline_paris_seed_") and f.endswith(".json")]
    n_seeds = len(baseline_files)
    if n_seeds == 0:
        print("Error: No baseline seed files found. Run the campaign script first!")
        return
        
    print(f"Detected Campaign Size: N = {n_seeds} seeds")
    fit_seeds = list(range(1, n_seeds // 2 + 1))
    heldout_seeds = list(range(n_seeds // 2 + 1, n_seeds + 1))
    print(f"Fit Set Seeds (Training): {fit_seeds}")
    print(f"Held-out Benign Set Seeds (Threshold tuning): {heldout_seeds}")
    
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
            
    # Set tau to 95th and 99th percentile of heldout distances
    tau_95 = np.percentile(heldout_distances, 95)
    tau_99 = np.percentile(heldout_distances, 99)
    print(f"Calibrated Thresholds on Held-out Benign Set: tau_95 = {tau_95:.4f} (FPR = 5%), tau_99 = {tau_99:.4f} (FPR = 1%)")
    
    # Collect baseline active window distances for statistical significance tests
    baseline_active_distances = []
    for s in range(1, n_seeds + 1):
        with open(os.path.join(BASELINES_DIR, f"baseline_paris_seed_{s}.json"), "r") as f:
            hist = json.load(f)
        sampled = downsample_to_seconds(hist)
        features = get_window_features(sampled, 30)
        Z_base = (features - mu_base) / std_base
        for w in range(10, 20):
            baseline_active_distances.append(np.sqrt(Z_base[w].dot(cov_inv).dot(Z_base[w])))
            
    # 2. Analyze attacks
    attack_names = {
        "sybil": "Sybil injection",
        "traffic_light": "Traffic-light tampering",
        "sensor_spoofing": "Sensor spoofing",
        "universal_perturbation": "Universal adv. perturbation",
        "fake_safety": "Fake safety message",
        "fake_emergency": "Fake emergency message"
    }
    
    # We will accumulate frontier data: attack -> level -> list of (tpr, latency)
    frontier_data = {att: {"L1": [], "L2": [], "L3": []} for att in attack_names.keys()}
    
    # Also collect Table V metrics for L3 (standard full intensity)
    table_v_results = {}
    
    for att in attack_names.keys():
        table_v_results[att] = {
            "tprs": [], "fprs": [], "precisions": [], "f1s": [],
            "auc_rocs": [], "auc_prs": [], "latencies": [],
            "distances_active": [], # Collect active attack window Mahalanobis distances
            "deltas": {k: [] for k in range(5)} # For each of 5 KPIs
        }
        
        for lvl in ["L1", "L2", "L3"]:
            tpr_list = []
            lat_list = []
            
            for s in range(1, n_seeds + 1):
                run_file = f"run_paris_{att}_{lvl}_seed_{s}.json"
                run_path = os.path.join(RUNS_DIR, run_file)
                if not os.path.exists(run_path):
                    continue
                    
                with open(run_path, "r") as f:
                    hist = json.load(f)
                sampled = downsample_to_seconds(hist)
                X_live = get_window_features(sampled, 30)
                Z_live = (X_live - mu_base) / std_base
                distances = [np.sqrt(z.dot(cov_inv).dot(z)) for z in Z_live]
                
                # Ground truth: dynamic baseline deviation (matches Version A logic)
                with open(os.path.join(BASELINES_DIR, f"baseline_paris_seed_{s}.json"), "r") as f:
                    b_hist = json.load(f)
                b_sampled = downsample_to_seconds(b_hist)
                X_base = get_window_features(b_sampled, 30)
                
                y_live = []
                for w in range(30):
                    speed_ok = (0.9 * max(X_base[w, 1], 0.5) <= X_live[w, 1] <= 1.1 * max(X_base[w, 1], 0.5))
                    stopped_ok = (abs(X_live[w, 0] - X_base[w, 0]) <= 0.05)
                    y_live.append(0 if (speed_ok and stopped_ok) else 1)
                
                # If there are no positive windows, set attack window as active
                if sum(y_live) == 0:
                    y_live = [1 if (10 <= w < 20) else 0 for w in range(30)]
                    
                # Metrics at tau_95
                tp = fp = tn = fn = 0
                for dist, y in zip(distances, y_live):
                    pred = 1 if dist > tau_95 else 0
                    if pred == 1 and y == 1: tp += 1
                    elif pred == 1 and y == 0: fp += 1
                    elif pred == 0 and y == 0: tn += 1
                    elif pred == 0 and y == 1: fn += 1
                    
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
                
                tpr_list.append(tpr)
                
                # Latency (elapsed seconds between attack start (t=100s, w=10) and first detection window >= 10)
                lat = 100.0 # Default max penalty
                for w in range(10, 20):
                    if distances[w] > tau_95:
                        lat = float((w - 10) * 10)
                        if lat == 0.0: lat = 5.0
                        break
                lat_list.append(lat)
                
                # If L3, save stats for Table V
                if lvl == "L3":
                    # Curves & AUC
                    fprs, tprs_curve, recalls, precisions = compute_curves(distances, y_live)
                    auc_roc = compute_auc(fprs, tprs_curve)
                    auc_pr = compute_auc(recalls, precisions)
                    
                    table_v_results[att]["tprs"].append(tpr)
                    table_v_results[att]["fprs"].append(fpr)
                    table_v_results[att]["precisions"].append(precision)
                    table_v_results[att]["f1s"].append(f1)
                    table_v_results[att]["auc_rocs"].append(auc_roc)
                    table_v_results[att]["auc_prs"].append(auc_pr)
                    table_v_results[att]["latencies"].append(lat)
                    # Collect active distances
                    for w in range(10, 20):
                        table_v_results[att]["distances_active"].append(distances[w])
                    
                    # Cliff's delta for the 5 KPIs
                    for k in range(5):
                        # Compare active attack windows (w=10..19) of live run vs baseline run
                        delta_val = compute_cliffs_delta(X_live[10:20, k], X_base[10:20, k])
                        table_v_results[att]["deltas"][k].append(delta_val)
                        
            # Store mean level results for frontier plotting
            mean_tpr = np.mean(tpr_list) * 100.0 if tpr_list else 0.0
            mean_lat = np.mean(lat_list) if lat_list else 100.0
            frontier_data[att][lvl] = (mean_tpr, mean_lat)
            
    # 3. Print completed Table V
    print("\n" + "=" * 60)
    print("TABLE V: DETECTION PERFORMANCE PER ATTACK ON THE PARIS NETWORK")
    print("=" * 60)
    print(f"{'Attack':<30} | {'TPR (%)':<8} | {'FPR (%)':<8} | {'Prec. (%)':<9} | {'F1 (%)':<8} | {'ROC AUC':<7} | {'PR AUC':<7} | {'Latency (s)':<11} | Top KPIs (Cliff's d)")
    print("-" * 150)
    
    kpi_labels = ["stopped", "speed", "braking", "fuel", "collisions"]
    
    table_v_rows = []
    avg_tpr = []
    avg_fpr = []
    avg_prec = []
    avg_f1 = []
    avg_roc = []
    avg_pr = []
    avg_lat = []
    for att, res in table_v_results.items():
        if not res["tprs"]:
            continue
        m_tpr = np.mean(res["tprs"]) * 100.0
        m_fpr = np.mean(res["fprs"]) * 100.0
        m_prec = np.mean(res["precisions"]) * 100.0
        m_f1 = np.mean(res["f1s"]) * 100.0
        m_roc = np.mean(res["auc_rocs"])
        m_pr = np.mean(res["auc_prs"])
        m_lat = np.mean(res["latencies"])
        
        avg_tpr.append(m_tpr)
        avg_fpr.append(m_fpr)
        avg_prec.append(m_prec)
        avg_f1.append(m_f1)
        avg_roc.append(m_roc)
        avg_pr.append(m_pr)
        avg_lat.append(m_lat)
        
        # Sort KPIs by absolute Cliff's delta
        kpi_deltas = []
        for k in range(5):
            mean_d = np.mean(res["deltas"][k])
            kpi_deltas.append((kpi_labels[k], mean_d))
        kpi_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
        top_kpis_str = f"{kpi_deltas[0][0]} (d={kpi_deltas[0][1]:+.2f}), {kpi_deltas[1][0]} (d={kpi_deltas[1][1]:+.2f})"
        
        print(f"{attack_names[att]:<30} | {m_tpr:7.1f}% | {m_fpr:7.1f}% | {m_prec:8.1f}% | {m_f1:7.1f}% | {m_roc:7.3f} | {m_pr:7.3f} | {m_lat:10.1f}s | {top_kpis_str}")
        
        table_v_rows.append([
            attack_names[att],
            f"{m_tpr:.1f}%",
            f"{m_fpr:.1f}%",
            f"{m_prec:.1f}%",
            f"{m_f1:.1f}%",
            f"{m_roc:.3f}",
            f"{m_pr:.3f}",
            f"{m_lat:.1f}s",
            top_kpis_str
        ])
        
    print("-" * 150)
    print(f"{'Mean':<30} | {np.mean(avg_tpr):7.1f}% | {np.mean(avg_fpr):7.1f}% | {np.mean(avg_prec):8.1f}% | {np.mean(avg_f1):7.1f}% | {np.mean(avg_roc):7.3f} | {np.mean(avg_pr):7.3f} | {np.mean(avg_lat):10.1f}s | —")
    print("=" * 60)
    
    table_v_rows.append([
        "Mean",
        f"{np.mean(avg_tpr):.1f}%",
        f"{np.mean(avg_fpr):.1f}%",
        f"{np.mean(avg_prec):.1f}%",
        f"{np.mean(avg_f1):.1f}%",
        f"{np.mean(avg_roc):.3f}",
        f"{np.mean(avg_pr):.3f}",
        f"{np.mean(avg_lat):.1f}s",
        "―"
    ])
    
    # 3b. Holm-Bonferroni correction printout
    print("\n" + "=" * 90)
    print("HOLM-BONFERRONI STATISTICAL SIGNIFICANCE (ALPHA = 0.05)")
    print("=" * 90)
    print(f"{'Attack':<30} | {'U-statistic':<12} | {'Raw p-value':<12} | {'Adj. p-value':<12} | Significant?")
    print("-" * 90)
    
    # Compute raw p-values
    raw_p_values = []
    for att, res in table_v_results.items():
        if not res["tprs"] or not res["distances_active"]:
            continue
        u_val, p_val = mann_whitney_u(res["distances_active"], baseline_active_distances)
        raw_p_values.append((att, u_val, p_val))
        
    # Sort by p-value (ascending)
    raw_p_values.sort(key=lambda x: x[2])
    m = len(raw_p_values)
    
    # Holm-Bonferroni correction
    adjusted_p_values = []
    for rank, (att, u_val, p_val) in enumerate(raw_p_values):
        factor = m - rank
        adj_p = p_val * factor
        adjusted_p_values.append((att, u_val, p_val, adj_p))
        
    # Enforce monotonicity
    monotonized_p_values = []
    prev_adj = 0.0
    for att, u_val, p_val, adj_p in adjusted_p_values:
        adj_p = max(adj_p, prev_adj)
        adj_p = min(adj_p, 1.0)
        prev_adj = adj_p
        sig_str = "YES (p < 0.05)" if adj_p < 0.05 else "NO"
        monotonized_p_values.append((att, u_val, p_val, adj_p, sig_str))
        
    hb_rows = []
    for att, u_val, p_val, adj_p, sig in monotonized_p_values:
        print(f"{attack_names[att]:<30} | {u_val:12.1f} | {p_val:12.6f} | {adj_p:12.6f} | {sig}")
        raw_p_str = f"{p_val:.6f}" if p_val >= 0.000001 else "< 0.000001"
        adj_p_str = f"{adj_p:.6f}" if adj_p >= 0.000001 else "0.000000"
        hb_rows.append([
            attack_names[att],
            f"{u_val:.1f}",
            raw_p_str,
            adj_p_str,
            "YES" if adj_p < 0.05 else "NO"
        ])
    print("=" * 90)
    
    # Save campaign summary json
    summary_path = "/home/mehdi/VANET_Project/Docker_files/post_treatment/table/campaign_analysis_summary.json"
    summary_data = {
        "campaign_size": int(n_seeds),
        "fit_seeds": [int(s) for s in fit_seeds],
        "tuning_seeds": [int(s) for s in heldout_seeds],
        "tau_95": float(tau_95),
        "tau_99": float(tau_99),
        "table_v": table_v_rows,
        "significance": hb_rows
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    
    # 4. Generate Figure 10 (Frontier Plot)
    fig, ax = plt.subplots(figsize=(9, 6))
    
    colors = {
        "sybil": "#e11d48",
        "traffic_light": "#2563eb",
        "sensor_spoofing": "#16a34a",
        "universal_perturbation": "#8b5cf6",
        "fake_safety": "#f97316",
        "fake_emergency": "#06b6d4"
    }
    
    levels = ["L1", "L2", "L3"]
    
    # Draw High-Performance Detection Zone (Shaded region in data coordinates)
    rect = plt.Rectangle((0, 80), 20, 20, facecolor='#22c55e', alpha=0.1, edgecolor='#22c55e', linestyle='--', linewidth=1.5, label="Optimal Detection Zone (TPR >= 80%, Latency <= 20s)")
    ax.add_patch(rect)
    
    for att, lvl_data in frontier_data.items():
        tprs = [lvl_data[l][0] for l in levels]
        lats = [lvl_data[l][1] for l in levels]
        
        # Plot trajectory line connecting L1 -> L2 -> L3
        ax.plot(lats, tprs, color=colors[att], linestyle="-", linewidth=2.0, alpha=0.6)
        
        # Plot points for L1, L2, L3 with different marker sizes
        sizes = [40, 100, 180]
        markers = ["o", "p", "*"]
        for idx, lvl in enumerate(levels):
            lbl = attack_names[att] if idx == 2 else "" # Only label in legend once
            ax.scatter(lats[idx], tprs[idx], color=colors[att], s=sizes[idx], 
                       marker=markers[idx], edgecolor="black", linewidth=0.8, alpha=0.9, label=lbl)
            
            # Label individual points with L1, L2, L3
            ax.annotate(lvl, (lats[idx], tprs[idx]), textcoords="offset points", 
                        xytext=(0, 6), ha="center", fontsize=8, fontweight="bold", color="#334155")
            
    ax.set_xlabel("Mean Detection Latency (s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Detection Rate (TPR %)", fontsize=11, fontweight="bold")
    ax.set_xlim([-5, 105])
    ax.set_ylim([-5, 105])
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # Legend deduplication
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower left", fontsize=8.5, frameon=True)
    
    plt.title("Figure 10: The Detectability Frontier (TPR vs. Latency)\nTrajectory paths from L1 (Low) \u2192 L2 (Medium) \u2192 L3 (High) Intensity", 
              fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    
    # Save visuals
    output_png = "/home/mehdi/VANET_Project/Docker_files/post_treatment/figure/runs_roc_pr_frontier.png"
    output_pdf = "/home/mehdi/VANET_Project/Docker_files/post_treatment/figure/runs_roc_pr_frontier.pdf"
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf, format="pdf", dpi=300)
    # LaTeX copy
    plt.savefig("/home/mehdi/VANET_Project/Docker_files/post_treatment/figure/frontier.pdf", format="pdf", dpi=300)
    print("\n✓ Saved Frontier visuals to runs_roc_pr_frontier.png/pdf & frontier.pdf")

if __name__ == "__main__":
    main()
