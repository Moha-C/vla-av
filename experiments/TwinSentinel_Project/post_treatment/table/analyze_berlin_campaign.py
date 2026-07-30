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
    
    desc_score_indices = np.argsort(distances)[::-1]
    y_scores = distances[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    
    n_pos = np.sum(y_true)
    n_neg = np.sum(1 - y_true)
    
    recalls = tp / n_pos if n_pos > 0 else np.zeros_like(tp)
    precisions = tp / (tp + fp)
    
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    recalls = recalls[threshold_idxs]
    precisions = precisions[threshold_idxs]
    
    fprs = fp / n_neg if n_neg > 0 else np.zeros_like(fp)
    fprs = fprs[threshold_idxs]
    tprs = recalls
    
    fprs = np.concatenate([[0.0], fprs, [1.0]])
    tprs = np.concatenate([[0.0], tprs, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    
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
    """Computes Mann-Whitney U statistic and returns U and p-value (normal approximation)."""
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
        
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined)) + 1
    r1 = np.sum(ranks[:n1])
    
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    
    # Normal approximation with continuity correction
    mu = n1 * n2 / 2.0
    std = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u - mu) / std
    
    from scipy.stats import norm
    p_val = 2.0 * norm.cdf(z)
    return u, p_val

def main():
    print("=" * 60)
    print("      Berlin VANET simulation campaign analysis      ")
    print("=" * 60)
    
    # 1. Load Berlin baselines
    baseline_files = [f for f in os.listdir(BASELINES_DIR) if f.startswith("baseline_berlin_seed_") and f.endswith(".json")]
    n_seeds = len(baseline_files)
    if n_seeds == 0:
        print("No Berlin baseline files found in baselines/ directory. Run campaign script first.")
        return
        
    print(f"Loading {n_seeds} Berlin baseline runs...")
    
    # Split seeds: 10 seeds for calibration (fit), 10 seeds for validation
    seeds = sorted([int(f.replace("baseline_berlin_seed_", "").replace(".json", "")) for f in baseline_files])
    fit_seeds = seeds[:10]
    heldout_seeds = seeds[10:]
    
    print(f"  Calibration Seeds (Fit): {fit_seeds}")
    print(f"  Validation Seeds (Heldout): {heldout_seeds}")
    
    # Train Mahalanobis parameters on fit seeds
    fit_features = []
    for s in fit_seeds:
        with open(os.path.join(BASELINES_DIR, f"baseline_berlin_seed_{s}.json"), "r") as f:
            hist = json.load(f)
        sampled = downsample_to_seconds(hist)
        sampled = [pt for pt in sampled if get_point_time(pt) <= 300]
        feats = get_window_features(sampled, 30)
        fit_features.append(feats)
        
    fit_features = np.vstack(fit_features)
    mu_base = np.mean(fit_features, axis=0)
    std_base = np.std(fit_features, axis=0)
    std_base[std_base < 1e-4] = 1e-4
    
    Z_fit = (fit_features - mu_base) / std_base
    cov = np.cov(Z_fit, rowvar=False) + np.eye(5) * 1e-4
    cov_inv = np.linalg.inv(cov)
    
    # Determine anomaly threshold (tau) using heldout baseline runs
    heldout_distances = []
    for s in heldout_seeds:
        with open(os.path.join(BASELINES_DIR, f"baseline_berlin_seed_{s}.json"), "r") as f:
            hist = json.load(f)
        sampled = downsample_to_seconds(hist)
        sampled = [pt for pt in sampled if get_point_time(pt) <= 300]
        feats = get_window_features(sampled, 30)
        Z_heldout = (feats - mu_base) / std_base
        for z in Z_heldout:
            heldout_distances.append(np.sqrt(z.dot(cov_inv).dot(z)))
            
    tau_95 = np.percentile(heldout_distances, 95)
    tau_99 = np.percentile(heldout_distances, 99)
    print(f"✓ Calibration complete:")
    print(f"  - 95th percentile threshold (tau_95): {tau_95:.4f}")
    print(f"  - 99th percentile threshold (tau_99): {tau_99:.4f}")
    
    # Collect validation baseline distance distribution
    baseline_active_distances = []
    for s in heldout_seeds:
        with open(os.path.join(BASELINES_DIR, f"baseline_berlin_seed_{s}.json"), "r") as f:
            hist = json.load(f)
        sampled = downsample_to_seconds(hist)
        sampled = [pt for pt in sampled if get_point_time(pt) <= 300]
        feats = get_window_features(sampled, 30)
        Z = (feats - mu_base) / std_base
        # Extract features for active attack window (10 to 20 representing 100s-200s)
        for w in range(10, 20):
            baseline_active_distances.append(np.sqrt(Z[w].dot(cov_inv).dot(Z[w])))
            
    # 2. Analyze Attack Campaign
    attacks = ["sybil", "traffic_light", "sensor_spoofing", "universal_perturbation", "fake_safety", "fake_emergency"]
    attack_names = {
        "sybil": "Sybil Attack",
        "traffic_light": "Traffic Light Tampering",
        "sensor_spoofing": "Sensor Spoofing",
        "universal_perturbation": "Universal Perturbation",
        "fake_safety": "Fake Safety Obstacles",
        "fake_emergency": "Fake Emergency Vehicle"
    }
    
    table_v_results = {}
    frontier_data = {}
    
    for att in attacks:
        table_v_results[att] = {
            "tprs": [], "fprs": [], "precs": [], "f1s": [],
            "rocs": [], "prs": [], "latencies": [],
            "distances_active": [], "deltas": [[] for _ in range(5)]
        }
        frontier_data[att] = {
            "L1": [0.0, 0.0],
            "L2": [0.0, 0.0],
            "L3": [0.0, 0.0]
        }
        
        # Track metrics per intensity level (L1, L2, L3)
        for lvl in ["L1", "L2", "L3"]:
            lvl_tprs = []
            lvl_lats = []
            
            for s in range(1, n_seeds + 1):
                run_file = f"run_berlin_{att}_{lvl}_seed_{s}.json"
                run_path = os.path.join(RUNS_DIR, run_file)
                if not os.path.exists(run_path):
                    continue
                    
                try:
                    with open(run_path, "r") as f:
                        live_hist = json.load(f)
                    with open(os.path.join(BASELINES_DIR, f"baseline_berlin_seed_{s}.json"), "r") as f:
                        base_hist = json.load(f)
                except Exception as e:
                    print(f"Error loading {run_file}: {e}")
                    continue
                    
                base_sampled = downsample_to_seconds(base_hist)
                live_sampled = downsample_to_seconds(live_hist)
                
                base_sampled = [pt for pt in base_sampled if get_point_time(pt) <= 300]
                live_sampled = [pt for pt in live_sampled if get_point_time(pt) <= 300]
                
                if len(base_sampled) < 30 or len(live_sampled) < 30:
                    continue
                    
                X_base = get_window_features(base_sampled, 30)
                X_live = get_window_features(live_sampled, 30)
                
                Z_live = (X_live - mu_base) / std_base
                distances = [np.sqrt(z.dot(cov_inv).dot(z)) for z in Z_live]
                
                # Active attack window: w in [10..19] (100s to 200s)
                y_true = [1 if 10 <= w < 20 else 0 for w in range(30)]
                
                # Detection stats at tau_95
                tp = fp = tn = fn = 0
                for w in range(30):
                    pred = 1 if distances[w] > tau_95 else 0
                    if y_true[w] == 1:
                        if pred == 1: tp += 1
                        else: fn += 1
                    else:
                        if pred == 1: fp += 1
                        else: tn += 1
                        
                tpr = tp / 10.0
                fpr = fp / 20.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                f1 = 2 * (prec * tpr) / (prec + tpr) if (prec + tpr) > 0 else 0.0
                
                # Curves and AUC
                crv_fprs, crv_tprs, crv_recalls, crv_precs = compute_curves(distances, y_true)
                auc_roc = compute_auc(crv_fprs, crv_tprs)
                auc_pr = compute_auc(crv_recalls, crv_precs)
                
                # Latency
                latency = 100.0  # Max penalty
                for w in range(10, 20):
                    if distances[w] > tau_95:
                        latency = (w - 10) * 10.0
                        if latency == 0.0: latency = 5.0
                        break
                        
                # Accumulate for Table V (all seeds of this attack and level)
                if lvl == "L3":
                    table_v_results[att]["tprs"].append(tpr * 100.0)
                    table_v_results[att]["fprs"].append(fpr * 100.0)
                    table_v_results[att]["precs"].append(prec * 100.0)
                    table_v_results[att]["f1s"].append(f1 * 100.0)
                    table_v_results[att]["rocs"].append(auc_roc)
                    table_v_results[att]["prs"].append(auc_pr)
                    table_v_results[att]["latencies"].append(latency)
                    
                    # Collect distances of active attack windows
                    for w in range(10, 20):
                        table_v_results[att]["distances_active"].append(distances[w])
                        
                    # Collect Cliff's delta metrics per feature
                    for k in range(5):
                        d = compute_cliffs_delta(X_live[10:20, k], X_base[10:20, k])
                        table_v_results[att]["deltas"][k].append(d)
                    
                lvl_tprs.append(tpr * 100.0)
                lvl_lats.append(latency)
                
            # Average metrics for Frontier plot at this specific intensity level
            if lvl_tprs:
                frontier_data[att][lvl] = [np.mean(lvl_tprs), np.mean(lvl_lats)]
                
    # 3. Print Results (Table V)
    print("\n" + "=" * 150)
    print("TABLE V: DETECTION PERFORMANCE PER ATTACK ON THE BERLIN NETWORK")
    print("=" * 150)
    print(f"{'Attack Scenario':<30} | {'TPR':<8} | {'FPR':<8} | {'Precision':<9} | {'F1-Score':<8} | {'AUC-ROC':<7} | {'AUC-PR':<7} | {'Latency':<10} | {'Primary Perturbed Features (Cliff\'s d)'}")
    print("-" * 150)
    
    kpi_labels = ["Stopped Ratio", "Avg Speed", "Emergency Braking", "Fuel Consumption", "Collision Count"]
    
    table_v_rows = []
    avg_tpr, avg_fpr, avg_prec, avg_f1, avg_roc, avg_pr, avg_lat = [], [], [], [], [], [], []
    
    for att in attacks:
        res = table_v_results[att]
        if not res["tprs"]:
            print(f"{attack_names[att]:<30} | No simulation runs found.")
            continue
            
        m_tpr = np.mean(res["tprs"])
        m_fpr = np.mean(res["fprs"])
        m_prec = np.mean(res["precs"])
        m_f1 = np.mean(res["f1s"])
        m_roc = np.mean(res["rocs"])
        m_pr = np.mean(res["prs"])
        m_lat = np.mean(res["latencies"])
        
        avg_tpr.append(m_tpr)
        avg_fpr.append(m_fpr)
        avg_prec.append(m_prec)
        avg_f1.append(m_f1)
        avg_roc.append(m_roc)
        avg_pr.append(m_pr)
        avg_lat.append(m_lat)
        
        # Sort features by Cliff's delta
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
    if avg_tpr:
        print(f"{'Mean':<30} | {np.mean(avg_tpr):7.1f}% | {np.mean(avg_fpr):7.1f}% | {np.mean(avg_prec):8.1f}% | {np.mean(avg_f1):7.1f}% | {np.mean(avg_roc):7.3f} | {np.mean(avg_pr):7.3f} | {np.mean(avg_lat):10.1f}s | —")
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
    print("=" * 60)
    
    # 3b. Holm-Bonferroni correction printout
    print("\n" + "=" * 90)
    print("HOLM-BONFERRONI STATISTICAL SIGNIFICANCE (BERLIN - ALPHA = 0.05)")
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
    summary_path = "/home/mehdi/VANET_Project/Docker_files/post_treatment/table/berlin_campaign_analysis_summary.json"
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
        
    # 4. Generate Berlin Figure 10 (Frontier Plot)
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
    
    # Draw High-Performance Detection Zone
    rect = plt.Rectangle((0, 80), 20, 20, facecolor='#22c55e', alpha=0.1, edgecolor='#22c55e', linestyle='--', linewidth=1.5, label="Optimal Detection Zone (TPR >= 80%, Latency <= 20s)")
    ax.add_patch(rect)
    
    has_plotted_frontier = False
    for att, lvl_data in frontier_data.items():
        # Check if we have data for this attack
        if lvl_data["L1"] == [0.0, 0.0] and lvl_data["L2"] == [0.0, 0.0] and lvl_data["L3"] == [0.0, 0.0]:
            continue
        has_plotted_frontier = True
        
        tprs = [lvl_data[l][0] for l in levels]
        lats = [lvl_data[l][1] for l in levels]
        
        ax.plot(lats, tprs, color=colors[att], linestyle="-", linewidth=2.0, alpha=0.6)
        
        sizes = [40, 100, 180]
        markers = ["o", "p", "*"]
        for idx, lvl in enumerate(levels):
            lbl = attack_names[att] if idx == 2 else ""
            ax.scatter(lats[idx], tprs[idx], color=colors[att], s=sizes[idx], 
                       marker=markers[idx], edgecolor="black", linewidth=0.8, alpha=0.9, label=lbl)
            
            ax.annotate(lvl, (lats[idx], tprs[idx]), textcoords="offset points", 
                        xytext=(0, 6), ha="center", fontsize=8, fontweight="bold", color="#334155")
            
    if has_plotted_frontier:
        ax.set_xlabel("Mean Detection Latency (s)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Detection Rate (TPR %)", fontsize=11, fontweight="bold")
        ax.set_xlim([-5, 105])
        ax.set_ylim([-5, 105])
        ax.grid(True, linestyle=":", alpha=0.6)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower left", fontsize=8.5, frameon=True)
        
        plt.title("Figure 10 (Berlin): The Detectability Frontier (TPR vs. Latency)\nTrajectory paths from L1 (Low) \u2192 L2 (Medium) \u2192 L3 (High) Intensity", 
                  fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        
        output_png = "/home/mehdi/VANET_Project/Docker_files/post_treatment/figure/berlin_runs_roc_pr_frontier.png"
        output_pdf = "/home/mehdi/VANET_Project/Docker_files/post_treatment/figure/berlin_runs_roc_pr_frontier.pdf"
        plt.savefig(output_png, dpi=300)
        plt.savefig(output_pdf, format="pdf", dpi=300)
        plt.savefig("/home/mehdi/VANET_Project/Docker_files/post_treatment/figure/berlin_frontier.pdf", format="pdf", dpi=300)
        print("\n✓ Saved Berlin Frontier visuals to berlin_runs_roc_pr_frontier.png/pdf & berlin_frontier.pdf")

if __name__ == "__main__":
    main()
