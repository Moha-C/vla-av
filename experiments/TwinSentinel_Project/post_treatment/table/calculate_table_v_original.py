import os
import json
import glob
import numpy as np

RUNS_DIR = "/home/mehdi/VANET_Project/Docker_files/runs"
BASELINES_DIR = "/home/mehdi/VANET_Project/Docker_files/baselines"

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
    thresholds = np.linspace(0.0, 40.0, 201)
    fprs = []
    tprs = []
    precisions = []
    recalls = []
    
    for tau in thresholds:
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
                
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        
        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(tpr)
        
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
    run_files = sorted(glob.glob(os.path.join(RUNS_DIR, "run_*.json")))
    if not run_files:
        print("No run files found.")
        return
        
    print(f"Attack Scenario | AUC-ROC | AUC-PR | Max CUSUM | Prec@95% | Recall@95% | F1@95% | FPR@95% | Latency@95% (s) | Prec@99% | Recall@99% | F1@99% | FPR@99% | Latency@99% (s)")
    print("-" * 150)
    
    attack_names = {
        "universal_perturbation": "Universal Perturbation",
        "sybil": "Sybil Attack",
        "fake_safety": "Fake Safety Obstacles",
        "light": "Traffic Light Tampering",
        "sensor_spoofing": "Sensor Spoofing",
        "fake_emergency": "Fake Emergency Vehicle"
    }
    
    for filepath in run_files:
        filename = os.path.basename(filepath)
        parts = filename.replace(".json", "").split("_")
        if len(parts) < 3:
            continue
        map_name = parts[1]
        last_part = parts[-1]
        if last_part.isdigit() or (len(last_part) > 10 and last_part.isalnum() and any(c.isdigit() for c in last_part)):
            attack_type = "_".join(parts[2:-1])
        else:
            attack_type = "_".join(parts[2:])
            
        if attack_type not in attack_names:
            continue
            
        baseline_path = os.path.join(BASELINES_DIR, "baseline_paris.json")
        with open(filepath, "r", encoding="utf-8") as f:
            live_history = json.load(f)
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_history = json.load(f)
            
        baseline_sampled = downsample_to_seconds(baseline_history)
        live_sampled = downsample_to_seconds(live_history)
        
        baseline_sampled = [pt for pt in baseline_sampled if get_point_time(pt) <= 300]
        live_sampled = [pt for pt in live_sampled if get_point_time(pt) <= 300]
        
        max_elapsed = min(300, int(np.floor(len(live_sampled))))
        num_windows = max_elapsed // 10
        
        baseline_filtered = [pt for pt in baseline_sampled if get_point_time(pt) <= max_elapsed]
        live_filtered = [pt for pt in live_sampled if get_point_time(pt) <= max_elapsed]
        
        X_baseline = get_window_features(baseline_filtered, num_windows)
        X_live = get_window_features(live_filtered, num_windows)
        
        mu_base = np.mean(X_baseline, axis=0)
        std_base = np.std(X_baseline, axis=0)
        std_base[std_base < 1e-4] = 1e-4
        
        Z_baseline = (X_baseline - mu_base) / std_base
        Z_live = (X_live - mu_base) / std_base
        
        cov = np.cov(Z_baseline, rowvar=False) + np.eye(5) * 1e-4
        cov_inv = np.linalg.inv(cov)
        
        distances = [np.sqrt(z.dot(cov_inv).dot(z)) for z in Z_live]
        baseline_distances = [np.sqrt(z.dot(cov_inv).dot(z)) for z in Z_baseline]
        
        # Original Dynamic Ground Truth Logic from plot_runs_roc_pr.py
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
        
        # Curves
        fprs, tprs, recalls, precisions = compute_curves(distances, y_live)
        auc_roc = compute_auc(fprs, tprs)
        auc_pr = compute_auc(recalls, precisions)
        
        # CUSUM
        g = []
        g_prev = 0.0
        k = 0.5
        for z in Z_live:
            z_bar = np.mean(z)
            g_curr = max(0.0, g_prev + z_bar - k)
            g.append(g_curr)
            g_prev = g_curr
        max_cusum = max(g)
        
        # Threshold stats
        tau_95 = np.percentile(baseline_distances, 95)
        p95, r95, f95, fpr95 = compute_metrics_at_threshold(distances, y_live, tau_95)
        
        tau_99 = np.percentile(baseline_distances, 99)
        p99, r99, f99, fpr99 = compute_metrics_at_threshold(distances, y_live, tau_99)
        
        # Latencies (based on active attack window as first indicator)
        active_windows = []
        for w in range(num_windows):
            window_pts = live_filtered[w*10 : (w+1)*10]
            is_active = any(pt.get("active_attack_count", 0) > 0 or len(pt.get("active_attack_types", [])) > 0 for pt in window_pts)
            if is_active:
                active_windows.append(w)
        if not active_windows:
            active_windows = list(range(10, 20))
            
        onset_w = min(active_windows)
        lat95 = lat99 = len(active_windows) * 10
        
        for w in active_windows:
            if distances[w] > tau_95:
                lat95 = (w - onset_w) * 10
                if lat95 == 0: lat95 = 5.0
                break
                
        for w in active_windows:
            if distances[w] > tau_99:
                lat99 = (w - onset_w) * 10
                if lat99 == 0: lat99 = 5.0
                break
                
        print(f"{attack_names[attack_type]:<25} | {auc_roc:.4f} | {auc_pr:.4f} | {max_cusum:.4f} | "
              f"{p95:5.1f}% | {r95:9.1f}% | {f95:5.1f}% | {fpr95:6.1f}% | {lat95:13.1f}s | "
              f"{p99:5.1f}% | {r99:9.1f}% | {f99:5.1f}% | {fpr99:6.1f}% | {lat99:13.1f}s")

if __name__ == "__main__":
    main()
