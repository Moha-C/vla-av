import json
import glob
import os
import numpy as np

BASELINES_DIR = "baselines"
RUNS_DIR = "runs"

def downsample_to_seconds(history):
    sampled = []
    seen_secs = set()
    for pt in history:
        sec = int(np.floor(pt.get("simulation_time", pt.get("step", 0) * 0.05)))
        if sec not in seen_secs:
            seen_secs.add(sec)
            sampled.append(pt)
    return sampled

def get_window_features(sampled, num_windows):
    features = []
    for w in range(num_windows):
        window_points = sampled[w * 10 : (w + 1) * 10]
        if not window_points:
            features.append([0.0, 0.0, 0.0, 0.0])
            continue
        sum_stopped = sum(pt.get("stopped_ratio", 0.0) for pt in window_points)
        sum_speed = sum(pt.get("avg_speed", 0.0) for pt in window_points)
        sum_jammed = sum(pt.get("jammed_lanes", 0.0) for pt in window_points)
        sum_eb = sum(pt.get("metrics", {}).get("emergency_breaking", 0.0) for pt in window_points)
        n = len(window_points)
        features.append([
            sum_stopped / n,
            sum_speed / n,
            sum_jammed / n,
            sum_eb / n
        ])
    return np.array(features)

def process_run(live_history, baseline_history):
    baseline_sampled = downsample_to_seconds(baseline_history)
    live_sampled = downsample_to_seconds(live_history)
    
    baseline_filtered = [pt for pt in baseline_sampled if pt.get("simulation_time", pt.get("step", 0)*0.05) <= 300]
    live_filtered = [pt for pt in live_sampled if pt.get("simulation_time", pt.get("step", 0)*0.05) <= 300]
    
    max_elapsed = min(300, len(live_filtered))
    num_windows = max_elapsed // 10
    
    X_baseline = get_window_features(baseline_filtered[:max_elapsed], num_windows)
    X_live = get_window_features(live_filtered[:max_elapsed], num_windows)
    
    mu = np.mean(X_baseline, axis=0)
    cov = np.cov(X_baseline, rowvar=False)
    cov += np.eye(4) * 1e-4
    cov_inv = np.linalg.inv(cov)
    
    distances = []
    for x in X_live:
        diff = x - mu
        dist = np.sqrt(diff.dot(cov_inv).dot(diff))
        distances.append(dist)
        
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
            y_live.append(0)  # Saine
        else:
            y_live.append(1)  # Perturbed
            
    return distances, y_live

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

def compute_curves(distances, y_true):
    thresholds = np.linspace(0.0, 40.0, 201)
    fprs = []
    tprs = []
    precisions = []
    recalls = []
    for tau in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
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

for filepath in sorted(glob.glob(os.path.join(RUNS_DIR, "run_*.json"))):
    filename = os.path.basename(filepath)
    baseline_filename = f"baseline_paris.json"
    baseline_path = os.path.join(BASELINES_DIR, baseline_filename)
    if not os.path.exists(baseline_path):
        continue
    try:
        with open(filepath) as f:
            live_history = json.load(f)
        with open(baseline_path) as f:
            baseline_history = json.load(f)
    except:
        continue
    distances, y_live = process_run(live_history, baseline_history)
    
    num_pos = sum(y_live)
    num_neg = len(y_live) - num_pos
    
    fprs, tprs, recalls, precisions = compute_curves(distances, y_live)
    auc_roc = compute_auc(fprs, tprs)
    auc_pr = compute_auc(recalls, precisions)
    print(f"{filename}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f} (Positives: {num_pos}/{len(y_live)})")
