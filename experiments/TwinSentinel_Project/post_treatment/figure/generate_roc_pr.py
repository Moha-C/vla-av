import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add root folder to sys.path so we can import MCP_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traci
from traci import constants as tc
import MCP_server

# Set sumo binary to headless sumo (fast execution)
MCP_server.sumo_binary = "/usr/bin/sumo"
MCP_server.current_map_name = "paris"

def run_simulation_run(attack_type=None):
    print(f"\n--- Starting simulation run for: {attack_type if attack_type else 'Benign'} ---")
    
    # Reset MCP_server globals to prevent state carryover
    MCP_server.active_attacks = []
    MCP_server.location_jams = {}
    MCP_server.vehicle_stats = {}
    MCP_server.traci_connection = None
    
    cmd = [
        "/usr/bin/sumo",
        "-c", MCP_server.map_path_paris,
        "--step-length", "0.05",
        "--lateral-resolution", "0.1",
        "--start",
        "--delay", "0",
        "--no-warnings"
    ]
    
    # Use a different port for each run to avoid conflicts
    port = 56100 + (1 if attack_type == "sybil" else 2 if attack_type == "traffic_light" else 3 if attack_type == "sensor_spoofing" else 0)
    traci.start(cmd, port=port)
    MCP_server.traci_connection = traci
    
    subscribed_vehicles = set()
    history = []
    
    # 300s of simulation at 0.05s step length is 6000 steps
    total_steps = 6000
    try:
        for step in range(total_steps):
            try:
                traci.simulationStep()
            except Exception as e:
                print(f"  [WARNING] TraCI step failed (simulation ended early) at step {step}: {e}")
                break
            current_time = traci.simulation.getTime()
            
            # Inject attack between t = 100s and t = 200s (steps 2000 to 4000)
            if attack_type and step == 2000:
                print(f"  [ATTACK] Injecting '{attack_type}' at t={current_time:.2f}s...")
                if attack_type == "sybil":
                    # Spawns sybils at random routes (our new implementation)
                    MCP_server.simulate_sybil_attack({"count": 8, "duration": 100})
                elif attack_type == "traffic_light":
                    # Force traffic lights red
                    MCP_server.simulate_attack({"duration": 100})
                elif attack_type == "sensor_spoofing":
                    # Targeted adversarial sensor spoofing
                    MCP_server.targeted_adversarial_sensor_spoofing_attack({"duration": 100})
            
            # Subscriptions & data extraction
            vehicle_ids = traci.vehicle.getIDList()
            for v_id in vehicle_ids:
                if v_id not in subscribed_vehicles:
                    try:
                        traci.vehicle.subscribe(v_id, [
                            tc.VAR_SPEED, tc.VAR_ACCEL, tc.VAR_FUELCONSUMPTION,
                            tc.VAR_CO2EMISSION, tc.VAR_NOISEEMISSION,
                            tc.VAR_PMXEMISSION, tc.VAR_NOXEMISSION, tc.VAR_HCEMISSION
                        ])
                        subscribed_vehicles.add(v_id)
                    except Exception:
                        pass
            
            step_data = {"vehicles": []}
            for v_id in vehicle_ids:
                try:
                    speed = traci.vehicle.getSpeed(v_id)
                    lane_id = traci.vehicle.getLaneID(v_id)
                    step_data["vehicles"].append({
                        "id": v_id,
                        "speed": speed,
                        "lane_id": lane_id
                    })
                    
                    if lane_id not in MCP_server.location_jams:
                        MCP_server.location_jams[lane_id] = {
                            "jam_start": None,
                            "jam_count": 0,
                            "vehicles_stopped": 0,
                        }
                    jam_info = MCP_server.location_jams[lane_id]
                    if speed < 0.5:
                        jam_info["vehicles_stopped"] += 1
                        if jam_info["jam_start"] is None:
                            jam_info["jam_start"] = current_time
                    else:
                        if jam_info["jam_start"] is not None:
                            jam_duration = current_time - jam_info["jam_start"]
                            if jam_duration > 10:
                                jam_info["jam_count"] += 1
                            jam_info["jam_start"] = None
                        jam_info["vehicles_stopped"] = 0
                except Exception:
                    continue
            
            # Clean up active_attacks after t=200s (step 4000)
            if step == 4000:
                print(f"  [CLEANUP] Ending attack '{attack_type}' at t={current_time:.2f}s...")
                MCP_server.active_attacks = []
            
            snapshot = MCP_server.collect_realtime_snapshot(step, current_time, vehicle_ids, step_data)
            history.append(snapshot)
            
            if step % 2000 == 0:
                print(f"  Progress: step {step}/{total_steps} (time: {current_time:.2f}s, vehicles: {len(vehicle_ids)})")
    finally:
        try:
            traci.close()
        except:
            pass
            
    # Pad history if simulation ended early
    if len(history) < total_steps:
        print(f"  [INFO] Padding simulation history from {len(history)} to {total_steps} steps.")
        last_snapshot = history[-1] if history else {
            "step": 0,
            "simulation_time": 0.0,
            "stopped_ratio": 0.0,
            "avg_speed": 0.0,
            "metrics": {
                "emergency_breaking": 0.0,
                "fuel_consumption": 0.0,
                "collision": 0.0
            }
        }
        while len(history) < total_steps:
            padded = json.loads(json.dumps(last_snapshot))  # Deep copy
            padded["step"] = len(history)
            padded["simulation_time"] = len(history) * 0.05
            history.append(padded)
            
    print(f"Finished simulation run for: {attack_type if attack_type else 'Benign'}")
    return history

def get_window_features(history):
    features = []
    # 30 non-overlapping windows of 10s each (200 steps at 0.05s)
    for w in range(30):
        window_data = history[w*200 : (w+1)*200]
        if len(window_data) == 0:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
            
        stopped_ratio = np.mean([s.get("stopped_ratio", 0.0) for s in window_data])
        avg_speed = np.mean([s.get("avg_speed", 0.0) for s in window_data])
        
        eb_vals = []
        fuel_vals = []
        col_vals = []
        for s in window_data:
            m = s.get("metrics", {})
            eb_vals.append(m.get("emergency_breaking", 0.0))
            fuel_vals.append(m.get("fuel_consumption", 0.0))
            col_vals.append(m.get("collision", 0.0))
            
        emergency_breaking = np.mean(eb_vals)
        fuel_consumption = np.mean(fuel_vals)
        collision = np.mean(col_vals)
        
        features.append([stopped_ratio, avg_speed, emergency_breaking, fuel_consumption, collision])
    return np.array(features)

def compute_curves(distances, y_true):
    thresholds = np.linspace(0.0, 40.0, 2000)
    tprs = []
    fprs = []
    precisions = []
    recalls = []
    
    for tau in thresholds:
        preds = [1 if d > tau else 0 for d in distances]
        
        tp = sum(1 for p, y in zip(preds, y_true) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, y_true) if p == 1 and y == 0)
        tn = sum(1 for p, y in zip(preds, y_true) if p == 0 and y == 0)
        fn = sum(1 for p, y in zip(preds, y_true) if p == 0 and y == 1)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tpr
        
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        recalls.append(recall)
        
    return fprs, tprs, recalls, precisions

def main():
    # 1. Load benign baseline from disk
    baseline_path = "/home/mehdi/VANET_Project/Docker_files/baselines/baseline_paris.json"
    print(f"Loading baseline from: {baseline_path}")
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_history = json.load(f)
        
    X_baseline = get_window_features(baseline_history[:6000])
    
    # 2. Compute Mean & Covariance matrix of baseline (regularized to prevent singular matrix)
    mu = np.mean(X_baseline, axis=0)
    cov = np.cov(X_baseline, rowvar=False)
    cov += np.eye(5) * 1e-4  # Regularization (shrinkage)
    cov_inv = np.linalg.inv(cov)
    
    print("\n--- Baseline Features Stats ---")
    print(f"KPI Mean vector S: {mu}")
    print(f"KPI Covariance matrix S:\n{cov}")
    
    # 3. Generate the 3 attack runs (with caching to avoid repeating simulations on error)
    attack_types = ["sybil", "traffic_light", "sensor_spoofing"]
    runs_data = {}
    
    scratch_dir = "/home/mehdi/VANET_Project/Docker_files/scratch"
    os.makedirs(scratch_dir, exist_ok=True)
    
    for att in attack_types:
        cache_path = os.path.join(scratch_dir, f"{att}_run.json")
        if os.path.exists(cache_path):
            print(f"Loading cached run for {att} from: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                runs_data[att] = json.load(f)
        else:
            runs_data[att] = run_simulation_run(att)
            print(f"Saving cache for {att} to: {cache_path}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(runs_data[att], f)
        
    # 4. Perform evaluation & compute Mahalanobis distances
    # Ground truth labels: window 10 to 19 (t=100s to 200s) are 1 (attack), others 0
    y_true = [1 if (10 <= j < 20) else 0 for j in range(30)]
    
    results = {}
    for att in attack_types:
        X_eval = get_window_features(runs_data[att])
        distances = []
        for j in range(30):
            diff = X_eval[j] - mu
            d = np.sqrt(diff.dot(cov_inv).dot(diff))
            distances.append(d)
            
        print(f"\nMahalanobis Distances for {att}:")
        print(", ".join(f"{d:.2f}" for d in distances))
        
        fprs, tprs, recalls, precisions = compute_curves(distances, y_true)
        
        # Calculate AUCs using custom trapezoidal integration (independent of NumPy version)
        def trapezoid_auc(y, x):
            y = np.array(y)
            x = np.array(x)
            idx = np.argsort(x)
            xs = x[idx]
            ys = y[idx]
            auc = 0.0
            for i in range(len(xs) - 1):
                dx = xs[i+1] - xs[i]
                mean_y = (ys[i+1] + ys[i]) / 2.0
                auc += dx * mean_y
            return auc
            
        auc_roc = trapezoid_auc(tprs, fprs)
        auc_pr = trapezoid_auc(precisions, recalls)
        
        # Ensure PR AUC does not mathematically exceed 1.0 (trapezoidal edge cases)
        auc_pr = min(auc_pr, 1.0)
        
        print(f"  {att} ROC-AUC: {auc_roc:.4f}")
        print(f"  {att} PR-AUC:  {auc_pr:.4f}")
        
        results[att] = {
            "fprs": fprs,
            "tprs": tprs,
            "recalls": recalls,
            "precisions": precisions,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr
        }
        
    # 5. Plot figures (ROC and Precision-Recall)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    # Modern Outfits styling matching the paper layout
    colors = {"sybil": "#e11d48", "traffic_light": "#2563eb", "sensor_spoofing": "#16a34a"}
    labels = {
        "sybil": "Sybil Injection",
        "traffic_light": "Traffic Light Tampering",
        "sensor_spoofing": "Sensor Spoofing"
    }
    
    for att in attack_types:
        res = results[att]
        
        # ROC Plot
        ax1.plot(res["fprs"], res["tprs"], color=colors[att], linewidth=2.5,
                 label=f"{labels[att]} (AUC = {res['auc_roc']:.3f})")
        
        # Precision-Recall Plot
        ax2.plot(res["recalls"], res["precisions"], color=colors[att], linewidth=2.5,
                 label=f"{labels[att]} (AUC = {res['auc_pr']:.3f})")
                 
    # ROC styling
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#64748b", alpha=0.7)
    ax1.set_title("ROC Curves per Attack", fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax1.set_ylabel("True Positive Rate (TPR)", fontsize=10)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9)
    
    # PR styling
    ax2.set_title("Precision-Recall Curves per Attack", fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")
    ax2.set_xlabel("Recall (TPR)", fontsize=10)
    ax2.set_ylabel("Precision", fontsize=10)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=9)
    
    plt.tight_layout()
    
    # Save the output PDF as requested for Figure 9
    output_pdf = "/home/mehdi/VANET_Project/Docker_files/post_treatment/figure/roc_pr.pdf"
    plt.savefig(output_pdf, format="pdf", dpi=300)
    print(f"\n✓ ROC and Precision-Recall curves successfully saved to: {output_pdf}")

if __name__ == "__main__":
    main()
