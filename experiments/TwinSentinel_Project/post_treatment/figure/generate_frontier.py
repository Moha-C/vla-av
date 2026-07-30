import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

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

def analyze_intensity(X_live, X_baseline, mu_base, std_base, cov_inv, active_windows, beta, noise_scale=0.02):
    """
    Simulates the features at a given intensity beta:
    X_perturbed = X_baseline + beta * (X_live - X_baseline) + noise
    Returns Mahalanobis distances and CUSUM stats.
    """
    num_windows = len(X_live)
    X_perturbed = np.copy(X_baseline)
    
    # Apply intensity scaling for active windows
    for w in range(num_windows):
        if w in active_windows:
            deviation = X_live[w] - X_baseline[w]
            # Add minor noise to represent sensor fluctuation at lower signal-to-noise ratios
            noise = np.random.normal(0, noise_scale * std_base)
            X_perturbed[w] = X_baseline[w] + beta * deviation + noise

    # Standardize
    Z_perturbed = (X_perturbed - mu_base) / std_base
    
    # Calculate Mahalanobis distances
    distances = []
    for z in Z_perturbed:
        dist = np.sqrt(z.dot(cov_inv).dot(z))
        distances.append(dist)
        
    # Calculate CUSUM
    g = []
    g_prev = 0.0
    k = 0.5  # Slack
    for z in Z_perturbed:
        z_bar = np.mean(z)
        g_curr = max(0.0, g_prev + z_bar - k)
        g.append(g_curr)
        g_prev = g_curr
        
    return distances, g

def main():
    print("==================================================")
    print("      VANET Detectability Frontier Generator       ")
    print("==================================================")
    
    # Scan run files
    run_pattern = os.path.join(RUNS_DIR, "run_*.json")
    run_files = sorted(glob.glob(run_pattern))
    if not run_files:
        print(f"No run files found in {RUNS_DIR}")
        return
        
    print(f"Found {len(run_files)} run file(s) to process.")
    
    # Define intensities
    intensities = ["L1 (Low)", "L2 (Medium)", "L3 (High)"]
    betas = [0.25, 0.60, 1.00]
    
    # Define attack mapping for colors and styling
    attack_styles = {
        "universal_perturbation": {"name": "Universal Perturbation", "color": "#2563eb"},
        "sybil": {"name": "Sybil Attack", "color": "#e11d48"},
        "fake_safety": {"name": "Fake Safety Obstacles", "color": "#16a34a"},
        "light": {"name": "Traffic Light Tampering", "color": "#9333ea"},
        "sensor_spoofing": {"name": "Sensor Spoofing", "color": "#0891b2"},
        "fake_emergency": {"name": "Fake Emergency Vehicle", "color": "#ea580c"}
    }
    
    frontier_results = {}
    
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
            
        if attack_type not in attack_styles:
            print(f"  [WARN] Unknown attack type {attack_type}. Skipping.")
            continue
            
        # Load baseline
        baseline_filename = f"baseline_{map_name}.json"
        baseline_path = os.path.join(BASELINES_DIR, baseline_filename)
        if not os.path.exists(baseline_path):
            baseline_path = os.path.join(BASELINES_DIR, "baseline_paris.json")
            if not os.path.exists(baseline_path):
                continue
                
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                live_history = json.load(f)
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline_history = json.load(f)
        except Exception as e:
            print(f"  [ERROR] Failed to load JSON: {e}")
            continue
            
        baseline_sampled = downsample_to_seconds(baseline_history)
        live_sampled = downsample_to_seconds(live_history)
        
        baseline_sampled = [pt for pt in baseline_sampled if get_point_time(pt) <= 300]
        live_sampled = [pt for pt in live_sampled if get_point_time(pt) <= 300]
        
        if len(baseline_sampled) < 150 or len(live_sampled) < 150:
            continue
            
        max_elapsed = min(300, int(np.floor(len(live_sampled))))
        num_windows = max_elapsed // 10
        
        baseline_filtered = [pt for pt in baseline_sampled if get_point_time(pt) <= max_elapsed]
        live_filtered = [pt for pt in live_sampled if get_point_time(pt) <= max_elapsed]
        
        X_baseline = get_window_features(baseline_filtered, num_windows)
        X_live = get_window_features(live_filtered, num_windows)
        
        # Train baseline parameters
        mu_base = np.mean(X_baseline, axis=0)
        std_base = np.std(X_baseline, axis=0)
        std_base[std_base < 1e-4] = 1e-4
        
        Z_baseline = (X_baseline - mu_base) / std_base
        cov = np.cov(Z_baseline, rowvar=False) + np.eye(5) * 1e-4
        cov_inv = np.linalg.inv(cov)
        
        # Determine baseline 95% threshold for Mahalanobis
        baseline_dists = [np.sqrt(z.dot(cov_inv).dot(z)) for z in Z_baseline]
        tau_95 = np.percentile(baseline_dists, 95)
        h_cusum = 2.0  # CUSUM alarm threshold
        
        # Identify active windows programmatically
        active_windows = []
        for w in range(num_windows):
            window_pts = live_filtered[w*10 : (w+1)*10]
            is_active = any(pt.get("active_attack_count", 0) > 0 or len(pt.get("active_attack_types", [])) > 0 for pt in window_pts)
            if is_active:
                active_windows.append(w)
                
        if not active_windows:
            print(f"  [WARN] No active attack windows found for {attack_type}. Defaulting to [10..19].")
            active_windows = list(range(10, 20))
            
        onset_window = min(active_windows)
        attack_duration_seconds = len(active_windows) * 10
        
        rates = []
        latencies = []
        
        # Evaluate for each beta
        for beta in betas:
            dists, g = analyze_intensity(X_live, X_baseline, mu_base, std_base, cov_inv, active_windows, beta)
            
            # Count detections in active windows
            detected_windows = 0
            first_alarm_window = None
            
            for w in active_windows:
                # Combined detector: Mahalanobis > tau OR CUSUM > h
                if dists[w] > tau_95 or g[w] > h_cusum:
                    detected_windows += 1
                    if first_alarm_window is None:
                        first_alarm_window = w
                        
            # Detection rate
            rate = detected_windows / len(active_windows)
            rates.append(rate)
            
            # Latency (seconds)
            if first_alarm_window is not None:
                latency = (first_alarm_window - onset_window) * 10
                # Clamp minimum latency to 5s if detected immediately (midway through first window)
                if latency == 0:
                    latency = 5.0
            else:
                latency = attack_duration_seconds  # Penalty for non-detection
            latencies.append(latency)
            
        frontier_results[attack_type] = {
            "rates": rates,
            "latencies": latencies
        }
        print(f"  -> Processed {attack_styles[attack_type]['name']}: Rates={[r*100 for r in rates]}, Latencies={latencies}")

    # Plot the Detectability Frontier (Dual Y-Axis)
    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    ax2 = ax1.twinx()
    
    # Set x positions
    x = np.arange(len(betas))
    
    for attack_type, res in frontier_results.items():
        style = attack_styles[attack_type]
        # Left axis (Detection Rate) - Solid Line
        ax1.plot(x, [r * 100.0 for r in res["rates"]], color=style["color"], linestyle="-", marker="o", linewidth=2.5, markersize=8,
                 label=f"{style['name']} (Rate)")
        # Right axis (Latency) - Dashed Line
        ax2.plot(x, res["latencies"], color=style["color"], linestyle="--", marker="s", linewidth=1.8, markersize=6, alpha=0.85,
                 label=f"{style['name']} (Latency)")
                 
    # Labeling left axis
    ax1.set_xlabel("Attack Intensity Level", fontsize=12, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Detection Rate (True Positive Rate %)", color="#0f172a", fontsize=12, fontweight="bold")
    ax1.set_ylim([-5, 105])
    ax1.set_xticks(x)
    ax1.set_xticklabels(intensities, fontsize=11, fontweight="bold")
    ax1.tick_params(axis='y', labelcolor="#0f172a")
    
    # Labeling right axis
    ax2.set_ylabel("Mean Detection Latency (Seconds)", color="#475569", fontsize=12, fontweight="bold")
    ax2.set_ylim([-5, 105])
    ax2.tick_params(axis='y', labelcolor="#475569")
    
    # Grid lines
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Title
    plt.title("Figure 10: VANET Attack Detectability Frontier\nDetection Rate and Latency vs. Attack Intensity Level", 
              fontsize=13, fontweight="bold", pad=15)
              
    # Create combined legend for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # To keep it clean, let's group by attack type
    combined_lines = []
    combined_labels = []
    for attack_type in attack_styles:
        if attack_type in frontier_results:
            style = attack_styles[attack_type]
            # Find the line handles
            h_rate = next(h for h, l in zip(lines1, labels1) if style["name"] in l)
            h_lat = next(h for h, l in zip(lines2, labels2) if style["name"] in l)
            combined_lines.extend([h_rate, h_lat])
            combined_labels.extend([f"{style['name']} (Rate)", f"{style['name']} (Latency)"])
            
    ax1.legend(combined_lines, combined_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), 
               ncol=3, framealpha=0.9, fontsize=9)
               
    plt.tight_layout()
    
    # Save outputs
    png_path = os.path.join(OUTPUT_DIR, "runs_roc_pr_frontier.png")
    pdf_path = os.path.join(OUTPUT_DIR, "runs_roc_pr_frontier.pdf")
    # Also save as frontier.pdf for LaTeX inclusion
    latex_pdf_path = os.path.join(OUTPUT_DIR, "frontier.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(latex_pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    
    print("\n==================================================")
    print("✓ Success! Detectability Frontier graphs generated:")
    print(f"  - PNG: {png_path}")
    print(f"  - PDF: {pdf_path}")
    print(f"  - LaTeX fallback (frontier.pdf): {latex_pdf_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
