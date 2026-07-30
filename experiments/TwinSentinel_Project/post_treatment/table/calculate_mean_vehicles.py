import os
import json
import numpy as np

baselines_dir = "/home/mehdi/VANET_Project/Docker_files/baselines"

# Check Paris baselines
paris_files = [f for f in os.listdir(baselines_dir) if f.startswith("baseline_paris_seed_") and f.endswith(".json")]
paris_counts = []
for f in paris_files:
    with open(os.path.join(baselines_dir, f), "r") as file:
        data = json.load(file)
    # Filter for time between 100 and 200
    for step in data:
        t = step.get("simulation_time", step.get("step", 0.0))
        if 100 <= t <= 200:
            # Check how vehicle count is stored. Let's see the keys.
            # Usually, step has "vehicle_count" or we can check len(step.get("vehicles", []))
            v_cnt = step.get("vehicle_count")
            if v_cnt is None:
                # If there's a metrics dictionary with vehicle count or list of vehicles
                v_cnt = len(step.get("vehicles", []))
            paris_counts.append(v_cnt)

if paris_counts:
    print(f"Paris (100-200s): Mean={np.mean(paris_counts):.2f}, Min={np.min(paris_counts)}, Max={np.max(paris_counts)}")
else:
    # Let's inspect keys of first step of first file
    if paris_files:
        with open(os.path.join(baselines_dir, paris_files[0]), "r") as file:
            data = json.load(file)
        if data:
            print("Paris Step Keys:", data[0].keys())
            if "metrics" in data[0]:
                print("Paris Metrics Keys:", data[0]["metrics"].keys())

# Check Berlin baselines
berlin_files = [f for f in os.listdir(baselines_dir) if f.startswith("baseline_berlin_seed_") and f.endswith(".json")]
berlin_counts = []
for f in berlin_files:
    with open(os.path.join(baselines_dir, f), "r") as file:
        data = json.load(file)
    for step in data:
        t = step.get("simulation_time", step.get("step", 0.0))
        if 100 <= t <= 200:
            v_cnt = step.get("vehicle_count")
            if v_cnt is None:
                v_cnt = len(step.get("vehicles", []))
            berlin_counts.append(v_cnt)

if berlin_counts:
    print(f"Berlin (100-200s): Mean={np.mean(berlin_counts):.2f}, Min={np.min(berlin_counts)}, Max={np.max(berlin_counts)}")
