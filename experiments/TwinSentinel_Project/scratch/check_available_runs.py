import os
import json

base_dir = "/home/mehdi/VANET_Project/Docker_files"
files_to_check = [
    os.path.join(base_dir, "baselines/baseline_paris.json"),
    os.path.join(base_dir, "baselines/baseline_paris1.json"),
    os.path.join(base_dir, "baselines/baseline_basic.json"),
    os.path.join(base_dir, "scratch/paris.json"),
    os.path.join(base_dir, "scratch/baselines/baseline_paris.json"),
    os.path.join(base_dir, "scratch/baselines/baseline_paris2.0.json")
]

for filepath in files_to_check:
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            # check if any step has attacks
            attack_steps = [d for d in data if d.get("active_attack_count", 0) > 0 or d.get("active_attack_types")]
            print(f"File: {os.path.basename(filepath)}")
            print(f"  Total records: {len(data)}")
            print(f"  First step time: {data[0].get('simulation_time')}s, Last step time: {data[-1].get('simulation_time')}s")
            print(f"  Steps with active attacks: {len(attack_steps)}")
            if attack_steps:
                all_attack_types = set()
                for step in attack_steps:
                    for t in step.get("active_attack_types", []):
                        all_attack_types.add(t)
                print(f"  Attack types present: {all_attack_types}")
        else:
            print(f"File: {os.path.basename(filepath)} - Not a valid list or empty")
    except Exception as e:
        print(f"Error checking {filepath}: {e}")
