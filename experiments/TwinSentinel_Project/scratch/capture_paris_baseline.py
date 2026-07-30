import os
import sys
import json
from datetime import datetime

# Add parent directory to path so we can import MCP_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traci
from traci import constants as tc
import MCP_server

# Override some configurations to run GUI
MCP_server.sumo_binary = "/usr/bin/sumo-gui"
# Initialize states
MCP_server.current_map_name = "paris"
MCP_server.location_jams = {}
MCP_server.vehicle_stats = {}
MCP_server.active_attacks = []
MCP_server.realtime_metrics.clear()

# Monkey patch traci.vehicle methods to use fast local subscription cache
subscribed_vehicles = set()

def get_cached_metric(veh_id, var_id, default=0.0):
    try:
        results = traci.vehicle.getSubscriptionResults(veh_id)
        if results and var_id in results:
            return results[var_id]
    except Exception:
        pass
    return default

# Override vehicle getters
traci.vehicle.getSpeed = lambda veh_id: get_cached_metric(veh_id, tc.VAR_SPEED, 0.0)
traci.vehicle.getAcceleration = lambda veh_id: get_cached_metric(veh_id, tc.VAR_ACCEL, 0.0)
traci.vehicle.getFuelConsumption = lambda veh_id: get_cached_metric(veh_id, tc.VAR_FUELCONSUMPTION, 0.0)
traci.vehicle.getCO2Emission = lambda veh_id: get_cached_metric(veh_id, tc.VAR_CO2EMISSION, 0.0)
traci.vehicle.getNoiseEmission = lambda veh_id: get_cached_metric(veh_id, tc.VAR_NOISEEMISSION, 0.0)
traci.vehicle.getPMxEmission = lambda veh_id: get_cached_metric(veh_id, tc.VAR_PMXEMISSION, 0.0)
traci.vehicle.getNOxEmission = lambda veh_id: get_cached_metric(veh_id, tc.VAR_NOXEMISSION, 0.0)
traci.vehicle.getHCEmission = lambda veh_id: get_cached_metric(veh_id, tc.VAR_HCEMISSION, 0.0)

# Launch simulation
print("Starting SUMO-GUI for Paris baseline (optimized subscription mode with no delay)...")
cmd = [
    "/usr/bin/sumo-gui",
    "-c", MCP_server.map_path_paris,
    "--step-length", "0.05",
    "--lateral-resolution", "0.1",
    "--start",
    "--delay", "0"
]
traci.start(cmd, port=55005)
print("Connected to TraCI successfully.")

simulation_data = []

try:
    # 600s of simulation at 0.05s step length is exactly 12000 steps
    total_steps = 12000
    for step in range(total_steps):
        traci.simulationStep()
        current_time = traci.simulation.getTime()
        
        # Get vehicle data
        vehicle_ids = traci.vehicle.getIDList()
        
        # Subscribe new vehicles to variables of interest to avoid round-trip network lag
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

        # Build step_data as needed by collect_realtime_snapshot
        step_data = {
            "vehicles": []
        }
        for v_id in vehicle_ids:
            try:
                # These calls will read from our fast local lambdas instead of making socket queries
                speed = traci.vehicle.getSpeed(v_id)
                lane_id = traci.vehicle.getLaneID(v_id)
                
                step_data["vehicles"].append({
                    "id": v_id,
                    "speed": speed,
                    "lane_id": lane_id
                })
                
                # Update location jams
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

        # Collect snapshot
        snapshot = MCP_server.collect_realtime_snapshot(step, current_time, vehicle_ids, step_data)
        simulation_data.append(snapshot)
        
        if step % 1000 == 0:
            print(f"Processed step {step}/{total_steps} (time: {current_time:.2f}s, vehicles: {len(vehicle_ids)})")

finally:
    try:
        traci.close()
    except Exception:
        pass

# Save baseline to the correct project baselines directory
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baselines", "baseline_paris_2.0.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(simulation_data, f, indent=2, default=str)

print(f"Baseline saved successfully to {output_path} ({len(simulation_data)} points)!")
