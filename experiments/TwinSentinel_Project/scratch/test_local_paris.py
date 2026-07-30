import os
import sys
import random
import time
import traceback
from datetime import datetime

# Add the workspace path to sys.path so we can import from it if needed
sys.path.append("/home/mehdi/VANET_Project/Docker_files")

import traci

# Global simulation variables mimicking MCP_server.py
running = True
step_counter = 0
simulation_data = []
latest_data = None
vehicle_stats = {}
location_jams = {}
active_attacks = []
current_map_name = "paris"
traffic = 0

def resolve_dynamic_vehicle_type():
    return "DEFAULT_VEHTYPE"

def collect_vehicle_data(step):
    step_data = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "vehicles": []
    }
    vehicle_ids = traci.vehicle.getIDList()
    for veh_id in vehicle_ids:
        try:
            vehicle_data = {
                "id": veh_id,
                "speed": traci.vehicle.getSpeed(veh_id),
                "position": traci.vehicle.getPosition(veh_id),
                "angle": traci.vehicle.getAngle(veh_id),
                "road_id": traci.vehicle.getRoadID(veh_id),
                "vehicle_type": traci.vehicle.getTypeID(veh_id),
                "acceleration": traci.vehicle.getAcceleration(veh_id),
                "length": traci.vehicle.getLength(veh_id),
                "color": traci.vehicle.getColor(veh_id),
                "lane_id": traci.vehicle.getLaneID(veh_id),
                "lane_position": traci.vehicle.getLanePosition(veh_id),
                "co2_emission": traci.vehicle.getCO2Emission(veh_id),
                "fuel_consumption": traci.vehicle.getFuelConsumption(veh_id),
                "noise_emission": traci.vehicle.getNoiseEmission(veh_id)
            }
            step_data["vehicles"].append(vehicle_data)
        except Exception as e:
            print(f"Error collecting data for vehicle {veh_id}: {e}")
    return step_data

def collect_realtime_snapshot(step, current_time, vehicle_ids, step_data):
    vehicle_count = len(vehicle_ids)
    speeds = [v.get("speed", 0.0) for v in step_data.get("vehicles", [])]
    stopped_count = sum(1 for s in speeds if s < 0.5)
    avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
    
    return {
        "step": step,
        "simulation_time": current_time,
        "vehicle_count": vehicle_count,
        "avg_speed": avg_speed,
        "stopped_count": stopped_count,
    }

def main():
    global step_counter, running, latest_data
    
    # 1. Clean up any existing SUMO processes first
    print("Cleaning up any running sumo processes...")
    os.system("killall -9 sumo sumo-gui 2>/dev/null")
    time.sleep(1)
    
    # 2. Start sumo
    map_path_paris = "maps/paris/map.sumocfg"
    cmd = [
        "sumo",  # command-line version
        "-c", map_path_paris,
        "--step-length", "0.05",
        "--delay", "1000",
        "--lateral-resolution", "0.1"
    ]
    
    print(f"Starting sumo: {' '.join(cmd)}")
    try:
        traci_conn = traci.start(cmd, port=55001)
        print("✓ Connected to TraCI successfully!")
    except Exception as e:
        print(f"Failed to start/connect to TraCI: {e}")
        traceback.print_exc()
        return

    # 3. Replicate simulation loop
    vehicles_spawned = False
    
    print("Starting simulation loop replication (up to 10 steps)...")
    for step in range(1, 11):
        print(f"\n--- Step {step} ---")
        try:
            # Spawning test vehicles logic
            if not vehicles_spawned and step_counter < 5:
                routes = traci.route.getIDList()
                vehicle_count = traci.vehicle.getIDCount()
                print(f"Routes available: {routes}")
                print(f"Current vehicle count: {vehicle_count}")
                
                if routes and vehicle_count == 0:
                    print(f"Spawning 5 test vehicles...")
                    for i in range(5):
                        route = random.choice(routes)
                        try:
                            traci.vehicle.add(f"test_vehicle_{i}", routeID=route, depart=0)
                            print(f"  ✓ Spawned test_vehicle_{i} on route {route}")
                        except Exception as e:
                            print(f"  Could not spawn test_vehicle_{i}: {e}")
                    vehicles_spawned = True
                elif vehicle_count > 0:
                    print(f"Vehicles detected in simulation: {vehicle_count}")
                    vehicles_spawned = True
            
            print("Calling traci.simulationStep()...")
            t_start = time.time()
            traci.simulationStep()
            t_end = time.time()
            print(f"traci.simulationStep() completed in {t_end - t_start:.4f}s")
            
            step_counter += 1
            
            # Collect data
            step_data = collect_vehicle_data(step_counter)
            simulation_data.append(step_data)
            latest_data = step_data
            
            current_time = traci.simulation.getTime()
            vehicle_ids = traci.vehicle.getIDList()
            print(f"Simulation time: {current_time:.2f}s, Vehicles: {len(vehicle_ids)}")
            
            # Realtime snapshot
            snapshot = collect_realtime_snapshot(step_counter, current_time, vehicle_ids, step_data)
            print("Snapshot:", snapshot)
            
        except Exception as e:
            print(f"❌ Error in simulation loop step {step}: {e}")
            traceback.print_exc()
            break
            
    print("\nClosing TraCI connection...")
    try:
        traci.close()
        print("✓ TraCI closed successfully.")
    except Exception as e:
        print(f"Error closing TraCI: {e}")

if __name__ == "__main__":
    main()
