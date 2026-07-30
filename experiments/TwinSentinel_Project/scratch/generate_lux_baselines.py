import os
import sys
import json
import time
import socket

# Set default socket timeout to 5 minutes to allow large network parsing
socket.setdefaulttimeout(300.0)

# Add root folder to sys.path so we can import MCP_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traci
from traci import constants as tc
import MCP_server

# Set sumo binary to headless sumo (fast execution) and set current map to luxembourg
MCP_server.sumo_binary = "/usr/bin/sumo"
MCP_server.current_map_name = "luxembourg"

BASELINES_DIR = "/home/mehdi/VANET_Project/Docker_files/baselines"
os.makedirs(BASELINES_DIR, exist_ok=True)

def run_single_lux_simulation(seed, port=56000):
    """Runs a single 300s SUMO simulation on Luxembourg with a given random seed."""
    MCP_server.active_attacks = []
    MCP_server.location_jams = {}
    MCP_server.vehicle_stats = {}
    MCP_server.traci_connection = None
    
    # We use the static config as defined in MCP_server
    cmd = [
        "/usr/bin/sumo",
        "-c", MCP_server.map_path_luxembourg,
        "--seed", str(seed),
        "--step-length", "0.05",
        "--lateral-resolution", "0.1",
        "--start",
        "--delay", "0",
        "--no-warnings"
    ]
    
    # Clean registries before starting
    for conn_dict_name in ['_connections', 'main._connections']:
        try:
            import traci.main
            dct = getattr(traci.main if 'main' in conn_dict_name else traci, '_connections')
            if 'default' in dct:
                del dct['default']
        except Exception:
            pass

    try:
        traci.start(cmd, port=port)
    except Exception as e:
        print(f"  [ERROR] Failed starting TraCI on port {port}: {e}. Retrying with another port...")
        try:
            traci.close()
        except Exception:
            pass
        try:
            import traci.main
            if 'default' in traci.main._connections:
                del traci.main._connections['default']
        except Exception:
            pass
        time.sleep(2)
        traci.start(cmd, port=port + 500)
        
    MCP_server.traci_connection = traci
    subscribed_vehicles = set()
    history = []
    
    total_steps = 6000  # 300 seconds at 0.05s step length
    try:
        for step in range(total_steps):
            try:
                traci.simulationStep()
            except Exception:
                break
                
            # Collect metrics ONCE per second (every 20 steps) to avoid socket bottleneck
            if step % 20 == 0:
                current_time = traci.simulation.getTime()
                vehicle_ids = traci.vehicle.getIDList()
                
                # Subscribe newly entered vehicles
                for v_id in vehicle_ids:
                    if v_id not in subscribed_vehicles:
                        try:
                            traci.vehicle.subscribe(v_id, [
                                tc.VAR_SPEED, tc.VAR_ACCEL, tc.VAR_FUELCONSUMPTION,
                                tc.VAR_CO2EMISSION, tc.VAR_NOISEEMISSION,
                                tc.VAR_PMXEMISSION, tc.VAR_NOXEMISSION, tc.VAR_HCEMISSION
                            ])
                            subscribed_vehicles.add(v_id)
                        except:
                            pass
                
                # Build step_data using the cached subscription values
                step_data = {"vehicles": []}
                for v_id in vehicle_ids:
                    try:
                        sub = traci.vehicle.getSubscriptionResults(v_id) or {}
                        speed = sub.get(tc.VAR_SPEED, 0.0)
                        step_data["vehicles"].append({
                            "id": v_id,
                            "speed": speed,
                            "lane_id": ""
                        })
                    except:
                        continue
                
                snapshot = MCP_server.collect_realtime_snapshot(step // 20, current_time, vehicle_ids, step_data)
                history.append(snapshot)
    finally:
        try:
            traci.close()
        except:
            pass
        try:
            import traci.main
            if 'default' in traci.main._connections:
                del traci.main._connections['default']
        except Exception:
            pass
        try:
            if 'default' in traci._connections:
                del traci._connections['default']
        except Exception:
            pass
        try:
            traci.connection._connections.clear()
        except:
            pass
            
    # Pad history to 300 seconds on early end
    while len(history) < 300:
        last = history[-1] if history else {"step": len(history), "simulation_time": float(len(history)), "avg_speed": 0.0, "stopped_ratio": 0.0, "metrics": {}}
        padded = json.loads(json.dumps(last))
        padded["step"] = len(history)
        padded["simulation_time"] = float(len(history))
        history.append(padded)
        
    return history

def main():
    n_seeds = 20
    print("=" * 60)
    print(f"Starting Luxembourg Baseline Generation Campaign: {n_seeds} Seeds")
    print("=" * 60)
    
    for s in range(1, n_seeds + 1):
        filename = f"baseline_lux_seed_{s}.json"
        filepath = os.path.join(BASELINES_DIR, filename)
        if os.path.exists(filepath):
            print(f"Luxembourg Baseline seed {s} already exists. Skipping.")
            continue
        print(f"-> Generating Luxembourg Baseline, Seed {s}/{n_seeds}...")
        t0 = time.time()
        # Use unique ports for baseline runs as well to prevent binding conflicts
        history = run_single_lux_simulation(seed=s, port=56600 + s)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f)
        print(f"   Saved baseline successfully to {filename} (took {time.time() - t0:.1f}s)")
        
    print("\n✓ Luxembourg Baseline Campaign complete!")

if __name__ == "__main__":
    main()
