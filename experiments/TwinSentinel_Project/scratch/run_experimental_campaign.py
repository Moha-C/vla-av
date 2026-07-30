import os
import sys
import json
import time
import numpy as np

# Add root folder to sys.path so we can import MCP_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traci
from traci import constants as tc
import MCP_server

# Set sumo binary to headless sumo (fast execution)
MCP_server.sumo_binary = "/usr/bin/sumo"
MCP_server.current_map_name = "paris"

# Clean runs directory
RUNS_DIR = "/home/mehdi/VANET_Project/Docker_files/runs"
BASELINES_DIR = "/home/mehdi/VANET_Project/Docker_files/baselines"
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(BASELINES_DIR, exist_ok=True)

# Configuration parameters for attacks at L1, L2, L3
ATTACK_CONFIGS = {
    "sybil": {
        "tool": MCP_server.simulate_sybil_attack,
        "L1": {"count": 2, "duration": 100},
        "L2": {"count": 9, "duration": 100},
        "L3": {"count": 26, "duration": 100}
    },
    "sensor_spoofing": {
        "tool": MCP_server.targeted_adversarial_sensor_spoofing_attack,
        "L1": {"num_obstacles": 2, "duration": 100},
        "L2": {"num_obstacles": 9, "duration": 100},
        "L3": {"num_obstacles": 26, "duration": 100}
    },
    "fake_safety": {
        "tool": MCP_server.simulate_fake_safety_alert,
        "L1": {"count": 2, "duration": 100},
        "L2": {"count": 9, "duration": 100},
        "L3": {"count": 26, "duration": 100}
    },
    "fake_emergency": {
        "tool": MCP_server.simulate_fake_emergency_vehicle,
        "L1": {"count": 2, "duration": 100},
        "L2": {"count": 9, "duration": 100},
        "L3": {"count": 26, "duration": 100}
    }
}

def run_single_simulation(seed, attack_name=None, intensity_level=None, port=56000):
    """Runs a single 300s SUMO simulation with a given random seed and attack setup."""
    # Reset MCP server globals to prevent pollution between seeds
    MCP_server.active_attacks = []
    MCP_server.location_jams = {}
    MCP_server.vehicle_stats = {}
    MCP_server.traci_connection = None
    
    cmd = [
        "/usr/bin/sumo",
        "-c", MCP_server.map_path_paris,
        "--seed", str(seed),
        "--step-length", "0.05",
        "--lateral-resolution", "0.1",
        "--start",
        "--delay", "0",
        "--no-warnings"
    ]
    
    try:
        traci.start(cmd, port=port)
    except Exception as e:
        print(f"  [ERROR] Failed starting TraCI on port {port}: {e}. Retrying with another port...")
        traci.start(cmd, port=port + 10)
        
    MCP_server.traci_connection = traci
    subscribed_vehicles = set()
    history = []
    
    def apply_active_attacks():
        import math
        for attack in MCP_server.active_attacks:
            try:
                if attack['type'] == 'traffic_light_tampering':
                    for tls in MCP_server._traffic_light_attack_targets(attack):
                        try:
                            state = traci.trafficlight.getRedYellowGreenState(tls)
                            traci.trafficlight.setRedYellowGreenState(tls, "r" * len(state))
                        except:
                            pass
                elif attack['type'] == 'fake_safety':
                    obstacles = attack['data'].get('obstacles', [])
                    if not obstacles:
                        obstacles = [{
                            'vehicle_id': attack['data'].get('vehicle_id'),
                            'route_id': attack['data'].get('route_id'),
                            'lane_id': attack['data'].get('lane_id'),
                            'position': attack['data'].get('position')
                        }]
                    for obs in obstacles:
                        veh_id = obs.get('vehicle_id')
                        if veh_id and veh_id not in traci.vehicle.getIDList():
                            try:
                                route_id = obs.get('route_id')
                                lane_id = obs.get('lane_id')
                                pos = obs.get('position')
                                vehicle_type = MCP_server.resolve_dynamic_vehicle_type()
                                traci.vehicle.add(vehID=veh_id, routeID=route_id, typeID=vehicle_type)
                                traci.vehicle.moveTo(veh_id, lane_id, pos)
                                traci.vehicle.setSpeed(veh_id, 0.0)
                                traci.vehicle.setColor(veh_id, (255, 255, 0))
                            except:
                                pass
                elif attack['type'] == 'fake_emergency':
                    veh_ids = attack['data'].get('vehicle_ids', [attack['data'].get('vehicle_id')])
                    active_evs = [ev for ev in veh_ids if ev and ev in traci.vehicle.getIDList()]
                    for ev_id in active_evs:
                        try:
                            traci.vehicle.setColor(ev_id, (0, 0, 255))
                        except:
                            pass
                    all_vehs = traci.vehicle.getIDList()
                    affected_vehs = set()
                    for ev_id in active_evs:
                        try:
                            ev_pos = traci.vehicle.getPosition(ev_id)
                        except:
                            continue
                        for v in all_vehs:
                            if v != ev_id and MCP_server.is_real_vehicle(v):
                                try:
                                    v_pos = traci.vehicle.getPosition(v)
                                    dist = ((ev_pos[0] - v_pos[0])**2 + (ev_pos[1] - v_pos[1])**2)**0.5
                                    if dist < 120.0:
                                        affected_vehs.add(v)
                                except:
                                    pass
                    prev_affected = attack['data'].setdefault('affected_vehicles', [])
                    for v in affected_vehs:
                        try:
                            traci.vehicle.setSpeed(v, 1.5)
                            traci.vehicle.setColor(v, (255, 128, 0))
                        except:
                            pass
                    for v in prev_affected:
                        if v not in affected_vehs and v in all_vehs:
                            try:
                                traci.vehicle.setSpeed(v, -1.0)
                                traci.vehicle.setColor(v, (255, 255, 255))
                            except:
                                pass
                    attack['data']['affected_vehicles'] = list(affected_vehs)
                elif attack['type'] == 'universal_perturbation':
                    vehicle_ids = traci.vehicle.getIDList()
                    original_states = attack['data'].setdefault('original_vehicle_states', {})
                    pert = attack['data'].get('perturbation', {})
                    pos_pert = pert.get('position', [0.0, 0.0])
                    dx, dy = pos_pert[0], pos_pert[1]
                    gps_error_magnitude = math.sqrt(dx*dx + dy*dy) * 50.0
                    for veh_id in vehicle_ids:
                        if not MCP_server.is_real_vehicle(veh_id):
                            continue
                        try:
                            if veh_id not in original_states:
                                original_states[veh_id] = {
                                    'max_speed': traci.vehicle.getMaxSpeed(veh_id),
                                    'color': traci.vehicle.getColor(veh_id),
                                    'lane_change_mode': traci.vehicle.getLaneChangeMode(veh_id)
                                }
                            sub = traci.vehicle.getSubscriptionResults(veh_id) or {}
                            speed = sub.get(tc.VAR_SPEED, 0.0) if sub else traci.vehicle.getSpeed(veh_id)
                            next_tls = traci.vehicle.getNextTLS(veh_id)
                            reacted_to_tls = False
                            if gps_error_magnitude > 5.0 and next_tls:
                                tls_id, tls_index, dist_to_tls, state = next_tls[0]
                                if dist_to_tls > 10.0 and dist_to_tls < 40.0 and state.lower() in ['r', 'y']:
                                    traci.vehicle.slowDown(veh_id, 0.0, 2.5)
                                    traci.vehicle.setColor(veh_id, (255, 69, 0))
                                    reacted_to_tls = True
                            if reacted_to_tls:
                                continue
                            leader_info = traci.vehicle.getLeader(veh_id, 50.0)
                            reacted_to_leader = False
                            if gps_error_magnitude > 5.0 and leader_info:
                                leader_id, physical_dist = leader_info
                                decel_factor = min(0.5, (gps_error_magnitude / 25.0) * 0.5)
                                target_speed = max(2.0, speed * (1.0 - decel_factor))
                                traci.vehicle.slowDown(veh_id, target_speed, 1.5)
                                traci.vehicle.setColor(veh_id, (255, 140, 0))
                                reacted_to_leader = True
                            if reacted_to_leader:
                                continue
                            if gps_error_magnitude > 5.0:
                                traci.vehicle.setLaneChangeMode(veh_id, 512)
                                traci.vehicle.setColor(veh_id, (255, 165, 0))
                            else:
                                traci.vehicle.setLaneChangeMode(veh_id, 1621)
                                traci.vehicle.setColor(veh_id, (255, 255, 255))
                        except:
                            continue
            except:
                pass

    # 300s of simulation at 0.05s step length is 6000 steps
    total_steps = 6000
    try:
        for step in range(total_steps):
            try:
                traci.simulationStep()
            except Exception as e:
                break
            
            # Inject attack between t = 100s and t = 200s (steps 2000 to 4000)
            if attack_name and intensity_level and step == 2000:
                config = ATTACK_CONFIGS[attack_name][intensity_level]
                tool = ATTACK_CONFIGS[attack_name]["tool"]
                tool(config)
            
            # Apply attack effects during active period (every 10 steps, i.e., 0.5s intervals, to avoid socket bottleneck)
            if 2000 <= step < 4000 and step % 10 == 0 and MCP_server.active_attacks:
                current_time = traci.simulation.getTime()
                expired_attacks = []
                for attack in MCP_server.active_attacks:
                    start_time = attack.get('start_time', 100.0)
                    duration = attack.get('duration', 100.0)
                    if current_time >= start_time + duration:
                        # Clean up this specific attack
                        if attack['type'] == 'traffic_light_tampering':
                            for tls in MCP_server._traffic_light_attack_targets(attack):
                                try:
                                    traci.trafficlight.setProgram(tls, "0")
                                    traci.trafficlight.setPhase(tls, 0)
                                    traci.trafficlight.setPhaseDuration(tls, 2)
                                except:
                                    pass
                        elif attack['type'] == 'universal_perturbation':
                            original_states = attack['data'].get('original_vehicle_states', {})
                            for veh_id, state in original_states.items():
                                if veh_id in traci.vehicle.getIDList():
                                    try:
                                        traci.vehicle.setMaxSpeed(veh_id, state.get('max_speed', 50.0))
                                        traci.vehicle.setSpeed(veh_id, -1.0)
                                        traci.vehicle.setLaneChangeMode(veh_id, state.get('lane_change_mode', 1621))
                                        traci.vehicle.setColor(veh_id, state.get('color', (255, 255, 255)))
                                    except:
                                        pass
                        elif attack['type'] == 'fake_emergency':
                            affected_vehs = attack['data'].get('affected_vehicles', [])
                            for v in affected_vehs:
                                if v in traci.vehicle.getIDList():
                                    try:
                                        traci.vehicle.setSpeed(v, -1.0)
                                        traci.vehicle.setColor(v, (255, 255, 255))
                                    except:
                                        pass
                            veh_ids = attack['data'].get('vehicle_ids', [attack['data'].get('vehicle_id')])
                            for veh_id in veh_ids:
                                if veh_id and veh_id in traci.vehicle.getIDList():
                                    try: traci.vehicle.remove(veh_id)
                                    except: pass
                        elif attack['type'] == 'fake_safety':
                            veh_ids = attack['data'].get('vehicle_ids', [attack['data'].get('vehicle_id')])
                            for veh_id in veh_ids:
                                if veh_id and veh_id in traci.vehicle.getIDList():
                                    try: traci.vehicle.remove(veh_id)
                                    except: pass
                        expired_attacks.append(attack)
                
                for att in expired_attacks:
                    if att in MCP_server.active_attacks:
                        MCP_server.active_attacks.remove(att)
                        
                if MCP_server.active_attacks:
                    apply_active_attacks()
                
            # Clean up active attacks at t = 200s (step 4000) and restore state
            if step == 4000:
                for attack in MCP_server.active_attacks:
                    if attack['type'] == 'traffic_light_tampering':
                        for tls in MCP_server._traffic_light_attack_targets(attack):
                            try:
                                traci.trafficlight.setProgram(tls, "0")
                                traci.trafficlight.setPhase(tls, 0)
                                traci.trafficlight.setPhaseDuration(tls, 2)
                            except:
                                pass
                    elif attack['type'] == 'universal_perturbation':
                        original_states = attack['data'].get('original_vehicle_states', {})
                        for veh_id, state in original_states.items():
                            if veh_id in traci.vehicle.getIDList():
                                try:
                                    traci.vehicle.setMaxSpeed(veh_id, state.get('max_speed', 50.0))
                                    traci.vehicle.setSpeed(veh_id, -1.0)
                                    traci.vehicle.setLaneChangeMode(veh_id, state.get('lane_change_mode', 1621))
                                    traci.vehicle.setColor(veh_id, state.get('color', (255, 255, 255)))
                                except:
                                    pass
                    elif attack['type'] == 'fake_emergency':
                        affected_vehs = attack['data'].get('affected_vehicles', [])
                        for v in affected_vehs:
                            if v in traci.vehicle.getIDList():
                                try:
                                    traci.vehicle.setSpeed(v, -1.0)
                                    traci.vehicle.setColor(v, (255, 255, 255))
                                except:
                                    pass
                        veh_ids = attack['data'].get('vehicle_ids', [attack['data'].get('vehicle_id')])
                        for veh_id in veh_ids:
                            if veh_id and veh_id in traci.vehicle.getIDList():
                                try: traci.vehicle.remove(veh_id)
                                except: pass
                    elif attack['type'] == 'fake_safety':
                        veh_ids = attack['data'].get('vehicle_ids', [attack['data'].get('vehicle_id')])
                        for veh_id in veh_ids:
                            if veh_id and veh_id in traci.vehicle.getIDList():
                                try: traci.vehicle.remove(veh_id)
                                except: pass
                MCP_server.active_attacks = []
                
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
            
    # Pad history to 300 seconds on early end
    while len(history) < 300:
        last = history[-1] if history else {"step": len(history), "simulation_time": float(len(history)), "avg_speed": 0.0, "stopped_ratio": 0.0, "metrics": {}}
        padded = json.loads(json.dumps(last))
        padded["step"] = len(history)
        padded["simulation_time"] = float(len(history))
        history.append(padded)
        
    return history

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_experimental_campaign.py <num_seeds>")
        print("Example: python3 run_experimental_campaign.py 20")
        sys.exit(1)
        
    n_seeds = int(sys.argv[1])
    print("=" * 60)
    print(f"Starting Experimental Campaign: {n_seeds} Seeds")
    print(f"Total runs to execute: {n_seeds} Baselines + {n_seeds * 6 * 3} Attack runs = {n_seeds * 19} runs")
    print("=" * 60)
    
    # 1. Generate N baseline files
    for s in range(1, n_seeds + 1):
        filename = f"baseline_paris_seed_{s}.json"
        filepath = os.path.join(BASELINES_DIR, filename)
        if os.path.exists(filepath):
            print(f"Baseline seed {s} already exists. Skipping.")
            continue
        print(f"-> Generating Baseline (Paris), Seed {s}/{n_seeds}...")
        history = run_single_simulation(seed=s, port=56200)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f)
            
    # 2. Generate N attack runs for each combination of attack & intensity
    run_idx = 1
    total_attack_runs = n_seeds * 6 * 3
    
    for att in ATTACK_CONFIGS.keys():
        for lvl in ["L1", "L2", "L3"]:
            for s in range(1, n_seeds + 1):
                filename = f"run_paris_{att}_{lvl}_seed_{s}.json"
                filepath = os.path.join(RUNS_DIR, filename)
                if os.path.exists(filepath):
                    print(f"[{run_idx}/{total_attack_runs}] Attack {att} ({lvl}), Seed {s} already exists. Skipping.")
                    run_idx += 1
                    continue
                    
                print(f"[{run_idx}/{total_attack_runs}] -> Running {att} ({lvl}), Seed {s}...")
                history = run_single_simulation(seed=s, attack_name=att, intensity_level=lvl, port=56300 + (s % 100))
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(history, f)
                run_idx += 1
                
    print("\n✓ Campaign complete! All runs generated successfully.")

if __name__ == "__main__":
    main()
