import os
import sys
import json
import time
import math
import random
import numpy as np
from multiprocessing import Pool

# Add root folder to sys.path so we can import MCP_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traci
import traci.connection
from traci import constants as tc
import MCP_server

# Set sumo binary to headless sumo (fast execution)
MCP_server.sumo_binary = "/usr/bin/sumo"
MCP_server.current_map_name = "berlin"

RUNS_DIR = "/home/mehdi/VANET_Project/Docker_files/runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Configuration parameters for attacks at L1, L2, L3 based on Berlin's density
# Average vehicle count between 100s and 200s in Berlin baselines is 65.84 (~66 vehicles)
# Injection/sensor spoofing: L1 = 2% (~1), L2 = 10% (~7), L3 = 30% (~20)
# Traffic Light: L1 = 30% (0.30), L2 = 70% (0.70), L3 = 100% (1.00)
# Universal Perturbation: L1 = 0.15, L2 = 0.30, L3 = 0.50
ATTACK_CONFIGS = {
    "sybil": {
        "tool": MCP_server.simulate_sybil_attack,
        "L1": {"count": 1, "duration": 100},
        "L2": {"count": 7, "duration": 100},
        "L3": {"count": 20, "duration": 100}
    },
    "sensor_spoofing": {
        "tool": MCP_server.targeted_adversarial_sensor_spoofing_attack,
        "L1": {"num_obstacles": 1, "duration": 100},
        "L2": {"num_obstacles": 7, "duration": 100},
        "L3": {"num_obstacles": 20, "duration": 100}
    },
    "fake_safety": {
        "tool": MCP_server.simulate_fake_safety_alert,
        "L1": {"count": 1, "duration": 100},
        "L2": {"count": 7, "duration": 100},
        "L3": {"count": 20, "duration": 100}
    },
    "fake_emergency": {
        "tool": MCP_server.simulate_fake_emergency_vehicle,
        "L1": {"count": 1, "duration": 100},
        "L2": {"count": 7, "duration": 100},
        "L3": {"count": 20, "duration": 100}
    },
    "traffic_light": {
        "tool": MCP_server.simulate_attack,
        "L1": {"ratio": 0.30, "duration": 100},
        "L2": {"ratio": 0.70, "duration": 100},
        "L3": {"ratio": 1.00, "duration": 100}
    },
    "universal_perturbation": {
        "tool": MCP_server.universal_perturbation_attack,
        "L1": {"epsilon": 0.15, "duration": 100},
        "L2": {"epsilon": 0.30, "duration": 100},
        "L3": {"epsilon": 0.50, "duration": 100}
    }
}

def apply_active_attacks():
    """Step-by-step active attack side-effect application (matches simulation_loop in MCP_server)."""
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
        except Exception as e:
            pass

def run_single_simulation(args):
    """Worker function to run a single simulation run on Berlin with the specified attack."""
    seed, attack_name, intensity_level, port = args
    filename = f"run_berlin_{attack_name}_{intensity_level}_seed_{seed}.json"
    filepath = os.path.join(RUNS_DIR, filename)
    
    # Check if run file already exists (caching)
    if os.path.exists(filepath):
        return {"status": "skipped", "file": filename}
        
    t0 = time.time()
    
    # Set up process environment isolation
    import socket
    socket.setdefaulttimeout(300.0)
    
    # Reset MCP server variables inside the worker process
    MCP_server.active_attacks = []
    MCP_server.location_jams = {}
    MCP_server.vehicle_stats = {}
    MCP_server.traci_connection = None
    
    cmd = [
        "/usr/bin/sumo",
        "-c", MCP_server.map_path_berlin,
        "--seed", str(seed),
        "--step-length", "0.05",
        "--lateral-resolution", "0.1",
        "--start",
        "--delay", "0",
        "--no-warnings"
    ]
    
    # Clear any leftover TraCI connections in this worker's process space before starting
    for conn_dict_name in ['_connections', 'main._connections']:
        try:
            import traci.main
            dct = getattr(traci.main if 'main' in conn_dict_name else traci, '_connections')
            if 'default' in dct:
                del dct['default']
        except Exception:
            pass

    # Attempt to start TraCI connection
    try:
        traci.start(cmd, port=port)
    except Exception as e:
        # Clear registries and retry with a fallback port
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
        try:
            traci.start(cmd, port=port + 500)
        except Exception as retry_err:
            return {"status": "failed", "file": filename, "error": str(retry_err)}
            
    MCP_server.traci_connection = traci
    subscribed_vehicles = set()
    history = []
    
    total_steps = 6000  # 300 seconds at 0.05s
    
    try:
        for step in range(total_steps):
            try:
                traci.simulationStep()
            except Exception:
                break
                
            # Inject attack at t = 100s (step 2000)
            if step == 2000:
                config = ATTACK_CONFIGS[attack_name][intensity_level]
                tool = ATTACK_CONFIGS[attack_name]["tool"]
                # In python, calling the tool registers the attack in MCP_server.active_attacks
                tool(config)
                
            # Apply attack side-effects at active phase (every 10 steps)
            if 2000 <= step < 4000 and step % 10 == 0 and MCP_server.active_attacks:
                current_time = traci.simulation.getTime()
                expired_attacks = []
                for attack in MCP_server.active_attacks:
                    start_time = attack.get('start_time', 100.0)
                    duration = attack.get('duration', 100.0)
                    if current_time >= start_time + duration:
                        # Restore TLS, universal perturbation, fake emergency, fake safety
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
                    
            # Final restoration at t = 200s (step 4000)
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
                
            # Collect metrics once per second (every 20 steps)
            if step % 20 == 0:
                current_time = traci.simulation.getTime()
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
                        except:
                            pass
                            
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
            
    # Pad history to 300 seconds
    while len(history) < 300:
        last = history[-1] if history else {"step": len(history), "simulation_time": float(len(history)), "avg_speed": 0.0, "stopped_ratio": 0.0, "metrics": {}}
        padded = json.loads(json.dumps(last))
        padded["step"] = len(history)
        padded["simulation_time"] = float(len(history))
        history.append(padded)
        
    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f)
        
    elapsed = time.time() - t0
    return {"status": "completed", "file": filename, "duration": elapsed}

def main():
    n_seeds = 20
    attacks = list(ATTACK_CONFIGS.keys())
    levels = ["L1", "L2", "L3"]
    
    # Build list of runs to execute
    tasks = []
    task_idx = 0
    for att in attacks:
        for lvl in levels:
            for s in range(1, n_seeds + 1):
                # Use a unique port for every single run to completely prevent TIME_WAIT port conflicts
                port = 57000 + task_idx
                tasks.append((s, att, lvl, port))
                task_idx += 1
                
    print("=" * 60)
    print("Starting Berlin Parallel Attack Campaign")
    print(f"Total runs to execute: {len(tasks)} runs (20 seeds * 6 attacks * 3 levels)")
    print(f"Running in parallel using 2 worker processes...")
    print("=" * 60)
    
    t0 = time.time()
    completed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Run the Pool
    with Pool(2) as pool:
        for result in pool.imap_unordered(run_single_simulation, tasks):
            if result["status"] == "skipped":
                skipped_count += 1
            elif result["status"] == "failed":
                failed_count += 1
                print(f"  [ERROR] Failed generating {result['file']}: {result['error']}")
            else:
                completed_count += 1
                print(f"  [{completed_count + skipped_count + failed_count}/{len(tasks)}] Generated {result['file']} (took {result['duration']:.1f}s)")
                
    elapsed_total = time.time() - t0
    print("\n" + "=" * 60)
    print("✓ Berlin Campaign Execution complete!")
    print(f"  Completed runs: {completed_count}")
    print(f"  Skipped (cached): {skipped_count}")
    print(f"  Failed runs: {failed_count}")
    print(f"  Total campaign duration: {elapsed_total / 60.0:.1f} minutes")
    print("=" * 60)

if __name__ == "__main__":
    main()
