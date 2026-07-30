#!/usr/bin/env python3
import os
import json
import time
import argparse
from datetime import datetime
import traci
import sys

# Config
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAP_NAME = os.getenv('BASELINE_MAP_NAME', 'basic').strip() or 'basic'
DEFAULT_MAP_CFG = os.getenv('SUMO_MAP_CFG', os.getenv('SUMO_MAP_BASIC', os.path.join(ROOT, 'maps', 'basic_simulation', 'osm.sumocfg')))
DEFAULT_SUMO_BIN = os.getenv('SUMO_BINARY', 'sumo-gui')
DEFAULT_STEP = float(os.getenv('BASELINE_STEP', '0.05'))
DEFAULT_TARGET_TIME = float(os.getenv('BASELINE_TARGET', '600.0'))
DEFAULT_OUT_DIR = os.path.join(ROOT, 'node_dashboard', 'baselines')
DEFAULT_OUT_FILE = os.getenv('BASELINE_OUT_FILE', os.path.join(DEFAULT_OUT_DIR, f'baseline_{DEFAULT_MAP_NAME}.json'))

parser = argparse.ArgumentParser(description='Generate a SUMO baseline snapshot with metrics.')
parser.add_argument('--map-name', default=DEFAULT_MAP_NAME)
parser.add_argument('--map-cfg', default=DEFAULT_MAP_CFG)
parser.add_argument('--sumo-bin', default=DEFAULT_SUMO_BIN)
parser.add_argument('--step', type=float, default=DEFAULT_STEP)
parser.add_argument('--target-time', type=float, default=DEFAULT_TARGET_TIME)
parser.add_argument('--out-file', default=DEFAULT_OUT_FILE)
args = parser.parse_args()

MAP_NAME = args.map_name.strip() or 'basic'
MAP_CFG = args.map_cfg
SUMO_BIN = args.sumo_bin
STEP = float(args.step)
TARGET_TIME = float(args.target_time)
OUT_FILE = args.out_file
OUT_DIR = os.path.dirname(OUT_FILE)

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Using SUMO: {SUMO_BIN}")
print(f"Map name: {MAP_NAME}")
print(f"Config: {MAP_CFG}")
print(f"Output: {OUT_FILE}")

cmd = [SUMO_BIN, '-c', MAP_CFG, '--step-length', str(STEP), '--no-warnings']
print('Launching SUMO:', ' '.join(cmd))
try:
    traci.start(cmd)
except Exception as e:
    print('Failed to start SUMO:', e)
    sys.exit(1)

snapshots = []
step = 0

def safe_get(getter, *args, default=0.0):
    try:
        return float(getter(*args))
    except Exception:
        return float(default)

while True:
    try:
        traci.simulationStep()
    except Exception as e:
        print('Simulation step error:', e)
        break
    step += 1
    sim_time = traci.simulation.getTime()
    vehicle_ids = traci.vehicle.getIDList()

    # Collect vehicle-level snapshot (minimal)
    vehicles = []
    speeds = []
    for vid in vehicle_ids:
        try:
            s = safe_get(traci.vehicle.getSpeed, vid)
            speeds.append(s)
            veh = {
                'id': vid,
                'speed': s,
                'acceleration': safe_get(traci.vehicle.getAcceleration, vid),
                'lane_id': traci.vehicle.getLaneID(vid) if traci.vehicle.getLaneID(vid) else None,
                'co2_emission': safe_get(traci.vehicle.getCO2Emission, vid),
                'fuel_consumption': safe_get(traci.vehicle.getFuelConsumption, vid),
                'noise_emission': safe_get(traci.vehicle.getNoiseEmission, vid),
            }
            vehicles.append(veh)
        except Exception:
            continue

    vehicle_count = len(vehicle_ids)
    stopped_count = sum(1 for s in speeds if s < 0.5)
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    fuel = 0.0
    co2 = 0.0
    noise = 0.0
    pm = 0.0
    nox = 0.0
    nvmoc = 0.0
    emergency = 0
    for vid in vehicle_ids:
        try:
            fuel += max(safe_get(traci.vehicle.getFuelConsumption, vid) / 1000.0, 0.0)
            co2 += max(safe_get(traci.vehicle.getCO2Emission, vid), 0.0)
            noise += max(safe_get(traci.vehicle.getNoiseEmission, vid), 0.0)
            pm += max(safe_get(traci.vehicle.getPMxEmission, vid), 0.0)
            nox += max(safe_get(traci.vehicle.getNOxEmission, vid), 0.0)
            nvmoc += max(safe_get(traci.vehicle.getHCEmission, vid), 0.0)
            acc = safe_get(traci.vehicle.getAcceleration, vid)
            if acc < -3.0:
                emergency += 1
        except Exception:
            continue

    try:
        collision = int(traci.simulation.getCollidingVehiclesNumber())
    except Exception:
        collision = 0

    jammed_lanes = 0
    jam_events = 0

    snapshot = {
        'step': step,
        'simulation_time': sim_time,
        'timestamp': datetime.now().isoformat(),
        'vehicle_count': vehicle_count,
        'avg_speed': avg_speed,
        'stopped_count': stopped_count,
        'stopped_ratio': (stopped_count / vehicle_count) if vehicle_count else 0.0,
        'active_attack_count': 0,
        'active_attack_types': [],
        'tls_under_attack': [],
        'jammed_lanes': jammed_lanes,
        'jam_events': jam_events,
        'metrics': {
            'fuel_consumption': fuel,
            'co2': co2,
            'noise': noise,
            'jam': float(jammed_lanes),
            'emergency_breaking': float(emergency),
            'pm': pm,
            'nox': nox,
            'congestion': (stopped_count / vehicle_count) if vehicle_count else 0.0,
            'collision': float(collision),
            'nvmoc': nvmoc,
        },
    }
    snapshots.append(snapshot)

    if sim_time >= TARGET_TIME:
        print(f"Target time reached: {sim_time}s (steps={step}), saving baseline...")
        break

print(f"Writing {len(snapshots)} points to {OUT_FILE}")
with open(OUT_FILE + '.tmp', 'w', encoding='utf-8') as f:
    json.dump(snapshots, f, indent=2, default=str)
os.replace(OUT_FILE + '.tmp', OUT_FILE)
print('Saved baseline successfully.')
traci.close()
