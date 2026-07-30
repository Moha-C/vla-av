import traci
import numpy as np

cmd = [
    "/usr/bin/sumo",
    "-c", "/home/mehdi/VANET_Project/Docker_files/maps/luxembourg/dua.actuated.sumocfg",
    "--start",
    "--delay", "0",
    "--no-warnings"
]

print("Starting traci for Luxembourg...")
try:
    traci.start(cmd, port=57200)
    counts = []
    # We want average vehicle count between 100s and 200s (2000 to 4000 steps at 0.05s step length)
    for step in range(4001):
        traci.simulationStep()
        t = traci.simulation.getTime()
        if 100 <= t <= 200:
            counts.append(traci.vehicle.getIDCount())
            
    if counts:
        print(f"Luxembourg (100-200s): Mean={np.mean(counts):.2f}")
    else:
        print("No vehicle count recorded.")
finally:
    try:
        traci.close()
    except:
        pass
