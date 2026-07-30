import sys
import os
import socket
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traci
import MCP_server

# Set up environment
MCP_server.sumo_binary = "/usr/bin/sumo"
MCP_server.current_map_name = "berlin"
socket.setdefaulttimeout(300.0)

MCP_server.active_attacks = []
MCP_server.location_jams = {}
MCP_server.vehicle_stats = {}
MCP_server.traci_connection = None

cmd = [
    "/usr/bin/sumo",
    "-c", MCP_server.map_path_berlin,
    "--seed", "1",
    "--step-length", "0.05",
    "--lateral-resolution", "0.1",
    "--start",
    "--delay", "0",
    "--no-warnings"
]

print("Starting traci on port 57123...")
try:
    traci.start(cmd, port=57123)
    MCP_server.traci_connection = traci
    print("✓ Successfully connected to TraCI!")
    
    # Run a few steps
    for i in range(5):
        traci.simulationStep()
        print(f"Step {i} simulated.")
finally:
    try:
        traci.close()
    except:
        pass
    print("Done.")
