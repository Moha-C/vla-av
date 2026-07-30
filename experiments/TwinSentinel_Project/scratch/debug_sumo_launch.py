import subprocess
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MCP_server

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

print("Launching sumo directly with subprocess.Popen...")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Wait a few seconds
time.sleep(5)

# Check if process is running
poll = proc.poll()
if poll is not None:
    print(f"SUMO exited with code {poll}")
    stdout, stderr = proc.communicate()
    print("STDOUT:")
    print(stdout)
    print("STDERR:")
    print(stderr)
else:
    print("SUMO is running. Terminating...")
    proc.terminate()
    stdout, stderr = proc.communicate()
    print("STDOUT:")
    print(stdout[:500])
    print("STDERR:")
    print(stderr[:500])
