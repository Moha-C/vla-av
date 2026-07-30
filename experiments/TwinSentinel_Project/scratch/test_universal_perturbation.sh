#!/bin/bash
# Test script for Universal Perturbation Attack
# Usage: bash test_universal_perturbation.sh

set -e

MCP_HOST="localhost"
MCP_PORT="8000"
DASHBOARD_PORT="3100"

echo "═══════════════════════════════════════════════════════════════"
echo "   VANET UNIVERSAL PERTURBATION ATTACK TEST"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if MCP server is running
echo -e "${YELLOW}[1/5] Checking MCP server...${NC}"
if curl -s http://$MCP_HOST:$MCP_PORT/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ MCP server is running${NC}"
else
    echo -e "${RED}✗ MCP server is NOT running. Start it first:${NC}"
    echo "   cd /home/mehdi/VANET_Project/Docker_files && python3 MCP_server.py"
    exit 1
fi

# Check if Dashboard is running
echo -e "${YELLOW}[2/5] Checking Dashboard...${NC}"
if curl -s http://$MCP_HOST:$DASHBOARD_PORT/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Dashboard is running${NC}"
else
    echo -e "${YELLOW}⚠ Dashboard not running yet (will start during test)${NC}"
fi

# Launch Basic simulation
echo -e "${YELLOW}[3/5] Launching Basic SUMO simulation...${NC}"
LAUNCH_RESPONSE=$(curl -s -X POST http://$MCP_HOST:$MCP_PORT/mcp/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "launch_basic_simulation",
      "arguments": {}
    }
  }')

echo "Response: $LAUNCH_RESPONSE"
echo -e "${GREEN}✓ Simulation launch initiated${NC}"
echo "   (Waiting 8 seconds for SUMO to start...)"
sleep 8

# Start simulation loop
echo -e "${YELLOW}[4/5] Starting simulation loop...${NC}"
START_RESPONSE=$(curl -s -X POST http://$MCP_HOST:$MCP_PORT/mcp/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "start_simulation",
      "arguments": {}
    }
  }')

echo "Response: $START_RESPONSE"
echo -e "${GREEN}✓ Simulation loop started${NC}"
echo "   (Waiting 5 seconds for vehicles to enter...)"
sleep 5

# Launch Universal Perturbation Attack
echo -e "${YELLOW}[5/5] Launching Universal Perturbation Attack...${NC}"
ATTACK_RESPONSE=$(curl -s -X POST http://$MCP_HOST:$MCP_PORT/mcp/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "universal_perturbation_attack",
      "arguments": {
        "duration": 30,
        "epsilon": 0.5,
        "scale_position": 0.7,
        "scale_velocity": 0.4
      }
    }
  }')

echo "Response: $ATTACK_RESPONSE"

if echo "$ATTACK_RESPONSE" | grep -q "Universal Perturbation attack started"; then
    echo -e "${GREEN}✓ ATTACK LAUNCHED SUCCESSFULLY${NC}"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "   MONITORING ATTACK FOR 35 SECONDS..."
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "WHAT TO OBSERVE:"
    echo "1. Vehicles in simulation should turn ORANGE"
    echo "2. Metrics should degrade (check dashboard)"
    echo "3. Fuel consumption, CO2 should increase"
    echo "4. Average speed should decrease"
    echo "5. After 30s, attack ends and vehicles return to normal"
    echo ""
    
    for i in {1..35}; do
        if [ $((i % 5)) -eq 0 ]; then
            echo "   [$i/35s] Monitoring active attack..."
        fi
        sleep 1
    done
    
    echo ""
    echo -e "${GREEN}✓ ATTACK TEST COMPLETED${NC}"
    echo ""
    echo "NEXT STEPS:"
    echo "1. Open dashboard: http://localhost:3100"
    echo "2. Check Baseline vs Live graphs"
    echo "3. Look for divergence in metrics during attack"
    echo "4. Compare directional arrows (↑ red = worse)"
    echo ""
    
else
    echo -e "${RED}✗ ATTACK LAUNCH FAILED${NC}"
    echo "Full response: $ATTACK_RESPONSE"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "   TEST COMPLETED"
echo "═══════════════════════════════════════════════════════════════"
