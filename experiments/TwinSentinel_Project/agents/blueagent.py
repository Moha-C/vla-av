
#!/usr/bin/env python3
"""Blue Agent - Defensive SUMO Simulation Controller using LLM + MCP Tools"""
import asyncio
import json
import httpx

MCP_URL = "http://127.0.0.1:8000/mcp/"
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL = "qwen:1.8b"
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

class BlueAgent:
    """Blue Team - Defensive traffic simulation control agent"""
    
    def __init__(self):
        self.name = "Blue Agent"
        self.role = "Defensive Traffic Simulation Control"
        self.session_id = None
        self.request_id = 1
        self.available_tools = [
            "launch_basic_simulation",
            "launch_Paris",
            "start_simulation",
            "stop_simulation",
            "simulation_stats",
            "adaptive_traffic_lights",
            "export_traffic_report"
        ]

    def _next_request_id(self) -> int:
        current = self.request_id
        self.request_id += 1
        return current

    def _session_headers(self) -> dict:
        headers = dict(MCP_HEADERS)
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        return headers

    def _parse_mcp_response(self, response: httpx.Response) -> dict:
        content_type = (response.headers.get("content-type") or "").lower()
        if "text/event-stream" in content_type:
            last_data = None
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    if payload:
                        last_data = payload
            if not last_data:
                return {"error": "Empty event-stream response", "raw": response.text}
            return json.loads(last_data)
        return response.json()

    async def _ensure_session(self) -> dict:
        """Initialize MCP streamable-http session if needed."""
        if self.session_id:
            return {"ok": True}

        init_payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "blueagent", "version": "1.0.0"},
            },
        }

        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                init_response = await client.post(MCP_URL, json=init_payload, headers=MCP_HEADERS)
                if init_response.status_code != 200:
                    return {"ok": False, "error": f"HTTP {init_response.status_code}", "details": init_response.text}

                self.session_id = init_response.headers.get("mcp-session-id")
                if not self.session_id:
                    return {"ok": False, "error": "Missing mcp-session-id", "details": init_response.text}

                # MCP lifecycle notification after initialize
                initialized_payload = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
                await client.post(MCP_URL, json=initialized_payload, headers=self._session_headers())

            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        
    async def call_mcp_tool(self, tool_name: str, arguments: dict = None) -> dict:
        """Call a tool on the MCP server via JSON-RPC"""
        try:
            session = await self._ensure_session()
            if not session.get("ok"):
                return {"error": "MCP session init failed", "details": session}

            payload = {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments or {}
                }
            }
            
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.post(MCP_URL, json=payload, headers=self._session_headers())
                
                if response.status_code == 200:
                    result = self._parse_mcp_response(response)
                    # Extract the tool result
                    if "result" in result:
                        tool_result = result["result"]
                        content = tool_result.get("content") if isinstance(tool_result, dict) else None
                        if isinstance(content, list) and content:
                            text_payload = content[0].get("text")
                            if isinstance(text_payload, str):
                                try:
                                    return json.loads(text_payload)
                                except Exception:
                                    return {"text": text_payload}
                        return tool_result
                    return result
                else:
                    return {"error": f"HTTP {response.status_code}", "details": response.text}
        except Exception as e:
            return {"error": str(e), "tool": tool_name}
    
    async def get_ai_decision(self, prompt: str) -> str:
        """Get decision from LLM via Ollama"""
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                response = await client.post(
                    OLLAMA_API,
                    json={"model": MODEL, "prompt": prompt, "stream": False},
                    timeout=60
                )
                if response.status_code == 200:
                    result = response.json().get("response", "").strip()
                    return result if result else "Processing..."
        except Exception as e:
            return f"[LLM error: {type(e).__name__}]"
        return "No response"
    
    async def test_server_connection(self) -> bool:
        """Test MCP server connection"""
        try:
            session = await self._ensure_session()
            if not session.get("ok"):
                return False

            async with httpx.AsyncClient(timeout=5, follow_redirects=True) as client:
                response = await client.post(
                    MCP_URL,
                    json={"jsonrpc": "2.0", "id": self._next_request_id(), "method": "tools/list"},
                    headers=self._session_headers(),
                )
                return response.status_code < 500
        except:
            return False
    
    async def run(self):
        """Run interactive blue agent"""
        print("=" * 70)
        print(f"🔵 {self.name} - {self.role}")
        print("=" * 70)
        print()
        
        # Check server connection
        if not await self.test_server_connection():
            print(f"❌ Cannot connect to MCP server at {MCP_URL}")
            print(f"   Start the server with: python3 MCP_server.py")
            print()
            return
        
        print(f"✅ Connected to MCP server at {MCP_URL}")
        print()
        print("📋 Available Defense Tools:")
        for i, tool in enumerate(self.available_tools, 1):
            print(f"   {i}. {tool}")
        print()
        print("Commands:")
        print("  help       - Show commands")
        print("  health     - Health check (MCP test endpoint)")
        print("  status     - Same as stats")
        print("  stats      - Simulation statistics")
        print("  analyze    - LLM analysis on latest status")
        print("  launch     - Launch basic simulation")
        print("  launch_paris - Launch Paris simulation")
        print("  start      - Start simulation")
        print("  stop       - Stop simulation")
        print("  defend     - Activate adaptive traffic lights")
        print("  report     - Export traffic report")
        print("  exit/quit  - Exit agent")
        print()
        print("=" * 70)
        print()
        
        # Interactive loop
        while True:
            try:
                cmd = input("🔵 > ").strip().lower()
                if not cmd:
                    continue
                    
                if cmd in ["exit", "quit"]:
                    print("👋 Goodbye!")
                    break
                
                if cmd == "help":
                    print("\n📋 Commands:")
                    print("   health   -> test_endpoint")
                    print("   status   -> simulation_stats")
                    print("   stats    -> simulation_stats")
                    print("   analyze  -> simulation_stats + Ollama")
                    print("   launch   -> launch_basic_simulation")
                    print("   launch_paris -> launch_Paris")
                    print("   start    -> start_simulation")
                    print("   stop     -> stop_simulation")
                    print("   defend   -> adaptive_traffic_lights")
                    print("   report   -> export_traffic_report")
                    print()
                    continue
                
                # Tool execution
                result = None
                
                if cmd == "launch":
                    print("⏳ Launching SUMO simulation...")
                    result = await self.call_mcp_tool("launch_basic_simulation", {})

                elif cmd in ["launch_paris", "paris"]:
                    print("⏳ Launching Paris SUMO simulation...")
                    result = await self.call_mcp_tool("launch_Paris", {})

                elif cmd == "health":
                    print("⏳ Running health check...")
                    result = await self.call_mcp_tool("test_endpoint", {})

                elif cmd == "status":
                    print("⏳ Retrieving simulation status...")
                    result = await self.call_mcp_tool("simulation_stats", {})

                elif cmd == "analyze":
                    print("⏳ Collecting status and asking Ollama...")
                    stats = await self.call_mcp_tool("simulation_stats", {})
                    if isinstance(stats, dict) and stats.get("error"):
                        result = stats
                    else:
                        prompt = (
                            "You are Blue Team analyst for VANET traffic simulation. "
                            "Given this simulation status, provide a concise analysis and one defense action "
                            "from this list only: adaptive_traffic_lights, export_traffic_report. "
                            f"Status: {stats}"
                        )
                        llm_text = await self.get_ai_decision(prompt)
                        result = {"status": stats, "llm_analysis": llm_text}
                
                elif cmd == "start":
                    print("⏳ Starting simulation loop...")
                    result = await self.call_mcp_tool("start_simulation", {})
                
                elif cmd == "stop":
                    print("⏳ Stopping simulation...")
                    result = await self.call_mcp_tool("stop_simulation", {})
                
                elif cmd == "stats":
                    print("⏳ Retrieving simulation statistics...")
                    result = await self.call_mcp_tool("simulation_stats", {})
                
                elif cmd == "defend":
                    print("⏳ Activating adaptive traffic lights...")
                    result = await self.call_mcp_tool("adaptive_traffic_lights", {"action": "enable"})
                
                elif cmd == "report":
                    print("⏳ Exporting traffic report...")
                    result = await self.call_mcp_tool("export_traffic_report", {})
                
                else:
                    print(f"❓ Unknown command: {cmd}")
                    print("   Type 'help' for available commands")
                    continue
                
                # Display result
                if result:
                    print("\n✅ Tool Result:")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            print(f"   {key}: {value}")
                    else:
                        print(f"   {result}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()


async def main():
    agent = BlueAgent()
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Shutdown")
