import urllib.request
import urllib.error
import json
import time

MCP_URL = "http://localhost:8000/mcp"

def parse_mcp_body(body_str, content_type):
    ct = content_type.lower()
    if "text/event-stream" in ct:
        lines = body_str.split("\n")
        last = None
        for line in lines:
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    last = payload
        if last:
            return json.loads(last)
        return {}
    return json.loads(body_str)

def extract_tool_payload(parsed):
    result = parsed.get("result", parsed) if isinstance(parsed, dict) else parsed
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list) and len(content) > 0:
            text = content[0].get("text")
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except:
                    return {"text": text}
    return result

def call_tool(name, arguments, headers):
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments
        }
    }
    req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as response:
            body = response.read().decode('utf-8')
            ct = response.info().get("Content-Type", "")
            parsed = parse_mcp_body(body, ct)
            return extract_tool_payload(parsed)
    except urllib.error.HTTPError as e:
        print(f"HTTP Error calling {name}: {e.code} - {e.reason}")
        print(e.read().decode('utf-8'))
        return None
    except Exception as e:
        print(f"Error calling {name}: {e}")
        return None

def check_mcp_sim():
    # 1. Initialize
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "check-script", "version": "1.0.0" }
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    print("Initializing session...")
    req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(init_payload).encode('utf-8'),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                print(f"Failed to initialize: HTTP {response.status}")
                return
            
            resp_headers = response.info()
            session_id = resp_headers.get("mcp-session-id")
            if not session_id:
                print("No session ID returned in headers!")
                return
                
            print(f"Session established: {session_id}")
            headers["mcp-session-id"] = session_id
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Initialized Notification
    initialized_payload = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    }
    req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(initialized_payload).encode('utf-8'),
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as response:
            pass
    except Exception as e:
        print(f"Initialized notification failed: {e}")

    # 3. Call start_simulation
    print("Calling start_simulation...")
    res = call_tool("start_simulation", {}, headers)
    print("start_simulation response:", res)

    # 4. Wait a bit and check stats
    print("Waiting 3 seconds...")
    time.sleep(3)
    
    print("Calling simulation_stats...")
    res = call_tool("simulation_stats", {}, headers)
    print("simulation_stats response:", res)

if __name__ == "__main__":
    check_mcp_sim()
