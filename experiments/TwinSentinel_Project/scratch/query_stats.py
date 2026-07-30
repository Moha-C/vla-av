import urllib.request
import urllib.error
import json

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

def query_stats():
    # 1. Initialize
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "query-script", "version": "1.0.0" }
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(init_payload).encode('utf-8'),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            resp_headers = response.info()
            session_id = resp_headers.get("mcp-session-id")
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
        urllib.request.urlopen(req)
    except Exception as e:
        pass

    # 3. Call simulation_stats
    stats_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "simulation_stats",
            "arguments": {}
        }
    }
    
    req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(stats_payload).encode('utf-8'),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            body = response.read().decode('utf-8')
            ct = response.info().get("Content-Type", "")
            parsed = parse_mcp_body(body, ct)
            payload = extract_tool_payload(parsed)
            print("Response Payload:")
            print(json.dumps(payload, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    query_stats()
