const express = require("express");
const http = require("http");
const path = require("path");
const fs = require("fs");
const axios = require("axios");
const { Server } = require("socket.io");
const { exec } = require("child_process");

const PORT = Number(process.env.PORT || 3100);
const MCP_URL = process.env.MCP_URL || "http://127.0.0.1:8000/mcp";
const POLL_MS = Number(process.env.POLL_MS || 1000);
const BASELINE_DIR = path.join(__dirname, "..", "baselines");
const LIVE_MODE = String(process.env.TWINSENTINEL_LIVE_MODE || "0") === "1";
const LIVE_STATE_FILE = process.env.TWINSENTINEL_STATE_FILE || path.resolve(__dirname, "../../../logs/sumo_mirror/live_state.json");
const LIVE_COMMAND_FILE = process.env.TWINSENTINEL_COMMAND_FILE || path.resolve(__dirname, "../../../logs/sumo_mirror/attack_commands.jsonl");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const server = http.createServer(app);
const io = new Server(server);

let sessionId = null;
let reqId = 1;
const baselineCache = new Map();
const liveHistory = [];

const LIVE_ATTACKS = {
  sybil: {
    label: "Sybil Injection",
    commandType: "sybil",
    defaultParams: { duration: 100, count: 10 },
  },
  traffic_light: {
    label: "Traffic Light Tampering",
    commandType: "traffic_light_tampering",
    defaultParams: { duration: 100, mode: "all_red", ratio: 1.0 },
  },
  universal_perturbation: {
    label: "Universal Perturbation",
    commandType: "universal_perturbation",
    defaultParams: { duration: 100, epsilon: 0.5, scale_position: 0.5, scale_velocity: 0.3 },
  },
  sensor_spoofing: {
    label: "Sensor Spoofing",
    commandType: "adversarial_sensor_spoofing",
    defaultParams: { duration: 100, num_obstacles: 2 },
  },
  fake_safety: {
    label: "Fake Safety Alert",
    commandType: "fake_safety",
    defaultParams: { duration: 100, count: 2 },
  },
  fake_emergency: {
    label: "Fake Emergency Vehicle",
    commandType: "fake_emergency",
    defaultParams: { duration: 100, count: 2, speed: 22.0 },
  },
  clear_attacks: {
    label: "Clear / Restore",
    commandType: "clear_attacks",
    defaultParams: {},
  },
};

function readJsonFile(filePath) {
  try {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function appendLiveCommand(command) {
  fs.mkdirSync(path.dirname(LIVE_COMMAND_FILE), { recursive: true });
  const payload = {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    created_at: new Date().toISOString(),
    source: "TwinSentinel_Project/node_dashboard",
    ...command,
  };
  fs.appendFileSync(LIVE_COMMAND_FILE, `${JSON.stringify(payload)}\n`, "utf8");
  return payload;
}

function liveSnapshot() {
  const latest = readJsonFile(LIVE_STATE_FILE);
  if (!latest) {
    return {
      status: "waiting_for_vla_av_mirror",
      bridge_mode: "vla_av_carla_sumo_live_bridge",
      state_file: LIVE_STATE_FILE,
      command_file: LIVE_COMMAND_FILE,
      latest: null,
    };
  }
  return {
    status: "connected",
    bridge_mode: "vla_av_carla_sumo_live_bridge",
    state_file: LIVE_STATE_FILE,
    command_file: LIVE_COMMAND_FILE,
    latest,
  };
}

function liveAttackCommand(attackType, reqBody = {}) {
  const attack = LIVE_ATTACKS[attackType];
  if (!attack) {
    throw new Error(`Unknown attack type: ${attackType}`);
  }
  const params = {
    ...attack.defaultParams,
    ...((reqBody && reqBody.params) || {}),
  };
  if (typeof reqBody.duration !== "undefined") params.duration = Number(reqBody.duration);
  if (typeof reqBody.count !== "undefined") params.count = Number(reqBody.count);
  if (typeof reqBody.speed !== "undefined") params.speed = Number(reqBody.speed);
  if (typeof reqBody.ratio !== "undefined") params.ratio = Number(reqBody.ratio);

  const command = {
    type: attack.commandType,
    attack: attack.commandType,
    twin_sentinel_attack: attackType,
    label: attack.label,
    ...params,
  };
  return appendLiveCommand(command);
}

function resetSession() {
  sessionId = null;
}

function normalizeMapName(mapName = "paris") {
  const key = String(mapName || "paris").trim().toLowerCase();
  if (key === "basic_simulation") return "basic";
  
  const seedMatch = key.match(/seed_?(\d+)/i);
  if (seedMatch) {
    const seedPart = `_seed_${seedMatch[1]}`;
    if (key.includes("luxembourg") || key.includes("lux")) {
      return "lux" + seedPart;
    }
    if (key.includes("paris")) {
      return "paris" + seedPart;
    }
    if (key.includes("berlin")) {
      return "berlin" + seedPart;
    }
  }
  
  if (key.includes("luxembourg") || key.includes("lux")) return "lux";
  return key;
}

function baselineFilePath(mapName, seed = 42) {
  const normMap = normalizeMapName(mapName);
  
  let baseMap = "paris";
  if (normMap.includes("paris")) baseMap = "paris";
  else if (normMap.includes("berlin")) baseMap = "berlin";
  else if (normMap.includes("lux")) baseMap = "lux";
  else if (normMap.includes("basic")) baseMap = "basic";
  else baseMap = normMap;
  
  let finalSeed = seed;
  const seedMatch = normMap.match(/seed_?(\d+)/i);
  if (seedMatch) {
    finalSeed = parseInt(seedMatch[1], 10);
  }
  
  if (baseMap === "paris") {
    const defaultParis = path.join(BASELINE_DIR, "baseline_paris.json");
    if (finalSeed === 42 && fs.existsSync(defaultParis)) {
      return defaultParis;
    }
    return path.join(BASELINE_DIR, `baseline_paris_seed_${finalSeed}.json`);
  }
  
  if (baseMap === "berlin") {
    const defaultBerlin = path.join(BASELINE_DIR, "baseline_berlin.json");
    if (finalSeed === 42 && fs.existsSync(defaultBerlin)) {
      return defaultBerlin;
    }
    return path.join(BASELINE_DIR, `baseline_berlin_seed_${finalSeed}.json`);
  }
  
  if (baseMap === "lux") {
    const defaultLux = path.join(BASELINE_DIR, "baseline_luxembourg.json");
    if (finalSeed === 42 && fs.existsSync(defaultLux)) {
      return defaultLux;
    }
    return path.join(BASELINE_DIR, `baseline_lux_seed_${finalSeed}.json`);
  }
  
  return path.join(BASELINE_DIR, `baseline_${baseMap}.json`);
}

function loadBaselineFromDisk(mapName, seed = 42, forceReload = false) {
  const normalized = normalizeMapName(mapName);
  
  let baseMap = "paris";
  if (normalized.includes("paris")) baseMap = "paris";
  else if (normalized.includes("berlin")) baseMap = "berlin";
  else if (normalized.includes("lux")) baseMap = "lux";
  else if (normalized.includes("basic")) baseMap = "basic";
  else baseMap = normalized;
  
  let finalSeed = seed;
  const seedMatch = normalized.match(/seed_?(\d+)/i);
  if (seedMatch) {
    finalSeed = parseInt(seedMatch[1], 10);
  }
  
  const cacheKey = `${baseMap}_seed_${finalSeed}`;
  if (!forceReload && baselineCache.has(cacheKey)) {
    return baselineCache.get(cacheKey);
  }
  const filePath = baselineFilePath(normalized, seed);
  if (!fs.existsSync(filePath)) return null;
  const raw = fs.readFileSync(filePath, "utf8");
  const parsed = JSON.parse(raw);
  baselineCache.set(cacheKey, parsed);
  return parsed;
}

function preloadBaselines() {
  if (!fs.existsSync(BASELINE_DIR)) {
    fs.mkdirSync(BASELINE_DIR, { recursive: true });
  }
  const loaded = [];
  for (const file of fs.readdirSync(BASELINE_DIR)) {
    if (!file.startsWith("baseline_") || !file.endsWith(".json")) continue;
    const mapName = file.replace("baseline_", "").replace(/\.json$/i, "");
    const data = loadBaselineFromDisk(mapName, 42, true);
    if (Array.isArray(data)) {
      loaded.push({ map_name: mapName, count: data.length });
    }
  }
  return loaded;
}

const mcpHeaders = () => {
  const headers = {
    Accept: "application/json, text/event-stream",
    "Content-Type": "application/json",
  };
  if (sessionId) {
    headers["mcp-session-id"] = sessionId;
  }
  return headers;
};

const nextId = () => reqId++;

function parseMcpBody(response) {
  const ct = String(response.headers["content-type"] || "").toLowerCase();
  if (ct.includes("text/event-stream") && typeof response.data === "string") {
    const lines = response.data.split(/\r?\n/);
    let last = null;
    for (const line of lines) {
      if (line.startsWith("data:")) {
        const payload = line.slice(5).trim();
        if (payload) last = payload;
      }
    }
    return last ? JSON.parse(last) : {};
  }
  return response.data;
}

function extractToolPayload(parsed) {
  const result = parsed && parsed.result ? parsed.result : parsed;
  const content = result && result.content;
  if (Array.isArray(content) && content.length > 0) {
    const text = content[0] && content[0].text;
    if (typeof text === "string") {
      try {
        return JSON.parse(text);
      } catch {
        return { text };
      }
    }
  }
  return result;
}

async function ensureSession() {
  if (sessionId) return;

  const initPayload = {
    jsonrpc: "2.0",
    id: nextId(),
    method: "initialize",
    params: {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "node-dashboard", version: "1.0.0" },
    },
  };

  const initRes = await axios.post(MCP_URL, initPayload, {
    headers: mcpHeaders(),
    timeout: 30000,
    maxRedirects: 5,
    validateStatus: () => true,
  });

  if (initRes.status !== 200) {
    throw new Error(`MCP initialize failed: HTTP ${initRes.status}`);
  }

  sessionId = initRes.headers["mcp-session-id"];
  if (!sessionId) {
    throw new Error("MCP initialize failed: missing mcp-session-id header");
  }

  await axios.post(
    MCP_URL,
    {
      jsonrpc: "2.0",
      method: "notifications/initialized",
      params: {},
    },
    {
      headers: mcpHeaders(),
      timeout: 30000,
      maxRedirects: 5,
      validateStatus: () => true,
    }
  );
}

async function mcpToolCall(name, argumentsObj = {}) {
  const postToolCall = async () => {
    const payload = {
      jsonrpc: "2.0",
      id: nextId(),
      method: "tools/call",
      params: {
        name,
        arguments: argumentsObj,
      },
    };

    return axios.post(MCP_URL, payload, {
      headers: mcpHeaders(),
      timeout: 180000,
      maxRedirects: 5,
      validateStatus: () => true,
    });
  };

  try {
    await ensureSession();
    let response = await postToolCall();

    if (response.status === 400 || response.status === 401 || response.status === 403 || response.status === 404 || response.status === 409) {
      resetSession();
      await ensureSession();
      response = await postToolCall();
    }

    if (response.status !== 200) {
      throw new Error(`MCP tools/call failed: HTTP ${response.status}`);
    }

    return extractToolPayload(parseMcpBody(response));
  } catch (error) {
    if (error && error.response && (error.response.status === 400 || error.response.status === 401 || error.response.status === 403 || error.response.status === 404 || error.response.status === 409)) {
      resetSession();
    }
    throw error;
  }
}

app.get("/api/health", async (req, res) => {
  try {
    if (LIVE_MODE) {
      return res.json({
        ok: true,
        live_mode: true,
        mcp: null,
        attacks: Object.keys(LIVE_ATTACKS),
        simulation: liveSnapshot(),
      });
    }
    const data = await mcpToolCall("simulation_stats", {});
    res.json({ ok: true, mcp: MCP_URL, simulation: data });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message, mcp: MCP_URL });
  }
});

app.post("/api/simulation/launch", async (req, res) => {
  try {
    if (LIVE_MODE) {
      return res.json({
        ok: true,
        live_mode: true,
        message: "TwinSentinel is attached to the active VLA-AV CARLA/SUMO bridge. Launch CARLA+SUMO from the VLA-AV dashboard, then inject attacks here.",
        simulation: liveSnapshot(),
      });
    }

    const selectedBaseline = String(req.body && req.body.map_name ? req.body.map_name : "paris").toLowerCase().trim();
    
    // Parse map and seed from baseline name
    let map_name = "paris";
    let seed = 42;
    
    if (selectedBaseline.includes("paris")) {
      map_name = "paris";
    } else if (selectedBaseline.includes("berlin")) {
      map_name = "berlin";
    } else if (selectedBaseline.includes("luxembourg") || selectedBaseline.includes("lux")) {
      map_name = "luxembourg";
    } else if (selectedBaseline.includes("basic")) {
      map_name = "basic";
    } else {
      map_name = "paris";
    }
    
    const seedMatch = selectedBaseline.match(/seed_?(\d+)/i);
    if (seedMatch) {
      seed = parseInt(seedMatch[1], 10);
    } else if (selectedBaseline === "paris1") {
      seed = 1;
    } else if (req.body && req.body.seed) {
      seed = parseInt(req.body.seed, 10) || 42;
    }

    let mcpToolName = "launch_Paris";
    if (map_name === "basic") {
      mcpToolName = "launch_basic_simulation";
    } else if (map_name === "berlin") {
      mcpToolName = "launch_Berlin";
    } else if (map_name === "luxembourg") {
      mcpToolName = "launch_Luxembourg";
    } else if (map_name === "paris") {
      mcpToolName = "launch_Paris";
    }

    console.log(`[dashboard] Received launch request. selectedBaseline='${selectedBaseline}', resolved map_name='${map_name}', seed=${seed}`);

    // Stop current running simulation if any to avoid port conflicts
    try {
      console.log(`[dashboard] Attempting to stop any running simulation first...`);
      await mcpToolCall("stop_simulation", {});
    } catch (stopErr) {
      console.log("[dashboard] stop_simulation failed/was not active:", stopErr.message);
    }

    const headless = !!(req.body && req.body.headless);
    console.log(`[dashboard] Calling MCP tool '${mcpToolName}' with headless=${headless}, seed=${seed}...`);
    const launchData = await mcpToolCall(mcpToolName, { headless, seed });
    console.log(`[dashboard] MCP tool '${mcpToolName}' result:`, JSON.stringify(launchData));

    console.log(`[dashboard] Calling MCP tool 'start_simulation'...`);
    const startData = await mcpToolCall("start_simulation", {});
    console.log(`[dashboard] MCP 'start_simulation' result:`, JSON.stringify(startData));

    console.log(`[dashboard] Launch simulation sequence completed successfully.`);
    res.json({
      ok: true,
      launch: launchData,
      start: startData,
      message: `Launched simulation on map '${map_name}' successfully (headless: ${headless}, seed: ${seed}).`
    });
  } catch (error) {
    console.error(`[dashboard] Error launching simulation:`, error);
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/simulation/attack", async (req, res) => {
  try {
    const attackType = String(req.body && req.body.type ? req.body.type : "").trim().toLowerCase();
    const mapName = String(req.body && req.body.map_name ? req.body.map_name : "paris").trim().toLowerCase();

    if (LIVE_MODE) {
      const command = liveAttackCommand(attackType, req.body || {});
      return res.json({
        ok: true,
        live_mode: true,
        status: "queued",
        mode: "vla_av_live_bridge",
        command,
        simulation: liveSnapshot(),
        message: `Queued ${attackType} for the active CARLA/SUMO mirror.`,
      });
    }

    let mcpToolName = "";
    
    // Increased duration (100s) to give the classifier more windows of observation
    let params = { duration: 100 }; 

    // Scale up the attack severity based on map size
    let scaleMultiplier = 1;
    if (mapName === "paris") {
      scaleMultiplier = 8;  // Spawns 40 Sybils, 16 obstacles, etc.
    } else if (mapName === "luxembourg") {
      scaleMultiplier = 4;  // Spawns 20 Sybils, 8 obstacles, etc.
    }

    switch (attackType) {
      case "sybil":
        mcpToolName = "sybil_attack";
        params.count = 5 * scaleMultiplier;
        break;
      case "traffic_light":
        mcpToolName = "traffic_light_tampering_attack";
        break;
      case "universal_perturbation":
        mcpToolName = "universal_perturbation_attack";
        params.epsilon = 0.5; // Stronger deceleration (50% reduction)
        break;
      case "sensor_spoofing":
        mcpToolName = "targeted_adversarial_sensor_spoofing";
        params.num_obstacles = 2 * scaleMultiplier;
        break;
      case "fake_safety":
        mcpToolName = "fake_safety_message_attack";
        params.count = 2 * scaleMultiplier;
        break;
      case "fake_emergency":
        mcpToolName = "fake_emergency_vehicle_broadcast";
        params.speed = 22.0;
        params.count = 2 * scaleMultiplier;
        break;
      default:
        return res.status(400).json({ error: `Unknown attack type: ${attackType}` });
    }

    const data = await mcpToolCall(mcpToolName, { params });
    res.json({ ok: true, data });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/live/attacks", (_req, res) => {
  res.json({
    live_mode: LIVE_MODE,
    attacks: Object.entries(LIVE_ATTACKS).map(([id, spec]) => ({
      id,
      label: spec.label,
      command_type: spec.commandType,
      default_params: spec.defaultParams,
    })),
  });
});

app.post("/api/attack/clear", async (_req, res) => {
  try {
    if (!LIVE_MODE) {
      return res.status(400).json({ error: "Clear/restore is only available in VLA-AV live bridge mode." });
    }
    const command = liveAttackCommand("clear_attacks", {});
    res.json({
      ok: true,
      live_mode: true,
      status: "queued",
      mode: "vla_av_live_bridge",
      command,
      message: "Clear/restore command queued for the active CARLA/SUMO mirror.",
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/benchmark/:label", async (req, res) => {
  try {
    if (LIVE_MODE) {
      return res.json({
        status: "captured",
        label: String(req.params.label || "live"),
        snapshot: liveSnapshot().latest,
        count: liveHistory.length,
        source: "vla_av_live_bridge",
      });
    }
    const label = String(req.params.label || "baseline");
    const window_steps = Number(req.body && req.body.window_steps ? req.body.window_steps : 300);
    const data = await mcpToolCall("capture_benchmark", { params: { label, window_steps } });
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/history", async (req, res) => {
  try {
    if (LIVE_MODE) {
      const limit = Number(req.query.limit || 0);
      const history = limit > 0 ? liveHistory.slice(-limit) : liveHistory;
      return res.json({ status: "ok", history, count: history.length, source: "vla_av_live_bridge" });
    }
    const limit = Number(req.query.limit || 0);
    const data = await mcpToolCall("metric_history", { params: { limit } });
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/compare", async (req, res) => {
  try {
    if (LIVE_MODE) {
      return res.json({
        status: "live_bridge_only",
        baseline: null,
        candidate: liveSnapshot().latest,
        message: "Use the VLA-AV dashboard KPI table for model-to-model comparison.",
      });
    }
    const baseline = String(req.query.baseline || "baseline");
    const candidate = String(req.query.candidate || "attacked");
    const data = await mcpToolCall("compare_benchmarks", { params: { baseline, candidate } });
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/metric-documentation", async (req, res) => {
  try {
    if (LIVE_MODE) {
      return res.json({
        status: "ok",
        source: "vla_av_live_bridge",
        note: "Live mode streams CARLA-owned mirror metrics from VLA-AV logs/sumo_mirror/live_state.json.",
      });
    }
    const data = await mcpToolCall("metric_documentation", { params: {} });
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/baseline/load", async (req, res) => {
  try {
    const map_name = normalizeMapName(req.body && req.body.map_name ? req.body.map_name : "paris");
    const seed = Number(req.body && req.body.seed ? req.body.seed : 42);
    const forceReload = Boolean(req.body && req.body.force_reload);
    let baseline = loadBaselineFromDisk(map_name, seed, forceReload);

    if (LIVE_MODE && !baseline) {
      return res.status(404).json({
        error: `Baseline not available for '${map_name}' in VLA-AV live bridge mode.`,
        live_mode: true,
        source: "TwinSentinel_Project/node_dashboard",
      });
    }

    if (!baseline) {
      const saveResult = await mcpToolCall("baseline_reference_load", { params: { map_name, seed } });
      if (saveResult && saveResult.error) {
        return res.status(404).json({
          error: saveResult.error,
          available_baselines: saveResult.available_baselines || [],
        });
      }
      const currentFromMcp = await mcpToolCall("baseline_get_current", { params: {} });
      if (currentFromMcp && currentFromMcp.status === "loaded" && Array.isArray(currentFromMcp.baseline)) {
        baseline = currentFromMcp.baseline;
        const cacheKey = map_name === "paris" ? `paris_seed_${seed}` : map_name;
        baselineCache.set(cacheKey, baseline);
      }
    }

    if (!baseline) {
      return res.status(404).json({
        error: `Baseline not available for map '${map_name}' (seed: ${seed})`,
      });
    }

    const data = {
      status: "loaded",
      map_name,
      count: baseline.length,
      first_time: baseline.length ? baseline[0].simulation_time : null,
      last_time: baseline.length ? baseline[baseline.length - 1].simulation_time : null,
      source: "node_disk_cache",
    };
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/baseline/save", async (req, res) => {
  try {
    if (LIVE_MODE) {
      return res.json({
        status: "live_bridge_only",
        message: "Baseline saving is handled by the main VLA-AV dashboard/logs in live bridge mode.",
      });
    }
    const map_name = normalizeMapName(req.body && req.body.map_name ? req.body.map_name : "custom");
    const data = await mcpToolCall("baseline_current_save", { params: { map_name } });
    if (!data.error) {
      loadBaselineFromDisk(map_name, true);
    }
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/baseline/current", async (req, res) => {
  try {
    const map_name = normalizeMapName(req.query.map_name || "paris");
    const seed = Number(req.query.seed || 42);
    const baseline = loadBaselineFromDisk(map_name, seed);
    if (!baseline) {
      return res.status(404).json({
        status: "no_baseline",
        message: `No baseline found for map '${map_name}'`,
      });
    }
    const data = {
      status: "loaded",
      map_name,
      count: baseline.length,
      baseline,
    };
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});


app.post("/api/simulation/save_run", (req, res) => {
  try {
    const { history, map_name, attack_type } = req.body;
    if (!Array.isArray(history) || !history.length) {
      return res.status(400).json({ error: "Empty or invalid history data." });
    }
    
    // Create folder runs/ if it doesn't exist
    const RUNS_DIR = path.join(__dirname, "..", "runs");
    if (!fs.existsSync(RUNS_DIR)) {
      fs.mkdirSync(RUNS_DIR, { recursive: true });
    }
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const attackLabel = attack_type ? String(attack_type).replace(/[^a-zA-Z0-9]/g, "_") : "normal";
    const filename = `run_${map_name || "custom"}_${attackLabel}_${timestamp}.json`;
    const filepath = path.join(RUNS_DIR, filename);
    
    fs.writeFileSync(filepath, JSON.stringify(history, null, 2), "utf8");
    console.log(`[dashboard] Saved simulation run to: ${filepath}`);
    
    res.json({
      status: "saved",
      filename,
      filepath,
      count: history.length
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/baseline/maps", (req, res) => {
  try {
    const loaded = preloadBaselines();
    res.json({
      status: "ok",
      baseline_dir: BASELINE_DIR,
      maps: loaded,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

io.on("connection", (socket) => {
  socket.emit("hello", { connected: true, pollMs: POLL_MS });
});

setInterval(async () => {
  try {
    if (LIVE_MODE) {
      const snapshot = liveSnapshot();
      if (snapshot.latest) {
        liveHistory.push(snapshot.latest);
        if (liveHistory.length > 15000) liveHistory.shift();
        io.emit("metrics", snapshot.latest);
      } else {
        io.emit("metrics_error", {
          error: "Waiting for VLA-AV CARLA/SUMO mirror live state",
          state_file: LIVE_STATE_FILE,
        });
      }
      return;
    }
    const response = await mcpToolCall("realtime_metrics", { params: { window_steps: 200 } });
    // Emit only the latest snapshot (which includes map_name)
    if (response && response.latest) {
      io.emit("metrics", response.latest);
    }
  } catch (error) {
    io.emit("metrics_error", { error: error.message, mcp: MCP_URL });
  }
}, POLL_MS);

server.listen(PORT, () => {
  const preloaded = preloadBaselines();
  console.log(`[dashboard] running on http://localhost:${PORT}`);
  if (LIVE_MODE) {
    console.log("[dashboard] mode: VLA-AV CARLA/SUMO live bridge");
    console.log(`[dashboard] live state: ${LIVE_STATE_FILE}`);
    console.log(`[dashboard] live command file: ${LIVE_COMMAND_FILE}`);
  } else {
    console.log(`[dashboard] MCP endpoint: ${MCP_URL}`);
  }
  console.log(`[dashboard] Baseline cache preloaded: ${preloaded.length} map(s)`);
});
