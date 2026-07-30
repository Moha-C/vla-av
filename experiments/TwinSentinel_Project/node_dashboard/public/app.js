const socket = io();

const METRICS = {
  fuel_consumption: { label: "Fuel Consumption", unit: "L", decimals: 3, better: "lower" },
  co2: { label: "CO2", unit: "g", decimals: 1, better: "lower" },
  noise: { label: "Noise", unit: "dB", decimals: 1, better: "lower" },
  jam: { label: "Jam", unit: "lanes", decimals: 0, better: "lower" },
  emergency_breaking: { label: "Emergency Braking", unit: "events", decimals: 0, better: "lower" },
  pm: { label: "PM", unit: "g", decimals: 3, better: "lower" },
  nox: { label: "NOx", unit: "g", decimals: 3, better: "lower" },
  congestion: { label: "Congestion", unit: "%", decimals: 1, better: "lower", isPercent: true },
  collision: { label: "Collision", unit: "events", decimals: 0, better: "lower" },
  nvmoc: { label: "NVMOC", unit: "g", decimals: 3, better: "lower" },
};

const simTimeEl = document.getElementById("simTime");
const vehiclesEl = document.getElementById("vehicles");
const avgSpeedEl = document.getElementById("avgSpeed");
const stopRatioEl = document.getElementById("stopRatio");
const attacksEl = document.getElementById("attacks");
const compareTableBody = document.querySelector("#compareTable tbody");
const mapSelect = document.getElementById("mapSelect");

const HISTORY_ATTACKED_KEY = "vanetAttackedHistory";
const SELECTED_METRIC_KEY = "vanetSelectedMetric";
const LAST_MAP_KEY = "vanetLastMap";

const state = {
  liveMode: false,
  liveHistory: [],
  baselineHistory: [],
  attackedHistory: loadStoredHistory(HISTORY_ATTACKED_KEY),
  selectedMetric: localStorage.getItem(SELECTED_METRIC_KEY) || "fuel_consumption",
  latestLive: null,
  currentMap: localStorage.getItem(LAST_MAP_KEY) || "paris",
  currentBaseline: null,
  allBaselines: [],
};

async function detectRuntimeMode() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    state.liveMode = !!(data && data.live_mode);
    if (state.liveMode) {
      const subtitle = document.querySelector(".subtitle");
      if (subtitle) {
        subtitle.textContent = "Attached to the active VLA-AV CARLA/SUMO bridge. Launch the route from VLA-AV, then inject attacks here.";
      }
      const mapControls = document.querySelector(".map-selector-container");
      if (mapControls) {
        mapControls.style.display = "none";
      }
      document.getElementById("saveBaselineBtn")?.setAttribute("disabled", "disabled");
    }
  } catch (err) {
    console.warn("[RUNTIME] health check failed:", err.message);
  }
}

function getBaseMapName(mapName) {
  const norm = String(mapName || "").toLowerCase().replace("_simulation", "");
  if (norm.startsWith("paris")) return "paris";
  if (norm.startsWith("berlin")) return "berlin";
  if (norm.startsWith("luxembourg")) return "luxembourg";
  if (norm === "basic") return "basic";
  return norm;
}

let metricChart = null;

function loadStoredHistory(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function getMetricValue(point, metric) {
  if (!point) return 0;
  if (point.metrics && Object.prototype.hasOwnProperty.call(point.metrics, metric)) {
    return Number(point.metrics[metric]) || 0;
  }
  if (Object.prototype.hasOwnProperty.call(point, metric)) {
    return Number(point[metric]) || 0;
  }
  return 0;
}

function getPointTime(point) {
  const time = Number(point && (point.simulation_time ?? point.step ?? 0));
  return Number.isFinite(time) ? time : 0;
}

function ensureZeroOrigin(points) {
  if (!points.length) return points;
  if (points[0].x > 0) return [{ x: 0, y: points[0].y }, ...points];
  return points;
}

function seriesToPoints(history, metric) {
  return ensureZeroOrigin(
    (history || [])
      .map((point) => ({ x: getPointTime(point), y: getMetricValue(point, metric) }))
      .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  );
}

function downsampleToSeconds(history) {
  if (!history || !history.length) return [];
  const sampled = [];
  for (let i = 0; i < history.length; i++) {
    const pt = history[i];
    const t = getPointTime(pt);
    const sec = Math.floor(t);

    const nextPt = history[i + 1];
    const nextSec = nextPt ? Math.floor(getPointTime(nextPt)) : -1;

    if (nextSec !== sec) {
      sampled.push(pt);
    }
  }
  return sampled;
}

function averageMetricUpToTime(history, metric, maxTime) {
  const limit = Number(maxTime);
  if (!Number.isFinite(limit)) return 0;

  const sampled = downsampleToSeconds(history);
  const filtered = sampled.filter((point) => getPointTime(point) <= limit);
  if (!filtered.length) return 0;

  const total = filtered.reduce((sum, point) => sum + getMetricValue(point, metric), 0);
  return total / filtered.length;
}

function getAverageSnapshot(history, metric, maxTime) {
  return averageMetricUpToTime(history, metric, maxTime);
}

function formatValue(metric, value) {
  const config = METRICS[metric];
  if (!config) return String(value ?? 0);
  const numeric = Number(value ?? 0);
  if (config.isPercent) return `${(numeric * 100).toFixed(config.decimals)}%`;
  return `${numeric.toFixed(config.decimals)} ${config.unit}`;
}

function formatDeltaPercent(liveValue, baselineValue) {
  const baseline = Number(baselineValue);
  const live = Number(liveValue);
  if (!Number.isFinite(baseline) || !Number.isFinite(live)) return "n/a";
  if (Math.abs(baseline) < 1e-9) return "n/a";
  const deltaPercent = ((live - baseline) / Math.abs(baseline)) * 100;
  const sign = deltaPercent > 0 ? "+" : "";
  return `${sign}${deltaPercent.toFixed(1)}%`;
}

function updateTopCards() {
  const latest = state.latestLive;
  if (!latest) {
    simTimeEl.textContent = "-";
    vehiclesEl.textContent = "-";
    avgSpeedEl.textContent = "-";
    stopRatioEl.textContent = "-";
    attacksEl.textContent = "0";
    return;
  }

  const simulationTime = Number(latest.simulation_time);
  const vehicleCount = Number(latest.vehicle_count);
  const avgSpeed = Number(latest.avg_speed);
  const stoppedRatio = Number(latest.stopped_ratio);
  const attackCount = Number(latest.active_attack_count);

  simTimeEl.textContent = Number.isFinite(simulationTime) ? `${simulationTime.toFixed(1)}s` : "-";
  vehiclesEl.textContent = Number.isFinite(vehicleCount) ? `${vehicleCount}` : "-";
  avgSpeedEl.textContent = Number.isFinite(avgSpeed) ? `${avgSpeed.toFixed(1)} m/s` : "-";
  stopRatioEl.textContent = Number.isFinite(stoppedRatio) ? `${(stoppedRatio * 100).toFixed(1)}%` : "-";
  attacksEl.textContent = Number.isFinite(attackCount) ? `${attackCount}` : "0";
}

function renderComparisonTable() {
  if (!compareTableBody) return;

  const latestLive = latestPoint(state.liveHistory);
  const currentTime = latestLive ? getPointTime(latestLive) : 0;

  compareTableBody.innerHTML = "";
  Object.entries(METRICS).forEach(([metric, config]) => {
    const liveValue = getAverageSnapshot(state.liveHistory, metric, currentTime);
    const baselineValue = getAverageSnapshot(state.baselineHistory, metric, currentTime);
    const delta = liveValue - baselineValue;
    const trend = delta < 0 ? "better" : delta > 0 ? "worse" : "same";
    const trendClass = trend === "better" ? "trend-better" : trend === "worse" ? "trend-worse" : "trend-same";
    const arrow = delta < 0 ? "▼" : delta > 0 ? "▲" : "→";
    const arrowColor = delta < 0 ? "ok" : delta > 0 ? "bad" : "";
    const deltaPercentText = formatDeltaPercent(liveValue, baselineValue);
    const comparisonText = deltaPercentText === "n/a" ? `${arrow} n/a` : `${arrow} ${deltaPercentText}`;
    const isActive = metric === state.selectedMetric;

    const row = document.createElement("tr");
    if (isActive) {
      row.classList.add("active-row");
    }
    row.innerHTML = `
      <td style="font-weight: 500;">${config.label}</td>
      <td>${formatValue(metric, liveValue)}</td>
      <td>${formatValue(metric, baselineValue)}</td>
      <td class="${arrowColor}">${comparisonText}</td>
      <td><span class="${trendClass}">${trend}</span></td>
    `;
    row.addEventListener("click", () => {
      state.selectedMetric = metric;
      localStorage.setItem(SELECTED_METRIC_KEY, metric);
      applyAllViews();
    });
    compareTableBody.appendChild(row);
  });
}

function initChart() {
  const canvas = document.getElementById("speedChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  metricChart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Baseline",
          data: [],
          borderColor: "#06b6d4",
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.1,
          fill: false,
        },
        {
          label: "Attack",
          data: [],
          borderColor: "#ec4899",
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.1,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          type: "linear",
          title: {
            display: true,
            text: "Time (s)",
            color: "#475569",
            font: { family: "Outfit", size: 13, weight: "bold" }
          },
          grid: {
            color: "rgba(0, 0, 0, 0.08)"
          },
          ticks: {
            color: "#475569",
            font: { family: "Outfit" }
          },
          min: 0,
          max: 300,
        },
        y: {
          title: {
            display: true,
            text: "Metric Value",
            color: "#475569",
            font: { family: "Outfit", size: 13, weight: "bold" }
          },
          grid: {
            color: "rgba(0, 0, 0, 0.08)"
          },
          ticks: {
            color: "#475569",
            font: { family: "Outfit" }
          }
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: {
            color: "#0f172a",
            font: { family: "Outfit", size: 12, weight: "500" }
          }
        },
        zoom: {
          pan: {
            enabled: true,
            mode: 'x',
            modifierKey: 'shift',
          },
          zoom: {
            drag: {
              enabled: true,
              backgroundColor: 'rgba(99, 102, 241, 0.15)',
              borderColor: 'rgba(99, 102, 241, 0.4)',
              borderWidth: 1,
            },
            wheel: {
              enabled: true,
              speed: 0.05,
            },
            pinch: {
              enabled: true
            },
            mode: 'x',
          }
        }
      },
    },
  });

  // Provide visual feedback for cursor shift
  window.addEventListener("keydown", (e) => {
    if (e.key === "Shift") {
      canvas.style.cursor = "grab";
    }
  });

  window.addEventListener("keyup", (e) => {
    if (e.key === "Shift") {
      canvas.style.cursor = "default";
    }
  });

  // Trackpad horizontal swipe to pan horizontally
  canvas.addEventListener("wheel", (e) => {
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      e.preventDefault();
      if (metricChart) {
        const xMin = metricChart.options.scales.x.min ?? 0;
        const xMax = metricChart.options.scales.x.max ?? 300;
        const range = xMax - xMin;
        const shift = (e.deltaX / canvas.clientWidth) * range * 0.5;
        metricChart.options.scales.x.min = xMin + shift;
        metricChart.options.scales.x.max = xMax + shift;
        metricChart.update("none");
      }
    }
  }, { passive: false });
}

function renderChart() {
  if (!metricChart) return;

  const metric = state.selectedMetric;
  const config = METRICS[metric];
  const chartTitleEl = document.getElementById("chartTitle");
  if (chartTitleEl && config) {
    chartTitleEl.textContent = `Attack vs Baseline - ${config.label}`;
  }

  const baselinePoints = seriesToPoints(state.baselineHistory, metric);
  const livePoints = seriesToPoints(state.liveHistory, metric);

  metricChart.data.datasets[0].data = baselinePoints;
  metricChart.data.datasets[1].data = livePoints;

  if (metricChart.options.plugins.zoom) {
    // If the chart isn't zoomed, update scale bounds to match simulation length
    const maxTime = Math.max(
      baselinePoints.length ? baselinePoints[baselinePoints.length - 1].x : 300,
      livePoints.length ? livePoints[livePoints.length - 1].x : 300
    );
    if (metricChart.scales.x.min === 0 && metricChart.scales.x.max === 300 && maxTime > 305) {
      metricChart.options.scales.x.max = Math.ceil(maxTime / 100) * 100;
    }
  }

  metricChart.update("none");
}

function latestPoint(history) {
  return history && history.length ? history[history.length - 1] : null;
}

function applyAllViews() {
  updateTopCards();
  renderChart();
  renderComparisonTable();
}

function getMapFolder(baselineName) {
  const norm = String(baselineName || "paris").toLowerCase();
  if (norm.includes("paris")) return "paris";
  if (norm.includes("berlin")) return "berlin";
  if (norm.includes("lux")) return "luxembourg";
  if (norm.includes("basic")) return "basic_simulation";
  return "paris";
}

function getBaselineSeed(baselineName) {
  const norm = String(baselineName || "").toLowerCase();
  const match = norm.match(/seed_?(\d+)/i);
  if (match) return parseInt(match[1], 10);
  if (norm.endsWith("1")) return 1;
  return 42;
}

async function loadBaselineForMap(baselineName, forceReload = false) {
  const normalized = String(baselineName || "paris").toLowerCase().replace("_simulation", "");
  const seedVal = getBaselineSeed(normalized);
  const mapFolder = getMapFolder(normalized);

  try {
    const loadRes = await fetch("/api/baseline/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ map_name: normalized, force_reload: !!forceReload, seed: seedVal }),
    });
    const loadData = await loadRes.json();
    if (!loadRes.ok || loadData.error) throw new Error(loadData.error || `HTTP ${loadRes.status}`);

    const currentRes = await fetch(`/api/baseline/current?map_name=${encodeURIComponent(normalized)}&seed=${seedVal}`);
    const currentData = await currentRes.json();
    if (!currentRes.ok || currentData.status !== "loaded" || !Array.isArray(currentData.baseline)) {
      throw new Error(currentData.message || currentData.error || "Baseline unavailable");
    }

    state.baselineHistory = currentData.baseline;
    state.currentBaseline = baselineName;
    state.currentMap = mapFolder;
    localStorage.setItem(LAST_MAP_KEY, mapFolder);

    if (mapSelect) {
      mapSelect.value = mapFolder;
    }
    const baselineSelect = document.getElementById("baselineSelect");
    if (baselineSelect) {
      baselineSelect.value = baselineName;
    }

    applyAllViews();
    return true;
  } catch (err) {
    console.error("[BASELINE] Error:", err.message);
    return false;
  }
}

function populateBaselinesDropdown() {
  const baselineSelect = document.getElementById("baselineSelect");
  if (!baselineSelect) return;

  const selectedMap = mapSelect ? mapSelect.value : "paris";
  const baseMapName = getBaseMapName(selectedMap);

  const filtered = state.allBaselines.filter((b) => {
    const normB = String(b.map_name).toLowerCase();
    if (baseMapName === "paris") return normB.includes("paris");
    if (baseMapName === "berlin") return normB.includes("berlin");
    if (baseMapName === "luxembourg" || baseMapName === "lux") return normB.includes("lux");
    if (baseMapName === "basic") return normB.includes("basic");
    return normB.includes(baseMapName);
  });

  filtered.sort((a, b) => {
    const nameA = String(a.map_name).toLowerCase();
    const nameB = String(b.map_name).toLowerCase();
    const isSeedA = nameA.includes("seed");
    const isSeedB = nameB.includes("seed");
    if (!isSeedA && isSeedB) return -1;
    if (isSeedA && !isSeedB) return 1;
    if (isSeedA && isSeedB) {
      const numA = parseInt(nameA.match(/\d+/)?.[0] || 0, 10);
      const numB = parseInt(nameB.match(/\d+/)?.[0] || 0, 10);
      return numA - numB;
    }
    return nameA.localeCompare(nameB);
  });

  baselineSelect.innerHTML = filtered
    .map((b) => {
      let displayName = b.map_name.toUpperCase();
      displayName = displayName.replace("BASELINE_", "").replace("_SEED_", " Seed ");
      if (displayName === "PARIS") displayName = "PARIS DEFAULT (Seed 42)";
      if (displayName === "BERLIN") displayName = "BERLIN DEFAULT (Seed 42)";
      if (displayName === "LUXEMBOURG") displayName = "LUXEMBOURG DEFAULT (Seed 42)";
      if (displayName === "BASIC") displayName = "BASIC DEFAULT (Seed 42)";
      return `<option value="${b.map_name}">${displayName}</option>`;
    })
    .join("");

  if (filtered.length === 0) {
    baselineSelect.innerHTML = `<option value="${baseMapName}">${baseMapName.toUpperCase()} DEFAULT (Seed 42)</option>`;
  }
}

async function preloadBaselineAtStartup() {
  if (state.liveMode) {
    state.baselineHistory = [];
    applyAllViews();
    return;
  }
  try {
    const mapsRes = await fetch("/api/baseline/maps");
    const mapsData = await mapsRes.json();
    state.allBaselines = Array.isArray(mapsData && mapsData.maps) ? mapsData.maps : [];

    if (mapSelect) {
      mapSelect.value = state.currentMap || "paris";
    }

    populateBaselinesDropdown();

    const baselineSelect = document.getElementById("baselineSelect");
    const startupBaseline = baselineSelect && baselineSelect.value ? baselineSelect.value : (state.currentMap || "paris");
    await loadBaselineForMap(startupBaseline);
  } catch (err) {
    console.error("[BASELINE] Startup preload failed:", err.message);
  }
}

function normalizeSnapshotPayload(payload) {
  if (!payload) return null;
  if (payload.latest && typeof payload.latest === "object") return payload.latest;
  if (payload.status === "ok" && payload.latest && typeof payload.latest === "object") return payload.latest;
  if (typeof payload === "object" && (payload.metrics || payload.simulation_time !== undefined || payload.step !== undefined)) {
    return payload;
  }
  return null;
}

function bindSocket() {
  socket.on("metrics", async (payload) => {
    const snapshot = normalizeSnapshotPayload(payload);
    if (!snapshot) return;

    const last = state.liveHistory[state.liveHistory.length - 1];
    if (!last || Number(last.step) !== Number(snapshot.step)) {
      state.liveHistory.push(snapshot);
      if (state.liveHistory.length > 5000) state.liveHistory.shift();
    } else {
      state.liveHistory[state.liveHistory.length - 1] = snapshot;
    }

    state.latestLive = snapshot;
    applyAllViews();

    const currentTime = getPointTime(snapshot);

    if (snapshot.map_name) {
      const inferredMap = String(snapshot.map_name).toLowerCase().replace("_simulation", "");
      const baselineMissing = state.baselineHistory.length === 0;
      const currentBase = getBaseMapName(state.currentMap);
      const inferredBase = getBaseMapName(inferredMap);
      const mapChanged = !state.currentMap || currentBase !== inferredBase;
      if (baselineMissing || mapChanged) {
        await loadBaselineForMap(inferredMap);
      }
    }
  });

  socket.on("metrics_error", (err) => {
    const message = err && err.error ? err.error : "unknown";
    attacksEl.textContent = `error: ${message}`;
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  initChart();
  await detectRuntimeMode();


  document.getElementById("zoomInBtn")?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (metricChart) metricChart.zoom(1.2);
  });
  document.getElementById("zoomOutBtn")?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (metricChart) metricChart.zoom(0.8);
  });
  document.getElementById("resetViewBtn")?.addEventListener("click", () => {
    if (metricChart) {
      metricChart.options.scales.x.min = 0;
      metricChart.options.scales.x.max = 300;
      metricChart.update();
    }
  });

  document.getElementById("saveBaselineBtn")?.addEventListener("click", async () => {
    if (!state.currentMap) {
      alert("No map loaded.");
      return;
    }
    if (state.liveHistory.length === 0) {
      alert("No simulation data to save.");
      return;
    }

    try {
      const res = await fetch("/api/baseline/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ map_name: state.currentMap })
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || "Save failed.");
      }

      await loadBaselineForMap(state.currentMap, true);
      alert(`Current simulation successfully saved as reference baseline for ${state.currentMap.toUpperCase()}!`);
    } catch (err) {
      console.error(err);
      alert(`Failed to save baseline: ${err.message}`);
    }
  });

  document.getElementById("saveRunBtn")?.addEventListener("click", async () => {
    if (state.liveHistory.length === 0) {
      alert("No simulation data to export.");
      return;
    }
    
    // Determine active attack type if any
    let activeAttackType = "";
    const lastPoint = state.liveHistory[state.liveHistory.length - 1];
    if (lastPoint && lastPoint.active_attack_types && lastPoint.active_attack_types.length > 0) {
      activeAttackType = lastPoint.active_attack_types.join("_");
    }

    let savedOnServer = false;
    let serverPath = "";
    try {
      const res = await fetch("/api/simulation/save_run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          history: state.liveHistory,
          map_name: state.currentMap || "custom",
          attack_type: activeAttackType
        })
      });
      const data = await res.json();
      if (res.ok && !data.error) {
        savedOnServer = true;
        serverPath = data.filepath;
      }
    } catch (err) {
      console.warn("Server-side save failed, falling back to local download:", err);
    }

    try {
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.liveHistory, null, 2));
      const downloadAnchor = document.createElement('a');
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const attackLabel = activeAttackType ? activeAttackType.replace(/[^a-zA-Z0-9]/g, "_") : "normal";
      
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `run_${state.currentMap || "custom"}_${attackLabel}_${timestamp}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();

      if (savedOnServer) {
        alert(`Simulation run data exported successfully!\n\n1. Saved on Server: ${serverPath}\n2. Downloaded locally to your computer.`);
      } else {
        alert(`Simulation run data downloaded locally to your computer!\n(Note: Server-side write failed/skipped).`);
      }
    } catch (err) {
      console.error(err);
      alert(`Failed to export run data: ${err.message}`);
    }
  });

  applyAllViews();

  await preloadBaselineAtStartup();

  mapSelect?.addEventListener("change", async (e) => {
    const selectedMap = e.target.value;
    if (selectedMap) {
      state.currentMap = selectedMap;
      localStorage.setItem(LAST_MAP_KEY, selectedMap);
      populateBaselinesDropdown();

      const baselineSelect = document.getElementById("baselineSelect");
      if (baselineSelect && baselineSelect.value) {
        await loadBaselineForMap(baselineSelect.value);
      }
    }
  });

  document.getElementById("baselineSelect")?.addEventListener("change", async (e) => {
    const selectedBaseline = e.target.value;
    if (selectedBaseline) {
      await loadBaselineForMap(selectedBaseline);
    }
  });

  const launchSimBtn = document.getElementById("launchSimBtn");
  launchSimBtn?.addEventListener("click", async () => {
    if (state.liveMode) {
      alert("TwinSentinel is attached to the active VLA-AV CARLA/SUMO bridge. Launch CARLA+SUMO from the VLA-AV dashboard first.");
      return;
    }
    const selectedMap = mapSelect?.value;
    if (!selectedMap) {
      alert("Please select a map first!");
      return;
    }
    const selectedBaseline = document.getElementById("baselineSelect")?.value || selectedMap;

    const originalText = launchSimBtn.innerHTML;
    launchSimBtn.disabled = true;
    launchSimBtn.innerHTML = `
      <svg style="animation: spin 1s linear infinite;" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line></svg>
      Launching...
    `;

    try {
      const headlessVal = !!document.getElementById("headlessCheckbox")?.checked;
      const seedVal = getBaselineSeed(selectedBaseline);

      const res = await fetch("/api/simulation/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ map_name: selectedMap, headless: headlessVal, seed: seedVal })
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || "Simulation launch failed.");
      }

      // Clear current histories to start fresh with new live data
      state.liveHistory = [];
      state.latestLive = null;

      // Load corresponding baseline
      await loadBaselineForMap(selectedBaseline, true);

      alert(`Simulation successfully launched for map: ${selectedMap.toUpperCase()} (Seed: ${seedVal})`);
    } catch (err) {
      console.error(err);
      alert(`Failed to launch simulation: ${err.message}`);
    } finally {
      launchSimBtn.disabled = false;
      launchSimBtn.innerHTML = originalText;
    }
  });

  document.querySelectorAll(".attack-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const attackType = btn.getAttribute("data-attack");
      if (!attackType) return;

      const originalText = btn.innerHTML;
      const attackLabel = btn.textContent.trim();
      btn.disabled = true;
      btn.classList.add("loading");
      btn.innerHTML = `
        <svg style="animation: spin 1s linear infinite;" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line></svg>
        Injecting...
      `;

      try {
        const res = await fetch("/api/simulation/attack", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            type: attackType,
            map_name: state.currentMap || "paris"
          })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          throw new Error(data.error || "Attack injection failed.");
        }

        alert(`Successfully injected: ${attackLabel}`);
      } catch (err) {
        console.error(err);
        alert(`Failed to inject attack: ${err.message}`);
      } finally {
        btn.disabled = false;
        btn.classList.remove("loading");
        btn.innerHTML = originalText;
      }
    });
  });

  bindSocket();



  // Local JSON files comparison logic
  const importBaselineInput = document.getElementById("importBaselineInput");
  const importRunInput = document.getElementById("importRunInput");
  const compareJsonBtn = document.getElementById("compareJsonBtn");
  const clearCompareBtn = document.getElementById("clearCompareBtn");

  compareJsonBtn?.addEventListener("click", () => {
    const baselineFile = importBaselineInput?.files[0];
    const runFile = importRunInput?.files[0];

    if (!baselineFile || !runFile) {
      alert("Please select both a baseline JSON file and a run JSON file to compare!");
      return;
    }

    let loadedBaseline = null;
    let loadedRun = null;

    const readBaseline = new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const parsed = JSON.parse(e.target.result);
          loadedBaseline = parsed;
          resolve();
        } catch (err) {
          reject(new Error("Error parsing baseline JSON: " + err.message));
        }
      };
      reader.onerror = () => reject(new Error("Error reading baseline file"));
      reader.readAsText(baselineFile);
    });

    const readRun = new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const parsed = JSON.parse(e.target.result);
          loadedRun = parsed;
          resolve();
        } catch (err) {
          reject(new Error("Error parsing run JSON: " + err.message));
        }
      };
      reader.onerror = () => reject(new Error("Error reading run file"));
      reader.readAsText(runFile);
    });

    Promise.all([readBaseline, readRun])
      .then(() => {
        if (!Array.isArray(loadedBaseline) || !Array.isArray(loadedRun)) {
          throw new Error("Both JSON files must contain arrays of simulation steps.");
        }

        // Apply loaded histories to local state
        state.baselineHistory = loadedBaseline;
        state.liveHistory = loadedRun;
        state.latestLive = loadedRun[loadedRun.length - 1];

        const currentTime = getPointTime(state.latestLive);

        applyAllViews();
        alert("JSON files loaded and compared successfully!");
      })
      .catch((err) => {
        alert(err.message);
      });
  });

  clearCompareBtn?.addEventListener("click", () => {
    state.liveHistory = [];
    state.latestLive = null;
    if (importBaselineInput) importBaselineInput.value = "";
    if (importRunInput) importRunInput.value = "";
    preloadBaselineAtStartup();
    applyAllViews();
    alert("Comparison cleared.");
  });
});
