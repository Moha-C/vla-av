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

const socketStatus = document.getElementById("attacks");
const simTimeEl = document.getElementById("simTime");
const vehiclesEl = document.getElementById("vehicles");
const avgSpeedEl = document.getElementById("avgSpeed");
const stopRatioEl = document.getElementById("stopRatio");
const jamsEl = document.getElementById("jams");
const metricSelect = document.getElementById("metricSelect");
const metricGrid = document.getElementById("metricGrid");
const baselineStatusEl = document.getElementById("baselineStatus");
const mapSelect = document.getElementById("mapSelect");
const loadBaselineBtn = document.getElementById("loadBaseline");
const captureAttackedBtn = document.getElementById("captureAttacked");
const compareBtn = document.getElementById("compareBtn");

const HISTORY_ATTACKED_KEY = "vanetAttackedHistory";
const SELECTED_METRIC_KEY = "vanetSelectedMetric";
const SELECTED_MAP_KEY = "vanetSelectedMap";

const state = {
  liveHistory: [],
  baselineHistory: [],
  attackedHistory: loadStoredHistory(HISTORY_ATTACKED_KEY),
  selectedMetric: localStorage.getItem(SELECTED_METRIC_KEY) || "fuel_consumption",
  selectedMap: localStorage.getItem(SELECTED_MAP_KEY) || "paris",
  latestLive: null,
  baselineLoaded: false,
};

function loadStoredHistory(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveStoredHistory(key, value) {
  localStorage.setItem(key, JSON.stringify(value || []));
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
  const time = Number(point?.simulation_time ?? point?.step ?? 0);
  return Number.isFinite(time) ? time : 0;
}

function ensureZeroOrigin(points) {
  if (!points.length) return points;
  if (points[0].x > 0) {
    return [{ x: 0, y: points[0].y }, ...points];
  }
  return points;
}

function seriesToPoints(history, metric) {
  return ensureZeroOrigin(
    (history || [])
      .map((point) => ({ x: getPointTime(point), y: getMetricValue(point, metric) }))
      .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  );
}

function formatValue(metric, value) {
  const config = METRICS[metric];
  if (!config) return String(value ?? 0);
  const numeric = Number(value ?? 0);
  if (config.isPercent) {
    return `${(numeric * 100).toFixed(config.decimals)}%`;
  }
  return `${numeric.toFixed(config.decimals)} ${config.unit}`;
}

function latestPoint(history) {
  return history && history.length ? history[history.length - 1] : null;
}

function renderMetricGrid() {
  const latestLive = latestPoint(state.liveHistory);
  const latestBaseline = latestPoint(state.baselineHistory);

  metricGrid.innerHTML = Object.entries(METRICS)
    .map(([metric, config]) => {
      const liveValue = latestLive ? getMetricValue(latestLive, metric) : 0;
      const baselineValue = latestBaseline ? getMetricValue(latestBaseline, metric) : 0;
      const activeClass = metric === state.selectedMetric ? "style=\"border-color: #d97706; box-shadow: 0 0 0 2px rgba(217,119,6,0.15);\"" : "";
      return `
        <div class="card" ${activeClass}>
          <div class="label">${config.label}</div>
          <div class="value">${formatValue(metric, liveValue)}</div>
          <div class="label">Baseline: ${formatValue(metric, baselineValue)}</div>
        </div>
      `;
    })
    .join("");
}

const chartCtx = document.getElementById("speedChart").getContext("2d");
const metricChart = new Chart(chartCtx, {
  type: "line",
  data: {
    datasets: [
      {
        label: "Live",
        data: [],
        borderColor: "#d97706",
        backgroundColor: "rgba(217, 119, 6, 0.15)",
        fill: false,
        tension: 0.25,
        pointRadius: 0,
      },
      {
        label: "Baseline",
        data: [],
        borderColor: "#0f766e",
        backgroundColor: "rgba(15, 118, 110, 0.12)",
        fill: false,
        tension: 0.25,
        pointRadius: 0,
      },
    ],
  },
  options: {
    responsive: true,
    animation: false,
    parsing: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Simulation time (s)" },
        beginAtZero: true,
      },
      y: {
        beginAtZero: true,
        title: { display: true, text: METRICS[state.selectedMetric].label },
      },
    },
    plugins: {
      legend: { position: "top" },
    },
  },
});

function renderChart() {
  const livePoints = seriesToPoints(state.liveHistory, state.selectedMetric);
  const baselinePoints = seriesToPoints(state.baselineHistory, state.selectedMetric);
  metricChart.data.datasets[0].label = `Live - ${METRICS[state.selectedMetric].label}`;
  metricChart.data.datasets[1].label = `Baseline - ${METRICS[state.selectedMetric].label}`;
  metricChart.data.datasets[0].data = livePoints;
  metricChart.data.datasets[1].data = baselinePoints;
  metricChart.options.scales.y.title.text = METRICS[state.selectedMetric].label;
  metricChart.update();
}

function renderComparisonTable() {
  const tbody = document.querySelector("#compareTable tbody");
  tbody.innerHTML = "";
  const latestLive = latestPoint(state.liveHistory);
  const latestBaseline = latestPoint(state.baselineHistory);

  Object.entries(METRICS).forEach(([metric, config]) => {
    const liveValue = latestLive ? getMetricValue(latestLive, metric) : 0;
    const baselineValue = latestBaseline ? getMetricValue(latestBaseline, metric) : 0;
    const delta = liveValue - baselineValue;
    const trend = delta < 0 ? "better" : delta > 0 ? "worse" : "same";
    const trendClass = trend === "better" ? "ok" : trend === "worse" ? "bad" : "";
    const deltaText = formatValue(metric, delta);

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${config.label}</td>
      <td>${formatValue(metric, liveValue)}</td>
      <td>${formatValue(metric, baselineValue)}</td>
      <td>${deltaText}</td>
      <td class="${trendClass}">${trend}</td>
    `;
    tbody.appendChild(row);
  });
}

function updateStatusLine() {
  const liveCount = state.liveHistory.length;
  const baselineCount = state.baselineHistory.length;
  const baselineStatus = state.baselineLoaded 
    ? `loaded from server (${baselineCount} points, map: ${state.selectedMap})`
    : baselineCount > 0
    ? `loaded from browser (${baselineCount} points)`
    : "not loaded";
  const attackedCount = state.attackedHistory.length;
  baselineStatusEl.textContent = `Baseline: ${baselineStatus} | Scenario: ${attackedCount ? `saved (${attackedCount} points)` : "not saved"} | Live points: ${liveCount}`;
}

function applyAllViews() {
  renderMetricGrid();
  renderChart();
  renderComparisonTable();
  updateStatusLine();
}

function syncLatestSummary() {
  const latest = state.latestLive;
  if (!latest) return;

  simTimeEl.textContent = `${Number(getPointTime(latest)).toFixed(1)} s`;
  vehiclesEl.textContent = String(latest.vehicle_count ?? 0);
  avgSpeedEl.textContent = `${Number(latest.avg_speed ?? 0).toFixed(2)} m/s`;
  stopRatioEl.textContent = `${((latest.stopped_ratio ?? 0) * 100).toFixed(1)}%`;
  jamsEl.textContent = String(latest.jammed_lanes ?? 0);
  socketStatus.textContent = `${latest.active_attack_count ?? 0} (${(latest.active_attack_types || []).join(", ") || "none"})`;
}

async function loadLiveHistory() {
  const res = await fetch("/api/history?limit=0");
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  state.liveHistory = data.history || [];
  state.latestLive = latestPoint(state.liveHistory);
  syncLatestSummary();
  applyAllViews();
}

async function loadBaselineFromServer(mapName) {
  try {
    loadBaselineBtn.disabled = true;
    loadBaselineBtn.textContent = "Loading...";
    
    const response = await fetch("/api/baseline/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ map_name: mapName }),
    });
    
    const data = await response.json();
    
    if (data.error) {
      baselineStatusEl.textContent = `Error: ${data.error}`;
      loadBaselineBtn.textContent = "Load Baseline";
      loadBaselineBtn.disabled = false;
      return;
    }
    
    if (data.status === "loaded") {
      const baselineResponse = await fetch("/api/baseline/current");
      const baselineData = await baselineResponse.json();
      
      if (baselineData.status === "loaded") {
        state.baselineHistory = baselineData.baseline || [];
        state.baselineLoaded = true;
        updateStatusLine();
        applyAllViews();
        loadBaselineBtn.textContent = "✓ Loaded";
        setTimeout(() => {
          loadBaselineBtn.textContent = "Load Baseline";
          loadBaselineBtn.disabled = false;
        }, 2000);
      }
    }
  } catch (error) {
    baselineStatusEl.textContent = `Error loading baseline: ${error.message}`;
    loadBaselineBtn.textContent = "Load Baseline";
    loadBaselineBtn.disabled = false;
  }
}

function saveAttackedFromLive() {
  state.attackedHistory = [...state.liveHistory];
  saveStoredHistory(HISTORY_ATTACKED_KEY, state.attackedHistory);
  updateStatusLine();
}

// UI Initialization
metricSelect.innerHTML = Object.entries(METRICS)
  .map(([metric, config]) => `<option value="${metric}">${config.label}</option>`)
  .join("");
metricSelect.value = state.selectedMetric;

mapSelect.value = state.selectedMap;

// Event listeners
metricSelect.addEventListener("change", () => {
  state.selectedMetric = metricSelect.value;
  localStorage.setItem(SELECTED_METRIC_KEY, state.selectedMetric);
  renderChart();
});

mapSelect.addEventListener("change", () => {
  state.selectedMap = mapSelect.value;
  localStorage.setItem(SELECTED_MAP_KEY, state.selectedMap);
});

loadBaselineBtn.addEventListener("click", async () => {
  await loadBaselineFromServer(state.selectedMap);
});

captureAttackedBtn.addEventListener("click", () => {
  saveAttackedFromLive();
  captureAttackedBtn.textContent = "✓ Scenario saved";
  setTimeout(() => {
    captureAttackedBtn.textContent = "Capture Scenario";
  }, 2000);
});

compareBtn.addEventListener("click", async () => {
  await loadLiveHistory();
  applyAllViews();
});

// Socket.IO listeners
socket.on("metrics", (payload) => {
  if (!payload || payload.status !== "ok" || !payload.latest) return;
  const latest = payload.latest;
  const last = state.liveHistory[state.liveHistory.length - 1];
  if (!last || Number(last.step) !== Number(latest.step)) {
    state.liveHistory.push(latest);
  } else {
    state.liveHistory[state.liveHistory.length - 1] = latest;
  }
  state.latestLive = latest;
  syncLatestSummary();
  applyAllViews();
});

socket.on("metrics_error", (err) => {
  socketStatus.textContent = `error: ${err.error}`;
});

// Initialization
(async () => {
  try {
    await loadLiveHistory();
    
    // Try to load baseline from server
    await loadBaselineFromServer(state.selectedMap);
    
    if (state.attackedHistory.length) {
      captureAttackedBtn.textContent = `Scenario saved (${state.attackedHistory.length} points)`;
    }
    applyAllViews();
  } catch (error) {
    socketStatus.textContent = `error: ${error.message}`;
  }
})();
