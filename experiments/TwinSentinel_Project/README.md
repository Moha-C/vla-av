# TwinSentinel: AI-Driven Intrusion Detection & Adversarial Simulation in VANETs

TwinSentinel is a comprehensive pairs-simulation environment designed to evaluate and secure Vehicular Ad-Hoc Networks (VANETs). The platform enables the orchestration of realistic vehicular networks, the execution of multi-layer cyber attacks (Red Agent), and the real-time detection and mitigation of threats (Blue Agent) using Machine Learning and cryptographic verification.

---

## 🚀 Key Features

*   **Decoupled Map & Seed Workflows**: Run simulations on distinct city topologies (**Paris**, **Berlin**, and **Luxembourg**) while testing against specific baseline seeds ($N=1\text{ to }20$) to maintain experimental consistency.
*   **Adversarial Campaign Simulation (Red vs Blue)**:
    *   **Red Agent (Attacks)**: Multi-layer threat models including Sybil attacks, DDoS, GPS spoofing, and sensor noise injection.
    *   **Blue Agent (Defense)**: Real-time anomaly detection, physical speed deviations tracking, and adaptive network/traffic-light overrides.
*   **TwinSentinel Live Monitor**: A Web-based Node.js dashboard with live telemetry (speed, fuel consumption, stopped vehicle ratio, jam counts) mapped dynamically against selected reference baselines.
*   **Headless & GUI Execution**: Run simulations in headless mode for high-throughput batch experiments, or in GUI mode to visually inspect traffic behavior in SUMO.
*   **Automated Campaign Reporting**: Automated statistical testing (Holm-Bonferroni correction, Cliff's delta effect sizes) and report generation directly to formatted `.docx` and `.pdf` documents.

---

## 📂 Workspace Organization

The workspace is structured to keep data, source code, and post-processing separate and clean:

```bash
/home/mehdi/VANET_Project/TwinSentinel_Project/
├── MCP_server.py             # Model Context Protocol (MCP) Python orchestration server
├── baselines/                # Stored benign baseline JSON runs (Seeds 1–20) for Paris, Berlin, & Luxembourg
├── runs/                     # Directory where active and historical simulation logs are exported
├── maps/                     # SUMO network topology, trip files, and configurations for each map
├── node_dashboard/           # Node.js backend (server.js) and frontend dashboard (public/app.js)
├── post_treatment/           # Post-processing files, split into:
│   ├── figure/               # ROC curves, detectability frontiers, and figure generators
│   └── table/                # Campaign metrics JSONs, Table V calculations, and statistical scripts
├── scripts/                  # Base simulation execution and helper scripts
├── scratch/                  # Temporary development scratchpad and tests
└── .venv/                    # Python virtual environment containing simulation dependencies
```

---

## ⚙️ Getting Started

### 1. Prerequisites
Ensure you have **SUMO (Simulation of Urban MObility)** version 1.22+ installed on your system.

### 2. Environment Activation & Dependencies
Activate the virtual environment and install standard requirements:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launching the TwinSentinel Suite

First, start the Python MCP Server which manages the SUMO subprocesses and handles the TraCI connection:
```bash
python MCP_server.py
```

In a separate terminal, navigate to the dashboard directory and launch the web server:
```bash
cd node_dashboard
npm install
npm start
```

Open your browser and navigate to **`http://localhost:3100`** to view the live dashboard.

---

## 📊 Live Dashboard Workflow

1.  **Select Topology (Map)**: Choose between **Paris**, **Berlin**, **Luxembourg**, or **Basic Simulation**.
2.  **Select Baseline (Reference Seed)**: Choose the specific seed reference you want to compare the live run against (e.g. `BERLIN Seed 5`).
3.  **Simulation Mode**: Toggle **Headless (Fast)** mode off to launch the SUMO GUI interface visually, or leave it checked for background runs.
4.  **Launch**: Click the **Launch** button to start the simulator. The dashboard will automatically stream real-time metrics.
5.  **Inject Attacks**: Trigger adversarial attacks (Sybil, DDoS, GPS noise) dynamically using the Attack Injection controls on the dashboard.
6.  **Evaluate & Save**: Watch the live metrics diverge from the benign reference line on the chart. Save the run data directly using **Export Run Data**.

---

## 📈 Post-Processing & Reporting

Offline evaluation and campaign analysis are run via standalone post-processing scripts:

### 1. Re-run Campaign Anomaly Analysis
Processes all simulation runs, calculates detection accuracy metrics, and exports statistical summaries:
```bash
python post_treatment/table/analyze_berlin_campaign.py
python post_treatment/table/analyze_lux_campaign.py
python post_treatment/table/analyze_campaign.py          # Paris
```

### 2. Regenerate Curve Plots & Results Table
Generates the ROC-PR curves and `detailed_table_results.json` files for paper publication:
```bash
python post_treatment/figure/generate_berlin_figure_and_table.py
python post_treatment/figure/generate_lux_figure_and_table.py
python post_treatment/figure/generate_figure_and_table.py       # Paris
```

---

## 🤝 Git LFS & Collaboration

This repository uses **Git LFS (Large File Storage)** to manage large SUMO network topologies (`*.net.xml`, `*.rou.gz`) and simulation run logs (`runs/*.json`, `baselines/*.json`).

### 1. Prerequisites for Collaborators
To work on this project and fetch all assets correctly, you must have Git LFS installed on your system.

**On Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install git-lfs
```

### 2. Cloning the Repository
When cloning this repository, initialize Git LFS and pull the large objects:
```bash
# Clone the repository
git clone https://github.com/E-Mehdi-Boulharts/TwinSentinel_Project.git
cd TwinSentinel_Project

# Initialize Git LFS and download the tracked assets (baselines, maps, runs)
git lfs install
git lfs pull
```

### 3. Adding New Large Files
If you add new simulation runs, baselines, or maps, Git LFS is configured to track them automatically via `.gitattributes`. Simply add, commit, and push as usual:
```bash
git add runs/new_run.json
git commit -m "Add new simulation run data"
git push origin main
```

---

## 📝 Contributors
*   **Nassim Anemiche**
*   **Mehdi Boulharts**
