#!/usr/bin/env python3
import csv
import glob
import json
import mimetypes
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SIMLINGO_ROOT = ROOT / "external" / "simlingo"
ROUTE_DIR = SIMLINGO_ROOT / "leaderboard" / "data" / "bench2drive_split"
CARLA_ROOT = Path(os.environ.get("CARLA_ROOT", str(Path.home() / "carla_simulator")))
DREAMER_ROOT = ROOT / "experiments" / "dreamer_ppo_carla"
SDBS_DREAMER_ROOT = ROOT / "experiments" / "dreamer_ppo_carla_sdbs_fresh"
LOG_DIR = ROOT / "logs" / "simlingo_dashboard"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_STABLE_TOWNS = "Town12,Town13"
STABLE_TOWNS = {
    town.strip()
    for town in os.environ.get("SIMLINGO_STABLE_TOWNS", DEFAULT_STABLE_TOWNS).split(",")
    if town.strip()
}
SHOW_EXPERIMENTAL_TOWNS = os.environ.get("SIMLINGO_DASHBOARD_SHOW_EXPERIMENTAL", "1").lower() in ("1", "true", "yes")

STATE = {
    "process": None,
    "route": None,
    "route_town": None,
    "scenario": None,
    "mode": None,
    "dreamer_mode": "off",
    "cot_mode": "off",
    "seed": None,
    "port": 2000,
    "started_at": None,
    "launch_log": None,
    "last_error": None,
}
STATE_LOCK = threading.Lock()

ASSET_FILES = {
    "simlingo_teaser.png": SIMLINGO_ROOT / "assets" / "simlingo_teaser.png",
    "simlingo_thumbnail.png": SIMLINGO_ROOT / "assets" / "thumbnail.png",
    "bench2drive_overview.jpg": SIMLINGO_ROOT / "Bench2Drive" / "assets" / "overview.jpg",
    "bench2drive_benchmark.jpg": SIMLINGO_ROOT / "Bench2Drive" / "assets" / "benchmark.jpg",
    "carla_header.png": SIMLINGO_ROOT / "leaderboard" / "docs" / "img" / "carla_header.png",
}


def installed_towns():
    maps_dir = CARLA_ROOT / "CarlaUE4" / "Content" / "Carla" / "Maps"
    towns = set()
    for path in glob.glob(str(maps_dir / "**" / "Town*.umap"), recursive=True):
        town = Path(path).stem
        if "_Tile_" not in town:
            towns.add(town)
    return towns


def route_catalog():
    installed = installed_towns()
    routes = []
    for path in sorted(ROUTE_DIR.glob("bench2drive_*.xml")):
        text = path.read_text(errors="ignore")
        town_match = re.search(r'town="([^"]+)"', text)
        scenario_match = re.search(r'<scenario name="([^"]+)" type="([^"]+)"', text)
        town = town_match.group(1) if town_match else ""
        scenario_name = scenario_match.group(1) if scenario_match else ""
        scenario_type = scenario_match.group(2) if scenario_match else ""
        route_id = path.stem.replace("bench2drive_", "")
        vru = any(token in scenario_type for token in ("Pedestrian", "Bicycle", "DynamicObject", "Crossing"))
        traffic_light = (
            ("Signalized" in scenario_type and "NonSignalized" not in scenario_type)
            or "RedLight" in scenario_type
            or "GreenLight" in scenario_type
        )
        stop = "Stopsign" in scenario_type or "Stop" in scenario_type
        junction = "Junction" in scenario_type or "T_Junction" in scenario_type
        accident = "Accident" in scenario_type
        cut_in = "CutIn" in scenario_type
        actor_flow = "ActorFlow" in scenario_type
        stable = town in STABLE_TOWNS
        compatible = town in installed and (stable or SHOW_EXPERIMENTAL_TOWNS)
        routes.append({
            "id": route_id,
            "file": str(path),
            "town": town,
            "scenario_name": scenario_name,
            "scenario_type": scenario_type,
            "compatible": compatible,
            "installed": town in installed,
            "stable": stable,
            "disabled_reason": "" if compatible else (
                "not installed" if town not in installed else "hidden experimental town"
            ),
            "vru": vru,
            "traffic_light": traffic_light,
            "stop": stop,
            "junction": junction,
            "accident": accident,
            "cut_in": cut_in,
            "actor_flow": actor_flow,
        })
    return routes


def choose_route(payload):
    all_routes = route_catalog()
    routes = [r for r in all_routes if r["compatible"]]
    town = payload.get("town", "any")
    scenario = payload.get("scenario", "any")
    route_id = payload.get("route_id", "random")
    if town not in ("any", "", None):
        routes = [r for r in routes if r["town"] == town]
    if scenario == "vru":
        routes = [r for r in routes if r["vru"]]
    elif scenario == "light":
        routes = [r for r in routes if r["traffic_light"]]
    elif scenario == "stop":
        routes = [r for r in routes if r["stop"]]
    elif scenario == "junction":
        routes = [r for r in routes if r["junction"]]
    elif scenario == "accident":
        routes = [r for r in routes if r["accident"]]
    elif scenario == "cut_in":
        routes = [r for r in routes if r["cut_in"]]
    elif scenario == "actor_flow":
        routes = [r for r in routes if r["actor_flow"]]

    if route_id and route_id != "random":
        exact = [r for r in all_routes if r["id"] == str(route_id) and r["compatible"]]
        if not exact:
            raise ValueError(f"Route {route_id} is not compatible with the installed CARLA maps.")
        return exact[0]

    if not routes:
        raise ValueError("No compatible route for this filter. Choose another town/scenario.")
    return random.choice(routes)


def stop_current(kill_carla=False):
    with STATE_LOCK:
        proc = STATE.get("process")
        STATE["process"] = None
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=8)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
    if kill_carla:
        patterns = (
            "scripts/carla_ego_viewer.py",
            "scripts/carla_sumo_mirror.py",
            "scripts/vlm_cot_sidecar.py",
            "scripts/play_recorded_video.py",
            "scripts/action_dreaming_collect_normal.py",
            "scripts/run_simlingo_with_action_dreaming_collect.sh",
            "scripts/run_simlingo_with_sumo_mirror.sh",
            "scripts/run_simlingo_with_pov.sh",
            "leaderboard_evaluator.py",
            "sumo-gui",
            "sumo -c",
            "CarlaUE4",
        )
        for pattern in patterns:
            subprocess.run(["pkill", "-TERM", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        for pattern in patterns:
            subprocess.run(["pkill", "-9", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_run(payload):
    stop_current(kill_carla=True)
    route = choose_route(payload)
    seed = int(payload.get("seed") or random.randint(1, 999999))
    port = int(payload.get("port") or 2000)
    tm_port = int(payload.get("tm_port") or 8000)
    quality = payload.get("quality", "Low")
    camera = payload.get("camera", "chase")
    visual_weather = payload.get("visual_weather", "day")
    prompt_mode = payload.get("prompt_mode", "native")
    playback_speed = payload.get("playback_speed", "5")
    video_quality = payload.get("video_quality")
    run_mode = payload.get("run_mode", "pov")
    dreamer_mode = payload.get("dreamer_mode", "off")
    cot_mode = payload.get("cot_mode", "off")
    if video_quality == "epic":
        width, height = (1920, 1080)
    elif video_quality == "hd":
        width, height = (1280, 720)
    else:
        width, height = (960, 540)

    env = os.environ.copy()
    env.update({
        "ROUTE_FILE": route["file"],
        "ROUTE_ID": route["id"],
        "SEED": str(seed),
        "PORT": str(port),
        "TM_PORT": str(tm_port),
        "CARLA_QUALITY": quality,
        "SIMLINGO_RENDER_MODE": "offscreen",
        "SIMLINGO_VIEW_MODE": camera,
        "SIMLINGO_VIEW_WIDTH": str(width),
        "SIMLINGO_VIEW_HEIGHT": str(height),
        "SIMLINGO_VIEW_FOV": str(payload.get("fov", "95")),
        "SIMLINGO_VIEW_FPS": str(payload.get("view_fps", "45")),
        "SIMLINGO_VIEW_BRIGHTNESS": str(payload.get("brightness", "8")),
        "SIMLINGO_VIEW_CONTRAST": str(payload.get("contrast", "1.08")),
        "SIMLINGO_VIEW_SATURATION": str(payload.get("saturation", "1.10")),
        "SIMLINGO_VISUAL_WEATHER": visual_weather,
        "SIMLINGO_DRAW_WAYPOINTS": "1",
        "SIMLINGO_TRAFFIC_LIGHT_OVERLAY": str(payload.get("traffic_light_overlay", "1")),
        "SIMLINGO_TRAFFIC_LIGHT_OVERLAY_DISTANCE": str(payload.get("traffic_light_overlay_distance", "160")),
        "SIMLINGO_TRAFFIC_LIGHT_OVERLAY_MAX": str(payload.get("traffic_light_overlay_max", "80")),
        "SIMLINGO_PLAYBACK_AFTER": "1",
        "SIMLINGO_PLAYBACK_SPEED": str(playback_speed),
        "SIMLINGO_OUT_DIR": str(ROOT / "logs" / "simlingo_eval"),
        "SIMLINGO_DREAMER_STATUS_PATH": str(ROOT / "logs" / "simlingo_eval" / "dreamer_guard_status.json"),
        "SIMLINGO_DREAMER_GUARD": "0",
        "SIMLINGO_VLM_COT": str(cot_mode),
        "SIMLINGO_VLM_COT_MODEL": str(payload.get("cot_model") or "Qwen/Qwen2-VL-7B-Instruct"),
        "SIMLINGO_VLM_COT_INTERVAL": str(payload.get("cot_interval") or "2.0"),
        "SIMLINGO_VLM_COT_FRAME_INTERVAL": str(payload.get("cot_frame_interval") or "1.0"),
        "SIMLINGO_VLM_COT_FRAME_WIDTH": str(payload.get("cot_frame_width") or "1280"),
        "SIMLINGO_VLM_COT_STATUS_PATH": str(ROOT / "logs" / "simlingo_eval" / "vlm_cot_status.json"),
        "SIMLINGO_VLM_COT_FRAME_PATH": str(ROOT / "logs" / "simlingo_eval" / "vlm_cot_frame.jpg"),
        "SIMLINGO_VLM_COT_LOG_PATH": str(ROOT / "logs" / "simlingo_eval" / "vlm_cot_reasoning.jsonl"),
        "SUMO_HOME": os.environ.get("SUMO_HOME", "/usr/share/sumo"),
    })
    if cot_mode in ("qwen2_vl", "auto"):
        env["SIMLINGO_VLM_COT_LOCAL_ONLY"] = str(payload.get("cot_local_only") or "1")
    dreamer_presets = {
        "shadow": {
            "SIMLINGO_DREAMER_GUARD": "shadow",
            "SIMLINGO_DREAMER_GUARD_MODE": "shadow",
            "SIMLINGO_DREAMER_RISK_MARGIN": "0.05",
            "SIMLINGO_DREAMER_MAX_PROGRESS_DROP": "0.01",
            "SIMLINGO_DREAMER_MAX_STEER_DELTA": "0.12",
        },
        "guard": {
            "SIMLINGO_DREAMER_GUARD": "1",
            "SIMLINGO_DREAMER_GUARD_MODE": "apply",
            "SIMLINGO_DREAMER_RISK_MARGIN": "0.05",
            "SIMLINGO_DREAMER_MAX_PROGRESS_DROP": "0.01",
            "SIMLINGO_DREAMER_MAX_STEER_DELTA": "0.12",
        },
        "balanced": {
            "SIMLINGO_DREAMER_GUARD": "1",
            "SIMLINGO_DREAMER_GUARD_MODE": "apply",
            "SIMLINGO_DREAMER_RISK_MARGIN": "0.03",
            "SIMLINGO_DREAMER_MAX_PROGRESS_DROP": "0.015",
            "SIMLINGO_DREAMER_MAX_STEER_DELTA": "0.16",
        },
        "accident_overtake": {
            "SIMLINGO_DREAMER_GUARD": "1",
            "SIMLINGO_DREAMER_GUARD_MODE": "apply",
            "SIMLINGO_DREAMER_VARIANT": "dreamer_guard_v1_accident_overtake_adapter",
            "SIMLINGO_DREAMER_RISK_MARGIN": "0.025",
            "SIMLINGO_DREAMER_MAX_PROGRESS_DROP": "0.03",
            "SIMLINGO_DREAMER_MAX_STEER_DELTA": "0.34",
            "SIMLINGO_DREAMER_MAX_BRAKE_INCREASE": "1.0",
            "SIMLINGO_DREAMER_HAZARD_FRONT_M": "26.0",
            "SIMLINGO_DREAMER_W_PROGRESS": "1.05",
            "SIMLINGO_DREAMER_W_RISK": "2.0",
            "SIMLINGO_DREAMER_ACTION_PENALTY": "0.065",
            "SIMLINGO_DREAMER_RECOVERY": "1",
            "SIMLINGO_DREAMER_RECOVERY_MIN_TICKS": "5",
            "SIMLINGO_DREAMER_RECOVERY_FRONT_M": "18.0",
            "SIMLINGO_DREAMER_RECOVERY_CLEARANCE_M": "14.0",
            "SIMLINGO_DREAMER_RECOVERY_ONCOMING_CLEARANCE_M": "48.0",
            "SIMLINGO_DREAMER_RECOVERY_ONCOMING_MIN_TTC": "6.5",
            "SIMLINGO_DREAMER_RECOVERY_MIN_TTC": "3.2",
            "SIMLINGO_DREAMER_RECOVERY_THROTTLE": "0.38",
            "SIMLINGO_DREAMER_RECOVERY_STEER": "0.30",
            "SIMLINGO_DREAMER_RECOVERY_HOLD_TICKS": "44",
            "SIMLINGO_DREAMER_RECOVERY_EXIT_FRONT_M": "22.0",
            "SIMLINGO_DREAMER_RECOVERY_REQUIRE_DRIVING_LANE": "1",
            "SIMLINGO_DREAMER_RECOVERY_GAP": "1",
            "SIMLINGO_DREAMER_RECOVERY_GAP_CLEARANCE_M": "7.5",
            "SIMLINGO_DREAMER_RECOVERY_GAP_MIN_TTC": "1.7",
            "SIMLINGO_DREAMER_RECOVERY_GAP_ONCOMING_CLEARANCE_M": "42.0",
            "SIMLINGO_DREAMER_RECOVERY_GAP_ONCOMING_MIN_TTC": "5.5",
            "SIMLINGO_DREAMER_RECOVERY_GAP_THROTTLE": "0.48",
            "SIMLINGO_DREAMER_RECOVERY_COMMIT_LOCK_TICKS": "72",
            "SIMLINGO_DREAMER_RECOVERY_COMMIT_ENTRY_TICKS": "18",
            "SIMLINGO_DREAMER_RECOVERY_COMMIT_CRUISE_TICKS": "34",
            "SIMLINGO_DREAMER_RECOVERY_COMMIT_EMERGENCY_CLEARANCE_M": "3.2",
            "SIMLINGO_DREAMER_RECOVERY_COMMIT_EMERGENCY_TTC": "1.6",
            "SIMLINGO_DREAMER_RECOVERY_COMMIT_ONCOMING_MIN_TTC": "4.8",
            "SIMLINGO_DREAMER_RECOVERY_FINISH_TICKS": "42",
            "SIMLINGO_DREAMER_RECOVERY_FINISH_STEER_SCALE": "0.38",
            "SIMLINGO_DREAMER_RECOVERY_FINISH_THROTTLE": "0.48",
            "SIMLINGO_DREAMER_COLLISION_SHIELD": "1",
            "SIMLINGO_DREAMER_COLLISION_SHIELD_FRONT_M": "12.0",
            "SIMLINGO_DREAMER_COLLISION_SHIELD_RISK": "0.70",
            "SIMLINGO_DREAMER_COLLISION_SHIELD_MIN_SPEED": "0.25",
            "SIMLINGO_DREAMER_COLLISION_SHIELD_BRAKE": "0.78",
        },
        "sdbs_fresh_accident_overtake": {
            "SIMLINGO_DREAMER_GUARD": "1",
            "SIMLINGO_DREAMER_GUARD_MODE": "apply",
            "SIMLINGO_DREAMER_VARIANT": "youma_sdbs_fresh_accident_overtake_v1_runtime_adapter",
            "SIMLINGO_DREAMER_RISK_MARGIN": "0.025",
            "SIMLINGO_DREAMER_MAX_PROGRESS_DROP": "0.03",
            "SIMLINGO_DREAMER_MAX_STEER_DELTA": "0.34",
            "SIMLINGO_DREAMER_MAX_BRAKE_INCREASE": "1.0",
            "SIMLINGO_DREAMER_HAZARD_FRONT_M": "26.0",
            "SIMLINGO_DREAMER_W_PROGRESS": "1.05",
            "SIMLINGO_DREAMER_W_RISK": "2.0",
            "SIMLINGO_DREAMER_ACTION_PENALTY": "0.065",
            "SIMLINGO_DREAMER_RECOVERY": "1",
            "SIMLINGO_DREAMER_RECOVERY_MIN_TICKS": "5",
            "SIMLINGO_DREAMER_RECOVERY_FRONT_M": "18.0",
            "SIMLINGO_DREAMER_RECOVERY_CLEARANCE_M": "14.0",
            "SIMLINGO_DREAMER_RECOVERY_ONCOMING_CLEARANCE_M": "48.0",
            "SIMLINGO_DREAMER_RECOVERY_ONCOMING_MIN_TTC": "6.5",
            "SIMLINGO_DREAMER_RECOVERY_MIN_TTC": "3.2",
            "SIMLINGO_DREAMER_RECOVERY_THROTTLE": "0.38",
            "SIMLINGO_DREAMER_RECOVERY_STEER": "0.30",
            "SIMLINGO_DREAMER_RECOVERY_USE_BASE_THROTTLE": "1",
            "SIMLINGO_DREAMER_RECOVERY_HOLD_TICKS": "44",
            "SIMLINGO_DREAMER_RECOVERY_EXIT_FRONT_M": "22.0",
            "SIMLINGO_DREAMER_RECOVERY_REQUIRE_DRIVING_LANE": "1",
            "SIMLINGO_DREAMER_RECOVERY_GAP": "1",
            "SIMLINGO_DREAMER_RECOVERY_GAP_CLEARANCE_M": "5.8",
            "SIMLINGO_DREAMER_RECOVERY_GAP_MIN_TTC": "1.7",
            "SIMLINGO_DREAMER_RECOVERY_GAP_ONCOMING_CLEARANCE_M": "42.0",
            "SIMLINGO_DREAMER_RECOVERY_GAP_ONCOMING_MIN_TTC": "5.5",
            "SIMLINGO_DREAMER_RECOVERY_GAP_THROTTLE": "0.52",
            "SIMLINGO_DREAMER_RECOVERY_GAP_INITIATIVE_TICKS": "22",
            "SIMLINGO_DREAMER_RECOVERY_GAP_INITIATIVE_CLEARANCE_M": "5.4",
            "SIMLINGO_DREAMER_RECOVERY_GAP_INITIATIVE_MIN_TTC": "1.45",
            "SIMLINGO_DREAMER_RECOVERY_GAP_INITIATIVE_ONCOMING_CLEARANCE_M": "42.0",
            "SIMLINGO_DREAMER_RECOVERY_MAX_RISK": "1.01",
            "SIMLINGO_DREAMER_RECOVERY_MIN_RISK_DROP": "-1.0",
            "SIMLINGO_DREAMER_RECOVERY_RISK_WEIGHT": "0.0",
            "SIMLINGO_DREAMER_RECOVERY_COMMIT_LOCK_TICKS": "72",
            "SIMLINGO_DREAMER_RECOVERY_COMMIT_EMERGENCY_CLEARANCE_M": "3.2",
            "SIMLINGO_DREAMER_RECOVERY_COMMIT_EMERGENCY_TTC": "1.6",
            "SIMLINGO_DREAMER_RECOVERY_COMMIT_ONCOMING_MIN_TTC": "4.8",
            "SIMLINGO_DREAMER_RECOVERY_FINISH_TICKS": "42",
            "SIMLINGO_DREAMER_RECOVERY_FINISH_STEER_SCALE": "0.38",
            "SIMLINGO_DREAMER_RECOVERY_FINISH_THROTTLE": "0.52",
            "SIMLINGO_DREAMER_COLLISION_SHIELD": "1",
            "SIMLINGO_DREAMER_COLLISION_SHIELD_FRONT_M": "12.0",
            "SIMLINGO_DREAMER_COLLISION_SHIELD_RISK": "0.70",
            "SIMLINGO_DREAMER_COLLISION_SHIELD_MIN_SPEED": "0.25",
            "SIMLINGO_DREAMER_COLLISION_SHIELD_BRAKE": "0.78",
        },
        "full": {
            "SIMLINGO_DREAMER_GUARD": "1",
            "SIMLINGO_DREAMER_GUARD_MODE": "full",
            "SIMLINGO_DREAMER_RISK_MARGIN": "0.0",
            "SIMLINGO_DREAMER_MAX_PROGRESS_DROP": "1.0",
            "SIMLINGO_DREAMER_MAX_STEER_DELTA": "0.24",
        },
    }
    dreamer_presets["dreamer_ppo"] = {
        **dreamer_presets["accident_overtake"],
        "SIMLINGO_DREAMER_VARIANT": "dreamer_ppo_unified",
    }
    dreamer_presets["dreamer_sdbs"] = {
        **dreamer_presets["sdbs_fresh_accident_overtake"],
        "SIMLINGO_DREAMER_VARIANT": "dreamer_sdbs_unified",
    }
    dreamer_presets["dreamer_ppo_rl_noguard"] = {
        "SIMLINGO_DREAMER_GUARD": "1",
        "SIMLINGO_DREAMER_GUARD_MODE": "full",
        "SIMLINGO_DREAMER_VARIANT": "dreamer_ppo_rl_noguard",
        "SIMLINGO_DREAMER_RISK_MARGIN": "0.0",
        "SIMLINGO_DREAMER_MAX_PROGRESS_DROP": "1.0",
        "SIMLINGO_DREAMER_MAX_STEER_DELTA": "1.0",
        "SIMLINGO_DREAMER_MAX_BRAKE_INCREASE": "1.0",
        "SIMLINGO_DREAMER_HAZARD_FRONT_M": "26.0",
        "SIMLINGO_DREAMER_W_PROGRESS": "1.05",
        "SIMLINGO_DREAMER_W_RISK": "2.0",
        "SIMLINGO_DREAMER_ACTION_PENALTY": "0.065",
        "SIMLINGO_DREAMER_RECOVERY": "0",
        "SIMLINGO_DREAMER_COLLISION_SHIELD": "0",
    }
    dreamer_presets["dreamer_sdbs_rl_noguard"] = {
        **dreamer_presets["dreamer_ppo_rl_noguard"],
        "SIMLINGO_DREAMER_VARIANT": "dreamer_sdbs_rl_noguard",
    }
    dreamer_aliases = {
        "shadow": "dreamer_ppo",
        "guard": "dreamer_ppo",
        "balanced": "dreamer_ppo",
        "accident_overtake": "dreamer_ppo",
        "full": "dreamer_ppo",
        "sdbs_fresh_accident_overtake": "dreamer_sdbs",
    }
    dreamer_mode = dreamer_aliases.get(dreamer_mode, dreamer_mode)
    if dreamer_mode in dreamer_presets:
        env.update(dreamer_presets[dreamer_mode])
        checkpoint_map = {
            "dreamer_ppo": {
                "path": SIMLINGO_ROOT / "checkpoints" / "dreamer_guard" / "best_world_model.pt",
                "source": "dreamer_ppo",
                "help": "Dreamer PPO checkpoint missing: external/simlingo/checkpoints/dreamer_guard/best_world_model.pt",
            },
            "dreamer_sdbs": {
                "path": SIMLINGO_ROOT / "checkpoints" / "dreamer_sdbs_fresh" / "best_world_model.pt",
                "source": "dreamer_sdbs",
                "help": "Dreamer SDBS checkpoint missing: external/simlingo/checkpoints/dreamer_sdbs_fresh/best_world_model.pt",
            },
            "dreamer_ppo_rl_noguard": {
                "path": SIMLINGO_ROOT / "checkpoints" / "dreamer_ppo_rl_noguard" / "latest_rl_model.pt",
                "source": "dreamer_ppo_rl_noguard",
                "help": (
                    "Dreamer PPO RL no-guard checkpoint missing. Run a real RL training, "
                    "then install its best checkpoint as "
                    "external/simlingo/checkpoints/dreamer_ppo_rl_noguard/latest_rl_model.pt."
                ),
            },
            "dreamer_sdbs_rl_noguard": {
                "path": SIMLINGO_ROOT / "checkpoints" / "dreamer_sdbs_rl_noguard" / "latest_rl_model.pt",
                "source": "dreamer_sdbs_rl_noguard",
                "help": (
                    "Dreamer SDBS RL no-guard checkpoint missing. Run a real RL training, "
                    "then install its best checkpoint as "
                    "external/simlingo/checkpoints/dreamer_sdbs_rl_noguard/latest_rl_model.pt."
                ),
            },
        }
        checkpoint_info = checkpoint_map.get(dreamer_mode, checkpoint_map["dreamer_ppo"])
        checkpoint = checkpoint_info["path"]
        if not checkpoint.exists():
            raise RuntimeError(checkpoint_info["help"])
        env["SIMLINGO_DREAMER_CHECKPOINT"] = str(checkpoint)
        env["SIMLINGO_DREAMER_CHECKPOINT_SOURCE"] = checkpoint_info["source"]
    launch_script = "run_simlingo_with_pov.sh"
    if run_mode == "sumo_mirror":
        launch_script = "run_simlingo_with_sumo_mirror.sh"
        env.update({
            "SUMO_MIRROR_GUI": str(payload.get("sumo_mirror_gui", "1")),
            "SUMO_MIRROR_SYNC_TLS": str(payload.get("sumo_mirror_sync_tls", "1")),
            "SUMO_MIRROR_POLL": str(payload.get("sumo_mirror_poll", "0.05")),
            "SUMO_MIRROR_NO_WARNINGS": "1",
        })
    elif run_mode == "action_dreaming":
        launch_script = "run_simlingo_with_action_dreaming_collect.sh"
        env.update({
            "ACTION_DREAMING_SAMPLE_INTERVAL": str(payload.get("action_dreaming_sample_interval", "0.25")),
            "ACTION_DREAMING_K": str(payload.get("action_dreaming_k", "5")),
            "ACTION_DREAMING_GENERATE_AFTER": "1",
        })
    if prompt_mode == "obstacle":
        env.update({
            "SIMLINGO_USER_FLAG": "1",
            "SIMLINGO_CUSTOM_PROMPT": (
                "If there is an accident, parked vehicle, construction obstacle, or blocked lane ahead, "
                "go around it safely when the neighbouring lane is clear, then return to the route. "
                "What should the ego do next?"
            ),
        })
    launch_log = LOG_DIR / "latest_launch.log"
    env["SIMLINGO_VIEWER_LOG"] = str(ROOT / "logs" / "simlingo_eval" / "latest_pov_viewer.log")
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    launch_fh = launch_log.open("w", buffering=1, encoding="utf-8")
    print(f"[dashboard] started_at={time.strftime('%Y-%m-%d %H:%M:%S')}", file=launch_fh)
    print(f"[dashboard] root={ROOT}", file=launch_fh)
    print(f"[dashboard] script={launch_script}", file=launch_fh)
    print(f"[dashboard] route={route['id']} town={route['town']} scenario={route['scenario_type']}", file=launch_fh)
    print(f"[dashboard] mode={run_mode} dreamer={dreamer_mode} cot={cot_mode} seed={seed}", file=launch_fh)
    print(f"[dashboard] port={port} tm_port={tm_port} display={env.get('DISPLAY', '<unset>')}", file=launch_fh)
    print(f"[dashboard] viewer_log={env['SIMLINGO_VIEWER_LOG']}", file=launch_fh)
    print("[dashboard] --- child output ---", file=launch_fh)
    proc = subprocess.Popen(
        ["bash", str(ROOT / "scripts" / launch_script)],
        cwd=str(ROOT),
        env=env,
        stdout=launch_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    launch_fh.close()
    with STATE_LOCK:
        STATE.update({
            "process": proc,
            "route": route["id"],
            "route_town": route["town"],
            "scenario": route["scenario_type"],
            "mode": run_mode,
            "dreamer_mode": dreamer_mode,
            "cot_mode": cot_mode,
            "seed": seed,
            "port": port,
            "started_at": time.time(),
            "launch_log": str(launch_log),
            "last_error": None,
        })
    return {"ok": True, "route": route, "seed": seed}


def replay_latest(payload):
    latest_path = ROOT / "logs" / "simlingo_eval" / "latest_pygame_recording.txt"
    if not latest_path.exists():
        raise RuntimeError("No recorded SimLingo video found yet.")
    video_path = Path(latest_path.read_text().strip())
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"Recorded video is missing or empty: {video_path}")
    speed = str(payload.get("playback_speed") or "5")
    replay_log = ROOT / "logs" / "simlingo_eval" / "latest_replay.log"
    subprocess.run(["pkill", "-TERM", "-f", "scripts/play_recorded_video.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log_file = open(replay_log, "w", buffering=1)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "play_recorded_video.py"),
            str(video_path),
            "--speed",
            speed,
            "--title",
            "SimLingo replay",
        ],
        cwd=str(ROOT),
        env=os.environ.copy(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (ROOT / "logs" / "simlingo_eval" / "latest_replay.pid").write_text(str(proc.pid) + "\n")
    return {"ok": True, "video": str(video_path), "speed": speed, "pid": proc.pid, "log": str(replay_log)}


def start_twinsentinel_console():
    port = os.environ.get("TWINSENTINEL_PORT", "3100")
    env = os.environ.copy()
    env.update({
        "TWINSENTINEL_PORT": port,
        "TWINSENTINEL_STATE_FILE": str(ROOT / "logs" / "sumo_mirror" / "live_state.json"),
        "TWINSENTINEL_COMMAND_FILE": str(ROOT / "logs" / "sumo_mirror" / "attack_commands.jsonl"),
    })
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_twinsentinel_attack_console.sh")],
        cwd=str(ROOT),
        env=env,
        check=True,
        timeout=10,
    )
    return {
        "ok": True,
        "url": f"http://127.0.0.1:{port}",
        "state_file": env["TWINSENTINEL_STATE_FILE"],
        "command_file": env["TWINSENTINEL_COMMAND_FILE"],
    }


def load_json_file(path):
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def as_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


def fmt_percent(value, digits=1):
    if value is None:
        return "-"
    return f"{100.0 * float(value):.{digits}f}%"


def fmt_number(value, digits=3):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def mean_metric(summary, metric_name, field="mean"):
    try:
        return summary["metrics"][metric_name][field]
    except Exception:
        return None


def bench2drive_result_summary():
    paths = sorted((ROOT / "logs" / "simlingo_eval").glob("results_*.json"))
    rows = []
    for path in paths:
        data = load_json_file(path)
        if not data:
            continue
        record = data.get("_checkpoint", {}).get("global_record", {})
        scores = record.get("scores_mean", {})
        infractions = record.get("infractions", {})
        if not scores:
            continue
        rows.append({
            "score": as_float(scores.get("score_composed")),
            "route": as_float(scores.get("score_route")),
            "penalty": as_float(scores.get("score_penalty")),
            "ped": as_float(infractions.get("collisions_pedestrian")),
            "veh": as_float(infractions.get("collisions_vehicle")),
            "layout": as_float(infractions.get("collisions_layout")),
            "red": as_float(infractions.get("red_light")),
            "offroad": as_float(infractions.get("outside_route_lanes")),
            "blocked": as_float(infractions.get("vehicle_blocked")),
        })
    if not rows:
        return {
            "count": 0,
            "avg_score": None,
            "avg_route": None,
            "collisions": None,
            "red_lights": None,
            "offroad": None,
            "blocked": None,
        }
    return {
        "count": len(rows),
        "avg_score": sum(r["score"] for r in rows) / len(rows),
        "avg_route": sum(r["route"] for r in rows) / len(rows),
        "collisions": sum(r["ped"] + r["veh"] + r["layout"] for r in rows),
        "red_lights": sum(r["red"] for r in rows),
        "offroad": sum(r["offroad"] for r in rows),
        "blocked": sum(r["blocked"] for r in rows),
    }


def read_training_csv(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def compute_runme_kpis(rows, tail=0.25):
    if not rows:
        return None
    if 0 < tail < 1:
        keep = max(1, int(round(len(rows) * tail)))
        rows = rows[-keep:]
    n = len(rows)

    def avg(key, default=0.0):
        return sum(as_float(row.get(key), default) for row in rows) / max(1, n)

    mean_return = avg("return")
    route_completion = avg("route_completion")
    success_rate = sum(1 for row in rows if as_float(row.get("route_completion")) >= 0.99) / max(1, n)
    vru_collisions = avg("vru_collisions")
    vru_near_misses = avg("vru_near_misses")
    min_ttc = avg("min_ttc_vru", 3.0)
    distance_vru = avg("avg_distance_to_vru", 8.0)
    vehicle_collisions = avg("vehicle_collisions")
    vehicle_near_misses = avg("vehicle_near_misses")
    rear_incidents = avg("rear_incidents")
    lane_departures = avg("lane_departures")
    lane_change_success = avg("lane_change_success_rate", 1.0)

    vru_safety = clamp(
        100.0
        - 45.0 * vru_collisions
        - 12.0 * vru_near_misses
        - 10.0 * max(0.0, 2.0 - min_ttc)
    )
    vehicle_safety = clamp(
        100.0
        - 35.0 * vehicle_collisions
        - 10.0 * vehicle_near_misses
        - 8.0 * rear_incidents
    )
    progress = clamp(route_completion * 100.0)
    comfort = clamp(100.0 - 10.0 * lane_departures)
    composite = clamp(0.55 * vru_safety + 0.25 * progress + 0.12 * vehicle_safety + 0.08 * comfort)
    return {
        "episodes_evaluated": n,
        "mean_return": mean_return,
        "mean_route_completion": route_completion,
        "success_rate": success_rate,
        "vru_collisions_per_ep": vru_collisions,
        "vru_near_misses_per_ep": vru_near_misses,
        "mean_min_ttc_vru": min_ttc,
        "mean_distance_to_vru": distance_vru,
        "vehicle_collisions_per_ep": vehicle_collisions,
        "vehicle_near_misses_per_ep": vehicle_near_misses,
        "lane_change_success_rate": lane_change_success,
        "vru_safety_score": vru_safety,
        "vehicle_safety_score": vehicle_safety,
        "comfort_score": comfort,
        "composite_score": composite,
    }


def csv_kpi_for(label):
    logs_dir = DREAMER_ROOT / "logs"
    candidates = [
        logs_dir / f"{label}.csv",
        DREAMER_ROOT / "logs" / f"{label}.csv",
    ]
    for path in candidates:
        kpi = compute_runme_kpis(read_training_csv(path))
        if kpi:
            kpi["path"] = str(path)
            return kpi
    return None


SAFE_DREAM_ACTION_TAXONOMY = (
    "base",
    "model_nearby",
    "model_cautious",
    "model_steer_delta",
    "hazard_hold",
    "hazard_strong_hold",
    "recovery_overtake",
    "recovery_gap_commit",
    "recovery_commit_continue",
    "recovery_finish_pass",
    "recovery_hold",
    "recovery_creep",
    "collision_shield_hold",
)


def clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return None


def fmt_score(value, digits=1):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}%"


def fmt_ratio(value, digits=3):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_rate(value, digits=1):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def result_key_from_path(path):
    match = re.match(r"results_(.+)_seed_(\d+)\.json$", path.name)
    if not match:
        return None
    return match.group(1), match.group(2)


def run_log_for_result(result_path):
    key = result_key_from_path(result_path)
    if not key:
        return None
    route_label, seed = key
    path = ROOT / "logs" / "simlingo_eval" / f"run_{route_label}_seed_{seed}.log"
    return path if path.exists() else None


def infraction_count(value):
    if isinstance(value, list):
        return float(len(value))
    if isinstance(value, (int, float)):
        return float(value)
    if value in ("", None):
        return 0.0
    return as_float(value)


def parse_bench2drive_result(path):
    data = load_json_file(path)
    if not data:
        return None
    if data.get("eligible") is False:
        return None
    checkpoint = data.get("_checkpoint", {})
    records = checkpoint.get("records") or []
    record = records[0] if records else checkpoint.get("global_record", {})
    global_record = checkpoint.get("global_record", {})
    scores = record.get("scores") or record.get("scores_mean") or global_record.get("scores_mean", {})
    if not scores:
        return None
    infractions = record.get("infractions") or global_record.get("infractions", {})
    meta = record.get("meta") or global_record.get("meta", {})
    length_m = as_float(meta.get("route_length"), as_float(meta.get("total_length"), 0.0))
    collisions = (
        infraction_count(infractions.get("collisions_pedestrian"))
        + infraction_count(infractions.get("collisions_vehicle"))
        + infraction_count(infractions.get("collisions_layout"))
    )
    red_lights = infraction_count(infractions.get("red_light"))
    stop_infractions = infraction_count(infractions.get("stop_infraction"))
    offroad = infraction_count(infractions.get("outside_route_lanes"))
    blocked = infraction_count(infractions.get("vehicle_blocked"))
    scenario_timeouts = infraction_count(infractions.get("scenario_timeouts"))
    route_timeouts = infraction_count(infractions.get("route_timeout"))
    min_speed = infraction_count(infractions.get("min_speed_infractions"))
    route_score = as_float(scores.get("score_route"))
    driving_score = as_float(scores.get("score_composed"))
    return {
        "path": str(path),
        "route_label": result_key_from_path(path)[0] if result_key_from_path(path) else path.stem,
        "town": record.get("town_name") or "",
        "scenario": record.get("scenario_name") or "",
        "status": record.get("status") or global_record.get("status") or data.get("entry_status", ""),
        "length_km": max(0.0, length_m / 1000.0),
        "route_score": route_score,
        "driving_score": driving_score,
        "penalty": as_float(scores.get("score_penalty")),
        "collisions": collisions,
        "red_lights": red_lights,
        "stop_infractions": stop_infractions,
        "offroad": offroad,
        "blocked": blocked,
        "scenario_timeouts": scenario_timeouts,
        "route_timeouts": route_timeouts,
        "min_speed_infractions": min_speed,
        "success": 1.0 if route_score >= 99.0 and collisions == 0 and offroad == 0 and blocked == 0 else 0.0,
    }


def parse_arrow_pair(value):
    if "->" not in value:
        return None
    left, right = value.split("->", 1)
    return as_float(left, None), as_float(right, None)


def parse_guard_line(line):
    fields = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
        fields[key] = value
    return fields


def parse_ttc_value(value):
    if not value:
        return None
    try:
        if "/" in value:
            value = value.split("/")[-1]
        ttc = float(value)
    except Exception:
        return None
    if ttc <= 0.0 or ttc >= 90.0:
        return None
    return ttc


def parse_dreamer_log(path):
    info = {
        "group": "native",
        "variant": "native",
        "guard_rows": 0,
        "applied": 0,
        "shield": 0,
        "recovery": 0,
        "commit": 0,
        "finish": 0,
        "risk_deltas": [],
        "progress_deltas": [],
        "kinds": set(),
        "candidate_ids": set(),
        "min_ttc": None,
        "latest_step": None,
    }
    if not path or not path.exists():
        return info
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "SIMLINGO_DREAMER_GUARD enabled" in line:
                    fields = parse_guard_line(line)
                    info["group"] = "dreamer_v1"
                    variant = re.search(r"variant=([^\s]+)", line)
                    backend = fields.get("backend", "")
                    profile = fields.get("profile", "")
                    if variant:
                        info["variant"] = variant.group(1)
                    elif backend:
                        info["variant"] = backend + (f"_{profile}" if profile else "")
                    elif profile:
                        info["variant"] = f"dreamer_guard_v1_{profile}"
                    if "sdbs" in info["variant"].lower() or "sdbs" in backend.lower():
                        info["group"] = "dreamer_sdbs"
                    if "rl_noguard" in info["variant"].lower():
                        info["group"] = "dreamer_sdbs_rl" if "sdbs" in info["variant"].lower() else "dreamer_ppo_rl"
                if "SIMLINGO_DREAMER_GUARD step=" not in line:
                    continue
                fields = parse_guard_line(line)
                backend = fields.get("backend", "")
                profile = fields.get("profile", "")
                if "sdbs" in backend.lower():
                    info["group"] = "dreamer_sdbs"
                    if info["variant"] == "native":
                        info["variant"] = backend + (f"_{profile}" if profile else "")
                if "rl_noguard" in (info.get("variant") or "").lower() or "rl_noguard" in backend.lower():
                    is_sdbs = "sdbs" in (info.get("variant") or "").lower() or "sdbs" in backend.lower()
                    info["group"] = "dreamer_sdbs_rl" if is_sdbs else "dreamer_ppo_rl"
                info["guard_rows"] += 1
                kind = fields.get("kind", "")
                if kind:
                    info["kinds"].add(kind)
                candidate = fields.get("candidate")
                if candidate not in (None, ""):
                    info["candidate_ids"].add(candidate)
                if fields.get("applied") == "1":
                    info["applied"] += 1
                if fields.get("shield") == "1" or kind == "collision_shield_hold":
                    info["shield"] += 1
                if kind.startswith("recovery_"):
                    info["recovery"] += 1
                if kind in ("recovery_gap_commit", "recovery_commit_continue", "recovery_commit_recenter"):
                    info["commit"] += 1
                if kind == "recovery_finish_pass":
                    info["finish"] += 1
                if "step" in fields:
                    info["latest_step"] = as_float(fields.get("step"), info.get("latest_step"))
                risk_pair = parse_arrow_pair(fields.get("risk", ""))
                if risk_pair and risk_pair[0] is not None and risk_pair[1] is not None:
                    info["risk_deltas"].append(risk_pair[0] - risk_pair[1])
                progress_pair = parse_arrow_pair(fields.get("progress", ""))
                if progress_pair and progress_pair[0] is not None and progress_pair[1] is not None:
                    info["progress_deltas"].append(progress_pair[1] - progress_pair[0])
                for key in ("ttcL", "ttcR", "onL", "onR"):
                    ttc = parse_ttc_value(fields.get(key))
                    if ttc is not None:
                        info["min_ttc"] = ttc if info["min_ttc"] is None else min(info["min_ttc"], ttc)
    except Exception:
        return info
    return info


def mean_or_none(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def safe_dream_model_groups():
    groups = {
        "simlingo": {"id": "simlingo", "name": "SimLingo native", "subtitle": "Baseline VLA closed-loop"},
        "dreamer_v1": {"id": "dreamer_v1", "name": "SimLingo + Dreamer PPO", "subtitle": "Unified PPO Dreamer runtime guard"},
        "dreamer_sdbs": {"id": "dreamer_sdbs", "name": "SimLingo + Dreamer SDBS", "subtitle": "SDBS checkpoint with unified runtime guard"},
        "dreamer_ppo_rl": {"id": "dreamer_ppo_rl", "name": "SimLingo + Dreamer PPO RL", "subtitle": "No-guard RL policy, pending training"},
        "dreamer_sdbs_rl": {"id": "dreamer_sdbs_rl", "name": "SimLingo + Dreamer SDBS RL", "subtitle": "No-guard SDBS RL policy, pending training"},
    }
    for group in groups.values():
        group.update({
            "runs": [],
            "logs": [],
            "trace_logs": [],
            "incomplete_results": [],
            "observed_runs": [],
            "variants": set(),
            "guard_rows": 0,
            "applied": 0,
            "shield": 0,
            "recovery": 0,
            "commit": 0,
            "finish": 0,
            "risk_deltas": [],
            "progress_deltas": [],
            "kinds": set(),
            "candidate_ids": set(),
            "min_ttc": None,
            "latest_result": None,
        })

    for result_path in sorted((ROOT / "logs" / "simlingo_eval").glob("results_*.json")):
        log_path = run_log_for_result(result_path)
        log_info = parse_dreamer_log(log_path)
        group_key = log_info["group"]
        if group_key == "native":
            group_key = "simlingo"
        group = groups.get(group_key, groups["simlingo"])
        if log_path:
            group["trace_logs"].append(str(log_path))
        group["variants"].add(log_info.get("variant") or "native")

        result = parse_bench2drive_result(result_path)
        raw_result = load_json_file(result_path) or {}
        key = result_key_from_path(result_path)
        observed_status = "scored" if result else "unscored"
        entry_status = raw_result.get("entry_status")
        if not result and entry_status:
            observed_status = entry_status
        group["observed_runs"].append({
            "file": result_path.name,
            "log": log_path.name if log_path else "-",
            "route": key[0] if key else result_path.stem,
            "seed": key[1] if key else "-",
            "status": observed_status,
            "scored": bool(result),
            "variant": log_info.get("variant") or "native",
        })
        if not result:
            if group_key != "simlingo":
                group["incomplete_results"].append(str(result_path))
            continue
        group["runs"].append(result)
        if log_path:
            group["logs"].append(str(log_path))
        group["guard_rows"] += log_info["guard_rows"]
        group["applied"] += log_info["applied"]
        group["shield"] += log_info["shield"]
        group["recovery"] += log_info["recovery"]
        group["commit"] += log_info["commit"]
        group["finish"] += log_info["finish"]
        group["risk_deltas"].extend(log_info["risk_deltas"])
        group["progress_deltas"].extend(log_info["progress_deltas"])
        group["kinds"].update(log_info["kinds"])
        group["candidate_ids"].update(log_info["candidate_ids"])
        if log_info["min_ttc"] is not None:
            group["min_ttc"] = log_info["min_ttc"] if group["min_ttc"] is None else min(group["min_ttc"], log_info["min_ttc"])
        group["latest_result"] = result

    for group in groups.values():
        runs = group["runs"]
        n = len(runs)
        total_km = sum(r["length_km"] for r in runs)
        collisions = sum(r["collisions"] for r in runs)
        offroad = sum(r["offroad"] for r in runs)
        red_lights = sum(r["red_lights"] for r in runs)
        stops = sum(r["stop_infractions"] for r in runs)
        blocked = sum(r["blocked"] for r in runs)
        route_timeouts = sum(r["route_timeouts"] for r in runs)
        scenario_timeouts = sum(r["scenario_timeouts"] for r in runs)
        min_speed = sum(r["min_speed_infractions"] for r in runs)
        group.update({
            "n": n,
            "trace_run_count": len(group["trace_logs"]),
            "incomplete_result_count": len(group["incomplete_results"]),
            "total_km": total_km,
            "avg_route": mean_or_none([r["route_score"] for r in runs]),
            "avg_score": mean_or_none([r["driving_score"] for r in runs]),
            "success_rate": mean_or_none([r["success"] for r in runs]),
            "collisions": collisions,
            "collisions_per_ep": collisions / n if n else None,
            "collision_rate_mkm": collisions / total_km * 1e6 if total_km > 0 else None,
            "offroad_per_ep": offroad / n if n else None,
            "red_light_per_ep": red_lights / n if n else None,
            "blocked_per_ep": blocked / n if n else None,
            "timeout_per_ep": (route_timeouts + scenario_timeouts) / n if n else None,
            "min_speed_per_ep": min_speed / n if n else None,
            "traffic_rule_pass_rate": mean_or_none([1.0 if (r["red_lights"] + r["stop_infractions"]) == 0 else 0.0 for r in runs]),
            "blocked_pass_rate": mean_or_none([1.0 if r["blocked"] == 0 else 0.0 for r in runs]),
            "override_rate": group["applied"] / group["guard_rows"] if group["guard_rows"] else None,
            "safety_gain": mean_or_none(group["risk_deltas"]),
            "progress_gain": mean_or_none(group["progress_deltas"]),
        })
        if group["guard_rows"]:
            cc = min(1.0, len(group["kinds"]) / max(1, len(SAFE_DREAM_ACTION_TAXONOMY)))
            fd = min(1.0, len(group["candidate_ids"]) / 8.0)
            dc = clamp01(1.0 - ((collisions + offroad) / max(1, n)))
            unsafe_total = group["shield"] + collisions + offroad
            ufrr = group["shield"] / unsafe_total if unsafe_total > 0 else None
            sg_norm = clamp01(max(0.0, group["safety_gain"] or 0.0) / 0.20)
            dqi_terms = [term for term in (cc, fd, dc, ufrr, sg_norm) if term is not None]
            dqi = sum(dqi_terms) / len(dqi_terms) if dqi_terms else None
        else:
            cc = fd = ufrr = sg_norm = dqi = None
            dc = clamp01(1.0 - ((collisions + offroad) / max(1, n))) if n else None
        group.update({
            "counterfactual_coverage": cc,
            "future_diversity": fd,
            "dreaming_consistency": dc,
            "unsafe_rejection_rate": ufrr,
            "safety_gain_norm": sg_norm,
            "dreaming_quality_index": dqi,
        })
    return groups


def dreamer_comparison_payload():
    groups = safe_dream_model_groups()
    comparison_keys = ["simlingo", "dreamer_v1", "dreamer_sdbs", "dreamer_ppo_rl", "dreamer_sdbs_rl"]
    wm = load_json_file(DREAMER_ROOT / "outputs" / "simlingo_world_model_20260616" / "summary.json") or {}
    guard = load_json_file(DREAMER_ROOT / "outputs" / "simlingo_dreamer_guard_rm005_md005" / "summary.json") or {}
    guard_loose = load_json_file(DREAMER_ROOT / "outputs" / "simlingo_dreamer_guard_rm003_md003" / "summary.json") or {}
    pure = load_json_file(DREAMER_ROOT / "outputs" / "simlingo_vs_dreamer_benchmark_wrisk2" / "summary.json") or {}
    sdbs_summary = load_json_file(SIMLINGO_ROOT / "checkpoints" / "dreamer_sdbs_fresh" / "summary.json") or {}
    sdbs_manifest = SIMLINGO_ROOT / "checkpoints" / "dreamer_sdbs_fresh" / "manifest.txt"
    sdbs_checkpoint = SIMLINGO_ROOT / "checkpoints" / "dreamer_sdbs_fresh" / "best_world_model.pt"
    ppo_rl_checkpoint = SIMLINGO_ROOT / "checkpoints" / "dreamer_ppo_rl_noguard" / "latest_rl_model.pt"
    sdbs_rl_checkpoint = SIMLINGO_ROOT / "checkpoints" / "dreamer_sdbs_rl_noguard" / "latest_rl_model.pt"

    wm_best = wm.get("best", {})
    legacy_guard_override = guard.get("override_rate")
    legacy_loose_override = guard_loose.get("override_rate")
    pure_agreement = pure.get("dreamer_same_as_simlingo_rate")

    def card_for(key, checkpoint_note=""):
        group = groups[key]
        headline = "no runs yet"
        if group["n"]:
            headline = f"{fmt_score(group['avg_route'])} route / {fmt_number(group['collisions'], 0)} coll"
        metrics = [
            {"label": "Runs evaluated", "value": str(group["n"])},
            {"label": "Driving score", "value": fmt_score(group["avg_score"])},
            {"label": "Route completion", "value": fmt_score(group["avg_route"])},
            {"label": "Success rate", "value": fmt_percent(group["success_rate"])},
            {"label": "Collision rate / Mkm", "value": fmt_rate(group["collision_rate_mkm"])},
            {"label": "Override rate", "value": fmt_percent(group["override_rate"])},
            {"label": "SAFE-DREAM DQI", "value": fmt_ratio(group["dreaming_quality_index"])},
        ]
        status = "reference" if key == "simlingo" else "active"
        if key in ("dreamer_ppo_rl", "dreamer_sdbs_rl"):
            status = "rl pending" if not group["n"] else "rl evaluated"
        if key == "dreamer_sdbs" and not sdbs_checkpoint.exists():
            status = "needs training"
        note = checkpoint_note or "Computed from local Bench2Drive JSON results and matching run logs."
        if group.get("incomplete_result_count"):
            note += f" {group['incomplete_result_count']} trace run(s) detected but not counted because Bench2Drive did not write eligible scores."
        if group["latest_result"]:
            note += f" Latest: {Path(group['latest_result']['path']).name}."
        return {
            "id": group["id"],
            "name": group["name"],
            "subtitle": group["subtitle"],
            "status": status,
            "headline": headline,
            "metrics": metrics,
            "note": note,
        }

    def observed_for(key, limit=12):
        runs = groups[key].get("observed_runs", [])
        return list(reversed(runs[-limit:]))

    cards = [
        card_for("simlingo", "Native baseline: no Dreamer/guard intervention, only Bench2Drive closed-loop metrics."),
        card_for("dreamer_v1", "Unified Dreamer PPO guard. Legacy offline override rate: " + fmt_percent(legacy_guard_override) + "."),
        card_for(
            "dreamer_sdbs",
            (
                "Dreamer SDBS checkpoint installed; "
                if sdbs_checkpoint.exists() else "Dreamer SDBS checkpoint missing; "
            )
            + f"state={sdbs_summary.get('state_dim', 28)}D, transitions={sdbs_summary.get('transitions', '-')}, best_loss={fmt_number((sdbs_summary.get('best') or {}).get('loss'), 4)}.",
        ),
        card_for(
            "dreamer_ppo_rl",
            (
                "Dreamer PPO RL no-guard checkpoint installed; "
                if ppo_rl_checkpoint.exists() else
                "Dreamer PPO RL no-guard checkpoint missing; train/install latest_rl_model.pt first. "
            )
            + "Uses the no-guard full candidate scorer path.",
        ),
        card_for(
            "dreamer_sdbs_rl",
            (
                "Dreamer SDBS RL no-guard checkpoint installed; "
                if sdbs_rl_checkpoint.exists() else
                "Dreamer SDBS RL no-guard checkpoint missing; train/install latest_rl_model.pt first. "
            )
            + "Uses the no-guard full candidate scorer path.",
        ),
    ]

    def val(key, metric, formatter=fmt_ratio):
        return formatter(groups[key].get(metric))

    def row(label, metric, formatter=fmt_ratio):
        return {
            "label": label,
            "simlingo": val("simlingo", metric, formatter),
            "dreamer_v1": val("dreamer_v1", metric, formatter),
            "dreamer_sdbs": val("dreamer_sdbs", metric, formatter),
            "dreamer_ppo_rl": val("dreamer_ppo_rl", metric, formatter),
            "dreamer_sdbs_rl": val("dreamer_sdbs_rl", metric, formatter),
        }

    rows = [
        row("Family E - Runs evaluated", "n", lambda v: "-" if v is None else str(int(v))),
        row("Runtime - Trace logs detected", "trace_run_count", lambda v: "-" if v is None else str(int(v))),
        row("Runtime - Incomplete/unscored results", "incomplete_result_count", lambda v: "-" if v is None else str(int(v))),
        row("Family E - Driving score", "avg_score", fmt_score),
        row("Family E - Route completion", "avg_route", fmt_score),
        row("Family E - Scenario success rate", "success_rate", fmt_percent),
        row("Family E Eq.12 - Collision rate / 1M km", "collision_rate_mkm", fmt_rate),
        row("Family E Eq.12 - Collisions / episode", "collisions_per_ep", fmt_rate),
        row("Family E - Off-road infractions / episode", "offroad_per_ep", fmt_rate),
        row("Family E Eq.18 - Traffic-rule pass rate", "traffic_rule_pass_rate", fmt_percent),
        row("Family E Eq.17 - Agent-blocked pass rate", "blocked_pass_rate", fmt_percent),
        row("Family E - Min-speed infractions / episode", "min_speed_per_ep", fmt_rate),
        row("Family E Eq.13 - Min TTC observed in Dreamer log", "min_ttc", lambda v: "-" if v is None else f"{float(v):.2f}s"),
        row("Runtime - Dreamer override rate", "override_rate", fmt_percent),
        row("Family D Eq.4 - Counterfactual coverage CC", "counterfactual_coverage", fmt_ratio),
        row("Family D Eq.5 - Future diversity FD proxy", "future_diversity", fmt_ratio),
        row("Family D Eq.6 - Dreaming consistency DC proxy", "dreaming_consistency", fmt_ratio),
        row("Family D Eq.9 - Unsafe future rejection UFRR proxy", "unsafe_rejection_rate", fmt_ratio),
        row("Family D Eq.10 - Safety gain SG risk delta", "safety_gain", fmt_number),
        row("Family D Eq.11 - Dreaming Quality Index DQI", "dreaming_quality_index", fmt_ratio),
        {
            "label": "Evidence - Latest result file",
            "simlingo": Path(groups["simlingo"]["latest_result"]["path"]).name if groups["simlingo"]["latest_result"] else "-",
            "dreamer_v1": Path(groups["dreamer_v1"]["latest_result"]["path"]).name if groups["dreamer_v1"]["latest_result"] else "-",
            "dreamer_sdbs": Path(groups["dreamer_sdbs"]["latest_result"]["path"]).name if groups["dreamer_sdbs"]["latest_result"] else "-",
            "dreamer_ppo_rl": Path(groups["dreamer_ppo_rl"]["latest_result"]["path"]).name if groups["dreamer_ppo_rl"]["latest_result"] else "-",
            "dreamer_sdbs_rl": Path(groups["dreamer_sdbs_rl"]["latest_result"]["path"]).name if groups["dreamer_sdbs_rl"]["latest_result"] else "-",
        },
        {
            "label": "Evidence - Variants detected",
            "simlingo": ", ".join(sorted(groups["simlingo"]["variants"])) or "-",
            "dreamer_v1": ", ".join(sorted(groups["dreamer_v1"]["variants"])) or "-",
            "dreamer_sdbs": ", ".join(sorted(groups["dreamer_sdbs"]["variants"])) or "-",
            "dreamer_ppo_rl": ", ".join(sorted(groups["dreamer_ppo_rl"]["variants"])) or "-",
            "dreamer_sdbs_rl": ", ".join(sorted(groups["dreamer_sdbs_rl"]["variants"])) or "-",
        },
    ]

    return {
        "ok": True,
        "source": "SAFE-DREAM dashboard adapter over local Bench2Drive result JSONs and SimLingo/Dreamer run logs",
        "runme_kpis": [
            "Family E metrics are direct Bench2Drive outcomes: driving score, route completion, collisions, off-road, traffic-rule and blocked-agent rates.",
            "Family D metrics are derived from Dreamer traces: CC from observed candidate kinds, FD from candidate diversity, SG from base-risk minus chosen-risk, and UFRR from shielded unsafe futures.",
            "Metrics marked as proxy are valid for comparison inside this pipeline but should be reported as log-derived SAFE-DREAM proxies, not full rollout-ground-truth measurements.",
            "RL no-guard columns are reserved for the next training stage; they remain empty until their backend policies/checkpoints are wired and evaluated.",
            "Legacy PPO offline summary: override " + fmt_percent(legacy_guard_override) + ", loose override " + fmt_percent(legacy_loose_override) + ", candidate agreement " + fmt_percent(pure_agreement) + ", WM loss " + fmt_number(wm_best.get("loss"), 4) + ".",
        ],
        "all_runs": [
            {
                "id": key,
                "name": groups[key]["name"],
                "runs": observed_for(key),
            }
            for key in comparison_keys
        ],
        "cards": cards,
        "rows": rows,
        "raw": {
            key: {k: v for k, v in groups[key].items() if k not in ("runs", "logs", "variants", "kinds", "candidate_ids", "risk_deltas", "progress_deltas")}
            for key in comparison_keys
        },
    }


HTML = r"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VLA-AV SimLingo World</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #05070a;
      --ink: #f5f7fb;
      --muted: #9ca7b8;
      --line: rgba(255,255,255,.14);
      --line-soft: rgba(255,255,255,.08);
      --glass: rgba(11,15,23,.62);
      --glass-strong: rgba(15,20,31,.82);
      --cyan: #6ee7f9;
      --green: #7ef2a2;
      --red: #ff5d73;
      --amber: #ffd166;
      --blue: #7aa2ff;
      --shadow: 0 34px 80px rgba(0,0,0,.38);
    }
    * { box-sizing: border-box; }
    html { background: var(--bg); }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 12% 4%, rgba(110,231,249,.24), transparent 30%),
        radial-gradient(circle at 88% 14%, rgba(126,242,162,.14), transparent 32%),
        linear-gradient(180deg, #090d13 0%, #06080c 42%, #10131a 100%);
      overflow-x: hidden;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(to bottom, transparent 0, black 18%, black 72%, transparent 100%);
      opacity: .45;
    }
    button, select, input { font: inherit; }
    .shell { width: min(1440px, 100%); margin: 0 auto; padding: 22px 24px 36px; }
    .topbar {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 0 16px;
      background: rgba(255,255,255,.045);
      backdrop-filter: blur(20px);
      box-shadow: 0 14px 50px rgba(0,0,0,.18);
    }
    .brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .mark {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background:
        radial-gradient(circle at 50% 50%, #f5f7fb 0 3px, transparent 4px),
        conic-gradient(from 120deg, var(--cyan), var(--green), var(--blue), var(--cyan));
      box-shadow: 0 0 30px rgba(110,231,249,.5);
    }
    .brand strong { letter-spacing: .03em; font-size: .98rem; }
    .brand span { color: var(--muted); font-size: .82rem; margin-left: 8px; }
    .live {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-weight: 700;
      font-size: .82rem;
    }
    .live-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 0 0 rgba(126,242,162,.8);
      animation: livePulse 1.7s infinite;
    }
    @keyframes livePulse { to { box-shadow: 0 0 0 12px rgba(126,242,162,0); } }
    .hero {
      position: relative;
      min-height: 520px;
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(380px, .8fr);
      gap: 22px;
      align-items: stretch;
      padding: 22px 0;
    }
    .world {
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 520px;
      background: #0b0f17;
      box-shadow: var(--shadow);
      isolation: isolate;
    }
    .world img {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      filter: saturate(1.04) contrast(1.04) brightness(.82);
      transform: scale(1.035);
      animation: cinematicDrift 14s ease-in-out infinite alternate;
    }
    @keyframes cinematicDrift {
      from { transform: scale(1.035) translate3d(-.6%, -.4%, 0); }
      to { transform: scale(1.075) translate3d(.9%, .7%, 0); }
    }
    .world::after {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(90deg, rgba(5,7,10,.58) 0%, rgba(5,7,10,.10) 42%, rgba(5,7,10,.70) 100%),
        linear-gradient(180deg, rgba(5,7,10,.08) 0%, rgba(5,7,10,.82) 100%);
      z-index: 1;
    }
    .world-copy {
      position: absolute;
      left: clamp(20px, 4vw, 54px);
      bottom: clamp(22px, 5vw, 58px);
      z-index: 2;
      width: min(660px, calc(100% - 40px));
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: rgba(255,255,255,.82);
      font-size: .78rem;
      font-weight: 800;
      letter-spacing: .14em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }
    .eyebrow::before {
      content: "";
      width: 30px;
      height: 1px;
      background: linear-gradient(90deg, var(--cyan), transparent);
    }
    h1 {
      margin: 0;
      font-size: clamp(2.6rem, 7vw, 6.6rem);
      line-height: .88;
      letter-spacing: 0;
      max-width: 850px;
    }
    .world-copy p {
      margin: 18px 0 0;
      max-width: 620px;
      color: rgba(245,247,251,.76);
      line-height: 1.55;
      font-size: clamp(.98rem, 1.5vw, 1.13rem);
    }
    .scanline {
      position: absolute;
      z-index: 2;
      left: 8%;
      right: 8%;
      bottom: 36%;
      height: 2px;
      background: linear-gradient(90deg, transparent, rgba(110,231,249,.78), rgba(126,242,162,.70), transparent);
      box-shadow: 0 0 22px rgba(110,231,249,.55);
      animation: scan 3.8s ease-in-out infinite alternate;
    }
    @keyframes scan { from { transform: translateY(-92px); opacity: .35; } to { transform: translateY(88px); opacity: .85; } }
    .path-dots {
      position: absolute;
      inset: 0;
      z-index: 2;
      pointer-events: none;
    }
    .path-dots i {
      position: absolute;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--red);
      box-shadow: 0 0 18px rgba(255,93,115,.72);
      animation: dotFloat 1.8s ease-in-out infinite alternate;
    }
    .path-dots i:nth-child(1) { left: 52%; top: 62%; animation-delay: 0s; }
    .path-dots i:nth-child(2) { left: 54%; top: 58%; animation-delay: .12s; }
    .path-dots i:nth-child(3) { left: 57%; top: 54%; animation-delay: .24s; background: var(--green); }
    .path-dots i:nth-child(4) { left: 60%; top: 50%; animation-delay: .36s; background: var(--green); }
    .path-dots i:nth-child(5) { left: 64%; top: 47%; animation-delay: .48s; background: var(--blue); }
    @keyframes dotFloat { to { transform: translateY(-8px); filter: brightness(1.25); } }
    .launch-pad {
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 14px;
      min-height: 520px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.045));
      backdrop-filter: blur(26px);
      box-shadow: var(--shadow);
      padding: 16px;
    }
    .launch-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--line-soft);
    }
    .launch-head h2 { margin: 0; font-size: 1rem; letter-spacing: 0; }
    .launch-head span { color: var(--muted); font-size: .82rem; font-weight: 700; }
    .form-grid { display: grid; gap: 12px; align-content: start; }
    .split { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
    .tri { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    label { display: grid; gap: 7px; color: rgba(245,247,251,.68); font-size: .74rem; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; }
    select, input {
      width: 100%;
      height: 46px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 12px;
      background: rgba(5,7,10,.54);
      color: var(--ink);
      outline: none;
      font-weight: 800;
      min-width: 0;
      transition: border-color .18s ease, background .18s ease, transform .18s ease;
    }
    select:focus, input:focus { border-color: rgba(110,231,249,.82); background: rgba(5,7,10,.78); }
    input:disabled { color: rgba(245,247,251,.44); }
    .route-strip {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 0 0 22px;
    }
    .route-card {
      position: relative;
      overflow: hidden;
      min-height: 158px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0c111b;
      cursor: pointer;
      box-shadow: 0 18px 54px rgba(0,0,0,.23);
      transition: transform .28s ease, border-color .28s ease, box-shadow .28s ease;
    }
    .route-card:hover { transform: translateY(-6px); border-color: rgba(110,231,249,.58); box-shadow: 0 28px 74px rgba(0,0,0,.36); }
    .route-card img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: .72; transition: transform .5s ease; }
    .route-card:hover img { transform: scale(1.07); }
    .route-card::after { content: ""; position: absolute; inset: 0; background: linear-gradient(180deg, transparent 12%, rgba(5,7,10,.88) 100%); }
    .route-card .copy { position: absolute; inset: auto 14px 14px; z-index: 1; }
    .route-card span { color: rgba(245,247,251,.65); font-size: .72rem; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; }
    .route-card strong { display: block; margin-top: 5px; font-size: 1.02rem; }
    .action-row { display: block; }
    .main-actions {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      align-items: stretch;
    }
    .main-actions .primary-action { grid-column: 1 / -1; }
    .go, .ghost {
      height: 56px;
      min-width: 0;
      border: 0;
      border-radius: 8px;
      cursor: pointer;
      color: #061018;
      font-weight: 950;
      font-size: .96rem;
      letter-spacing: 0;
      line-height: 1.12;
      padding: 0 10px;
      transition: transform .18s ease, filter .18s ease, opacity .18s ease;
    }
    .go { background: linear-gradient(135deg, var(--cyan), var(--green)); box-shadow: 0 14px 36px rgba(110,231,249,.28); }
    .ghost {
      color: var(--ink);
      background: rgba(255,255,255,.075);
      border: 1px solid var(--line);
      box-shadow: none;
    }
    .go:hover, .ghost:hover { transform: translateY(-2px); filter: brightness(1.06); }
    .go:active, .ghost:active { transform: translateY(1px); }
    .status {
      margin-top: 12px;
      min-height: 58px;
      display: flex;
      align-items: center;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 12px 14px;
      color: rgba(245,247,251,.80);
      background: rgba(5,7,10,.46);
      line-height: 1.4;
      font-size: .92rem;
    }
    .telemetry {
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }
    .metric {
      min-height: 92px;
      padding: 14px;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: rgba(255,255,255,.045);
      backdrop-filter: blur(18px);
    }
    .metric span { display: block; color: var(--muted); font-size: .72rem; font-weight: 850; letter-spacing: .08em; text-transform: uppercase; }
    .metric strong { display: block; margin-top: 9px; color: var(--ink); font-size: clamp(1rem, 1.6vw, 1.55rem); line-height: 1.06; word-break: break-word; }
    .world-bands {
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 18px;
      align-items: stretch;
    }
    .mirror, .briefing {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--glass);
      backdrop-filter: blur(18px);
      overflow: hidden;
    }
    .mirror img { display: block; width: 100%; height: 260px; object-fit: cover; filter: saturate(1.05) contrast(1.05); }
    .briefing { padding: 18px; }
    .briefing h3 { margin: 0 0 12px; font-size: 1.05rem; letter-spacing: 0; }
    .briefing p { margin: 0; color: rgba(245,247,251,.70); line-height: 1.55; }
    .pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
    .pill { border: 1px solid var(--line); border-radius: 999px; color: rgba(245,247,251,.78); padding: 8px 11px; font-size: .8rem; font-weight: 800; background: rgba(255,255,255,.06); }
    .compare-panel {
      margin-top: 22px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.035));
      backdrop-filter: blur(18px);
      box-shadow: 0 24px 70px rgba(0,0,0,.26);
      overflow: hidden;
    }
    .compare-head {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      padding: 18px;
      border-bottom: 1px solid var(--line-soft);
    }
    .compare-head h3 { margin: 0; font-size: 1.08rem; letter-spacing: 0; }
    .compare-head p { margin: 6px 0 0; color: rgba(245,247,251,.67); line-height: 1.45; max-width: 820px; }
    .compare-refresh {
      min-width: 132px;
      height: 42px;
      border: 1px solid rgba(110,231,249,.42);
      border-radius: 8px;
      color: var(--ink);
      background: rgba(110,231,249,.10);
      cursor: pointer;
      font-weight: 900;
    }
    .compare-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      padding: 14px;
    }
    .compare-card {
      min-width: 0;
      min-height: 280px;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: rgba(5,7,10,.42);
      padding: 14px;
      display: grid;
      gap: 12px;
      align-content: start;
      overflow: hidden;
    }
    .compare-card h4 { margin: 0; font-size: 1rem; line-height: 1.25; overflow-wrap: anywhere; }
    .compare-sub {
      min-width: 0;
      color: var(--muted);
      font-size: .78rem;
      font-weight: 850;
      text-transform: uppercase;
      letter-spacing: .07em;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .compare-state {
      display: inline-flex;
      width: max-content;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 9px;
      color: rgba(245,247,251,.78);
      background: rgba(255,255,255,.055);
      font-size: .78rem;
      font-weight: 900;
    }
    .compare-headline {
      min-width: 0;
      font-size: clamp(1.28rem, 2.1vw, 1.82rem);
      line-height: 1.08;
      font-weight: 950;
      color: var(--cyan);
      overflow-wrap: anywhere;
    }
    .score-list { display: grid; gap: 8px; }
    .score-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(72px, 42%);
      gap: 10px;
      align-items: baseline;
      border-bottom: 1px solid var(--line-soft);
      padding-bottom: 7px;
      color: rgba(245,247,251,.72);
      font-size: .86rem;
      min-width: 0;
    }
    .score-row span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .score-row strong {
      color: var(--ink);
      font-size: .92rem;
      white-space: nowrap;
      justify-self: end;
      font-variant-numeric: tabular-nums;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .compare-note {
      min-width: 0;
      color: rgba(245,247,251,.64);
      line-height: 1.45;
      margin: 0;
      font-size: .87rem;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .kpi-table-wrap { padding: 0 14px 14px; overflow-x: auto; }
    .kpi-table {
      width: 100%;
      min-width: 1260px;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 8px;
      border: 1px solid var(--line-soft);
    }
    .kpi-table th, .kpi-table td {
      padding: 12px;
      border-bottom: 1px solid var(--line-soft);
      text-align: left;
      vertical-align: top;
      font-size: .88rem;
      line-height: 1.35;
    }
    .kpi-table th {
      color: rgba(245,247,251,.78);
      background: rgba(255,255,255,.06);
      font-size: .75rem;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .kpi-table td:first-child { color: var(--muted); font-weight: 900; }
    .observed-runs {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      padding: 0 14px 14px;
    }
    .observed-card {
      min-width: 0;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: rgba(5,7,10,.30);
      overflow: hidden;
    }
    .observed-card h4 {
      margin: 0;
      padding: 12px;
      border-bottom: 1px solid var(--line-soft);
      font-size: .88rem;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .observed-list {
      display: grid;
      max-height: 250px;
      overflow: auto;
    }
    .observed-row {
      display: grid;
      grid-template-columns: 78px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line-soft);
      font-size: .78rem;
      line-height: 1.35;
      color: rgba(245,247,251,.70);
      min-width: 0;
    }
    .observed-row strong {
      color: var(--ink);
      font-size: .8rem;
      overflow-wrap: anywhere;
    }
    .observed-row span {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .observed-badge {
      display: inline-flex;
      width: 72px;
      min-height: 24px;
      padding: 0 6px;
      align-items: center;
      justify-content: center;
      border-radius: 12px;
      border: 1px solid rgba(110,231,249,.28);
      color: var(--cyan);
      background: rgba(110,231,249,.08);
      font-size: .68rem;
      font-weight: 950;
      text-transform: uppercase;
      letter-spacing: .05em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .observed-badge.unscored {
      border-color: rgba(251,191,36,.32);
      color: var(--amber);
      background: rgba(251,191,36,.09);
    }
    .runme-kpis {
      margin: 0;
      padding: 0 18px 18px;
      color: rgba(245,247,251,.68);
      display: grid;
      gap: 7px;
      font-size: .86rem;
    }
    .runme-kpis li { list-style: none; }
    .runme-kpis li::before { content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--green); margin-right: 9px; }
    @media (max-width: 1100px) {
      .hero, .world-bands { grid-template-columns: 1fr; }
      .launch-pad { min-height: auto; }
      .telemetry { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .compare-grid { grid-template-columns: 1fr; }
      .observed-runs { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      .shell { padding: 14px; }
      .topbar { height: auto; min-height: 58px; align-items: flex-start; flex-direction: column; padding: 12px; }
      .hero { padding-top: 14px; }
      .world { min-height: 460px; }
      .split, .tri, .route-strip, .main-actions, .action-row, .telemetry { grid-template-columns: 1fr; }
      h1 { font-size: clamp(2.4rem, 15vw, 4.4rem); }
    }
  </style>
</head>
<body>
  <main class="shell">
    <nav class="topbar">
      <div class="brand">
        <div class="mark"></div>
        <strong>VLA-AV SimLingo World</strong>
        <span>CARLA / Bench2Drive / SUMO</span>
      </div>
      <div class="live"><span class="live-dot"></span><span id="topStatus">idle</span></div>
    </nav>

    <section class="hero">
      <div class="world" id="world">
        <img id="heroImage" src="/assets/simlingo_teaser.png" alt="SimLingo CARLA scene">
        <div class="scanline"></div>
        <div class="path-dots"><i></i><i></i><i></i><i></i><i></i></div>
        <div class="world-copy">
          <div class="eyebrow" id="sceneEyebrow">Town12 / Traffic Light</div>
          <h1 id="sceneTitle">Choose The World.</h1>
          <p id="sceneText">Native SimLingo closed-loop driving with optional SUMO mirror, cinematic POV, route waypoints and replay capture.</p>
        </div>
      </div>

      <aside class="launch-pad">
        <div class="launch-head">
          <div>
            <h2>Mission Control</h2>
            <span id="missionMeta">route 08 / SUMO mirror ready</span>
          </div>
          <span id="mProcess">idle</span>
        </div>

        <div class="form-grid controls">
          <div class="split">
            <label>Map<select id="town"></select></label>
            <label>Scenario<select id="scenario">
              <option value="any">All native scenarios</option>
              <option value="vru">VRU / crossing</option>
              <option value="light">Traffic light</option>
              <option value="stop">Stop</option>
              <option value="junction">Junction</option>
              <option value="accident">Accident</option>
              <option value="cut_in">Cut-in / parking</option>
              <option value="actor_flow">Actor flow</option>
            </select></label>
          </div>
          <label>Route<select id="route"></select></label>
          <div class="split">
            <label>Launch mode<select id="run_mode">
              <option value="sumo_mirror">CARLA POV + SUMO mirror</option>
              <option value="action_dreaming">CARLA POV + Action Dreaming collect</option>
              <option value="pov">CARLA POV only</option>
            </select></label>
            <label>SUMO GUI<select id="sumo_mirror_gui">
              <option value="1">Open SUMO 2D window</option>
              <option value="0">Headless mirror logs</option>
            </select></label>
          </div>
          <div class="split">
            <label>Dreamer mode<select id="dreamer_mode">
              <option value="off">Off - native SimLingo</option>
              <option value="dreamer_ppo">Dreamer PPO</option>
              <option value="dreamer_sdbs">Dreamer SDBS</option>
              <option value="dreamer_ppo_rl_noguard">Dreamer PPO RL no guard</option>
              <option value="dreamer_sdbs_rl_noguard">Dreamer SDBS RL no guard</option>
            </select></label>
            <label>Dreamer overlay<input type="text" value="Pygame live panel + replay" disabled></label>
          </div>
          <div class="split">
            <label>External CoT<select id="cot_mode">
              <option value="off">Off</option>
              <option value="mock">Mock panel test</option>
              <option value="qwen2_vl">Qwen local VLM-CoT</option>
            </select></label>
            <label>CoT interval<select id="cot_interval">
              <option value="2.0">Every 2s</option>
              <option value="1.0">Every 1s</option>
              <option value="4.0">Every 4s</option>
            </select></label>
          </div>
          <div class="split">
            <label>Seed<input id="seed" type="number" min="1" placeholder="random"></label>
            <label>CARLA quality<select id="quality"><option>Epic</option><option>Low</option></select></label>
          </div>
          <div class="split">
            <label>POV Pygame<select id="camera"><option value="chase">Chase</option><option value="wheel">Wheel</option><option value="front">Front</option><option value="top">Top</option></select></label>
            <label>Resolution<select id="video_quality"><option value="epic">Epic 1080p</option><option value="hd">HD 720p</option><option value="low">Fast 960p</option></select></label>
          </div>
          <div class="split">
            <label>Visual weather<select id="visual_weather"><option value="day">Day</option><option value="soft">Soft clouds</option><option value="sunset">Sunset</option><option value="route">Route weather</option></select></label>
            <label>Max FPS<input id="view_fps" type="number" min="15" max="60" value="45"></label>
          </div>
          <div class="split">
            <label>CARLA traffic lights<select id="traffic_light_overlay">
              <option value="1">Show state badges</option>
              <option value="0">Hide badges</option>
            </select></label>
            <label>Light range<select id="traffic_light_overlay_distance">
              <option value="160">160 m</option>
              <option value="100">100 m</option>
              <option value="220">220 m</option>
            </select></label>
          </div>
          <div class="split">
            <label>VLA prompt<select id="prompt_mode"><option value="native">Native benchmark prompt</option><option value="obstacle">Action Dreaming obstacle demo</option></select></label>
            <label>Replay speed<select id="playback_speed"><option value="5">x5</option><option value="4">x4</option><option value="3">x3</option><option value="8">x8</option><option value="50">x50</option></select></label>
          </div>
          <div class="tri">
            <label>Cars<input type="text" value="route XML" disabled></label>
            <label>Walkers<input type="text" value="route XML" disabled></label>
            <label>Scooters<input type="text" value="route XML" disabled></label>
          </div>
        </div>

        <div class="action-row">
          <div class="main-actions">
            <button class="go primary-action" id="go">Launch</button>
            <button class="ghost" id="stop">Stop</button>
            <button class="ghost" id="replay">Replay</button>
            <button class="ghost" id="twinsentinel">TwinSentinel attacks</button>
          </div>
        </div>
        <div class="status" id="status">Ready.</div>
      </aside>
    </section>

    <section class="route-strip">
      <button class="route-card" data-scenario="light" data-town="Town12">
        <img src="/assets/simlingo_thumbnail.png" alt="CARLA city route">
        <span class="copy"><span>Signalized</span><strong>Town12 Traffic Flow</strong></span>
      </button>
      <button class="route-card" data-scenario="accident" data-town="Town12">
        <img src="/assets/bench2drive_overview.jpg" alt="Bench2Drive overview">
        <span class="copy"><span>Red-team</span><strong>Accident Response</strong></span>
      </button>
      <button class="route-card" data-scenario="vru" data-town="Town13">
        <img src="/assets/carla_header.png" alt="CARLA map teaser">
        <span class="copy"><span>VRU</span><strong>Crossing Scenario</strong></span>
      </button>
    </section>

    <section class="telemetry">
      <div class="metric"><span>Native routes</span><strong id="mCatalog">-</strong></div>
      <div class="metric"><span>Route</span><strong id="mRoute">-</strong></div>
      <div class="metric"><span>Town</span><strong id="mTown">-</strong></div>
      <div class="metric"><span>Scenario</span><strong id="mScenario">-</strong></div>
      <div class="metric"><span>Mode</span><strong id="mMode">-</strong></div>
      <div class="metric"><span>Dreamer</span><strong id="mDreamer">off</strong></div>
      <div class="metric"><span>External CoT</span><strong id="mCot">off</strong></div>
      <div class="metric"><span>Seed</span><strong id="mSeed">-</strong></div>
    </section>

    <section class="world-bands">
      <div class="mirror">
        <img src="/assets/bench2drive_benchmark.jpg" alt="Bench2Drive benchmark map">
      </div>
      <div class="briefing">
        <h3 id="briefingTitle">Closed-loop baseline</h3>
        <p id="briefingText">SimLingo remains the CARLA driver. SUMO mirror gives the 2D traffic view while Bench2Drive keeps the route, actors, criteria and scoring pipeline aligned.</p>
        <div class="pills">
          <span class="pill">Waypoints overlay</span>
          <span class="pill">Native VLA</span>
          <span class="pill">Dreamer PPO/SDBS</span>
          <span class="pill">External CoT</span>
          <span class="pill">SUMO GUI</span>
          <span class="pill">Replay capture</span>
        </div>
      </div>
    </section>

    <section class="compare-panel" aria-label="Dreamer comparison window">
      <div class="compare-head">
        <div>
          <h3>SAFE-DREAM KPI Comparison</h3>
          <p>Same Family D/E metrics for native SimLingo, guarded Dreamer PPO/SDBS, and upcoming RL no-guard variants.</p>
        </div>
        <button class="compare-refresh" id="refreshCompare">Refresh KPIs</button>
      </div>
      <div class="compare-grid" id="compareCards"></div>
      <div class="kpi-table-wrap">
        <table class="kpi-table">
          <thead>
            <tr>
              <th>KPI</th>
              <th>SimLingo native</th>
              <th>Dreamer PPO</th>
              <th>Dreamer SDBS</th>
              <th>PPO RL no guard</th>
              <th>SDBS RL no guard</th>
            </tr>
          </thead>
          <tbody id="compareRows">
            <tr><td colspan="6">Loading comparison...</td></tr>
          </tbody>
        </table>
      </div>
      <div class="observed-runs" id="observedRuns"></div>
      <ul class="runme-kpis" id="runmeKpis"></ul>
    </section>
  </main>

  <script>
    let routes = [];
    const $ = id => document.getElementById(id);
    const sceneMap = {
      any: {
        eyebrow: "Bench2Drive / Native Routes",
        title: "Choose The World.",
        text: "Native SimLingo closed-loop driving with optional SUMO mirror, cinematic POV, route waypoints and replay capture.",
        image: "/assets/simlingo_teaser.png"
      },
      light: {
        eyebrow: "Signalized Route / Red-team Ready",
        title: "Read The Light.",
        text: "Traffic-light scenarios keep the same route while the environment can later be perturbed through CARLA/SUMO experiments.",
        image: "/assets/simlingo_teaser.png"
      },
      accident: {
        eyebrow: "Blocked Lane / Decision Point",
        title: "Go Around It.",
        text: "Accident routes are the best visual demo for trajectory prediction, target points and future red-team evaluation.",
        image: "/assets/bench2drive_overview.jpg"
      },
      vru: {
        eyebrow: "VRU / Crossing Flow",
        title: "Yield Or Move.",
        text: "Pedestrian and bicycle flows expose the VLA to dynamic agents, occlusions and safety-critical timing.",
        image: "/assets/carla_header.png"
      },
      stop: {
        eyebrow: "Stopsign / Rule Context",
        title: "Stop Clean.",
        text: "Stop scenarios stress rule compliance, route following and low-speed control.",
        image: "/assets/simlingo_thumbnail.png"
      },
      junction: {
        eyebrow: "Junction / Route Intent",
        title: "Commit To The Turn.",
        text: "Junction routes surface target-point following, lane selection and route conditioning.",
        image: "/assets/bench2drive_benchmark.jpg"
      },
      cut_in: {
        eyebrow: "Cut-in / Parking",
        title: "Hold The Gap.",
        text: "Cut-in scenarios pressure the model with near-field vehicles and sharp control transitions.",
        image: "/assets/bench2drive_overview.jpg"
      },
      actor_flow: {
        eyebrow: "Actor Flow / Dense Traffic",
        title: "Sync The Flow.",
        text: "Actor-flow routes are good stress tests for SUMO mirror visualization and traffic robustness.",
        image: "/assets/simlingo_teaser.png"
      }
    };
    async function api(path, opts) {
      const r = await fetch(path, opts);
      if (!r.ok) throw new Error((await r.json()).error || await r.text());
      return r.json();
    }
    function esc(value) {
      return String(value ?? "-")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
    async function loadDreamerComparison() {
      const data = await api("/api/dreamer-comparison");
      $("compareCards").innerHTML = data.cards.map(card => `
        <article class="compare-card">
          <span class="compare-sub">${esc(card.subtitle)}</span>
          <h4>${esc(card.name)}</h4>
          <span class="compare-state">${esc(card.status)}</span>
          <div class="compare-headline">${esc(card.headline)}</div>
          <div class="score-list">
            ${(card.metrics || []).map(m => `
              <div class="score-row"><span>${esc(m.label)}</span><strong>${esc(m.value)}</strong></div>
            `).join("")}
          </div>
          <p class="compare-note">${esc(card.note)}</p>
        </article>
      `).join("");
      $("compareRows").innerHTML = data.rows.map(row => `
        <tr>
          <td>${esc(row.label)}</td>
          <td>${esc(row.simlingo)}</td>
          <td>${esc(row.dreamer_v1)}</td>
          <td>${esc(row.dreamer_sdbs)}</td>
          <td>${esc(row.dreamer_ppo_rl)}</td>
          <td>${esc(row.dreamer_sdbs_rl)}</td>
        </tr>
      `).join("");
      $("observedRuns").innerHTML = (data.all_runs || []).map(group => `
        <article class="observed-card">
          <h4>${esc(group.name)} - all observed runs</h4>
          <div class="observed-list">
            ${(group.runs || []).length ? group.runs.map(run => `
              <div class="observed-row">
                <span class="observed-badge ${run.scored ? "" : "unscored"}">${esc(run.scored ? "scored" : run.status)}</span>
                <span>
                  <strong>${esc(run.route)} / seed ${esc(run.seed)}</strong><br>
                  ${esc(run.variant)}<br>
                  ${esc(run.file)}
                </span>
              </div>
            `).join("") : `<div class="observed-row"><span class="observed-badge unscored">none</span><span>No run found for this model.</span></div>`}
          </div>
        </article>
      `).join("");
      $("runmeKpis").innerHTML = data.runme_kpis.map(item => `<li>${esc(item)}</li>`).join("");
    }
    function scenarioMatch(r, scen) {
      if (scen === "vru") return r.vru;
      if (scen === "light") return r.traffic_light;
      if (scen === "stop") return r.stop;
      if (scen === "junction") return r.junction;
      if (scen === "accident") return r.accident;
      if (scen === "cut_in") return r.cut_in;
      if (scen === "actor_flow") return r.actor_flow;
      return true;
    }
    function filteredRoutes() {
      const town = $("town").value, scen = $("scenario").value;
      return routes.filter(r => r.compatible && (town === "any" || r.town === town) && scenarioMatch(r, scen));
    }
    function updateScene() {
      const key = $("scenario")?.value || "any";
      const scene = sceneMap[key] || sceneMap.any;
      $("sceneEyebrow").textContent = `${$("town")?.value || "Any map"} / ${scene.eyebrow}`;
      $("sceneTitle").textContent = scene.title;
      $("sceneText").textContent = scene.text;
      $("heroImage").src = scene.image;
      const selected = routes.find(r => r.id === $("route").value);
      $("missionMeta").textContent = selected ? `route ${selected.id} / ${selected.town}` : "random compatible route";
      $("briefingTitle").textContent = key === "any" ? "Closed-loop baseline" : `${$("scenario").selectedOptions[0].textContent}`;
    }
    function updateRouteOptions() {
      const filtered = filteredRoutes();
      $("route").innerHTML = `<option value="random">Random compatible (${filtered.length})</option>` +
        filtered.map(r => `<option value="${r.id}">${r.id} | ${r.town} | ${r.scenario_type}</option>`).join("");
      if (filtered.some(r => r.id === "08")) $("route").value = "08";
      updateScene();
    }
    async function loadRoutes() {
      const data = await api("/api/routes");
      routes = data.routes;
      const compatible = routes.filter(r => r.compatible);
      const counts = compatible.reduce((acc, r) => {
        acc[r.town] = (acc[r.town] || 0) + 1;
        return acc;
      }, {});
      const towns = Object.keys(counts).sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
      $("town").innerHTML = [`<option value="any">All native maps (${compatible.length})</option>`]
        .concat(towns.map(t => `<option value="${t}">${t} (${counts[t]} routes)</option>`))
        .join("");
      if (towns.includes("Town12")) $("town").value = "Town12";
      $("scenario").value = "light";
      $("mCatalog").textContent = `${compatible.length} routes / ${towns.length} maps`;
      updateRouteOptions();
    }
    async function start(modeOverride) {
      const payload = {
        town: $("town").value,
        scenario: $("scenario").value,
        route_id: $("route").value,
        seed: $("seed").value || Math.floor(Math.random() * 999999) + 1,
        quality: $("quality").value,
        camera: $("camera").value,
        video_quality: $("video_quality").value,
        visual_weather: $("visual_weather").value,
        prompt_mode: $("prompt_mode").value,
        playback_speed: $("playback_speed").value,
        run_mode: modeOverride || $("run_mode").value,
        dreamer_mode: $("dreamer_mode").value,
        cot_mode: $("cot_mode").value,
        cot_interval: $("cot_interval").value,
        cot_model: "Qwen/Qwen2-VL-7B-Instruct",
        cot_local_only: "1",
        sumo_mirror_gui: $("sumo_mirror_gui").value,
        sumo_mirror_sync_tls: "1",
        traffic_light_overlay: $("traffic_light_overlay").value,
        traffic_light_overlay_distance: $("traffic_light_overlay_distance").value,
        traffic_light_overlay_max: "80",
        action_dreaming_sample_interval: "0.25",
        action_dreaming_k: "5",
        view_fps: $("view_fps").value || 45,
        port: 2000,
        tm_port: 8000
      };
      const data = await api("/api/start", {method:"POST", body:JSON.stringify(payload)});
      const modeText = payload.run_mode === "sumo_mirror"
        ? "CARLA POV + SUMO GUI"
        : (payload.run_mode === "action_dreaming" ? "CARLA POV + Action Dreaming collect" : "CARLA POV");
      const dreamerLabels = {
        off: "native SimLingo",
        dreamer_ppo: "Dreamer PPO",
        dreamer_sdbs: "Dreamer SDBS",
        dreamer_ppo_rl_noguard: "Dreamer PPO RL no guard",
        dreamer_sdbs_rl_noguard: "Dreamer SDBS RL no guard"
      };
      const dreamerText = dreamerLabels[payload.dreamer_mode] || payload.dreamer_mode;
      const cotText = payload.cot_mode === "off" ? "CoT off" : `CoT ${payload.cot_mode}`;
      $("status").textContent = `Launching ${modeText} / ${dreamerText} / ${cotText}: route ${data.route.id}, seed ${data.seed}.`;
      refreshStatus();
    }
    async function stopRun() {
      await api("/api/stop", {method:"POST"});
      $("status").textContent = "Stopped.";
      refreshStatus();
    }
    async function replayLatest() {
      const data = await api("/api/replay", {
        method:"POST",
        body:JSON.stringify({playback_speed: $("playback_speed").value || "5"})
      });
      $("status").textContent = `Replay x${data.speed}: ${data.video}`;
    }
    async function openTwinSentinel() {
      const data = await api("/api/twinsentinel/start", {method:"POST"});
      $("status").textContent = `TwinSentinel attack console ready: ${data.url}`;
      window.open(data.url, "_blank");
    }
    async function refreshStatus() {
      const data = await api("/api/status");
      $("mRoute").textContent = data.route || "-";
      $("mTown").textContent = data.route_town || "-";
      $("mScenario").textContent = data.scenario || "-";
      $("mMode").textContent = data.mode === "sumo_mirror"
        ? "CARLA + SUMO"
        : (data.mode === "action_dreaming" ? "Action Dreaming" : (data.mode || "-"));
      $("mDreamer").textContent = data.dreamer_mode || "off";
      $("mCot").textContent = data.cot_mode || "off";
      $("mSeed").textContent = data.seed || "-";
      $("mProcess").textContent = data.running ? "running" : "idle";
      $("topStatus").textContent = data.running ? "simulation running" : "idle";
      if (data.last_error) $("status").textContent = data.last_error;
    }
    $("town").onchange = updateRouteOptions;
    $("scenario").onchange = updateRouteOptions;
    $("route").onchange = updateScene;
    $("go").onclick = () => start().catch(e => $("status").textContent = e.message);
    $("stop").onclick = () => stopRun().catch(e => $("status").textContent = e.message);
    $("replay").onclick = () => replayLatest().catch(e => $("status").textContent = e.message);
    $("twinsentinel").onclick = () => openTwinSentinel().catch(e => $("status").textContent = e.message);
    $("refreshCompare").onclick = () => loadDreamerComparison().catch(e => $("status").textContent = e.message);
    document.querySelectorAll(".route-card").forEach(card => {
      card.addEventListener("click", () => {
        const town = card.dataset.town;
        const scenario = card.dataset.scenario;
        if ([...$("town").options].some(o => o.value === town)) $("town").value = town;
        $("scenario").value = scenario;
        updateRouteOptions();
        window.scrollTo({top: 0, behavior: "smooth"});
      });
    });
    window.addEventListener("pointermove", e => {
      const x = (e.clientX / Math.max(1, window.innerWidth) - .5) * 10;
      const y = (e.clientY / Math.max(1, window.innerHeight) - .5) * 10;
      $("world").style.transform = `perspective(1200px) rotateY(${x * .16}deg) rotateX(${-y * .12}deg)`;
    });
    loadRoutes().catch(e => $("status").textContent = e.message);
    loadDreamerComparison().catch(e => $("status").textContent = e.message);
    setInterval(refreshStatus, 2000);
    refreshStatus();
  </script>
</body>
</html>
"""


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/assets/"):
            name = Path(path).name
            asset_path = ASSET_FILES.get(name)
            if asset_path and asset_path.exists() and asset_path.is_file():
                body = asset_path.read_bytes()
                content_type = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)
            return
        if path == "/api/routes":
            self._json({
                "routes": route_catalog(),
                "installed_towns": sorted(installed_towns()),
                "stable_towns": sorted(STABLE_TOWNS),
                "show_experimental": SHOW_EXPERIMENTAL_TOWNS,
            })
            return
        if path == "/api/status":
            with STATE_LOCK:
                proc = STATE.get("process")
                exit_code = proc.poll() if proc else None
                running = bool(proc and exit_code is None)
                if proc and exit_code not in (None, 0) and not STATE.get("last_error"):
                    STATE["last_error"] = (
                        f"Simulation exited with code {exit_code}. "
                        f"Launch log: {STATE.get('launch_log') or 'unavailable'}"
                    )
                payload = {k: v for k, v in STATE.items() if k != "process"}
            payload["running"] = running
            payload["exit_code"] = exit_code
            self._json(payload)
            return
        if path == "/api/dreamer-comparison":
            self._json(dreamer_comparison_payload())
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        payload = {}
        if length:
            payload = json.loads(self.rfile.read(length).decode())
        path = urlparse(self.path).path
        try:
            if path == "/api/start":
                self._json(start_run(payload))
                return
            if path == "/api/stop":
                stop_current(kill_carla=True)
                self._json({"ok": True})
                return
            if path == "/api/replay":
                self._json(replay_latest(payload))
                return
            if path == "/api/twinsentinel/start":
                self._json(start_twinsentinel_console())
                return
            self.send_error(404)
        except Exception as exc:
            with STATE_LOCK:
                STATE["last_error"] = str(exc)
            self._json({"ok": False, "error": str(exc)}, status=400)

    def log_message(self, fmt, *args):
        return


def main():
    port = int(os.environ.get("SIMLINGO_DASHBOARD_PORT", "8765"))
    server = ReusableThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    (LOG_DIR / "dashboard_url.txt").write_text(url + "\n")
    print(f"[simlingo-dashboard] {url}", flush=True)
    try:
        while True:
            try:
                server.serve_forever()
                break
            except KeyboardInterrupt:
                with STATE_LOCK:
                    proc = STATE.get("process")
                    running = bool(proc and proc.poll() is None)
                if running:
                    print(
                        "\n[simlingo-dashboard] Ctrl-C caught: stopping current simulation only. "
                        f"Dashboard stays alive at {url}",
                        flush=True,
                    )
                    stop_current(kill_carla=True)
                    with STATE_LOCK:
                        STATE["last_error"] = "Simulation stopped from terminal. Dashboard still running."
                    continue
                print("\n[simlingo-dashboard] No simulation running; exiting dashboard.", flush=True)
                break
    finally:
        stop_current(kill_carla=False)


if __name__ == "__main__":
    main()
