#!/usr/bin/env python3
"""Build a self-contained, read-only dashboard snapshot for Streamlit Cloud."""

import argparse
import base64
import importlib.util
import json
import math
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_SOURCE = ROOT / "scripts" / "simlingo_dashboard.py"
DEFAULT_OUTPUT = ROOT / "streamlit_share" / "dashboard_snapshot.html"


def load_dashboard_module():
    os.environ["SIMLINGO_DASHBOARD_READ_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location("simlingo_dashboard_snapshot", DASHBOARD_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {DASHBOARD_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value.replace(str(ROOT), "<project>").replace(str(Path.home()), "~")
    return value


def asset_data_url(path):
    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_snapshot(module):
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    routes = [
        {key: value for key, value in route.items() if key != "file"}
        for route in module.route_catalog()
    ]
    installed = sorted({route["town"] for route in routes if route.get("installed")})
    snapshot = {
        "/api/config": {
            "read_only": True,
            "snapshot_generated_at": generated_at,
        },
        "/api/routes": {
            "routes": routes,
            "installed_towns": installed,
            "stable_towns": sorted(module.STABLE_TOWNS),
            "show_experimental": module.SHOW_EXPERIMENTAL_TOWNS,
        },
        "/api/status": {
            "running": False,
            "route": "snapshot",
            "route_town": "-",
            "scenario": "presentation",
            "mode": "read-only",
            "dreamer_mode": "snapshot",
            "cot_mode": "off",
            "seed": "-",
            "online_rl_enabled": False,
            "last_error": None,
        },
        "/api/dreamer-comparison": module.dreamer_comparison_payload(),
    }
    snapshot = json_safe(snapshot)

    snapshot_json = json.dumps(
        snapshot,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")

    html = module.HTML
    api_source = """    async function api(path, opts) {
      const r = await fetch(path, opts);
      if (!r.ok) throw new Error((await r.json()).error || await r.text());
      return r.json();
    }"""
    api_replacement = """    const presentationSnapshot = JSON.parse(
      document.getElementById("presentationSnapshot").textContent
    );
    async function api(path, opts) {
      const method = String((opts || {}).method || "GET").toUpperCase();
      if (method !== "GET") {
        throw new Error("Read-only presentation: server actions are disabled.");
      }
      if (!(path in presentationSnapshot)) {
        throw new Error(`Snapshot endpoint unavailable: ${path}`);
      }
      return JSON.parse(JSON.stringify(presentationSnapshot[path]));
    }"""
    if api_source not in html:
        raise RuntimeError("Dashboard API function changed; snapshot exporter must be updated.")
    html = html.replace(api_source, api_replacement, 1)

    script_marker = "  <script>\n    let routes = [];"
    script_injection = (
        f'  <script id="presentationSnapshot" type="application/json">{snapshot_json}</script>\n'
        "  <script>\n    let routes = [];"
    )
    if script_marker not in html:
        raise RuntimeError("Dashboard script marker not found.")
    html = html.replace(script_marker, script_injection, 1)

    banner_text = (
        "Read-only dashboard: configuration and KPIs are visible, execution controls are locked "
        f"server-side. Snapshot: {generated_at}."
    )
    html = html.replace(
        "Read-only dashboard: configuration and KPIs are visible, execution controls are locked server-side.",
        banner_text,
        1,
    )

    for name, path in module.ASSET_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Dashboard asset missing: {path}")
        html = html.replace(f"/assets/{name}", asset_data_url(path))

    return html, generated_at, len(routes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    module = load_dashboard_module()
    html, generated_at, route_count = build_snapshot(module)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"[streamlit-share] snapshot={output}")
    print(f"[streamlit-share] generated_at={generated_at}")
    print(f"[streamlit-share] routes={route_count} size_bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
