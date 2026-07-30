#!/usr/bin/env python3
"""Collect lightweight Dreamer/SimLingo decision traces during a live run.

This watcher is intentionally passive: it does not touch CARLA controls. It
samples the Dreamer status JSON written by the SimLingo agent/viewer and appends
time-ordered rows that can later be filtered into recovery/overtake examples.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Any, Dict


def read_json(path: pathlib.Path) -> Dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--route-id", default=os.environ.get("ROUTE_ID", ""))
    parser.add_argument("--route-file", default=os.environ.get("ROUTE_FILE", ""))
    parser.add_argument("--town", default=os.environ.get("TOWN", ""))
    parser.add_argument("--seed", default=os.environ.get("SEED", ""))
    args = parser.parse_args()

    status_path = pathlib.Path(args.status_path)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen_signature = None
    with output.open("a", encoding="utf-8") as handle:
        while True:
            payload = read_json(status_path)
            if payload:
                signature = (
                    payload.get("timestamp"),
                    payload.get("candidate_index"),
                    payload.get("applied"),
                    payload.get("chosen_kind"),
                    tuple(sorted((payload.get("chosen_action") or {}).items())),
                )
                if signature != seen_signature:
                    seen_signature = signature
                    row = {
                        "collector_time": time.time(),
                        "route_id": args.route_id,
                        "route_file": args.route_file,
                        "town": args.town,
                        "seed": args.seed,
                        "status": payload,
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
            time.sleep(max(0.05, args.interval))


if __name__ == "__main__":
    main()
