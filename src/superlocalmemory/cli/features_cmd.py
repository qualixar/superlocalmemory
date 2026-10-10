# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm features`` - what is on and what you can turn on, asked of the running daemon."""

from __future__ import annotations

import sys
from argparse import Namespace
from typing import Any, NoReturn

from superlocalmemory.cli.daemon import daemon_request

FEATURES_PATH = "/api/v3/features"
NOT_RUNNING = "The SLM daemon is not running. Start it with: slm restart"
EXIT_DAEMON_DOWN = 3


def die(args: Namespace, message: str, code: int) -> NoReturn:
    if getattr(args, "json", False):
        from superlocalmemory.cli.json_output import json_print

        json_print(getattr(args, "_command", "features"), error={"message": message})
    else:
        print(message)
    sys.exit(code)


def media_line(media: dict[str, Any]) -> str:
    if media.get("enabled"):
        state = media.get("env_state", "")
        if media.get("restart_required"):
            return "on, ready - restart to start using it (slm restart)"
        if state == "ready":
            return "on"
        step = str(media.get("step") or "").strip()
        if state == "unsupported":
            return "on, but images & documents can't be set up on this computer yet" + (f" ({step})" if step else "")
        if state == "failed":
            return "on, but set-up failed" + (f" ({step})" if step else "") + " - see: slm doctor"
        pct = int(float(media.get("progress") or 0) * 100)
        return f"on, setting up ({state}, {pct}%) {media.get('step', '')}".rstrip()
    if media.get("requested"):
        return "requested by the installer - it starts the next time SLM starts"
    return "off - turn on with: slm media enable"


def render(data: dict[str, Any]) -> str:
    media = data.get("media", {})
    lines = [f"Images & documents: {media_line(media)}"]
    if media.get("error"):
        lines.append(f"  problem: {media['error']}")
    lines.append(f"Mesh: {data.get('mesh', {}).get('apps_with_mesh', 0)} app(s) connected")
    lines.append("Read more: slm media status, slm doctor")
    return "\n".join(lines)


def fetch(args: Namespace) -> dict[str, Any]:
    data = daemon_request("GET", FEATURES_PATH)
    if data is None:
        die(args, NOT_RUNNING, EXIT_DAEMON_DOWN)
    return data


def running_daemon_media() -> dict[str, Any]:
    """The daemon's own view of the media feature, or {} when it cannot be asked."""
    try:
        data = daemon_request("GET", FEATURES_PATH, timeout_seconds=3.0)
        return dict((data or {}).get("media") or {})
    except Exception:  # noqa: BLE001 - a diagnostic must not fail because the daemon is away
        return {}


def cmd_features(args: Namespace) -> None:
    data = fetch(args)
    if getattr(args, "json", False):
        from superlocalmemory.cli.json_output import json_print

        json_print("features", data=data)
    else:
        print(render(data))


def register_features_parser(sub: Any) -> None:
    p = sub.add_parser("features", help="See what is on and what you can turn on")
    p.add_argument("--json", action="store_true", help="machine-readable output")
