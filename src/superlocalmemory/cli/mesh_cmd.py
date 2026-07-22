# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""`slm mesh` — inspect the local agent mesh from the terminal (M-03, v3.7.9).

Before 3.7.9 the mesh was reachable only via MCP tools, the dashboard, and
Claude Code skills — there was no terminal command to check broker health or
list peer sessions. This is a thin, read-only wrapper over the same
capability-authenticated `/mesh/*` daemon endpoints the MCP tools use.
"""
from __future__ import annotations

import json
from argparse import Namespace

_ACTION_ENDPOINTS = {
    "status": "/mesh/status",
    "peers": "/mesh/peers",
}


def cmd_mesh(args: Namespace) -> int:
    action = getattr(args, "mesh_action", None)
    endpoint = _ACTION_ENDPOINTS.get(action)
    if endpoint is None:
        print("Usage: slm mesh {status|peers}")
        return 2

    from superlocalmemory.cli.daemon import daemon_request

    result = daemon_request("GET", endpoint)
    if result is None:
        print(
            "Mesh: cannot reach the daemon broker. Is the daemon running? "
            "(slm serve start)"
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
