# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm health --memory``: what memory did for you, with intervals (W3-5, M-12, M-13).

Reads the store with ``mode=ro`` (never the engine, never a daemon), so it can
run beside a live daemon and on a copy (``--db``). The report itself is
``outcomes.memory_report``, the same one the dashboard and the MCP ``health``
tool show. Ids and numbers only; ``--json`` for agents.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


def _profile(args: Namespace) -> str:
    given = getattr(args, "profile", None)
    if given:
        return str(given)
    try:
        from superlocalmemory.core.config import SLMConfig

        return SLMConfig.load().active_profile or "default"
    except Exception:  # noqa: BLE001 - an unreadable config reads the default profile
        return "default"


def cmd_health_memory(args: Namespace) -> None:
    from superlocalmemory.cli.doctor_truth import _db_path
    from superlocalmemory.outcomes import memory_report
    from superlocalmemory.truth.invariants import open_read_only

    path: Path = _db_path(args)
    use_json = getattr(args, "json", False)
    if not path.is_file():
        message = f"No store at {path}"
        if use_json:
            from superlocalmemory.cli.json_output import json_print
            json_print("health", error={"code": "NO_STORE", "message": message})
        else:
            print(message, file=sys.stderr)
        sys.exit(2)
    conn = open_read_only(str(path))
    try:
        report = memory_report.build(conn, _profile(args), slm_home=path.parent)
    finally:
        conn.close()
    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("health", data={"memory": report}, next_actions=[
            {"command": "slm flags", "description": "Turn receipts or withholding on or off"},
            {"command": "slm doctor --truth --json", "description": "Truth-layer invariants"},
        ])
        return
    for line in memory_report.describe(report):
        print(line)


__all__ = ["cmd_health_memory"]
