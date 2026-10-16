# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The ``slm doctor`` row for mesh.db, the mesh's own file next to memory.db.

Read-only: doctor never creates, migrates or repairs mesh.db. The daemon does
that when it starts (``mesh.mesh_store.prepare_mesh_db``), so the fix for a
damaged file is a restart, which quarantines it and builds a new one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

FIX = ("Run `slm restart`: the daemon quarantines a damaged mesh.db and builds a new "
       "one (agent mesh messages and shared state are rebuilt as agents reconnect)")


def _legacy_left(memory_db: Path) -> list[str]:
    """Old mesh tables still in memory.db (dropped at the daemon's next start)."""
    if not memory_db.exists():
        return []
    conn = sqlite3.connect(f"{memory_db.absolute().as_uri()}?mode=ro", uri=True, timeout=2)
    try:
        return [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'mesh\\_%' "
            "ESCAPE '\\' ORDER BY name")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def mesh_finding(root: str | Path) -> tuple[str, str, str]:
    """``(status, detail, fix)`` for mesh.db in the data folder ``root``."""
    from superlocalmemory.mesh.mesh_store import MESH_DB_NAME, inspect

    root = Path(root)
    state = inspect(root / MESH_DB_NAME)
    if state["status"] == "missing":
        return "PASS", "mesh.db not yet created (the daemon makes it at start)", ""
    if state["status"] != "ok":
        detail = state.get("detail") or state["status"]
        return "FAIL", f"mesh.db is {state['status']}: {detail}", FIX
    size_kb = state["size_bytes"] / 1024
    detail = f"mesh.db OK ({size_kb:.0f} KB, quick_check; since {state['migrated_at'][:19]})"
    left = _legacy_left(root / "memory.db")
    if left:
        return ("WARN", f"{detail}; {len(left)} old mesh table(s) still in memory.db",
                "Restart the daemon (`slm restart`); it drops them once mesh.db is verified")
    return "PASS", detail, ""


__all__ = ["FIX", "mesh_finding"]
