# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm cues list``: the cues that bring memories back, with the memory each belongs to (W3-8).

    slm cues list            # one line per cue: kind, pattern, origin, the memory's start
    slm cues list --json     # the same as structured data

A cue comes from ``slm remember --cue KIND:PATTERN`` (origin ``explicit``) or from a file
path found in a plan (origin ``extracted``). They are shown to an agent only while the
``delivery.cues`` setting is on (``slm flags``). Reads the store directly, read-only.
"""

from __future__ import annotations

import sqlite3
import sys
from argparse import Namespace
from typing import Any

#: Characters of the memory shown beside a cue (display only).
PREVIEW = 80
LIMIT = 500

_LIST = """
SELECT c.cue_id, c.kind, c.pattern, c.origin, c.created_at, c.fact_id, f.content
  FROM memory_cues c JOIN atomic_facts f ON f.fact_id = c.fact_id
 WHERE c.profile_id = ? ORDER BY c.created_at DESC, c.cue_id LIMIT ?
"""


def register_cue_parsers(sub: Any) -> None:
    """Attach ``slm cues``. Called from cli/main.py."""
    p = sub.add_parser("cues", help="Cues that bring memories back at the right step")
    p.add_argument("action", nargs="?", default="list", choices=["list"])
    p.add_argument("--json", action="store_true", help="machine-readable output")


def list_cues(db_path, profile_id: str) -> list[dict]:
    """Each cue of ``profile_id`` with the start of its memory; ``[]`` before M078."""
    from superlocalmemory.truth.invariants import open_read_only

    conn = open_read_only(str(db_path))
    try:
        rows = conn.execute(_LIST, (profile_id, LIMIT)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [{"cue_id": cid, "kind": kind, "pattern": pattern, "origin": origin,
             "created_at": at, "fact_id": fid, "memory": " ".join(content.split())[:PREVIEW]}
            for cid, kind, pattern, origin, at, fid, content in rows]


def cmd_cues(args: Namespace) -> int:
    from superlocalmemory.cli.json_output import json_print
    from superlocalmemory.hooks.event_delivery import active_profile
    from superlocalmemory.infra.data_root import canonical_data_root

    use_json = bool(getattr(args, "json", False))
    db_path = canonical_data_root() / "memory.db"
    rows = list_cues(db_path, active_profile()) if db_path.is_file() else []
    if use_json:
        json_print("cues", data={"cues": rows, "count": len(rows)})
    elif not rows:
        print("No cues yet. Add one with: slm remember \"...\" --cue path:src/**/schema.py")
    else:
        for r in rows:
            print(f"{r['kind']}:{r['pattern']}  [{r['origin']}]  {r['memory']}")
    return 0


def run(args: Namespace) -> None:
    rc = cmd_cues(args)
    if rc:
        sys.exit(rc)


__all__ = ["cmd_cues", "list_cues", "register_cue_parsers", "run"]
