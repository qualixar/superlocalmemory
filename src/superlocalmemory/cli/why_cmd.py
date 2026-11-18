# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm why``: everything that explains one memory (WS-5).

    slm why FACT_ID [--json] [--verify] [--deliveries N] [--db FILE] [--profile P]

Where the memory came from (class, taint, the raw signals and the rule that decided), whether
the instruction firewall holds it back and who approved it, its signature, its disputes and
source file, derived work that mentions it, and the last times it was shown to an agent, each
with the reason in words. Read-only on the store (``mode=ro``), like ``slm health --memory``:
no daemon is started and nothing is written. An agent reads it with ``--json``.

``--verify`` reads the signing key from the keychain once to check the signature; without it
the signature shows as signed or unsigned and is not checked.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from typing import Any


def register_why_parsers(sub: Any) -> None:
    """Attach ``slm why``. Called from cli/main.py."""
    p = sub.add_parser("why", help="Everything that explains one memory: where it came from, "
                                   "whether to trust it, why it was shown")
    p.add_argument("fact_id", help="The memory's id (shown by slm recall --json and the dashboard)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--verify", action="store_true",
                   help="Check the signature (reads the signing key from your keychain once)")
    p.add_argument("--deliveries", type=int, default=10,
                   help="How many recent deliveries to list (1-50, default 10)")
    p.add_argument("--profile", default="", help="Only a memory in this profile")
    p.add_argument("--db", default=None, help="Store file (default: yours)")


def _fail(use_json: bool, code: str, message: str, exit_code: int) -> None:
    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("why", error={"code": code, "message": message})
    else:
        print(message, file=sys.stderr)
    sys.exit(exit_code)


def cmd_why(args: Namespace) -> None:
    from superlocalmemory.cli.doctor_truth import _db_path
    from superlocalmemory.trust import why
    from superlocalmemory.truth.invariants import open_read_only

    use_json = bool(getattr(args, "json", False))
    path = _db_path(args)
    if not path.is_file():
        _fail(use_json, "NO_STORE", f"No store at {path}", 2)
    signer = None
    if getattr(args, "verify", False):
        from superlocalmemory.cli import doctor_trust

        signer = doctor_trust.read_key().signer
    limit = max(1, min(int(getattr(args, "deliveries", 10) or 10), 50))
    conn = open_read_only(str(path))
    try:
        report = why.explain(conn, str(args.fact_id), profile_id=(getattr(args, "profile", "") or None),
                             signer=signer, deliveries=limit)
    finally:
        conn.close()
    if report is None:
        _fail(use_json, "NOT_FOUND", f"No memory {args.fact_id}"
              + (f" in profile {args.profile}" if getattr(args, "profile", "") else ""), 1)
    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("why", data=report, next_actions=[
            {"command": "slm trust pending", "description": "Memories held back until promoted"},
            {"command": "slm disputes", "description": "Disputes that need your answer"},
        ])
        return
    for line in why.lines(report):
        print(line)


def run(args: Namespace) -> None:
    cmd_why(args)


__all__ = ["cmd_why", "register_why_parsers", "run"]
