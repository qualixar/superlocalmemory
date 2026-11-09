# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm corrections overtaken``, ``restore-overtaken`` and ``dismiss``.

    slm corrections overtaken [--limit N] [--profile P] [--json]
    slm corrections restore-overtaken CASE_ID [--profile P] [--json]
    slm corrections dismiss [--reason R ...] [--older-than DAYS] [--before DATE]
                            [--profile P] [--apply] [--json]

When you delete, replace or edit a memory, a correction SLM proposed by itself
and nobody reviewed no longer blocks you: it is closed as "overtaken by a user
action" and kept. The first command lists those cases (ids, what you did, when,
and whether each can still come back); the second puts one back exactly as it
was. A case cannot come back once one of its memories is deleted, or while the
memory has another open correction (your own edit, until you reject it).

``dismiss`` clears the pile of correction cases SLM proposed by itself and
nobody reviewed (a dry run unless ``--apply``, which backs memory.db up first).
It closes them the same way, so each can be put back with ``restore-overtaken``.
There is deliberately no bulk "reject": a rejected case counts as decided and
would block deleting its memories for good.

All run inside SLM (it stays the only writer of its store), like
``slm review-correction``. Ids and counts only are printed, never memory text.
"""

from __future__ import annotations

import sys
import urllib.parse
from argparse import Namespace
from typing import Any


def register_corrections_parsers(sub: Any) -> None:
    """Attach ``slm corrections``. Called from cli/main.py."""
    p = sub.add_parser("corrections", help="Corrections a delete, replace or edit of yours "
                                           "closed: list them, put one back")
    csub = p.add_subparsers(dest="corrections_command")
    ls = csub.add_parser("overtaken", help="List correction cases your own actions closed")
    ls.add_argument("--limit", type=int, default=50, help="How many, newest first (1-500)")
    ls.add_argument("--profile", default="", help="Profile (default: the active one)")
    ls.add_argument("--json", action="store_true", help="machine-readable output")
    rs = csub.add_parser("restore-overtaken", help="Put an overtaken correction case back")
    rs.add_argument("case_id", help="Case id from slm corrections overtaken")
    rs.add_argument("--profile", default="", help="Profile (default: the active one)")
    rs.add_argument("--json", action="store_true", help="machine-readable output")
    dm = csub.add_parser("dismiss", help="Dismiss correction cases SLM proposed by itself and "
                                         "nobody reviewed (dry run unless --apply)")
    dm.add_argument("--reason", action="append", default=[], dest="reasons", metavar="R",
                    help="Only this reason (repeatable): temporal_contradiction, "
                         "consolidation_update, consolidation_supersede")
    dm.add_argument("--older-than", type=int, default=None, metavar="DAYS",
                    help="Only cases created more than DAYS days ago")
    dm.add_argument("--before", default=None, metavar="DATE",
                    help="Only cases created before this date (e.g. 2026-11-01)")
    dm.add_argument("--profile", default="", help="Profile (default: the active one)")
    dm.add_argument("--apply", action="store_true",
                    help="Do it (memory.db is backed up first); without it nothing changes")
    dm.add_argument("--json", action="store_true", help="machine-readable output")


def _out(command: str, use_json: bool, *, data: dict | None = None,
         error: dict | None = None) -> None:
    from superlocalmemory.cli.json_output import json_print

    if use_json:
        json_print(command, data=data, error=error)


def _refused(command: str, message: str, use_json: bool, code: str = "CONFLICT") -> int:
    if use_json:
        _out(command, True, error={"code": code, "message": message, "retryable": False})
    else:
        print(f"Refused: {message}", file=sys.stderr)
    return 1


def _render(rows: list[dict]) -> None:
    if not rows:
        print("No correction case was overtaken by your actions.")
        return
    for r in rows:
        state = ("restored" if r.get("restored_at") else
                 "can be restored" if r.get("restorable") else
                 f"cannot be restored: {r.get('not_restorable_because')}")
        print(f"{r['case_id']}  {r['user_action']:7}  {r['overtaken_at']}  "
              f"{r['predecessor_fact_id']} -> {r['successor_fact_id']}  ({state})")


def _daemon() -> bool:
    from superlocalmemory.cli.daemon import ensure_daemon, is_daemon_running

    return bool(is_daemon_running() or ensure_daemon())


def _dismiss_body(args: Namespace) -> dict:
    """The request, checked here so a typo never needs the daemon to be refused."""
    from superlocalmemory.core.dismissed_cases import DismissFilter, cutoff_for

    DismissFilter(profile_id="-", reasons=tuple(args.reasons))
    cutoff_for(older_than_days=args.older_than, before=args.before)
    body: dict = {"apply": bool(args.apply)}
    for key, value in (("reasons", list(args.reasons)), ("older_than_days", args.older_than),
                       ("before", args.before), ("profile_id", (args.profile or "").strip())):
        if value not in (None, "", []):
            body[key] = value
    return body


def _render_dismiss(result: dict) -> None:
    by_reason = result.get("by_reason") or {}
    if not result.get("dry_run"):
        print(f"Dismissed {result.get('dismissed', 0)} correction case(s) in profile "
              f"{result.get('profile_id')}.")
        for key, label in (("skipped_raced", "changed while it ran"),
                           ("skipped_changed", "no longer a pending machine proposal")):
            if result.get(key):
                print(f"  Left alone: {result[key]} ({label}).")
        if result.get("backup"):
            print(f"Backup of memory.db taken first: {result['backup']}")
        print("Put one back with: slm corrections restore-overtaken CASE_ID")
        return
    print(f"Dry run: {result.get('eligible', 0)} correction case(s) SLM proposed by itself "
          f"and nobody reviewed would be dismissed in profile {result.get('profile_id')}.")
    for reason, count in sorted(by_reason.items()):
        print(f"  {reason}: {count}")
    if result.get("eligible"):
        print(f"Oldest: {result.get('oldest')}   Newest: {result.get('newest')}")
    print("Nothing changed. Run again with --apply to dismiss them (memory.db is backed up "
          "first, and each case can be put back).")


def _dismiss(args: Namespace, command: str, use_json: bool) -> int:
    from superlocalmemory.cli.commands import _daemon_unavailable
    from superlocalmemory.cli.daemon import DaemonNotFound, DaemonUnprocessable, daemon_request

    try:
        body = _dismiss_body(args)
    except ValueError as exc:
        return _refused(command, str(exc), use_json, "INVALID")
    if body["apply"]:
        from superlocalmemory.core.admission import gate_cli_mutation
        from superlocalmemory.core.operation_request import OperationKind

        gate_cli_mutation(OperationKind.CORRECT)
    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    try:
        result = daemon_request("POST", "/api/overtaken-corrections/dismiss", body,
                                timeout_seconds=900.0 if body["apply"] else 60.0,
                                preserve_unprocessable=True, preserve_not_found=True)
    except DaemonUnprocessable as exc:
        return _refused(command, exc.message, use_json, "INVALID")
    except DaemonNotFound as exc:  # an unknown --profile
        return _refused(command, exc.error_message or exc.message, use_json,
                        (exc.error_code or exc.code).upper())
    if not isinstance(result, dict) or not result.get("success"):
        _daemon_unavailable(command, use_json)
        return 1
    if use_json:
        _out(command, True, data=result)
    else:
        _render_dismiss(result)
    return 0


def cmd_corrections(args: Namespace) -> int:
    sub = getattr(args, "corrections_command", None)
    use_json = bool(getattr(args, "json", False))
    if sub == "dismiss":
        return _dismiss(args, "corrections dismiss", use_json)
    if sub not in ("overtaken", "restore-overtaken"):
        print("Usage: slm corrections overtaken | slm corrections restore-overtaken CASE_ID | "
              "slm corrections dismiss", file=sys.stderr)
        return 2
    from superlocalmemory.cli.commands import _daemon_unavailable
    from superlocalmemory.cli.daemon import DaemonConflict, DaemonNotFound, daemon_request

    command = f"corrections {sub}"
    profile = (getattr(args, "profile", "") or "").strip()
    if sub == "overtaken" and not 1 <= int(args.limit) <= 500:
        return _refused(command, "--limit must be from 1 to 500", use_json, "INVALID")
    if sub == "restore-overtaken":
        from superlocalmemory.core.admission import gate_cli_mutation
        from superlocalmemory.core.operation_request import OperationKind

        gate_cli_mutation(OperationKind.CORRECT)
    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    try:
        if sub == "overtaken":
            path = f"/api/overtaken-corrections?limit={int(args.limit)}"
            if profile:
                path += "&profile_id=" + urllib.parse.quote(profile, safe="")
            result = daemon_request("GET", path, preserve_not_found=True)
        else:
            path = ("/api/overtaken-corrections/" + urllib.parse.quote(args.case_id, safe="")
                    + "/restore")
            result = daemon_request("POST", path, {"profile_id": profile} if profile else {},
                                    preserve_conflict=True, preserve_not_found=True)
    except DaemonConflict as exc:
        return _refused(command, exc.detail, use_json)
    except DaemonNotFound as exc:  # an unknown --profile
        return _refused(command, exc.error_message or exc.message, use_json,
                        (exc.error_code or exc.code).upper())
    if not isinstance(result, dict) or not result.get("success"):
        _daemon_unavailable(command, use_json)
        return 1
    if use_json:
        _out(command, True, data=result)
    elif sub == "overtaken":
        _render(list(result.get("overtaken") or []))
    else:
        print(f"Restored correction case {result['restored']} ({result['events']} event(s)); "
              "it is waiting for review again.")
    return 0


def run(args: Namespace) -> None:
    """The ``slm corrections`` entry point: exits with the command's status."""
    rc = cmd_corrections(args)
    if rc:
        sys.exit(rc)


__all__ = ["cmd_corrections", "register_corrections_parsers", "run"]
