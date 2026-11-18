# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm proposals``: changes SuperLocalMemory proposes to your memory (WS-6, I-10).

    slm proposals [list] [--state proposed|applied|rolled_back|rejected|all] [--json]
    slm proposals show SET_ID [--json]
    slm proposals apply SET_ID        # you, at a terminal
    slm proposals reject SET_ID       # you, at a terminal
    slm proposals rollback SET_ID     # you, at a terminal

A set is a group of changes that is decided together: retire an older memory,
merge duplicates, rewrite a derived summary. Nothing in it touches your memory
until you apply it, and applying is all-or-nothing. ``rollback`` undoes an
applied set exactly, restoring each thing it replaced.

``list`` and ``show`` work anywhere, with ``--json`` for agents. ``apply``,
``reject`` and ``rollback`` ask a person: they refuse to run without an
interactive terminal, so a script or an agent cannot decide in your place (the
same rule as ``slm audit``). Everything runs inside SLM, which stays the only
writer of its store.
"""

from __future__ import annotations

import sys
import urllib.parse
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.corrections_cmd import _daemon, _out, _refused

_DECISIONS = {"apply": "Apply these changes?", "reject": "Decline these changes?",
              "rollback": "Undo these changes?"}
_DONE = {"apply": "Applied. Undo it any time with: slm proposals rollback {id}",
         "reject": "Declined. Nothing was changed.",
         "rollback": "Undone. Everything it replaced is back as it was."}
_STATES = ("proposed", "applied", "rolled_back", "rejected", "all")


def register_proposals_parsers(sub: Any) -> None:
    """Attach ``slm proposals``. Called from cli/main.py."""
    p = sub.add_parser("proposals", help="Changes SuperLocalMemory proposes to your memory: "
                                         "review, apply, decline, undo")
    psub = p.add_subparsers(dest="proposals_command")
    ls = psub.add_parser("list", help="Proposed changes waiting for you (the default)")
    ls.add_argument("--state", default="proposed", choices=_STATES)
    ls.add_argument("--limit", type=int, default=50, help="How many, newest first (1-200)")
    ls.add_argument("--profile", default="", help="Profile (default: the active one)")
    ls.add_argument("--json", action="store_true", help="machine-readable output")
    sh = psub.add_parser("show", help="One set of proposed changes")
    sh.add_argument("set_id")
    sh.add_argument("--profile", default="")
    sh.add_argument("--json", action="store_true")
    for name, text in (("apply", "Apply a set (you, at a terminal)"),
                       ("reject", "Decline a set (you, at a terminal)"),
                       ("rollback", "Undo an applied set exactly (you, at a terminal)")):
        d = psub.add_parser(name, help=text)
        d.add_argument("set_id")
        d.add_argument("--profile", default="")
        d.add_argument("--json", action="store_true")


def _render_set(found: dict) -> None:
    basis = found.get("basis")
    by = f"  [{basis['kind']}: {basis['ref']}]" if basis else ""
    print(f"{found['set_id']}  {found['state']}{by}  {found.get('title') or ''}".rstrip())
    for item in found.get("proposals") or ():
        print(f"    - {item['summary']}  ({item['kind']}, {item['state']})")


def _fetch(path: str, **flags: Any) -> Any:
    from superlocalmemory.cli.daemon import daemon_request

    return daemon_request("GET", path, preserve_not_found=True, preserve_unprocessable=True,
                          **flags)


def _confirm(action: str) -> bool:
    try:
        return input(f"{_DECISIONS[action]} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _decide(args: Namespace, command: str, action: str, profile: str, use_json: bool) -> int:
    from superlocalmemory.cli.daemon import (
        DaemonConflict,
        DaemonNotFound,
        DaemonRefused,
        DaemonUnprocessable,
        daemon_request,
    )

    quoted = urllib.parse.quote(args.set_id, safe="")
    suffix = "?profile_id=" + urllib.parse.quote(profile, safe="") if profile else ""
    try:
        shown = _fetch(f"/api/proposals/{quoted}{suffix}")
        if not isinstance(shown, dict) or not shown.get("success"):
            return _refused(command, "No such proposed change", use_json, "NOT_FOUND")
        if not use_json:
            _render_set(shown["set"])
        if not _confirm(action):
            print("Nothing was changed.")
            return 0
        body = {"channel": "cli", **({"profile_id": profile} if profile else {})}
        result = daemon_request("POST", f"/api/proposals/{quoted}/{action}", body,
                                preserve_conflict=True, preserve_not_found=True,
                                preserve_unprocessable=True)
    except DaemonRefused:
        return _refused(command, "SuperLocalMemory refused: decisions are made by a person on "
                                 "this computer.", use_json, "NEEDS_A_PERSON")
    except DaemonConflict as exc:
        return _refused(command, exc.detail, use_json)
    except DaemonUnprocessable as exc:
        return _refused(command, exc.message, use_json, exc.code)
    except DaemonNotFound as exc:
        return _refused(command, exc.error_message or exc.message, use_json, "NOT_FOUND")
    if not isinstance(result, dict) or not result.get("success"):
        return _refused(command, str((result or {}).get("detail") or "SLM refused the request"),
                        use_json)
    if use_json:
        _out(command, True, data=result)
    else:
        print(_DONE[action].format(id=args.set_id))
    return 0


def cmd_proposals(args: Namespace) -> int:
    sub = getattr(args, "proposals_command", None) or "list"
    use_json = bool(getattr(args, "json", False))
    command = f"proposals {sub}"
    profile = (getattr(args, "profile", "") or "").strip()
    if sub in _DECISIONS and not (sys.stdin.isatty() and sys.stdout.isatty()):
        return _refused(command, f"slm proposals {sub} asks a person at a terminal; it cannot be "
                                 "done by a script or an agent. Use slm proposals list --json "
                                 "to read what is waiting.", use_json, "NEEDS_A_PERSON")
    limit = int(getattr(args, "limit", 50))
    if sub == "list" and not 1 <= limit <= 200:
        return _refused(command, "--limit must be from 1 to 200", use_json, "INVALID")
    from superlocalmemory.cli.commands import _daemon_unavailable

    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    if sub in _DECISIONS:
        return _decide(args, command, sub, profile, use_json)
    from superlocalmemory.cli.daemon import DaemonNotFound, DaemonUnprocessable

    query = {"profile_id": profile} if profile else {}
    try:
        if sub == "show":
            path = f"/api/proposals/{urllib.parse.quote(args.set_id, safe='')}"
        else:
            path = "/api/proposals"
            query.update(state=getattr(args, "state", "proposed"), limit=limit)
        result = _fetch(path + ("?" + urllib.parse.urlencode(query) if query else ""))
    except DaemonUnprocessable as exc:
        return _refused(command, exc.message, use_json, exc.code)
    except DaemonNotFound as exc:
        return _refused(command, exc.error_message or exc.message, use_json, "NOT_FOUND")
    if not isinstance(result, dict) or not result.get("success"):
        _daemon_unavailable(command, use_json)
        return 1
    if use_json:
        _out(command, True, data=result)
    elif sub == "show":
        _render_set(result["set"])
    elif not result["sets"]:
        print("No proposed changes.")
    else:
        for found in result["sets"]:
            _render_set(found)
        print(f"{result['count']} waiting. Decide with: slm proposals apply|reject SET_ID")
    return 0


def run(args: Namespace) -> None:
    """The ``slm proposals`` entry point: exits with the command's status."""
    rc = cmd_proposals(args)
    if rc:
        sys.exit(rc)


__all__ = ["cmd_proposals", "register_proposals_parsers", "run"]
