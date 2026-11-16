# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm disputes``: what needs your answer (D-37..D-39).

    slm disputes [list] [--limit N] [--profile P] [--json]
    slm disputes settle DISPUTE_ID --keep FACT_ID|both|unrelated [--profile P] [--json]
    slm disputes still-true FACT_ID [--profile P] [--json]

``list`` shows open disputes (two memories that say different values of the
same thing, both sides with when each was said) and memories whose source file
changed or went missing. ``settle`` answers one dispute: keep the side that is
true now (the other is retired, and can be rolled back with
``slm review-correction``), keep ``both``, or say they are ``unrelated``.
``still-true`` says you read the changed file and the memory still holds.

All run inside SLM (it stays the only writer of its store); agents use the MCP
``list_disputes`` and ``settle_dispute`` tools, people also the dashboard.
"""

from __future__ import annotations

import sys
import urllib.parse
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.corrections_cmd import _daemon, _out, _refused


def register_disputes_parsers(sub: Any) -> None:
    """Attach ``slm disputes``. Called from cli/main.py."""
    p = sub.add_parser("disputes", help="What needs your answer: disputed values and "
                                        "memories whose source file changed")
    dsub = p.add_subparsers(dest="disputes_command")
    ls = dsub.add_parser("list", help="Open disputes and changed source files (the default)")
    ls.add_argument("--limit", type=int, default=50, help="How many, newest first (1-200)")
    ls.add_argument("--profile", default="", help="Profile (default: the active one)")
    ls.add_argument("--json", action="store_true", help="machine-readable output")
    st = dsub.add_parser("settle", help="Answer one dispute")
    st.add_argument("dispute_id", help="Dispute id from slm disputes")
    st.add_argument("--keep", required=True,
                    help="The fact id of the side that is true now, 'both' or 'unrelated'")
    st.add_argument("--profile", default="", help="Profile (default: the active one)")
    st.add_argument("--json", action="store_true", help="machine-readable output")
    ok = dsub.add_parser("still-true", help="You read the changed file; the memory still holds")
    ok.add_argument("fact_id", help="Fact id from slm disputes")
    ok.add_argument("--profile", default="", help="Profile (default: the active one)")
    ok.add_argument("--json", action="store_true", help="machine-readable output")


def _side_line(side: dict) -> str:
    text = side.get("content") if side.get("visible") else "(not visible to you)"
    now = " (current)" if side.get("current") else ""
    when = str(side.get("said_at") or "")[:10] or "unknown date"
    return f"    - {when}{now}: {' '.join(str(text or '').split())[:160]}  [fact {side['fact_id']}]"


def _render(result: dict) -> None:
    disputes = list(result.get("disputes") or [])
    changed = list(result.get("changed_sources") or [])
    if not disputes and not changed:
        print("Nothing needs your answer.")
        return
    for d in disputes:
        print(f"Dispute {d['dispute_id']}: {d.get('question', 'Which is true now?')}")
        for side in d.get("sides") or ():
            print(_side_line(side))
        print(f"    settle: slm disputes settle {d['dispute_id']} --keep FACT_ID|both|unrelated")
    for c in changed:
        word = "changed" if c.get("state") == "changed" else "is missing"
        print(f"Source file {word}: {c.get('file')}  [fact {c['fact_id']}]")
        print(f"    read it again, then: slm disputes still-true {c['fact_id']}")


def _request(sub: str, args: Namespace, profile: str) -> Any:
    from superlocalmemory.cli.daemon import daemon_request

    flags = {"preserve_conflict": True, "preserve_not_found": True,
             "preserve_unprocessable": True}
    if sub == "list":
        path = f"/api/truth/attention?limit={int(getattr(args, 'limit', 50))}"
        if profile:
            path += "&profile_id=" + urllib.parse.quote(profile, safe="")
        return daemon_request("GET", path, **flags)
    body: dict = {"profile_id": profile} if profile else {}
    if sub == "settle":
        body.update({"keep": args.keep, "channel": "cli"})
        path = f"/api/disputes/{urllib.parse.quote(args.dispute_id, safe='')}/settle"
    else:
        path = f"/api/changed-sources/{urllib.parse.quote(args.fact_id, safe='')}/still-true"
    return daemon_request("POST", path, body, **flags)


def cmd_disputes(args: Namespace) -> int:
    sub = getattr(args, "disputes_command", None) or "list"
    use_json = bool(getattr(args, "json", False))
    command = f"disputes {sub}"
    profile = (getattr(args, "profile", "") or "").strip()
    if sub == "list" and not 1 <= int(getattr(args, "limit", 50)) <= 200:
        return _refused(command, "--limit must be from 1 to 200", use_json, "INVALID")
    if sub != "list":
        from superlocalmemory.core.admission import gate_cli_mutation
        from superlocalmemory.core.operation_request import OperationKind

        gate_cli_mutation(OperationKind.CORRECT)
    from superlocalmemory.cli.commands import _daemon_unavailable
    from superlocalmemory.cli.daemon import DaemonConflict, DaemonNotFound, DaemonUnprocessable

    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    try:
        result = _request(sub, args, profile)
    except DaemonConflict as exc:
        return _refused(command, exc.detail, use_json)
    except DaemonUnprocessable as exc:
        return _refused(command, exc.message, use_json, exc.code)
    except DaemonNotFound as exc:
        return _refused(command, exc.error_message or exc.message, use_json,
                        (exc.error_code or exc.code).upper())
    if not isinstance(result, dict) or not result.get("success"):
        _daemon_unavailable(command, use_json)
        return 1
    if use_json:
        _out(command, True, data=result)
    elif sub == "list":
        _render(result)
    elif sub == "settle":
        retired = result.get("retired") or []
        print(f"Settled dispute {result['dispute_id']} ({result['resolution']})."
              + (f" Retired {len(retired)} memory; slm review-correction can roll it back."
                 if retired else " Nothing was retired."))
    else:
        print(f"Noted: the memory {result['fact_id']} still holds for its source file.")
    return 0


def run(args: Namespace) -> None:
    """The ``slm disputes`` entry point: exits with the command's status."""
    rc = cmd_disputes(args)
    if rc:
        sys.exit(rc)


__all__ = ["cmd_disputes", "register_disputes_parsers", "run"]
