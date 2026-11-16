# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm audit``: answer the questions that let SuperLocalMemory's judges earn trust (W2, R-16).

    slm audit            # one question at a time, at an interactive terminal
    slm audit --count    # how many are waiting (works anywhere)

SuperLocalMemory's judges (rules, the on-device model, Jev) decide things like
"does this newer memory replace that one?". None of them may act on its own
until a person has checked a random sample of its answers. Each question here
is one such sample: two memories and the question the judge was asked, never
the judge's own answer, so your answer is independent. Pick the option that is
true; "I can't tell" is always fine.

Answering asks a person: it refuses to run without a terminal, so an agent or a
script cannot answer in your place.
"""

from __future__ import annotations

import sys
import urllib.parse
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.corrections_cmd import _daemon, _out, _refused


def register_audit_parsers(sub: Any) -> None:
    """Attach ``slm audit``. Called from cli/main.py."""
    p = sub.add_parser("audit", help="Answer a few questions so SuperLocalMemory's judges can "
                                     "earn trust (asks you, at a terminal)")
    p.add_argument("--count", action="store_true", help="Only say how many are waiting")
    p.add_argument("--profile", default="", help="Profile (default: the active one)")
    p.add_argument("--json", action="store_true", help="machine-readable output (with --count)")


def _show(item: dict) -> None:
    print()
    print(item["question"])
    for mem in item["memories"]:
        when = str(mem.get("said_at") or "")[:10] or "unknown date"
        print(f"  {mem['name']} ({when}): {' '.join(str(mem['content']).split())}")
    for i, option in enumerate(item["options"], 1):
        print(f"  {i}. {option['text']}")


def _ask(item: dict) -> str | None:
    """The chosen label, ``""`` to skip, ``None`` to stop."""
    while True:
        choice = input("Your answer (number, s to skip, q to stop): ").strip().lower()
        if choice == "q":
            return None
        if choice == "s":
            return ""
        if choice.isdigit() and 1 <= int(choice) <= len(item["options"]):
            return item["options"][int(choice) - 1]["label"]
        print("Please type one of the numbers, s or q.")


def cmd_audit(args: Namespace) -> int:
    from superlocalmemory.cli.commands import _daemon_unavailable
    from superlocalmemory.cli.daemon import DaemonNotFound, DaemonUnprocessable, daemon_request

    use_json = bool(getattr(args, "json", False))
    profile = (getattr(args, "profile", "") or "").strip()
    command = "audit"
    if not getattr(args, "count", False) and not (sys.stdin.isatty() and sys.stdout.isatty()):
        return _refused(command, "slm audit asks a person at a terminal; it cannot be answered "
                                 "by a script or an agent. Use slm audit --count to see how many "
                                 "are waiting.", use_json, "NEEDS_A_PERSON")
    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    query = "?limit=10" + ("&profile_id=" + urllib.parse.quote(profile, safe="") if profile else "")
    result = daemon_request("GET", "/api/audit/pending" + query, preserve_not_found=True)
    if not isinstance(result, dict) or not result.get("success"):
        _daemon_unavailable(command, use_json)
        return 1
    if getattr(args, "count", False):
        if use_json:
            _out(command, True, data={"count": result["count"]})
        else:
            print(f"{result['count']} question(s) waiting. Answer them with: slm audit")
        return 0
    items = list(result.get("items") or [])
    if not items:
        print("Nothing to answer right now. Thank you.")
        return 0
    print(f"{result['count']} question(s) waiting. Your answers stay on this computer.")
    answered = 0
    for item in items:
        _show(item)
        label = _ask(item)
        if label is None:
            break
        if not label:
            continue
        body = {"label": label, "channel": "cli", **({"profile_id": profile} if profile else {})}
        try:
            daemon_request("POST", f"/api/audit/{urllib.parse.quote(item['item_id'], safe='')}/answer",
                           body, preserve_not_found=True, preserve_unprocessable=True)
            answered += 1
        except (DaemonNotFound, DaemonUnprocessable) as exc:
            print(f"Not saved: {getattr(exc, 'message', exc)}")
    print(f"\nSaved {answered} answer(s). Thank you.")
    return 0


def run(args: Namespace) -> None:
    rc = cmd_audit(args)
    if rc:
        sys.exit(rc)


__all__ = ["cmd_audit", "register_audit_parsers", "run"]
