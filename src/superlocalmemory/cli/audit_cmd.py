# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm audit``: give the blind labels that let SuperLocalMemory's judges earn trust (W2, R-16, D-49).

    slm audit                                   # one question at a time, at a terminal
    slm audit --count                           # how many are waiting
    slm audit --list --model M [--json]         # the queue, without any verdict (agents)
    slm audit --answer ITEM LABEL --model M     # one label, no terminal needed (agents)
    slm audit --retract ITEM                    # take a label back

SuperLocalMemory's judges (rules, the on-device model, Jev) decide things like
"does this newer memory replace that one?". None of them may act on its own
until a random sample of its answers has been checked. Each question here is
one such sample: two memories and the question the judge was asked, never the
judge's own answer, so a label is independent. Any caller may label (D-49): an
agent, a script or a person. The label records who gave it (agent id or client
name, channel ``cli``) and the labeller's own model id (``--model``); it counts
toward a judge's certificate only when that model differs from the judged
tier's model, so a judge never grades itself. A person at the terminal is the
model ``human``. "I can't tell" is always fine, and ``--retract`` takes a label
back (recorded).
"""

from __future__ import annotations

import sys
import urllib.parse
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.corrections_cmd import _daemon, _out, _refused


def register_audit_parsers(sub: Any) -> None:
    """Attach ``slm audit``. Called from cli/main.py."""
    p = sub.add_parser("audit", help="Give blind labels so SuperLocalMemory's judges can "
                                     "earn trust (any caller; recorded; undo with --retract)")
    p.add_argument("--count", action="store_true", help="Only say how many are waiting")
    p.add_argument("--list", action="store_true", dest="list_queue",
                   help="List the queue without any verdict (for agents; add --json)")
    p.add_argument("--answer", nargs=2, metavar=("ITEM_ID", "LABEL"),
                   help="Give one label without a terminal (add --model)")
    p.add_argument("--retract", metavar="ITEM_ID", help="Take a label you gave back")
    p.add_argument("--model", default="", help="Your own model id (counts only when it differs "
                                               "from the judged tier's model)")
    p.add_argument("--profile", default="", help="Profile (default: the active one)")
    p.add_argument("--json", action="store_true", help="machine-readable output")


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


def _model(args: Namespace) -> str:
    """The labeller's own model: ``--model``, else ``SLM_LABELLER_MODEL``, else ``human`` at a
    terminal (a person typing answers), else none (recorded, never counted)."""
    import os

    named = (getattr(args, "model", "") or os.environ.get("SLM_LABELLER_MODEL", "")).strip()
    if named:
        return named
    return "human" if (sys.stdin.isatty() and sys.stdout.isatty()) else ""


def _who() -> str:
    import os

    name = os.environ.get("SLM_AGENT_ID", "").strip()
    if name:
        return f"agent:{name}"
    try:
        import getpass

        return getpass.getuser()
    except Exception:  # noqa: BLE001 - a name is a courtesy, never a reason to fail
        return "local user"


def _post(path: str, body: dict) -> Any:
    from superlocalmemory.cli.daemon import daemon_request

    return daemon_request("POST", path, body, preserve_not_found=True,
                          preserve_unprocessable=True)


def _answer_one(args: Namespace, command: str, use_json: bool, profile: str) -> int:
    from superlocalmemory.cli.daemon import DaemonNotFound, DaemonRefused, DaemonUnprocessable

    item, label = args.answer
    body = {"label": label, "model": _model(args), "by": _who(),
            **({"profile_id": profile} if profile else {})}
    try:
        got = _post(f"/api/audit/{urllib.parse.quote(item, safe='')}/answer", body)
    except DaemonRefused:
        return _refused(command, "SuperLocalMemory refused: your role on this workspace does "
                                 "not allow changing memories.", use_json, "FORBIDDEN")
    except (DaemonNotFound, DaemonUnprocessable) as exc:
        return _refused(command, getattr(exc, "message", str(exc)), use_json, "NOT_SAVED")
    if use_json:
        _out(command, True, data=got)
    else:
        counted = int((got or {}).get("counted", 0))
        print(f"Saved. It counts toward {counted} judge decision(s)." if counted else
              "Saved, but it does not count: pass --model with your own model id (it must "
              "differ from the judged tier's model).")
    return 0


def _retract_one(args: Namespace, command: str, use_json: bool, profile: str) -> int:
    from superlocalmemory.cli.daemon import DaemonNotFound, DaemonRefused, DaemonUnprocessable

    body = {"by": _who(), **({"profile_id": profile} if profile else {})}
    try:
        got = _post(f"/api/audit/{urllib.parse.quote(args.retract, safe='')}/retract", body)
    except DaemonRefused:
        return _refused(command, "SuperLocalMemory refused: your role on this workspace does "
                                 "not allow changing memories.", use_json, "FORBIDDEN")
    except (DaemonNotFound, DaemonUnprocessable) as exc:
        return _refused(command, getattr(exc, "message", str(exc)), use_json, "NOT_FOUND")
    if use_json:
        _out(command, True, data=got)
    else:
        print("Label taken back. The question is waiting again.")
    return 0


def cmd_audit(args: Namespace) -> int:
    from superlocalmemory.cli.commands import _daemon_unavailable
    from superlocalmemory.cli.daemon import DaemonNotFound, DaemonUnprocessable, daemon_request

    use_json = bool(getattr(args, "json", False))
    profile = (getattr(args, "profile", "") or "").strip()
    command = "audit"
    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    if getattr(args, "answer", None):
        return _answer_one(args, command, use_json, profile)
    if getattr(args, "retract", None):
        return _retract_one(args, command, use_json, profile)
    query = "?limit=10" + ("&profile_id=" + urllib.parse.quote(profile, safe="") if profile else "")
    model = _model(args)
    if model:
        query += "&model=" + urllib.parse.quote(model, safe="")
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
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not getattr(
        args, "list_queue", False)
    if not interactive:  # an agent or a script reads the queue; it labels with --answer
        if use_json:
            _out(command, True, data=result)
        else:
            print(f"{result['count']} question(s) waiting.")
            for item in result.get("items") or []:
                _show(item)
            print("Label one with: slm audit --answer ITEM_ID LABEL --model YOUR_MODEL_ID")
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
        body = {"label": label, "model": model, "by": _who(),
                **({"profile_id": profile} if profile else {})}
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
