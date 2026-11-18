# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm trust``: memories that wait for your OK before they can be instructions (WS-5).

    slm trust pending [--json]          # what is waiting (anyone, including an agent)
    slm trust status [--json]           # is the firewall acting; how many wait
    slm trust approve FACT_ID [...]     # you, at a terminal
    slm trust approve --all             # everything listed, one answer (you, at a terminal)

A memory is data. It becomes an instruction when SuperLocalMemory puts it somewhere an
agent obeys it: a pin, a standing rule or decision, your core memory, the start of every
session. With ``trust.taint`` on, a memory that came from a tool's output, a web page,
another agent's message, an import, a model's text, an agent's own words, or from before
4.2.30, cannot be one of those until you approve it. It is still found by recall, as
evidence. ``pending`` lists what is waiting and why.

Approving asks a person: it refuses to run without an interactive terminal, so a script
or an agent cannot approve in your place (the same rule as ``slm audit``). An agent reads
``slm trust pending --json`` and tells you what waits.
"""

from __future__ import annotations

import sys
import urllib.parse
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.corrections_cmd import _daemon, _out, _refused

_SUBS = ("pending", "status", "approve")


def register_trust_parsers(sub: Any) -> None:
    """Attach ``slm trust``. Called from cli/main.py."""
    p = sub.add_parser("trust", help="Memories waiting for your OK before they can be "
                                     "instructions: pending, status, approve")
    tsub = p.add_subparsers(dest="trust_command")
    ls = tsub.add_parser("pending", help="What is waiting for your OK (the default)")
    ls.add_argument("--limit", type=int, default=50, help="How many, newest first (1-200)")
    ls.add_argument("--profile", default="", help="Profile (default: the active one)")
    ls.add_argument("--json", action="store_true", help="machine-readable output")
    st = tsub.add_parser("status", help="Is the firewall acting, and how many memories wait")
    st.add_argument("--profile", default="")
    st.add_argument("--json", action="store_true")
    ap = tsub.add_parser("approve", help="Approve memories so they may be instructions "
                                         "(you, at a terminal)")
    ap.add_argument("fact_ids", nargs="*", help="Memory ids (see slm trust pending)")
    ap.add_argument("--all", action="store_true", help="Approve everything that is listed")
    ap.add_argument("--profile", default="")
    ap.add_argument("--json", action="store_true")


def _fetch(path: str) -> Any:
    from superlocalmemory.cli.daemon import daemon_request

    return daemon_request("GET", path, preserve_not_found=True, preserve_unprocessable=True)


def _listed(profile: str) -> dict[str, dict]:
    """What waits, by memory id (the preview a person is asked about)."""
    suffix = "&profile_id=" + urllib.parse.quote(profile, safe="") if profile else ""
    found = _fetch("/api/trust/pending?limit=200" + suffix)
    return {i["fact_id"]: i for i in (found or {}).get("items") or []}


def _show(item: dict) -> None:
    wanted = "; ".join(item.get("wanted_words") or [])
    print(f"{item['fact_id']}  {item['origin']}")
    print(f"    {item['preview']}")
    if wanted:
        print(f"    This memory {wanted}.")


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _who() -> str:
    try:
        import getpass

        return getpass.getuser()
    except Exception:  # noqa: BLE001 - a name is a courtesy, never a reason to fail
        return "local user"


def _approve(args: Namespace, command: str, profile: str, use_json: bool) -> int:
    from superlocalmemory.cli.daemon import (
        DaemonNotFound,
        DaemonRefused,
        DaemonUnprocessable,
        daemon_request,
    )

    ids = list(dict.fromkeys(args.fact_ids or []))
    if bool(ids) == bool(args.all):
        return _refused(command, "Name the memories to approve, or use --all (not both).",
                        use_json, "INVALID")
    items = _listed(profile)
    if args.all and not items:
        print("Nothing is waiting for your OK.")
        return 0
    for fact_id in (items if args.all else ids):
        if fact_id in items:
            _show(items[fact_id])
    question = (f"Approve all {len(items)} of these?" if args.all
                else f"Approve {len(ids)} memor{'y' if len(ids) == 1 else 'ies'} so they may be "
                     "instructions?")
    if not _confirm(question):
        print("Nothing was changed.")
        return 0
    body = {"channel": "cli", "by": _who(), **({"profile_id": profile} if profile else {})}
    targets = ["/api/trust/approve-all"] if args.all else [
        f"/api/trust/approve/{urllib.parse.quote(i, safe='')}" for i in ids]
    results = []
    try:
        for path in targets:
            results.append(daemon_request("POST", path, body, preserve_not_found=True,
                                          preserve_unprocessable=True))
    except DaemonRefused:
        return _refused(command, "SuperLocalMemory refused: approval is given by a person at "
                                 "a terminal on this computer.", use_json, "NEEDS_A_PERSON")
    except DaemonUnprocessable as exc:
        return _refused(command, exc.message, use_json, exc.code)
    except DaemonNotFound as exc:
        return _refused(command, exc.error_message or exc.message, use_json, "NOT_FOUND")
    if not all(isinstance(r, dict) and r.get("success") for r in results):
        return _refused(command, "SLM did not approve everything; run slm trust pending.",
                        use_json)
    done = sum(int(r.get("approved", 1)) for r in results)
    if use_json:
        _out(command, True, data={"approved": done, "results": results})
    else:
        print(f"Approved {done}. They may now be pins, rules and part of your core memory.")
    return 0


def _render(result: dict) -> None:
    items = result.get("items") or []
    if not items:
        print("Nothing is waiting for your OK.")
    for item in items:
        _show(item)
    if items:
        print(f"{result['count']} waiting. Approve with: slm trust approve FACT_ID "
              "(or --all), at a terminal")
    if not result.get("acting"):
        print("(The firewall is not acting yet: trust.taint is off or in shadow, so nothing "
              "is held back. This is what it would hold.)")
    if result.get("unclassified"):
        print(f"{result['unclassified']} memories have no origin class yet; they are "
              "classified in the background.")


def cmd_trust(args: Namespace) -> int:
    sub = getattr(args, "trust_command", None) or "pending"
    use_json = bool(getattr(args, "json", False))
    command = f"trust {sub}"
    profile = (getattr(args, "profile", "") or "").strip()
    if sub == "approve" and not (sys.stdin.isatty() and sys.stdout.isatty()):
        return _refused(command, "slm trust approve asks a person at a terminal; it cannot "
                                 "be done by a script or an agent. Use slm trust pending "
                                 "--json to read what is waiting.", use_json, "NEEDS_A_PERSON")
    limit = int(getattr(args, "limit", 50))
    if sub == "pending" and not 1 <= limit <= 200:
        return _refused(command, "--limit must be from 1 to 200", use_json, "INVALID")
    from superlocalmemory.cli.commands import _daemon_unavailable

    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    if sub == "approve":
        return _approve(args, command, profile, use_json)
    from superlocalmemory.cli.daemon import DaemonNotFound, DaemonUnprocessable

    query = {"profile_id": profile} if profile else {}
    if sub == "pending":
        query["limit"] = limit
    path = "/api/trust/" + sub + ("?" + urllib.parse.urlencode(query) if query else "")
    try:
        result = _fetch(path)
    except DaemonUnprocessable as exc:
        return _refused(command, exc.message, use_json, exc.code)
    except DaemonNotFound as exc:
        return _refused(command, exc.error_message or exc.message, use_json, "NOT_FOUND")
    if not isinstance(result, dict) or not result.get("success"):
        _daemon_unavailable(command, use_json)
        return 1
    if use_json:
        _out(command, True, data=result)
    elif sub == "status":
        print(f"Instruction firewall (trust.taint): {result['taint']}; secret screen "
              f"(trust.secret_screen): {result['secret_screen']}")
        print(f"{result['waiting']} waiting for your OK, {result['approved']} approved by you; "
              f"{result['unclassified']} not yet classified.")
    else:
        _render(result)
    return 0


def run(args: Namespace) -> None:
    """The ``slm trust`` entry point: exits with the command's status."""
    rc = cmd_trust(args)
    if rc:
        sys.exit(rc)


__all__ = ["cmd_trust", "register_trust_parsers", "run"]
