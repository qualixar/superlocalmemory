# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm trust``: memories that wait for a promotion before they can be instructions (WS-5, D-49).

    slm trust pending [--json]          # what is waiting
    slm trust status [--json]           # is the firewall acting; how many wait
    slm trust promote FACT_ID [...]     # one explicit call releases a memory (alias: approve)
    slm trust promote --all             # everything listed, one call
    slm trust withdraw FACT_ID [...]    # take a promotion back

A memory is data. It becomes an instruction when SuperLocalMemory puts it somewhere an
agent obeys it: a pin, a standing rule or decision, core memory, the start of every
session. With ``trust.taint`` on, a memory that came from a tool's output, a web page,
another agent's message, an import, a model's text, an agent's own words, or from before
4.2.30 (and was not already a pin or rule then), cannot be one of those by itself. It is
still found by recall, as evidence. ``pending`` lists what is waiting and why.

Any caller may promote: an agent, a script or a person. Nothing asks for a terminal or a
confirmation, because nothing here may need a person to work. Every promotion is recorded
(who, channel ``cli``, when), listed on the dashboard under "Recent decisions by your agents",
and ``withdraw`` takes it back. Promote text from an untrusted source only on purpose.
"""

from __future__ import annotations

import sys
import urllib.parse
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.corrections_cmd import _daemon, _out, _refused

_SUBS = ("pending", "status", "promote", "approve", "withdraw")


def register_trust_parsers(sub: Any) -> None:
    """Attach ``slm trust``. Called from cli/main.py."""
    p = sub.add_parser("trust", help="Memories waiting for a promotion before they can be "
                                     "instructions: pending, status, promote, withdraw")
    tsub = p.add_subparsers(dest="trust_command")
    ls = tsub.add_parser("pending", help="What is waiting for a promotion (the default)")
    ls.add_argument("--limit", type=int, default=50, help="How many, newest first (1-200)")
    ls.add_argument("--profile", default="", help="Profile (default: the active one)")
    ls.add_argument("--json", action="store_true", help="machine-readable output")
    st = tsub.add_parser("status", help="Is the firewall acting, and how many memories wait")
    st.add_argument("--profile", default="")
    st.add_argument("--json", action="store_true")
    for name, text in (("promote", "Promote memories on purpose so they may be instructions "
                                   "(recorded; undo with withdraw)"),
                       ("approve", "Same as promote")):
        ap = tsub.add_parser(name, help=text)
        ap.add_argument("fact_ids", nargs="*", help="Memory ids (see slm trust pending)")
        ap.add_argument("--all", action="store_true", help="Promote everything that is listed")
        ap.add_argument("--profile", default="")
        ap.add_argument("--json", action="store_true")
    wd = tsub.add_parser("withdraw", help="Take a promotion back")
    wd.add_argument("fact_ids", nargs="+", help="Memory ids")
    wd.add_argument("--profile", default="")
    wd.add_argument("--json", action="store_true")


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


def _who() -> str:
    """A client name for the record: the agent's own id when it set one, else this user's name."""
    import os

    name = os.environ.get("SLM_AGENT_ID", "").strip()
    if name:
        return f"agent:{name}"
    try:
        import getpass

        return getpass.getuser()
    except Exception:  # noqa: BLE001 - a name is a courtesy, never a reason to fail
        return "local user"


def _decide(args: Namespace, command: str, profile: str, use_json: bool, action: str) -> int:
    from superlocalmemory.cli.daemon import (
        DaemonNotFound,
        DaemonRefused,
        DaemonUnprocessable,
        daemon_request,
    )

    ids = list(dict.fromkeys(args.fact_ids or []))
    every = bool(getattr(args, "all", False))
    if bool(ids) == every:
        return _refused(command, "Name the memories, or use --all (not both).", use_json,
                        "INVALID")
    items = _listed(profile) if action == "promote" else {}
    if every and not items:
        print("Nothing is waiting for a promotion.")
        return 0
    if not use_json:
        for fact_id in (items if every else ids):
            if fact_id in items:
                _show(items[fact_id])
    body = {"by": _who(), **({"profile_id": profile} if profile else {})}
    verb = "withdraw" if action == "withdraw" else "promote"
    targets = [f"/api/trust/{verb}-all"] if every else [
        f"/api/trust/{verb}/{urllib.parse.quote(i, safe='')}" for i in ids]
    results = []
    try:
        for path in targets:
            results.append(daemon_request("POST", path, body, preserve_not_found=True,
                                          preserve_unprocessable=True))
    except DaemonRefused:
        return _refused(command, "SuperLocalMemory refused: your role on this workspace does "
                                 "not allow changing memories.", use_json, "FORBIDDEN")
    except DaemonUnprocessable as exc:
        return _refused(command, exc.message, use_json, exc.code)
    except DaemonNotFound as exc:
        return _refused(command, exc.error_message or exc.message, use_json, "NOT_FOUND")
    if not all(isinstance(r, dict) and r.get("success") for r in results):
        return _refused(command, "SLM did not finish; run slm trust pending.", use_json)
    done = sum(int(r.get("approved", 1)) for r in results)
    if use_json:
        _out(command, True, data={"changed": done, "results": results})
    elif action == "withdraw":
        print(f"Withdrew {done}. They are back under their own origin.")
    else:
        print(f"Promoted {done}. They may now be pins, rules and part of core memory. "
              "Recorded; undo with: slm trust withdraw FACT_ID")
    return 0


def _render(result: dict) -> None:
    items = result.get("items") or []
    if not items:
        print("Nothing is waiting for a promotion.")
    for item in items:
        _show(item)
    if items:
        print(f"{result['count']} waiting. Promote on purpose with: slm trust promote FACT_ID "
              "(or --all)")
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
    limit = int(getattr(args, "limit", 50))
    if sub == "pending" and not 1 <= limit <= 200:
        return _refused(command, "--limit must be from 1 to 200", use_json, "INVALID")
    from superlocalmemory.cli.commands import _daemon_unavailable

    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    if sub in ("promote", "approve", "withdraw"):
        return _decide(args, command, profile, use_json,
                       "withdraw" if sub == "withdraw" else "promote")
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
        print(f"{result['waiting']} waiting for a promotion, {result['approved']} promoted; "
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
