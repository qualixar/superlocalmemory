# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm import``: bring memories in from ChatGPT, Claude, Mem0, Letta, OKF and Markdown (WS-9, D-19).

    slm import preview FILE [--format F]    what the file holds; nothing is saved
    slm import run FILE [--format F] [--wait [S]]
                                            import in the background; prints the run to watch
    slm import status [RUN_ID]              progress of a run (the latest when no id)
    slm import list                         recent imports
    slm import cancel RUN_ID                stop it; what was saved is kept
    slm import undo RUN_ID [--wait [S]]     take the whole run back, exactly

FILE is a file or a folder on this computer: ``conversations.json`` (or the numbered
``conversations-000.json`` set, or the zip ChatGPT and Claude send), a Mem0 ``get_all`` dump, a
Letta ``.af`` agent file, a folder of OKF or Markdown notes (``AGENTS.md``, ``CLAUDE.md``,
``MEMORY.md`` and topic files), or a text file of Claude's memory. The format is detected; name it
with ``--format`` (chatgpt, claude, claude-memory, mem0, letta, okf, markdown) when you know it.

The pack is off until you turn it on: ``slm flags set portability.import on``. Any caller may
import or undo (an agent through the shell, a script, you); each is recorded with who and the
channel (``cli``) and listed on the dashboard under "Recent decisions by your agents". Imported
memories sit at the bottom of the origin lattice: they never replace what you already remember,
and a contradiction waits as a question. Everything runs inside SLM, which stays the only writer.
MCP has no import tool (the tool caps); an agent runs this command. ``--json`` is for agents.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.parse
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.corrections_cmd import _daemon, _out, _refused

FORMATS = ("chatgpt", "claude", "claude-memory", "mem0", "letta", "okf", "markdown")
_DONE_STATES = ("done", "cancelled", "failed", "rolled_back")
_STATE_WORDS = {"queued": "waiting to start", "running": "importing", "done": "done",
                "cancelled": "stopped (what was saved is kept)", "failed": "stopped on a problem",
                "rolling_back": "being undone", "rolled_back": "undone"}


def register_import_parsers(sub: Any) -> None:
    """Attach ``slm import``. Called from cli/main.py."""
    p = sub.add_parser("import", help="Bring memories in from ChatGPT, Claude, Mem0, Letta, "
                                      "OKF or Markdown files: preview, run, undo")
    isub = p.add_subparsers(dest="import_command")
    for name, text in (("preview", "What a file holds; nothing is saved"),
                       ("run", "Import a file in the background (recorded; undo with 'undo')")):
        c = isub.add_parser(name, help=text)
        c.add_argument("file", help="a file or folder on this computer")
        c.add_argument("--format", default="", choices=("",) + FORMATS,
                       help="the format, when detection is not enough")
        c.add_argument("--profile", default="", help="Profile (default: the active one)")
        c.add_argument("--json", action="store_true", help="machine-readable output")
        if name == "run":
            c.add_argument("--wait", nargs="?", type=float, const=600.0, default=0.0,
                           metavar="SECONDS", help="stay until it ends (default 10 minutes)")
    for name, text in (("status", "Progress of an import (the latest when no id)"),
                       ("list", "Recent imports")):
        c = isub.add_parser(name, help=text)
        if name == "status":
            c.add_argument("run_id", nargs="?", default="")
        c.add_argument("--profile", default="")
        c.add_argument("--json", action="store_true")
    for name, text in (("cancel", "Stop a running import; what was saved is kept"),
                       ("undo", "Take a whole import back, exactly")):
        c = isub.add_parser(name, help=text)
        c.add_argument("run_id")
        c.add_argument("--profile", default="")
        c.add_argument("--json", action="store_true")
        if name == "undo":
            c.add_argument("--wait", nargs="?", type=float, const=300.0, default=0.0,
                           metavar="SECONDS", help="stay until it is undone")


def _call(method: str, path: str, body: dict | None = None, *, timeout: float = 60.0) -> Any:
    from superlocalmemory.cli.daemon import daemon_request

    return daemon_request(method, path, body, timeout_seconds=timeout, preserve_conflict=True,
                          preserve_not_found=True, preserve_unprocessable=True)


def _query(profile: str, **extra: Any) -> str:
    items = {k: v for k, v in {"profile_id": profile, **extra}.items() if v not in ("", None)}
    return ("?" + urllib.parse.urlencode(items)) if items else ""


def _source(args: Namespace) -> dict:
    body: dict = {"path": os.path.abspath(os.path.expanduser(args.file)),
                  "format": getattr(args, "format", "") or ""}
    if (args.profile or "").strip():
        body["profile_id"] = args.profile.strip()
    return body


def _bar(run: dict) -> str:
    done = int(round(float(run.get("progress") or 0.0) * 20))
    return "[" + "#" * done + "." * (20 - done) + "]"


def _line(run: dict) -> str:
    saved, total = run["saved_items"], run["total_items"]
    return (f"{run['run_id']}  {_STATE_WORDS.get(run['state'], run['state'])}  "
            f"{run['format']}  {saved:,} of {total:,} saved  {run['source_label']}")


def _show_run(run: dict) -> None:
    print(_line(run))
    if run.get("active") and run.get("working"):
        print(f"  {_bar(run)} {int(float(run.get('progress') or 0) * 100)}%")
    for item in run.get("skipped_words") or ():
        print(f"  left out: {item['count']:,} {item['words']}")
    if run.get("disputes_opened") or run.get("disputes_unlisted"):
        print(f"  contradictions with what you already had: {run['disputes_opened']} to answer"
              + (f", {run['disputes_unlisted']} more not listed (cap)" if run["disputes_unlisted"]
                 else "") + ". See: slm disputes")
    if run.get("note"):
        print(f"  {run['note']}")


def _fail(command: str, use_json: bool, exc: Exception) -> int:
    from superlocalmemory.cli.daemon import (
        DaemonConflict,
        DaemonNotFound,
        DaemonRefused,
        DaemonUnprocessable,
    )

    if isinstance(exc, DaemonRefused):
        return _refused(command, "SuperLocalMemory refused: your role on this workspace does not "
                                 "allow this.", use_json, "FORBIDDEN")
    if isinstance(exc, DaemonConflict):
        return _refused(command, exc.detail, use_json)
    if isinstance(exc, DaemonUnprocessable):
        return _refused(command, exc.message, use_json, exc.code)
    if isinstance(exc, DaemonNotFound):
        return _refused(command, exc.error_message or exc.message, use_json, "NOT_FOUND")
    raise exc


def _preview(args: Namespace, command: str, use_json: bool) -> int:
    result = _call("POST", "/api/import/preview", _source(args), timeout=300.0)
    if not isinstance(result, dict) or not result.get("success"):
        return _refused(command, "SuperLocalMemory did not answer.", use_json, "UNAVAILABLE")
    if use_json:
        _out(command, True, data=result)
        return 0
    print(result["summary"])
    print("Nothing has been saved yet. To import it: "
          f"slm import run {args.file}" + (f" --format {args.format}" if args.format else ""))
    return 0


def _wait_for(run_id: str, profile: str, seconds: float, *, until: tuple[str, ...]) -> dict | None:
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        found = _call("GET", f"/api/import/runs/{urllib.parse.quote(run_id, safe='')}"
                      + _query(profile))
        last = found.get("run") if isinstance(found, dict) else last
        if last and last["state"] in until:
            return last
        time.sleep(1.0)
    return last


def _run(args: Namespace, command: str, use_json: bool) -> int:
    result = _call("POST", "/api/import/run", _source(args), timeout=300.0)
    if not isinstance(result, dict) or not result.get("success"):
        return _refused(command, "SuperLocalMemory did not answer.", use_json, "UNAVAILABLE")
    run = result["run"]
    if args.wait:
        run = _wait_for(run["run_id"], args.profile, args.wait, until=_DONE_STATES) or run
    if use_json:
        _out(command, True, data={**result, "run": run})
        return 0
    _show_run(run)
    if run["state"] not in _DONE_STATES:
        print(f"Importing in the background. Watch it: slm import status {run['run_id']}")
    print(f"Take it back any time: slm import undo {run['run_id']}")
    return 1 if run["state"] == "failed" else 0


def _status(args: Namespace, command: str, use_json: bool) -> int:
    run_id = urllib.parse.quote(getattr(args, "run_id", "") or "", safe="")
    if run_id:
        result = _call("GET", f"/api/import/runs/{run_id}" + _query(args.profile))
    else:
        result = _call("GET", "/api/import/runs" + _query(args.profile, limit=1))
    if not isinstance(result, dict) or not result.get("success"):
        return _refused(command, "SuperLocalMemory did not answer.", use_json, "UNAVAILABLE")
    if use_json:
        _out(command, True, data=result)
        return 0
    runs_ = [result["run"]] if run_id else result["runs"]
    if not runs_:
        print("No imports yet." + ("" if result.get("enabled") else
                                   " Importing is off: slm flags set portability.import on"))
        return 0
    for run in runs_:
        _show_run(run)
    return 0


def _list(args: Namespace, command: str, use_json: bool) -> int:
    result = _call("GET", "/api/import/runs" + _query(args.profile, limit=20))
    if not isinstance(result, dict) or not result.get("success"):
        return _refused(command, "SuperLocalMemory did not answer.", use_json, "UNAVAILABLE")
    if use_json:
        _out(command, True, data=result)
        return 0
    if not result["enabled"]:
        print("Importing from other apps is off. Turn it on: slm flags set portability.import on")
    if not result["runs"]:
        print("No imports yet.")
    for run in result["runs"]:
        print(_line(run))
    return 0


def _act(args: Namespace, command: str, use_json: bool, action: str) -> int:
    path = f"/api/import/runs/{urllib.parse.quote(args.run_id, safe='')}/{action}"
    body: dict = {"wait_seconds": min(60.0, args.wait)} if action == "undo" and args.wait else {}
    if (args.profile or "").strip():
        body["profile_id"] = args.profile.strip()
    result = _call("POST", path, body, timeout=120.0)
    if not isinstance(result, dict) or not result.get("success"):
        return _refused(command, "SuperLocalMemory did not answer.", use_json, "UNAVAILABLE")
    run = result["run"]
    if action == "undo" and args.wait and run["state"] != "rolled_back":
        run = _wait_for(args.run_id, args.profile, args.wait, until=("rolled_back",)) or run
    if use_json:
        _out(command, True, data={**result, "run": run})
        return 0
    _show_run(run)
    if action == "undo" and run["state"] == "rolling_back":
        print(f"Being undone in the background. Watch it: slm import status {run['run_id']}")
    elif action == "undo":
        print("Undone. The memory is as it was before this import.")
    return 0


def cmd_import(args: Namespace) -> int:
    sub = getattr(args, "import_command", None) or "list"
    use_json = bool(getattr(args, "json", False))
    command = f"import {sub}"
    from superlocalmemory.cli.commands import _daemon_unavailable

    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    try:
        if sub == "preview":
            return _preview(args, command, use_json)
        if sub == "run":
            return _run(args, command, use_json)
        if sub == "status":
            return _status(args, command, use_json)
        if sub == "list":
            return _list(args, command, use_json)
        return _act(args, command, use_json, sub)
    except Exception as exc:  # noqa: BLE001 - a refusal in words; anything else is a real bug
        return _fail(command, use_json, exc)


def run(args: Namespace) -> None:
    """The ``slm import`` entry point: exits with the command's status."""
    rc = cmd_import(args)
    if rc:
        sys.exit(rc)


__all__ = ["FORMATS", "cmd_import", "register_import_parsers", "run"]
