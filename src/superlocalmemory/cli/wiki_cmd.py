# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm wiki``: your memories as Markdown pages, and your edits to them as proposals (WS-6).

    slm wiki [status] [--json]      # pages, the folder, edits waiting
    slm wiki diff [--json]          # what a build would change; files holding your edits
    slm wiki build [--json]         # write the approved pages to the folder
    slm wiki rebuild                # you, at a terminal: compile the pages again and approve them
    slm wiki sync                   # you, at a terminal: read your edits back as proposals
    slm wiki dir [PATH|--default]   # show or choose the folder

Every sentence of a page is a word-for-word span of a memory it cites, or a fixed template
over slot values; no model writes any of it, and the page says how many sentences it
withheld. A page is derived: it is never saved as a memory. The Markdown folder is off until
you run ``slm flags set compiler.wiki on`` (``shadow`` shows what would be written).

Edit a sentence in a file and run ``slm wiki sync``: the edit becomes a proposal you review
with ``slm proposals``; nothing changes until you apply it. Deleting a line changes nothing,
and erasing a memory removes it from the pages and the files.

``status``, ``diff`` and ``build`` work anywhere (``--json`` for agents); ``rebuild`` and
``sync`` ask a person: they refuse to run without an interactive terminal.
"""

from __future__ import annotations

import sys
import urllib.parse
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.corrections_cmd import _daemon, _out, _refused

_PERSON = {"rebuild": "Compile the pages again from your memories and approve them?",
           "sync": "Read your edited Markdown back as proposals?"}
_STATES = {"new": "not written yet", "changed": "will be rewritten", "edited": "you edited it "
           "(kept; run slm wiki sync)", "adopted": "already holds this text", "stale": "its page "
           "is gone"}


def register_wiki_parsers(sub: Any) -> None:
    """Attach ``slm wiki``. Called from cli/main.py."""
    p = sub.add_parser("wiki", help="Your memories as Markdown pages; your edits become proposals")
    wsub = p.add_subparsers(dest="wiki_command")
    for name, text in (("status", "Pages, the folder and edits waiting (the default)"),
                       ("diff", "What a build would change, and files holding your edits"),
                       ("build", "Write the approved pages to the folder (needs compiler.wiki)"),
                       ("rebuild", "Compile the pages again and approve them (you, at a terminal)"),
                       ("sync", "Read your edits back as proposals (you, at a terminal)")):
        c = wsub.add_parser(name, help=text)
        c.add_argument("--profile", default="", help="Profile (default: the active one)")
        c.add_argument("--json", action="store_true", help="machine-readable output")
    d = wsub.add_parser("dir", help="Show or choose the folder the pages are written to")
    d.add_argument("path", nargs="?", default="", help="Folder to use")
    d.add_argument("--default", action="store_true", help="Go back to the default folder")
    d.add_argument("--json", action="store_true", help="machine-readable output")


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _query(profile: str) -> str:
    return "?profile_id=" + urllib.parse.quote(profile, safe="") if profile else ""


def _show_status(data: dict) -> None:
    status = data["status"]
    print(f"Markdown wiki: {data['flag']}   folder: {status['dir']}")
    kinds = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(status["by_kind"].items()))
    print(f"Pages: {status['pages']}" + (f" ({kinds})" if kinds else "")
          + f"   waiting for review: {status['pending_sets']}")
    print(f"Files: {status['files']}   edits waiting to be read: {status['differences'].get('edited', 0)}"
          f"   proposals from edits: {status['edits_waiting']}")
    if data["flag"] == "off":
        print("The folder is off. Turn it on with: slm flags set compiler.wiki on")


def _show_diff(data: dict) -> None:
    if not data["items"]:
        print("The folder matches the pages.")
    for item in data["items"]:
        print(f"  {item['path']}  {_STATES.get(item['state'], item['state'])}")


def _show_build(data: dict) -> None:
    lead = "Would write" if data["dry_run"] else "Wrote"
    print(f"{lead} {len(data['written'])} file(s) to {data['dir']}; "
          f"{len(data['unchanged'])} unchanged, {len(data['removed'])} removed.")
    for rel in data["kept_edited"]:
        print(f"  kept {rel}: you edited it. Run slm wiki sync first.")


def _show_sync(data: dict) -> None:
    lead = "Found" if data["dry_run"] else "Proposed"
    print(f"{lead} {data['found']} edit(s) in {data['files']} file(s); "
          f"{data['deleted_lines']} deleted line(s) ignored (a deleted line forgets nothing).")
    for item in data["queued"]:
        print(f"  proposal {item['set_id']} from {item['path']}")
    for item in data["skipped"]:
        print(f"  left alone: {item['path']}: {item['why']}")
    if data["queued"]:
        print("Review them with: slm proposals")


def _show_rebuild(data: dict) -> None:
    if not data["pages"]:
        print("The pages are up to date. Nothing to approve.")
    else:
        print(f"Compiled and approved {data['pages']} page(s).")
    if data.get("files"):
        _show_build(data["files"])


_RENDER = {"status": _show_status, "diff": _show_diff, "build": _show_build,
           "sync": _show_sync, "rebuild": _show_rebuild}


def _dir_command(args: Namespace, use_json: bool) -> int:
    from superlocalmemory.compiler import wiki

    command = "wiki dir"
    try:
        if getattr(args, "default", False):
            now = wiki.set_wiki_dir(None)
        elif getattr(args, "path", ""):
            now = wiki.set_wiki_dir(args.path)
        else:
            now = wiki.wiki_root()
    except (OSError, ValueError) as exc:
        return _refused(command, str(exc), use_json, "INVALID")
    if use_json:
        _out(command, True, data={"dir": str(now)})
    else:
        print(f"Wiki folder: {now}  (one folder per profile inside it)")
    return 0


def _call(command: str, sub: str, profile: str, use_json: bool) -> int:
    from superlocalmemory.cli.daemon import (
        DaemonConflict,
        DaemonNotFound,
        DaemonRefused,
        DaemonUnprocessable,
        daemon_request,
    )

    try:
        if sub == "status":
            data = daemon_request("GET", "/api/wiki/pages" + _query(profile),
                                  preserve_not_found=True, preserve_unprocessable=True)
        elif sub == "diff":
            data = daemon_request("GET", "/api/wiki/diff" + _query(profile),
                                  preserve_not_found=True, preserve_unprocessable=True)
        else:
            body = {"channel": "cli", **({"profile_id": profile} if profile else {})}
            data = daemon_request("POST", f"/api/wiki/{sub}", body, preserve_conflict=True,
                                  preserve_not_found=True, preserve_unprocessable=True)
    except DaemonRefused:
        return _refused(command, "SuperLocalMemory refused: the living wiki is for this "
                                 "computer, decided by a person.", use_json, "NEEDS_A_PERSON")
    except DaemonConflict as exc:
        return _refused(command, exc.detail, use_json)
    except DaemonUnprocessable as exc:
        return _refused(command, exc.message, use_json, exc.code)
    except DaemonNotFound as exc:
        return _refused(command, exc.error_message or exc.message, use_json, "NOT_FOUND")
    if not isinstance(data, dict) or not data.get("success"):
        return _refused(command, str((data or {}).get("detail") or "SLM refused the request"),
                        use_json)
    if use_json:
        _out(command, True, data=data)
    else:
        _RENDER[sub](data)
    return 0


def cmd_wiki(args: Namespace) -> int:
    sub = getattr(args, "wiki_command", None) or "status"
    use_json = bool(getattr(args, "json", False))
    command = f"wiki {sub}"
    if sub == "dir":
        return _dir_command(args, use_json)
    if sub in _PERSON and not (sys.stdin.isatty() and sys.stdout.isatty()):
        return _refused(command, f"slm wiki {sub} asks a person at a terminal; it cannot be done "
                                 "by a script or an agent. Use slm wiki status --json to read "
                                 "what is there.", use_json, "NEEDS_A_PERSON")
    from superlocalmemory.cli.commands import _daemon_unavailable

    if not _daemon():
        _daemon_unavailable(command, use_json)
        return 1
    profile = (getattr(args, "profile", "") or "").strip()
    if sub in _PERSON and not _confirm(_PERSON[sub]):
        print("Nothing was changed.")
        return 0
    return _call(command, sub, profile, use_json)


def run(args: Namespace) -> None:
    """The ``slm wiki`` entry point: exits with the command's status."""
    rc = cmd_wiki(args)
    if rc:
        sys.exit(rc)


__all__ = ["cmd_wiki", "register_wiki_parsers", "run"]
