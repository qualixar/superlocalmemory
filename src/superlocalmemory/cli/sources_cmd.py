# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm sources`` - connect folders and notes vaults from the terminal, through the running daemon.

    slm sources add PATH [--kind folder|obsidian] [--yes]
    slm sources list
    slm sources report ID
    slm sources rescan ID
    slm sources remove ID [--purge] [--yes]

Every subcommand is a thin client of ``/api/v3/sources`` (the routes the dashboard uses). Nothing
here reads the folder. ``add`` shows what would be read and asks first; ``--purge`` erases the
memories the folder gave and needs the typed word ``erase``. ``--yes`` answers both questions;
without a terminal and without ``--yes`` the command refuses. ``--json`` gives JSON.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.daemon import (
    DaemonConflict,
    DaemonNotFound,
    DaemonUnprocessable,
    daemon_request,
)
from superlocalmemory.cli.daemon_paths import InvalidDaemonId, describe, validate_daemon_id

_BASE = "/api/v3/sources"
_NOT_RUNNING = "The SLM daemon is not running (or did not answer). Start it with: slm serve"
_NEEDS_YES = "There is no terminal to ask on. Run again with --yes to confirm."
_ERASE_WORD = "erase"


class _Stop(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _is_tty() -> bool:
    return bool(sys.stdin and sys.stdin.isatty())


class _Out:
    def __init__(self, args: Namespace, command: str) -> None:
        self.as_json, self.command = bool(getattr(args, "json", False)), command

    def emit(self, data: dict, text: str) -> None:
        if self.as_json:
            from superlocalmemory.cli.json_output import json_print

            json_print(self.command, data=data)
        else:
            print(text)

    def fail(self, message: str, code: int = 1) -> None:
        if self.as_json:
            from superlocalmemory.cli.json_output import json_print

            json_print(self.command, error={"message": message})
        else:
            print(message)
        raise _Stop(code)


def _detail_text(raw: Any) -> str:
    """The sentence inside a ``{'code': ..., 'message': ...}`` detail, or the text as it is."""
    text = str(raw)
    if text.startswith("{"):
        try:
            found = ast.literal_eval(text)
            if isinstance(found, dict) and found.get("message"):
                return str(found["message"])
        except (ValueError, SyntaxError):
            pass
    return text


def _request(out: _Out, method: str, path: str, body: dict | None = None) -> dict:
    try:
        result = daemon_request(method, path, body, preserve_conflict=True, preserve_not_found=True,
                                preserve_unprocessable=True)
    except DaemonConflict as exc:
        out.fail(_detail_text(getattr(exc, "detail", exc)))
    except DaemonNotFound as exc:
        out.fail(exc.message)
    except DaemonUnprocessable as exc:
        out.fail(exc.message, 2)
    if result is None:
        out.fail(_NOT_RUNNING)
    return result  # type: ignore[return-value]


def _source_id(out: _Out, args: Namespace) -> str:
    try:
        return validate_daemon_id(args.source_id, label="folder ID")
    except InvalidDaemonId:
        out.fail(f"invalid folder ID {describe(args.source_id)} - run 'slm sources list' to see them", 2)
    return ""


def _preview_text(p: dict) -> str:
    types = ", ".join(f"{n} {ext or '(no extension)'}" for ext, n in sorted(p["files_by_type"].items()))
    lines = [f"Folder: {p['root']} ({p['kind']})",
             f"Would read: {sum(p['files_by_type'].values())} files ({types or 'none'}), about "
             f"{p['est_bytes'] / 1024:.0f} KB",
             f"Left out by rule: {sum(p['skipped_by_rule'].values())}"
             + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(p["skipped_by_rule"].items())) + ")"
                if p["skipped_by_rule"] else ""),
             f"Held back as holding a secret: {p['quarantined_count']}",
             f"Time: {p['est_seconds']} s. {p.get('estimate_note', '')}".strip()]
    lines += [f"Warning: {w}" for w in p.get("warnings") or []]
    return "\n".join(lines)


def _confirm(out: _Out, args: Namespace, question: str, word: str = "") -> bool:
    """``--yes``, or an answer typed at the terminal; refuses when there is no terminal to ask."""
    if args.yes:
        return True
    if not _is_tty():
        out.fail(_NEEDS_YES, 2)
    answer = input(question).strip()
    return answer == word if word else answer.lower() in ("y", "yes")


def _add(args: Namespace) -> None:
    out = _Out(args, "sources add")
    if not args.yes and not _is_tty():
        out.fail(_NEEDS_YES, 2)  # before the folder is even looked at
    body: dict[str, Any] = {"path": os.path.abspath(os.path.expanduser(args.path))}
    if args.kind:
        body["kind"] = args.kind
    preview = _request(out, "POST", _BASE, body)
    if not out.as_json:
        print(_preview_text(preview))
    if not _confirm(out, args, "Connect this folder? Its files will only be read, never changed. [y/N] "):
        out.fail("Not connected. Nothing was saved.")
    _request(out, "POST", f"{_BASE}/{preview['source_id']}/confirm")
    out.emit({"source_id": preview["source_id"], "confirmed": True, "preview": preview},
             f"Connected {preview['root']} as {preview['source_id']}. The first scan has started.")


def _list(args: Namespace) -> None:
    out = _Out(args, "sources list")
    found = _request(out, "GET", _BASE)
    rows = found.get("sources") or []
    lines = [f"{s['source_id']}  {s['state']:<8} {s['kind']:<8} {s['root_path']}  "
             f"files: {sum((s.get('files') or {}).values())}"
             + (f"  ({s['offline_reason']})" if s.get("offline_reason") else "") for s in rows]
    out.emit(found, "\n".join(lines) if lines else "No folders are connected. Add one with: slm sources add PATH")


def _report_text(r: dict) -> str:
    lines = [f"Folder {r['source_id']}: {r['state']}"
             + (f" ({r['offline_reason']})" if r.get("offline_reason") else "")
             + (f", paused: {r['paused_reason']}" if r.get("paused_reason") else ""),
             "Files: " + (", ".join(f"{k} {v}" for k, v in sorted(r["counts"].items())) or "none"),
             "Left out by rule: " + (", ".join(f"{k} {v}" for k, v in sorted(r["skipped_by_rule"].items())) or "none"),
             f"Last scan: {r.get('last_scan_at') or 'not yet'}",
             "Changes are noticed at once." if r.get("watch")
             else "Changes are picked up by the scan every 15 minutes."]
    lines += [f"Held back (secret): {q['relpath']}  {q['reason']}" for q in r["quarantined"]]
    lines += [f"Only in the cloud, not read: {rel}" for rel in r["cloud_only"]]
    lines += [f"Error: {e['relpath']}  {e['reason']}" for e in r["errors"]]
    if r.get("capped"):
        lines.append("The folder is over the file limit; only the first files are indexed.")
    return "\n".join(lines)


def _report(args: Namespace) -> None:
    out = _Out(args, "sources report")
    report = _request(out, "GET", f"{_BASE}/{_source_id(out, args)}/report")
    out.emit(report, _report_text(report))


def _rescan(args: Namespace) -> None:
    out = _Out(args, "sources rescan")
    job = _request(out, "POST", f"{_BASE}/{_source_id(out, args)}/rescan")
    out.emit(job, f"Scan {job.get('state', 'queued')}.")


def _remove(args: Namespace) -> None:
    out = _Out(args, "sources remove")
    sid = _source_id(out, args)
    purge = bool(args.purge)
    if purge and not _confirm(out, args, f"This erases every memory this folder gave. Type '{_ERASE_WORD}' to go on: ",
                              _ERASE_WORD):
        out.fail("Nothing was erased.")
    done = _request(out, "DELETE", f"{_BASE}/{sid}" + ("?purge=true" if purge else ""))
    out.emit(done, f"Removed {sid}." + (" Its memories were erased." if purge
                                        else " Its memories stay, hidden from the folder's files; add --purge to erase them."))


_COMMANDS = {"add": _add, "list": _list, "report": _report, "rescan": _rescan, "remove": _remove}


def cmd_sources(args: Namespace) -> int:
    action = getattr(args, "sources_command", None) or "list"
    try:
        _COMMANDS[action](args)
    except _Stop as stop:
        return stop.code
    return 0


def _json_flag(parser: Any) -> None:
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable output")


def register_sources_parser(sub: Any) -> None:
    p = sub.add_parser("sources", help="Connect folders and notes vaults (add/list/report/rescan/remove)")
    _json_flag(p)
    cmds = p.add_subparsers(dest="sources_command")
    add = cmds.add_parser("add", help="Check a folder, show what would be read, and connect it")
    add.add_argument("path")
    add.add_argument("--kind", choices=("folder", "obsidian"), default=None)
    add.add_argument("--yes", action="store_true", help="Connect without asking")
    lst = cmds.add_parser("list", help="Connected folders")
    for name, text in (("report", "What a folder skipped, held back or failed on"),
                       ("rescan", "Scan a folder now")):
        sp = cmds.add_parser(name, help=text)
        sp.add_argument("source_id")
        _json_flag(sp)
    rm = cmds.add_parser("remove", help="Disconnect a folder")
    rm.add_argument("source_id")
    rm.add_argument("--purge", action="store_true", help="Also erase the memories it gave")
    rm.add_argument("--yes", action="store_true", help="Do not ask")
    for sp in (add, lst, rm):
        _json_flag(sp)
    p.set_defaults(yes=False, purge=False)


__all__ = ["cmd_sources", "register_sources_parser"]
