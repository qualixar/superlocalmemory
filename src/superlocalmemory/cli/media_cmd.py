# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm media`` - turn images and documents on or off through the running daemon.

The install runs in the daemon, never in this process: when the daemon is not
running this says so and points at ``slm restart``.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.daemon import DaemonConflict, daemon_request
from superlocalmemory.cli.features_cmd import (
    EXIT_DAEMON_DOWN, FEATURES_PATH, NOT_RUNNING, die, media_line,
)

SIZE_TEXT = "about 1.5 GB"
EXIT_LOW_RAM = 4
REPAIR_TIMEOUT_S = 300.0  # one run works for up to four minutes


def _is_tty() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _call(args: Namespace, method: str, path: str, body: dict | None = None, **kwargs: Any) -> dict[str, Any]:
    data = daemon_request(method, path, body, **kwargs)
    if data is None:
        die(args, NOT_RUNNING, EXIT_DAEMON_DOWN)
    return data


def _emit(args: Namespace, data: dict[str, Any], text: str) -> None:
    if getattr(args, "json", False):
        from superlocalmemory.cli.json_output import json_print

        json_print(f"media {args.media_command}", data=data)
    else:
        print(text)


def _disk_text(precheck: dict[str, Any]) -> str:
    if not precheck:
        return "Disk check: not available"
    free = float(precheck.get("free_bytes", 0)) / 1024 ** 3
    verdict = "enough" if precheck.get("disk_ok") else "NOT enough"
    return f"Disk check: {verdict} free space ({free:.1f} GB free)"


def _confirmed(args: Namespace) -> bool:
    if getattr(args, "yes", False):
        return True
    if not _is_tty():
        die(args, "Turning this on downloads files. Run again with --yes to confirm.", 2)
    try:
        answer = input("Turn on images and documents now? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def _enable(args: Namespace) -> None:
    media = _call(args, "GET", FEATURES_PATH)["media"]
    if media.get("ram_ok") is False:  # refused here: say so plainly, ask nothing
        die(args, str(media.get("ram_message") or ""), EXIT_LOW_RAM)
    if not getattr(args, "json", False):
        print(f"Images & documents download {SIZE_TEXT} of models.")
        print(_disk_text(media.get("precheck", {})))
    if not _confirmed(args):
        print("Nothing changed.")
        return
    try:
        reply = _call(args, "POST", FEATURES_PATH + "/media/enable", {"yes": True, "source": "cli"},
                      preserve_conflict=True)
    except DaemonConflict as exc:  # the daemon refused (not enough memory)
        die(args, str(exc), EXIT_LOW_RAM)
    _emit(args, reply, _enable_text(reply.get("media") or {}))


def _enable_text(media: dict[str, Any]) -> str:
    state = media.get("env_state", "")
    step = str(media.get("step") or "").strip()
    if state == "unsupported":
        return ("Saved your choice, but images & documents can't be set up on this computer yet"
                + (f": {step}" if step else "") + ". Nothing is downloaded.")
    if state == "failed":
        return "Saved your choice, but set-up failed" + (f" ({step})" if step else "") + ". See: slm doctor"
    return "Turning on. The set-up runs in the background; check progress with: slm media status"


def _disable(args: Namespace) -> None:
    reply = _call(args, "POST", FEATURES_PATH + "/media/disable",
                  {"remove_files": bool(getattr(args, "remove_files", False))})
    _emit(args, reply, "Images & documents are off. Your memories are kept.")


def _status(args: Namespace) -> None:
    data = _call(args, "GET", FEATURES_PATH)
    _emit(args, data, "Images & documents: " + media_line(data["media"]))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _gc_text(report: dict[str, Any]) -> str:
    linked = int(report.get("anchors_filled", 0))
    if not report.get("dry_run", True):
        text = (f"Removed {_plural(int(report.get('rows_removed', 0)), 'picture record')} and "
                f"{_plural(int(report.get('files_removed', 0)), 'file')}. Your memories are untouched.")
        return text + (f" Linked {_plural(linked, 'picture')} to its memory." if linked else "")
    rows = len(report.get("rows_without_memory") or [])
    files = len(report.get("files_without_row") or [])
    if not rows and not files and not linked:
        return "Nothing to clean up."
    return (f"Found {_plural(rows, 'picture record')} without a memory, {_plural(files, 'file')} "
            f"without a record and {_plural(linked, 'picture')} not yet linked to its memory. "
            "Nothing was changed; to fix them run: slm media gc --apply")


def _gc(args: Namespace) -> None:
    """Report leftovers (default) or remove them with --apply (needs the owner or an admin)."""
    report = _call(args, "POST", "/api/v3/media/gc", {"dry_run": not getattr(args, "apply", False)})
    _emit(args, report, _gc_text(report))


def _repair_text(report: dict[str, Any]) -> str:
    missing, done = int(report.get("missing", 0)), int(report.get("repaired", 0))
    if report.get("dry_run"):
        extra = " The picture index also needs rebuilding for the current model." if report.get(
            "index_needs_rebuild") else ""
        return (f"{_plural(missing, 'picture')} can't be found by what they show yet.{extra} "
                "To fix: slm media repair") if missing or extra else "Nothing to repair."
    lines = []
    if report.get("rebuilt_index"):
        lines.append("Rebuilt the picture index for the current model.")
    lines.append(f"Repaired {done} of {_plural(missing, 'picture')}." if missing
                 else "Every picture can already be found by what it shows.")
    for key, text in (("skipped_no_file", "have no file kept, so they were skipped"),
                      ("failed", "could not be repaired")):
        if report.get(key):
            lines.append(f"{_plural(int(report[key]), 'picture')} {text}.")
    if report.get("remaining"):
        lines.append(f"{report['remaining']} left; run slm media repair again.")
    if report.get("reason"):
        lines.append(str(report["reason"]))
    return " ".join(lines)


def _repair(args: Namespace) -> None:
    """Make every saved picture findable by what it shows; --dry-run only counts."""
    report = _call(args, "POST", "/api/v3/media/repair", {"dry_run": bool(getattr(args, "dry_run", False))},
                   timeout_seconds=REPAIR_TIMEOUT_S)
    _emit(args, report, _repair_text(report))


def cmd_media(args: Namespace) -> None:
    sub = getattr(args, "media_command", None) or "status"
    args.media_command = sub
    {"enable": _enable, "disable": _disable, "status": _status, "gc": _gc, "repair": _repair}[sub](args)


def register_media_parser(sub: Any) -> None:
    p = sub.add_parser("media", help="Turn images and documents on or off")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    msub = p.add_subparsers(dest="media_command", title="media subcommands")
    flag = {"action": "store_true", "default": argparse.SUPPRESS, "help": "machine-readable output"}
    e = msub.add_parser("enable", help="turn images and documents on (downloads models)")
    e.add_argument("--yes", "-y", action="store_true", help="do not ask")
    e.add_argument("--json", **flag)
    d = msub.add_parser("disable", help="turn them off; memories are kept")
    d.add_argument("--remove-files", action="store_true", help="also delete the downloaded models")
    d.add_argument("--json", **flag)
    s = msub.add_parser("status", help="what is on and how set-up is going")
    s.add_argument("--json", **flag)
    g = msub.add_parser("gc", help="find picture leftovers (records without a memory, stray files)")
    g.add_argument("--apply", action="store_true", help="remove them (owner or admin)")
    g.add_argument("--json", **flag)
    r = msub.add_parser("repair", help="make pictures findable by what they show (re-embeds them; owner or admin)")
    r.add_argument("--dry-run", action="store_true", help="only count what needs repair")
    r.add_argument("--json", **flag)
