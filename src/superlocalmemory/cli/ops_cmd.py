# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""``slm ops`` — operational recovery & admin remediation.

Subcommands:

* ``slm ops list [--profile P]``
      Show all failed, stuck, or degraded operations grouped by category.
      Proxies GET /operations/failed on the running daemon.

* ``slm ops resolve <id> --action {retry|force_reconcile|cancel}``
      Admin action on a specific operation.
      Proxies POST /operations/<id>/resolve on the running daemon.

* ``slm ops status``
      Quick overview: failure counts + writer stall state from /status.
      No authentication required (status is public).

RBAC: list / resolve require OWNER or ADMIN role on the daemon.
Unauthenticated users see a clear permission error.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json as _json
import sys
import urllib.error as _uerr
import urllib.request as _urq
from argparse import Namespace
from typing import Any


_VALID_ACTIONS = ("retry", "force_reconcile", "cancel")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_daemon_port() -> int:
    """Return the active daemon port (default 8765)."""
    try:
        from superlocalmemory.cli.daemon import _get_port
        return _get_port()
    except Exception:
        return 8765


def _daemon_get(path: str, timeout_s: float = 10.0) -> dict | None:
    """HTTP GET to the daemon; return parsed JSON or None on failure."""
    port = _get_daemon_port()
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with _urq.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
            raw = resp.read().decode()
        return _json.loads(raw)
    except _uerr.HTTPError as exc:
        if exc.code == 403:
            _die(
                "Permission denied: list/resolve requires OWNER or ADMIN role.\n"
                "Check your SLM credentials or ask your administrator."
            )
        body = exc.read().decode(errors="replace") if hasattr(exc, "read") else str(exc)
        _die(f"Daemon returned HTTP {exc.code}: {body}")
    except _uerr.URLError as exc:
        _die(
            f"Could not reach SLM daemon at {url}: {exc.reason}\n"
            "Make sure the daemon is running: slm serve"
        )
    return None  # unreachable; _die exits


def _daemon_post(path: str, body: dict, timeout_s: float = 10.0) -> dict | None:
    """HTTP POST to the daemon; return parsed JSON or None on failure."""
    port = _get_daemon_port()
    url = f"http://127.0.0.1:{port}{path}"
    try:
        req = _urq.Request(
            url,
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urq.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            raw = resp.read().decode()
        return _json.loads(raw)
    except _uerr.HTTPError as exc:
        if exc.code == 403:
            _die(
                "Permission denied: resolve requires OWNER or ADMIN role.\n"
                "Check your SLM credentials or ask your administrator."
            )
        if exc.code == 400:
            body_txt = exc.read().decode(errors="replace") if hasattr(exc, "read") else str(exc)
            _die(f"Bad request: {body_txt}")
        body_txt = exc.read().decode(errors="replace") if hasattr(exc, "read") else str(exc)
        _die(f"Daemon returned HTTP {exc.code}: {body_txt}")
    except _uerr.URLError as exc:
        _die(
            f"Could not reach SLM daemon at {url}: {exc.reason}\n"
            "Make sure the daemon is running: slm serve"
        )
    return None  # unreachable; _die exits


def _die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def _print_json(data: Any) -> None:
    print(_json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_ops_list(args: Namespace) -> None:
    """List all failed, stuck, or degraded operations."""
    profile = getattr(args, "profile", None)
    path = "/operations/failed"
    if profile:
        path = f"{path}?profile={profile}"

    data = _daemon_get(path)
    if data is None:
        return

    if getattr(args, "json", False):
        _print_json(data)
        return

    total: int = data.get("total", 0)
    if total == 0:
        print("All operations healthy. No failures detected.")
        return

    print(f"Failed operations: {total} total\n")

    dead_letter = data.get("dead_letter", [])
    if dead_letter:
        print(f"--- Dead-letter (ingestion exhausted, {len(dead_letter)}) ---")
        for entry in dead_letter:
            print(
                f"  [{entry.get('operation_id', '?')}] "
                f"type={entry.get('operation_type', '?')} "
                f"attempts={entry.get('attempts', '?')} "
                f"profile={entry.get('profile_id', '?')}"
            )
            if entry.get("error"):
                print(f"    error: {entry['error']}")
        print()

    degraded = data.get("degraded_manifests", [])
    if degraded:
        print(f"--- Degraded manifests ({len(degraded)}) ---")
        for entry in degraded:
            print(
                f"  [{entry.get('operation_id', '?')}] "
                f"state={entry.get('state', '?')} "
                f"profile={entry.get('profile_id', '?')}"
            )
        print()

    exhausted = data.get("exhausted_obligations", [])
    if exhausted:
        print(f"--- Exhausted projection obligations ({len(exhausted)}) ---")
        for entry in exhausted:
            print(
                f"  [{entry.get('operation_id', '?')}] "
                f"kind={entry.get('kind', '?')} "
                f"attempts={entry.get('attempts', '?')} "
                f"profile={entry.get('profile_id', '?')}"
            )
        print()

    print("Use `slm ops resolve <id> --action cancel|retry|force_reconcile` to remediate.")


def _cmd_ops_resolve(args: Namespace) -> None:
    """Admin action on a specific operation."""
    operation_id: str = args.operation_id
    action: str = args.action

    if action not in _VALID_ACTIONS:
        _die(f"--action must be one of: {', '.join(_VALID_ACTIONS)}")

    result = _daemon_post(
        f"/operations/{operation_id}/resolve",
        {"action": action},
    )
    if result is None:
        return

    if getattr(args, "json", False):
        _print_json(result)
        return

    success = result.get("success", False)
    if success:
        print(
            f"OK: operation {operation_id!r} resolved with action={action!r}. "
            f"{result.get('message', '')}"
        )
    else:
        reason = result.get("reason") or result.get("error") or "unknown reason"
        print(f"Resolve failed: {reason}", file=sys.stderr)
        sys.exit(1)


def _cmd_ops_status(args: Namespace) -> None:
    """Quick ops-focused health status from the daemon."""
    data = _daemon_get("/status")
    if data is None:
        return

    fields = {
        "dead_letter_count": data.get("dead_letter_count", 0),
        "degraded_operations": data.get("degraded_operations", 0),
        "exhausted_obligations": data.get("exhausted_obligations", 0),
        "writer_stalled": data.get("writer_stalled", False),
        "writer_stalled_op_id": data.get("writer_stalled_op_id"),
        "writer_stalled_age_s": data.get("writer_stalled_age_s"),
    }

    if getattr(args, "json", False):
        _print_json(fields)
        return

    total_issues = (
        fields["dead_letter_count"]
        + fields["degraded_operations"]
        + fields["exhausted_obligations"]
    )
    stalled = fields["writer_stalled"]

    if total_issues == 0 and not stalled:
        print("Operations status: HEALTHY — no failures detected.")
        return

    print("Operations status: DEGRADED\n")
    if fields["dead_letter_count"]:
        print(f"  dead-letter entries    : {fields['dead_letter_count']}")
    if fields["degraded_operations"]:
        print(f"  degraded manifests     : {fields['degraded_operations']}")
    if fields["exhausted_obligations"]:
        print(f"  exhausted obligations  : {fields['exhausted_obligations']}")
    if stalled:
        op_id = fields["writer_stalled_op_id"] or "?"
        age = fields["writer_stalled_age_s"]
        age_str = f" (age {age:.1f}s)" if age is not None else ""
        print(f"  writer STALLED         : op={op_id}{age_str}")
    print("\nRun `slm ops list` to see details, or check the dashboard.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def cmd_ops(args: Namespace) -> None:
    """Dispatch ``slm ops`` subcommands."""
    sub = getattr(args, "ops_command", None)
    handlers = {
        "list": _cmd_ops_list,
        "resolve": _cmd_ops_resolve,
        "status": _cmd_ops_status,
    }
    handler = handlers.get(sub)
    if handler:
        handler(args)
    else:
        print("Usage: slm ops <list|resolve|status> [options]")
        print("  slm ops list [--profile P] [--json]")
        print("  slm ops resolve <id> --action {retry|force_reconcile|cancel} [--json]")
        print("  slm ops status [--json]")
        sys.exit(1)
