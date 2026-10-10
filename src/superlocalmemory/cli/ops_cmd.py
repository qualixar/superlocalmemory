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

import http.client as _hclient
import json as _json
import sys
import urllib.error as _uerr
import urllib.request as _urq
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.daemon_paths import InvalidDaemonId, describe, validate_daemon_id
from superlocalmemory.core import outbound_http as _outbound


_VALID_ACTIONS = ("retry", "force_reconcile", "cancel")

# Issue #148: building or sending a request can itself raise -- a non-ASCII
# path segment fails deep inside http.client's request-line encoding
# (UnicodeEncodeError), and an ASCII-but-unsafe one (a space, a control
# character) fails its path validation (http.client.InvalidURL). Both are
# malformed-input errors, never caught by the HTTPError/URLError handling
# below, so without this they reached main() as a raw traceback. Listed
# here as the one place every daemon call in this module converts them.
_REQUEST_BUILD_ERRORS = (UnicodeEncodeError, UnicodeDecodeError, ValueError, _hclient.InvalidURL)


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


def _owned_daemon_port() -> int:
    """The port of this account's own daemon, or exit with a plain message.

    On a computer shared by several accounts the port may be answered by
    another account's SuperLocalMemory, so the operation request waits until
    the daemon proves it is this account's.
    """
    port = _get_daemon_port()
    try:
        from superlocalmemory.cli.daemon import owned_daemon_answers

        owned = owned_daemon_answers(port)
    except Exception:
        owned = False
    if not owned:
        _die(
            f"Your SLM daemon is not answering on port {port} (if something "
            "answers there, it is not your SuperLocalMemory).\n"
            "Make sure the daemon is running: slm serve"
        )
    return port


def _daemon_get(path: str, timeout_s: float = 10.0) -> dict | None:
    """HTTP GET to the daemon; return parsed JSON or None on failure."""
    port = _owned_daemon_port()
    url = f"http://127.0.0.1:{port}{path}"
    try:
        # Only the connect+send phase is wrapped: this is where
        # http.client actually raises on a bad path (see
        # _REQUEST_BUILD_ERRORS below). Reading and decoding the response
        # happens after, outside this try, so a malformed response body
        # is never misreported as an invalid request path.
        resp = _outbound.urlopen(url, timeout=timeout_s)
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
    except _REQUEST_BUILD_ERRORS:
        # Defense in depth: every caller of _daemon_get is expected to
        # validate a user-supplied path segment first (see
        # cli/daemon_paths.py), so this should never fire -- but a request
        # that cannot be built or sent must still end in a friendly
        # one-liner, not whatever stdlib exception leaked out of
        # http.client.
        _die(f"invalid request path {describe(path)}")
        return None  # unreachable; _die exits
    with resp:
        raw = resp.read().decode()
    return _json.loads(raw)


def _daemon_post(path: str, body: dict, timeout_s: float = 10.0) -> dict | None:
    """HTTP POST to the daemon; return parsed JSON or None on failure."""
    port = _owned_daemon_port()
    url = f"http://127.0.0.1:{port}{path}"
    try:
        req = _urq.Request(
            url,
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Same split as _daemon_get: only build+connect+send is wrapped.
        resp = _outbound.urlopen(req, timeout=timeout_s)
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
    except _REQUEST_BUILD_ERRORS:
        # Same defense in depth as _daemon_get -- see the comment there.
        _die(f"invalid request path {describe(path)}")
        return None  # unreachable; _die exits
    with resp:
        raw = resp.read().decode()
    return _json.loads(raw)


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
        # Same bug class as operation_id: profile is interpolated into a
        # daemon URL (here, a query value). The daemon's own
        # validate_profile_name (server/routes/helpers.py) already
        # restricts every profile it will accept to ^[a-zA-Z0-9_-]+$.
        try:
            validate_daemon_id(profile, label="--profile value")
        except InvalidDaemonId:
            _die(f"invalid --profile value {describe(profile)}")
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
            if entry.get("error"):
                print(f"    error: {entry['error']}")
            if entry.get("what_happened"):
                print(f"    {entry['what_happened']}")
        print()

    print("Use `slm ops resolve <id> --action cancel|retry|force_reconcile` to remediate.")


def _cmd_ops_resolve(args: Namespace) -> None:
    """Admin action on a specific operation."""
    operation_id: str = args.operation_id
    action: str = args.action

    if action not in _VALID_ACTIONS:
        _die(f"--action must be one of: {', '.join(_VALID_ACTIONS)}")

    # Issue #148: operation_id is interpolated straight into a daemon URL
    # path below. The daemon only ever issues uuid.uuid4().hex-shaped IDs
    # (see core/operation_request.py), so anything outside
    # [A-Za-z0-9_-]+ -- a pasted "..." placeholder, a typo with a space or
    # slash -- cannot be a real ID. Reject it here, before it is ever
    # interpolated into a path or reaches a socket, instead of letting
    # http.client discover the problem by crashing on it.
    try:
        validate_daemon_id(operation_id, label="operation ID")
    except InvalidDaemonId:
        _die(
            f"invalid operation ID {describe(operation_id)} "
            "— run 'slm ops list' to see valid IDs"
        )

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


def _status_with_key() -> dict | None:
    """/status asked through the daemon client, which sends this computer's key."""
    try:
        from superlocalmemory.cli.daemon import daemon_request

        found = daemon_request("GET", "/status")
    except Exception:  # noqa: BLE001 - the caller reports a plain message
        return None
    return found if isinstance(found, dict) else None


def _cmd_ops_status(args: Namespace) -> None:
    """Quick ops-focused health status from the daemon."""
    data = _daemon_get("/status")
    if data is None:
        return
    if data.get("details_hidden"):
        # Where everyone must sign in, a bare request gets the short answer and
        # its zeros would read as "healthy". Ask again with this computer's key.
        data = _status_with_key() or data
        if data.get("details_hidden"):
            _die("This workspace requires sign-in to show operations status. "
                 "Sign in, or set SLM_USER_SESSION, then run it again.")

    fields = {
        "dead_letter_count": data.get("dead_letter_count", 0),
        "degraded_operations": data.get("degraded_operations", 0),
        "exhausted_obligations": data.get("exhausted_obligations", 0),
        "writer_stalled": data.get("writer_stalled", False),
        "writer_stalled_op_id": data.get("writer_stalled_op_id"),
        "writer_stalled_age_s": data.get("writer_stalled_age_s"),
        "unreadable_saves": data.get("unreadable_saves", 0),
    }

    if getattr(args, "json", False):
        _print_json(fields)
        return

    total_issues = (
        fields["dead_letter_count"]
        + fields["degraded_operations"]
        + fields["exhausted_obligations"]
        + max(0, int(fields["unreadable_saves"] or 0))
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
    if fields["unreadable_saves"] and fields["unreadable_saves"] > 0:
        print(f"  unreadable saves       : {fields['unreadable_saves']} (kept, set aside)")
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
