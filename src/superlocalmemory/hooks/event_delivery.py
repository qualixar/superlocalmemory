# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Hand hook events to the daemon's single writer; spool what it does not ack.

A hook runs on every tool call of every host, so this path is bounded: the
daemon gets ``SLM_HOOK_DELIVERY_TIMEOUT_MS`` (default 150 ms) to ack, end to
end, and anything else -- no daemon, a refusal, ``ok: false``, a timeout --
appends the same events to the spool, where the daemon picks them up on its
next pass. A late ack after the hook gave up only produces a duplicate event
id, which the writer skips.

Two ways in, both existing hook channels to the daemon:

* the hook socket (``hooks/hook_daemon.py``; POSIX): a 0600 socket in an
  owner-only folder that only this account can connect to. No HTTP, no
  extra imports: the fast path.
* ``POST /internal/hook-events`` through ``daemon_request`` (Windows, or no
  socket): only after ``daemon.json`` names the daemon and its /health proves
  it is this account's own, with the private capability header. With no
  ``daemon.json`` there is no network call at all.

Event builders live here too, so every producer emits the same shapes the
daemon validates (``server/hook_event_writer.py``).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Mapping, Sequence

from superlocalmemory.hooks import event_spool

HOOK_EVENTS_PATH = "/internal/hook-events"
#: Longest wait for the daemon's ack.
DEFAULT_DELIVERY_TIMEOUT_MS = 150
#: The hook's whole in-process time (stdin to exit) stays under the 200 ms cap:
#: the wait ends this long after the hook started, whatever the work before
#: it cost, leaving room for the spool append and the perf log.
HOOK_BUDGET_MS = 150
_MIN_WAIT_MS = 10
#: Each summary column keeps at most this many characters (as before).
MAX_SUMMARY_LEN = 500
#: Events per request; the route refuses larger batches.
MAX_BATCH = 32

# Compiled once: this path runs on every tool call.
_SECRET_PATTERNS = (
    (re.compile(r"\b(?:sk-|pk-|api[_-]?key[_-]?)[A-Za-z0-9_-]{10,}\b"), "[REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"), "[REDACTED]"),
    (re.compile(r"password\s*[=:]\s*\S+", re.IGNORECASE), "password=[REDACTED]"),
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _event_id() -> str:
    return uuid.uuid4().hex


def _scrub(text: str) -> str:
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def summary(raw: object) -> str:
    """Truncate, then scrub credential-shaped text (bounded regex cost)."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        try:
            raw = json.dumps(raw, default=str)
        except Exception:
            try:
                raw = str(raw)
            except Exception:
                return ""
    return _scrub(raw[:MAX_SUMMARY_LEN])


# ---------------------------------------------------------------------------
# Builders (the wire shapes)
# ---------------------------------------------------------------------------

def tool_event(*, session_id: str, tool_name: str, profile_id: str,
               tool_input: object = None, response_text: str = "",
               project_path: str = "", event_type: str = "complete") -> dict:
    return {
        "kind": "tool_event", "event_id": _event_id(), "profile_id": profile_id,
        "session_id": session_id, "tool_name": tool_name[:200], "event_type": event_type,
        "project_path": (project_path or "")[:MAX_SUMMARY_LEN],
        "input_summary": summary(tool_input), "output_summary": summary(response_text),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def fact_signal(*, session_id: str, fact_ids: Sequence[str], signal_name: str,
                signal_value: object, profile_id: str) -> dict:
    return {"kind": "fact_signal", "event_id": _event_id(), "profile_id": profile_id,
            "session_id": session_id, "fact_ids": list(fact_ids),
            "signal_name": signal_name, "signal_value": signal_value,
            "observed_ms": _now_ms()}


def outcome_signal(*, outcome_id: str, signal_name: str, signal_value: object,
                   profile_id: str) -> dict:
    return {"kind": "outcome_signal", "event_id": _event_id(), "profile_id": profile_id,
            "outcome_id": outcome_id, "signal_name": signal_name,
            "signal_value": signal_value, "observed_ms": _now_ms()}


def session_finalize(*, session_id: str, profile_id: str) -> dict:
    return {"kind": "session_finalize", "event_id": _event_id(), "profile_id": profile_id,
            "session_id": session_id, "observed_ms": _now_ms()}


def active_profile() -> str:
    try:
        from superlocalmemory.hooks.session_registry import resolve_active_profile

        return resolve_active_profile() or "default"
    except Exception:
        return "default"


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

def _deadline_s(started: float | None = None) -> float:
    """Seconds to wait for an ack: the configured timeout, cut to what is left
    of the hook's budget when ``started`` (``time.perf_counter()``) is known."""
    try:
        ms = int(os.environ.get("SLM_HOOK_DELIVERY_TIMEOUT_MS", DEFAULT_DELIVERY_TIMEOUT_MS))
    except (TypeError, ValueError):
        ms = DEFAULT_DELIVERY_TIMEOUT_MS
    ms = max(_MIN_WAIT_MS, min(ms, 2000))
    if started is not None:
        left = HOOK_BUDGET_MS - (time.perf_counter() - started) * 1000.0
        ms = max(_MIN_WAIT_MS, min(ms, left))
    return ms / 1000.0


def _acked(response: object) -> bool:
    return isinstance(response, dict) and response.get("ok") is True


def _send_http(events: list[Mapping], deadline_s: float) -> bool:
    """The capability-authenticated route, after /health proves the daemon."""
    try:
        from superlocalmemory.infra.daemon_identity import read_descriptor

        if read_descriptor() is None:
            return False  # no daemon published: do not touch the network
        from superlocalmemory.cli.daemon import daemon_request

        return _acked(daemon_request(
            "POST", HOOK_EVENTS_PATH, {"events": list(events)},
            timeout_seconds=deadline_s, start_wait_seconds=0,
        ))
    except Exception:  # refusal, conflict, transport: all mean "not acked"
        return False


def _send(events: list[Mapping], deadline_s: float, result: dict) -> None:
    """Socket when the daemon's hook socket is there (POSIX), else HTTP.

    A socket that exists but does not answer is not retried over HTTP: the
    daemon behind both is the same, and the cap is already being spent.
    """
    try:
        from superlocalmemory.hooks import hook_daemon

        if hook_daemon.owned_socket_path() is not None:
            result["ack"] = _acked(hook_daemon.try_socket_events(events, timeout=deadline_s))
            return
    except Exception:
        pass
    result["ack"] = _send_http(events, deadline_s)


def _acked_by_daemon(events: list[Mapping], deadline_s: float) -> bool:
    result: dict = {}
    worker = threading.Thread(target=_send, args=(events, deadline_s, result),
                              name="slm-hook-deliver", daemon=True)
    worker.start()
    worker.join(deadline_s)
    return bool(result.get("ack"))


def deliver(events: Sequence[Mapping], *, profile: str,
            started: float | None = None) -> str:
    """``daemon`` (acked), ``spooled``, ``dropped_<n>`` (spool full, counted),
    or ``spool_failed`` (spool unusable; reported on stderr). Never raises.

    ``started``: the hook's ``time.perf_counter()`` at entry, so the wait for
    the daemon ends inside the hook's budget (``HOOK_BUDGET_MS``).
    """
    batch = list(events)[:MAX_BATCH]
    if not batch:
        return "nothing"
    if _acked_by_daemon(batch, _deadline_s(started)):
        return "daemon"
    try:
        result = event_spool.append_events(batch, profile=profile)
    except Exception as exc:
        _breadcrumb(f"hook events not saved ({len(batch)}): {type(exc).__name__}")
        return "spool_failed"
    if result.dropped:
        _breadcrumb(f"hook spool full: {result.dropped} event(s) dropped (see `slm doctor`)")
        return f"dropped_{result.dropped}"
    return "spooled"


def _breadcrumb(message: str) -> None:
    import sys

    try:
        sys.stderr.write(f"slm-hook: {message}\n")
    except Exception:
        pass


__all__ = [
    "DEFAULT_DELIVERY_TIMEOUT_MS", "HOOK_BUDGET_MS", "HOOK_EVENTS_PATH", "MAX_BATCH",
    "MAX_SUMMARY_LEN",
    "active_profile", "deliver", "fact_signal", "outcome_signal", "session_finalize",
    "summary", "tool_event",
]
