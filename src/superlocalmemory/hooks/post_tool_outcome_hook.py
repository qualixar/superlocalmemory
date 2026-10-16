# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track A.2 (LLD-09 / LLD-00)

"""PostToolUse hook — record the tool call and any fact-usage signal.

Flow (hot path):
  1. Read the host's JSON from stdin.
  2. Build a ``tool_event`` for the invocation (summaries truncated + scrubbed).
  3. Cap tool_response to 100 KB (bounded scan, LLD-09 §7 failure-mode #4).
  4. Extract HMAC markers (``slm:fact:<id>:<hmac8>``) — validate each
     (LLD-00 §3). Bare substring scans are **banned** by the Stage-5b
     CI gate. Validated fact ids become one ``fact_signal`` event:
     ``'edit'`` for mutating tools (Edit/Write/NotebookEdit), else
     ``'dwell_ms'`` with a nominal 3000 ms value.
  5. Hand the events to the daemon's single writer (``event_delivery``);
     what it does not ack within the cap goes to the spool, never nowhere.
  6. Always emit ``{}`` on stdout and return 0. NEVER raise.

4.2.8: this hook no longer opens a write connection. It wrote ``tool_events``
with a 50 ms busy timeout and swallowed the failure; under a long memory
write that silently lost 29 of 145 and 63 of 88 events in a stress run.
"""

from __future__ import annotations

import re
import sys
import time

from superlocalmemory.hooks._outcome_common import (
    emit_empty_json,
    log_perf,
    read_stdin_json,
    session_state_file,
    summarize_response,
)

_HOOK_NAME = "post_tool_outcome"

# Tools that imply an "edit" signal (the agent acted on the fact).
_EDIT_TOOLS = frozenset({"Edit", "Write", "NotebookEdit"})

# Nominal dwell value for non-edit tool uses that hit a marker.
# The label formula clamps 2s..10s → 0.05..0.15 reward bonus.
_DEFAULT_DWELL_MS = 3000

# Marker regex — mirrors recall_pipeline._emit_marker but scoped locally
# so this module has no hot-path import of the full recall pipeline.
#
# S-L04 — ``fact_id`` is constrained to a conservative alphabet
# (alphanumerics, ``-`` and ``_``). The previous ``[^:\s]+`` allowed
# colons and let a malicious tool response emit markers like
# ``slm:fact:evil:deadbeef:abcdef01`` that the regex grouped wrong and
# still handed off to the validator. Defence-in-depth: the HMAC
# validator already rejects these, but disallowing colons keeps garbage
# from reaching it in the first place. The HMAC suffix stays lowercase
# hex (matches ``recall_pipeline._emit_marker``).
_MARKER_RE = re.compile(r"slm:fact:([A-Za-z0-9_\-]+):([0-9a-f]{8})")


def _validate(marker: str) -> str | None:
    """Delegate to the canonical validator (LLD-00 §3)."""
    try:
        from superlocalmemory.core.recall_pipeline import _validate_marker
    except Exception:
        return None
    try:
        return _validate_marker(marker)
    except Exception:
        return None


# --- Behavioural telemetry -------------------------------------------------
# ``tool_events`` is the record of what the agent did; assertion mining,
# skill-performance mining and the engagement features read it. This hook runs
# on exactly the right edge (after each tool, with its session), so it records
# the invocation whether or not the output named a memory: the marker decides
# what a reward is worth, not whether the action occurred.


def _validated_fact_ids(response_text: str) -> list[str]:
    """HMAC-validated fact ids named in the (capped) tool response."""
    if not response_text or "slm:fact:" not in response_text:
        return []
    # S9-W2 M-SEC-03: cap marker iteration to prevent adversarial
    # response_text floods (5,000 crafted markers x ~5 us HMAC = 25 ms).
    # LLD-09 says <=10 facts per recall; 100 is ample headroom.
    hits: list[str] = []
    for m in _MARKER_RE.finditer(response_text):
        if len(hits) >= _MAX_MARKERS:
            break
        fact_id = _validate(m.group(0))
        if fact_id:
            hits.append(fact_id)
    return hits


_MAX_MARKERS = 100


def _mark_session_active(session_id: str, profile_id: str) -> None:
    # S9-DASH-10: keep registry fresh on every PostToolUse so the MCP
    # server can pick up the current session even mid-turn.
    try:
        from superlocalmemory.hooks.session_registry import mark_active

        mark_active(session_id, agent_type="claude", profile_id=profile_id)
    except Exception:
        pass


def _inner_main(started: float | None = None) -> str:
    """Return an ``outcome`` string (for perf log); never raises."""
    payload = read_stdin_json()
    if payload is None:
        return "invalid_payload"

    session_id = payload.get("session_id")
    tool_name = payload.get("tool_name") or ""
    if not isinstance(session_id, str) or not session_id:
        return "no_session"

    from superlocalmemory.hooks import event_delivery as ed

    profile_id = ed.active_profile()
    _mark_session_active(session_id, profile_id)
    # Path-escape defence (SEC-C-02): this hook writes no session state;
    # the rehash / stop hooks are its writers and readers.
    _ = session_state_file(session_id)

    # Response scan — capped BEFORE regex (bound O(cap)).
    response_text = summarize_response(payload.get("tool_response"))
    events: list[dict] = []
    if isinstance(tool_name, str) and tool_name:
        raw_cwd = payload.get("cwd")
        events.append(ed.tool_event(
            session_id=session_id, tool_name=tool_name, profile_id=profile_id,
            tool_input=payload.get("tool_input"), response_text=response_text,
            project_path=raw_cwd if isinstance(raw_cwd, str) else "",
        ))
    hits = _validated_fact_ids(response_text)
    if hits:
        signal_name = "edit" if tool_name in _EDIT_TOOLS else "dwell_ms"
        events.append(ed.fact_signal(
            session_id=session_id, fact_ids=hits, signal_name=signal_name,
            signal_value=True if signal_name == "edit" else _DEFAULT_DWELL_MS,
            profile_id=profile_id,
        ))
    if not events:
        return "nothing"
    delivered = ed.deliver(events, profile=profile_id, started=started)
    return f"{delivered}_signal" if hits else delivered


def main() -> int:
    """Hook entry point — stdin JSON → signals_json update. Always exits 0."""
    t0 = time.perf_counter()
    outcome = "exception"
    try:
        outcome = _inner_main(t0)
    except Exception as exc:  # pragma: no cover — defensive
        try:
            sys.stderr.write(
                f"slm-hook {_HOOK_NAME}: {type(exc).__name__}\n"
            )
        except Exception:
            pass
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        emit_empty_json()
        try:
            log_perf(_HOOK_NAME, duration_ms, outcome)
        except Exception:
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry only
    sys.exit(main())
