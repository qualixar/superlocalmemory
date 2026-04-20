# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — Track A.2 (LLD-09 / LLD-00)

"""PostToolUse hook — detect fact usage + write engagement signal.

Flow (hot path, <10 ms typical, <20 ms hard):
  1. Read Claude Code JSON from stdin.
  2. Resolve session_id via ``safe_resolve_identifier`` (LLD-00 §4).
  3. Cap tool_response to 100 KB (bounded scan, LLD-09 §7 failure-mode #4).
  4. Extract HMAC markers (``slm:fact:<id>:<hmac8>``) — validate each
     (LLD-00 §3). Bare substring scans are **banned** by the Stage-5b
     CI gate.
  5. For each validated fact_id, find a pending_outcomes row where
     ``session_id`` matches AND ``fact_ids_json`` includes the fact_id
     AND ``status='pending'`` — call ``register_signal(outcome_id,
     signal_name, True)``. ``signal_name`` is ``'edit'`` for
     mutating tools (Edit/Write/NotebookEdit), else ``'dwell_ms'`` with
     a nominal 3000 ms value.
  6. Always emit ``{}`` on stdout and return 0. NEVER raise.

Crash-safety (LLD-09 §6):
  - Outer try/except around every code path. stderr breadcrumb (no
    stack trace, no payload echo). Always exit 0.
  - SQLite ``busy_timeout=50`` → fast-fail on DB contention.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from superlocalmemory.hooks._outcome_common import (
    emit_empty_json,
    log_perf,
    memory_db_path as _memory_db_path_fn,
    now_ms,
    open_memory_db,
    read_stdin_json,
    session_state_file,
    summarize_response,
)


_HOOK_NAME = "post_tool_outcome"

# Monkey-patchable indirection for tests.
def _memory_db_path() -> Path:
    return _memory_db_path_fn()


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


def _inner_main() -> str:
    """Return an ``outcome`` string (for perf log); never raises."""
    payload = read_stdin_json()
    if payload is None:
        return "invalid_payload"

    session_id = payload.get("session_id")
    tool_name = payload.get("tool_name") or ""
    if not isinstance(session_id, str) or not session_id:
        return "no_session"

    # S9-DASH-10: keep registry fresh on every PostToolUse so the MCP
    # server can pick up the current session even mid-turn.
    try:
        from superlocalmemory.hooks.session_registry import mark_active
        mark_active(session_id, agent_type="claude")
    except Exception:
        pass

    # Path-escape defence (SEC-C-02) — any unsafe session_id means we
    # must not touch the filesystem for this invocation. We still want
    # to safely query the DB (it uses parameterised SQL), so we only
    # gate the filesystem branch.
    _ = session_state_file(session_id)  # None → caller skips FS writes
    # Note: for post_tool_outcome we do NOT need to write session state.
    # Rehash / stop hooks are the writers/readers.

    # Response scan — capped BEFORE regex (bound O(cap)).
    response_text = summarize_response(payload.get("tool_response"))
    if not response_text:
        return "no_response"

    # Fast pre-check: if the HMAC prefix is absent, no marker can exist.
    if "slm:fact:" not in response_text:
        return "no_marker"

    # S9-W2 M-SEC-03: cap marker iteration to prevent adversarial
    # response_text floods. 5,000 crafted markers × ~5 μs HMAC = 25 ms
    # of CPU inside a 20 ms hook budget — enough to cascade budget
    # misses. LLD-09 says ≤10 facts per recall; 100 is ample headroom.
    _MAX_MARKERS = 100
    hits: list[str] = []
    for m in _MARKER_RE.finditer(response_text):
        if len(hits) >= _MAX_MARKERS:
            break
        marker = m.group(0)
        fact_id = _validate(marker)
        if fact_id:
            hits.append(fact_id)
    if not hits:
        return "no_validated_marker"

    # Persist signals via the canonical reward model — the DB write is
    # behind ``register_signal`` which enforces the schema contract and
    # the pending→settled state machine.
    try:
        from superlocalmemory.learning.reward import EngagementRewardModel
    except Exception:
        return "import_fail"

    signal_name = "edit" if tool_name in _EDIT_TOOLS else "dwell_ms"
    signal_value: object = True if signal_name == "edit" else _DEFAULT_DWELL_MS

    # S9-W3 C6: single connection for BOTH the pending-row match AND
    # the signal writes. Previously the hook opened ``open_memory_db()``
    # for the SELECT, closed it, then constructed EngagementRewardModel
    # which cached its own writer — two connects per invocation × 1-4 ms
    # each × FileVault contention = blown 20 ms hook budget.
    #
    # H-SKEP-03 / H-ARC-H4: pending-row window raised back to 50
    # (SEC-M2 had tightened it to 5 which silently dropped signals on
    # heavy Claude Code sessions). Outer cap on returned outcome_ids
    # caps UPDATE amplification at PENDING_WRITE_CAP × 10 by default.
    try:
        model = EngagementRewardModel(_memory_db_path())
    except Exception:
        return "model_init_fail"

    try:
        target_outcome_ids = model.match_pending_for_fact_ids(
            session_id=session_id, fact_ids=hits,
        )
        if not target_outcome_ids:
            # Distinguish "no pending rows exist" from "rows exist but
            # none matched" for perf-log observability.
            with model._lock:
                conn = model._get_conn()
                has_pending = conn.execute(
                    "SELECT 1 FROM pending_outcomes "
                    "WHERE session_id = ? AND status = 'pending' "
                    "LIMIT 1",
                    (session_id,),
                ).fetchone()
            return "no_match" if has_pending else "no_pending"

        wrote = 0
        for oid in target_outcome_ids:
            ok = model.register_signal(
                outcome_id=oid,
                signal_name=signal_name,
                signal_value=signal_value,
            )
            if ok:
                wrote += 1
        return f"signal_{signal_name}_x{wrote}"
    finally:
        try:
            model.close()
        except Exception:
            pass


def main() -> int:
    """Hook entry point — stdin JSON → signals_json update. Always exits 0."""
    t0 = time.perf_counter()
    outcome = "exception"
    try:
        outcome = _inner_main()
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
