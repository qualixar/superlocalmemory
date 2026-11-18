# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track A.2 (LLD-09 / LLD-00)

"""Stop hook — ask the daemon to finalize every pending outcome for this session.

Flow:
  1. Read stdin JSON {session_id}.
  2. Hand one ``session_finalize`` event to the daemon's single writer
     (``event_delivery``; spooled when the daemon does not ack in time).
     The daemon selects the session's pending outcomes and calls
     ``EngagementRewardModel.finalize_outcome(outcome_id=...)`` for each:
     the reward model owns the action_outcomes INSERT (profile_id populated
     per SEC-C-05) and the pending->settled transition.
  3. Cleanup: delete the per-session state file (best effort).
  4. Emit ``{}`` on stdout, exit 0.

The event is queued behind every signal this session already produced, so
the outcome is settled with them, not before them.

4.2.8: this hook no longer opens a write connection (it finalized directly
with a 500 ms busy timeout and reported nothing when that failed).

Contract (LLD-00 §2): the EngagementRewardModel finalize entry point
takes ``outcome_id=`` as its only keyword argument.
"""

from __future__ import annotations

import sys
import time

from superlocalmemory.hooks._outcome_common import (
    emit_empty_json,
    log_perf,
    read_stdin_json,
    session_state_file,
)

_HOOK_NAME = "stop_outcome"


def _inner_main(started: float | None = None) -> str:
    payload = read_stdin_json()
    if payload is None:
        return "invalid_payload"
    return settle(payload, started)


def settle(payload: dict, started: float | None = None) -> str:
    """The turn's overlap (if any) and its finalize, from an already-read payload.

    Shared by Claude Code's ``stop_outcome`` and Codex's installed ``codex-stop``
    (whose Stop input carries ``last_assistant_message`` too, per the Codex hooks
    reference), so both hosts settle the same way."""
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return "no_session"

    from superlocalmemory.hooks import event_delivery as ed

    profile_id = ed.active_profile()
    # The overlap goes first: the finalize is queued behind every signal of the turn.
    events = overlap_events(payload, session_id, profile_id)
    events.append(ed.session_finalize(session_id=session_id, profile_id=profile_id))
    delivered = ed.deliver(events, profile=profile_id, started=started)
    _cleanup_session_state(session_id)
    return f"finalize_{delivered}"


#: Longest the Stop hook waits on the store to read this session's deliveries
#: (routing only: a latency budget; the overlap is skipped past it).
READ_WAIT_S = 0.2


def overlap_events(payload: dict, session_id: str, profile_id: str) -> list[dict]:
    """W3-2b: one ``overlap_signal`` scoring this turn's reply against the memories
    delivered in the session, when ``outcomes.usage`` is on and the host sent the
    reply (``last_assistant_message``). Read-only; the reply stays in this
    process, only fact ids and scores leave. Never raises."""
    reply = payload.get("last_assistant_message")
    if not isinstance(reply, str) or not reply.strip():
        return []
    try:
        import sqlite3

        from superlocalmemory.core.feature_flags import FeatureFlags
        from superlocalmemory.hooks import event_delivery as ed
        from superlocalmemory.hooks._outcome_common import memory_db_path, slm_home
        from superlocalmemory.outcomes import usage

        if not FeatureFlags(slm_home()).observing("outcomes.usage"):
            return []
        conn = sqlite3.connect(f"file:{memory_db_path()}?mode=ro", uri=True,
                               timeout=READ_WAIT_S)
        try:
            conn.execute(f"PRAGMA busy_timeout={int(READ_WAIT_S * 1000)}")
            scores = usage.overlap_scores(conn, profile_id, session_id, reply)
        finally:
            conn.close()
    except Exception:
        return []
    if not scores:
        return []
    return [ed.overlap_signal(session_id=session_id, scores=scores, profile_id=profile_id)]


def overlap_only(payload: dict) -> str:
    """Hermes (``post_llm_call``, via ``slm hook overlap``): only the overlap signal.
    Hermes settles its outcomes itself at ``on_session_end``. Never raises."""
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return "no_session"
    try:
        from superlocalmemory.hooks import event_delivery as ed

        profile_id = ed.active_profile()
        events = overlap_events(payload, session_id, profile_id)
        return f"overlap_{ed.deliver(events, profile=profile_id)}" if events else "no_overlap"
    except Exception:
        return "failed"


def _cleanup_session_state(session_id: str) -> None:
    """Remove the session_state JSON file, if any. Best effort."""
    p = session_state_file(session_id)
    if p is None:
        return
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def main() -> int:
    t0 = time.perf_counter()
    outcome = "exception"
    try:
        outcome = _inner_main(t0)
    except Exception as exc:  # pragma: no cover
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
