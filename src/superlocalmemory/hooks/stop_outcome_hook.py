# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track A.2 (LLD-09 / LLD-00)

"""Stop hook — finalize every pending outcome for this session.

Flow (<500 ms typical, <1 s hard):
  1. Read stdin JSON {session_id}.
  2. SELECT outcome_id FROM pending_outcomes
     WHERE session_id = ? AND status = 'pending'.
  3. For each outcome_id, call
     ``EngagementRewardModel.finalize_outcome(outcome_id=...)``.
     The reward model owns the action_outcomes INSERT (profile_id
     populated per SEC-C-05) and the pending→settled transition.
  4. Cleanup: delete the per-session state file (best effort).
  5. Emit ``{}`` on stdout, exit 0.

Contract (LLD-00 §2): the EngagementRewardModel finalize entry point
takes ``outcome_id=`` as its only keyword argument. Any other form
(positional args, or a ``query_id=`` keyword) is forbidden and the
Stage-5b CI gate fails the build on sight.
"""

from __future__ import annotations

import sys
import time

from superlocalmemory.hooks._outcome_common import (
    emit_empty_json,
    log_perf,
    memory_db_path,
    open_memory_db,
    read_stdin_json,
    session_state_file,
)


_HOOK_NAME = "stop_outcome"


def _inner_main() -> str:
    payload = read_stdin_json()
    if payload is None:
        return "invalid_payload"

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return "no_session"

    # Enumerate pending outcomes for this session.
    try:
        with open_memory_db() as conn:
            rows = conn.execute(
                "SELECT outcome_id FROM pending_outcomes "
                "WHERE session_id = ? AND status = 'pending'",
                (session_id,),
            ).fetchall()
    except Exception:
        return "db_locked"

    if not rows:
        _cleanup_session_state(session_id)
        return "no_pending"

    # Delayed import of the model — the hot-path budget for Stop is 500 ms,
    # generous enough to pay the import tax once per session end.
    try:
        from superlocalmemory.learning.reward import EngagementRewardModel
    except Exception:
        return "import_fail"

    try:
        model = EngagementRewardModel(memory_db_path())
    except Exception:
        return "model_init_fail"

    finalized = 0
    try:
        for r in rows:
            oid = r["outcome_id"]
            # CRITICAL: kwarg-only per LLD-00 §2. Never positional.
            # Never the legacy ``query_id=``. Stage-5b CI gate enforces.
            try:
                model.finalize_outcome(outcome_id=oid)
                finalized += 1
            except Exception:  # pragma: no cover — reward returns 0.5 on error
                continue
    finally:
        try:
            model.close()
        except Exception:
            pass

    _cleanup_session_state(session_id)
    return f"finalized_{finalized}"


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
        outcome = _inner_main()
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
