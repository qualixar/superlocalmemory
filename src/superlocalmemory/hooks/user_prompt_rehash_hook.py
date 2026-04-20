# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — Track A.2 (LLD-09 / LLD-00)

"""UserPromptSubmit rehash hook — detect re-query within 60 s.

When the user re-asks the same thing within 60 s, that's a negative
signal: the prior recall did not satisfy. The hook writes a
``requery=True`` signal to the matching pending outcome.

Flow (hot path, <10 ms typical, <20 ms hard):
  1. Read stdin JSON {session_id, prompt}.
  2. Compute topic_signature(prompt) — stdlib regex, bounded.
  3. Read session_state/<session_id>.json (via safe_resolve_identifier).
  4. If prior sig == current sig AND prior age <= 60 s AND prior
     outcome_id is set → register_signal(requery=True).
  5. Always overwrite state with {sig, ts, outcome_id=current_best}
     so next turn has fresh context.
  6. Emit ``{}`` on stdout, exit 0.
"""

from __future__ import annotations

import sys
import time

from superlocalmemory.hooks._outcome_common import (
    REQUERY_WINDOW_MS,
    emit_empty_json,
    load_session_state,
    log_perf,
    memory_db_path,
    now_ms,
    open_memory_db,
    read_stdin_json,
    save_session_state,
    session_state_file,
)


_HOOK_NAME = "user_prompt_rehash"


def _current_latest_outcome_id(session_id: str) -> str | None:
    """Return the most-recent pending outcome_id for this session, or None.

    Kept for monkey-patch compatibility in tests. The main hot path in
    ``_inner_main`` now reuses a single DB connection — see H-12/H-P-04.

    S9-SKEP-08: index the result by position (``row[0]``) rather than
    by name. ``open_memory_db()`` returns a ``sqlite3.Row`` factory
    today, but downstream callers occasionally reset the factory (test
    fixtures, future connection pooling). Positional access works for
    both ``Row`` and the default tuple factory — future-proof, no cost.
    """
    try:
        with open_memory_db() as conn:
            row = conn.execute(
                "SELECT outcome_id FROM pending_outcomes "
                "WHERE session_id = ? AND status = 'pending' "
                "ORDER BY created_at_ms DESC LIMIT 1",
                (session_id,),
            ).fetchone()
    except Exception:
        return None
    return row[0] if row else None


def _current_latest_outcome_id_on(
    conn, session_id: str,
) -> str | None:
    """H-12/H-P-04: same as above but drives an existing connection —
    avoids a second ``sqlite3.connect`` on the UserPromptSubmit hot path.

    S9-SKEP-08: positional indexing — see sibling function docstring.
    """
    try:
        row = conn.execute(
            "SELECT outcome_id FROM pending_outcomes "
            "WHERE session_id = ? AND status = 'pending' "
            "ORDER BY created_at_ms DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    except Exception:
        return None
    return row[0] if row else None


def _inner_main() -> str:
    payload = read_stdin_json()
    if payload is None:
        return "invalid_payload"

    session_id = payload.get("session_id")
    prompt = payload.get("prompt")
    if not isinstance(session_id, str) or not session_id:
        return "no_session"
    if not isinstance(prompt, str) or not prompt:
        return "empty_prompt"

    # Path-escape defence — if session_id is unsafe we skip everything.
    if session_state_file(session_id) is None:
        return "unsafe_session_id"

    # Delayed import — hot-path cold start discipline.
    try:
        from superlocalmemory.core.topic_signature import compute_topic_signature
    except Exception:
        return "import_fail"

    try:
        sig_now = compute_topic_signature(prompt)
    except Exception:
        return "sig_fail"

    state = load_session_state(session_id)
    prior_sig = state.get("last_topic_sig")
    prior_ts = state.get("last_prompt_ts_ms")
    prior_oid = state.get("last_outcome_id")

    ts_now = now_ms()

    # H-12/H-P-04: share one connection for the outcome_id probe instead
    # of opening a fresh ``sqlite3.connect`` every hook call. The reward
    # model itself opens its own cached writer conn on the signal path
    # below, so the total connects-per-hook drops from 3 → 2 on the
    # UserPromptSubmit hot path (the flagship I1 path).
    new_oid = prior_oid
    try:
        with open_memory_db() as conn:
            fresh_oid = _current_latest_outcome_id_on(conn, session_id)
            if fresh_oid:
                new_oid = fresh_oid
    except Exception:
        pass

    # Update state first so even an early-return leaves fresh context.
    save_session_state(session_id, {
        "last_topic_sig": sig_now,
        "last_prompt_ts_ms": ts_now,
        "last_outcome_id": new_oid,
    })

    # Re-query detection
    if not (isinstance(prior_sig, str) and prior_sig == sig_now):
        return "no_rehash"
    if not isinstance(prior_ts, (int, float)):
        return "no_prior_ts"
    if ts_now - int(prior_ts) > REQUERY_WINDOW_MS:
        return "outside_window"
    if not isinstance(prior_oid, str) or not prior_oid:
        return "no_prior_outcome"

    # Register the negative signal via the canonical reward API.
    try:
        from superlocalmemory.learning.reward import EngagementRewardModel
    except Exception:
        return "import_fail"

    try:
        model = EngagementRewardModel(memory_db_path())
    except Exception:
        return "model_init_fail"

    try:
        ok = model.register_signal(
            outcome_id=prior_oid,
            signal_name="requery",
            signal_value=True,
        )
        return "requery_written" if ok else "requery_unknown"
    finally:
        try:
            model.close()
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
