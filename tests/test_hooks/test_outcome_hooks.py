# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track A.2 (LLD-09 / LLD-00)

"""Tests for outcome-population hooks (LLD-09).

13 manifest-locked tests for:
  - post_tool_outcome_hook
  - user_prompt_rehash_hook
  - stop_outcome_hook

Contract refs:
  - LLD-00 §1.2 — pending_outcomes schema (memory.db).
  - LLD-00 §2 — finalize_outcome(outcome_id=...) only.
  - LLD-00 §3 — HMAC validator only, no bare substring scan.
  - LLD-00 §4 — safe_resolve_identifier for session_state writes.
  - LLD-00 §1.1 — action_outcomes INSERT must include profile_id.
  - MANIFEST A.2 — 13 exact test names.

Tests are stdlib-only. Hooks are invoked in-process via `main()` + stdin/stdout
monkey-patch for speed (subprocess perf tests use real cold-start where
required).
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures — memory.db bootstrap + env isolation
# ---------------------------------------------------------------------------


def _bootstrap_memory_db(path: Path) -> None:
    """Create tables the outcome hooks touch (LLD-00 §1.1 + §1.2)."""
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE action_outcomes (
                outcome_id       TEXT PRIMARY KEY,
                profile_id       TEXT NOT NULL DEFAULT 'default',
                query            TEXT NOT NULL DEFAULT '',
                fact_ids_json    TEXT NOT NULL DEFAULT '[]',
                outcome          TEXT NOT NULL DEFAULT '',
                context_json     TEXT NOT NULL DEFAULT '{}',
                timestamp        TEXT NOT NULL DEFAULT (datetime('now')),
                reward           REAL,
                settled          INTEGER NOT NULL DEFAULT 0,
                settled_at       TEXT,
                recall_query_id  TEXT
            );
            CREATE TABLE pending_outcomes (
                outcome_id       TEXT PRIMARY KEY,
                profile_id       TEXT NOT NULL,
                session_id       TEXT NOT NULL,
                recall_query_id  TEXT NOT NULL,
                fact_ids_json    TEXT NOT NULL,
                query_text_hash  TEXT NOT NULL,
                created_at_ms    INTEGER NOT NULL,
                expires_at_ms    INTEGER NOT NULL,
                signals_json     TEXT NOT NULL DEFAULT '{}',
                status           TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE INDEX idx_pending_profile_expires
                ON pending_outcomes(profile_id, expires_at_ms);
            CREATE INDEX idx_pending_status
                ON pending_outcomes(status, expires_at_ms);
            """
        )
        # 4.2.8: PostToolUse records every invocation (drained by the daemon).
        from superlocalmemory.storage.schema_v347 import _TOOL_EVENTS_DDL
        conn.executescript(_TOOL_EVENTS_DDL)


@pytest.fixture()
def slm_home(tmp_path: Path, monkeypatch):
    """Isolate hook state under the winning canonical data-root alias."""
    home = tmp_path / "slm_home"
    home.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(home))
    monkeypatch.setenv("SL_MEMORY_PATH", str(tmp_path / "wrong-legacy-root"))
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "wrong-hook-root"))
    return home


@pytest.fixture()
def memory_db(slm_home: Path) -> Path:
    db = slm_home / "memory.db"
    _bootstrap_memory_db(db)
    return db


@pytest.fixture()
def install_token(slm_home: Path, monkeypatch) -> str:
    """Pin a deterministic install token for HMAC tests."""
    token_file = slm_home / ".install_token"
    token = "a" * 64
    token_file.write_text(token, encoding="utf-8")
    token_file.chmod(0o600)
    # Purge any cached token from earlier tests.
    import superlocalmemory.core.security_primitives as _sec
    try:
        _sec._cached_install_token = None  # type: ignore[attr-defined]
    except Exception:
        pass
    return token


def _make_marker(fact_id: str) -> str:
    """Build an HMAC marker using the current install token (LLD-00 §3)."""
    from superlocalmemory.core.recall_pipeline import _emit_marker
    return _emit_marker(fact_id)


def _invoke_hook(hook_main, payload: dict, monkeypatch) -> tuple[int, str]:
    """Run a hook's main() with injected stdin, capture stdout, return exit."""
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    rc = hook_main()
    return rc, stdout.getvalue()


def _seed_pending(
    db: Path,
    *,
    outcome_id: str,
    session_id: str,
    fact_ids: list[str],
    profile_id: str = "default",
    recall_query_id: str = "q-1",
    created_at_ms: int | None = None,
    status: str = "pending",
    # Scoreable by default: an outcome with no signal is deliberately finalized
    # WITHOUT a score, so a seed of '{}' cannot exercise the scoring path.
    signals_json: str = '{"cite": true}',
) -> None:
    if created_at_ms is None:
        created_at_ms = int(time.time() * 1000)
    expires_at_ms = created_at_ms + 60_000
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO pending_outcomes (outcome_id, profile_id, session_id, "
            "recall_query_id, fact_ids_json, query_text_hash, created_at_ms, "
            "expires_at_ms, signals_json, status) "
            "VALUES (?, ?, ?, ?, ?, '0' * 64, ?, ?, ?, ?)",
            (outcome_id, profile_id, session_id, recall_query_id,
             json.dumps(fact_ids), created_at_ms, expires_at_ms,
             signals_json, status),
        )


def _fetch_pending(db: Path, outcome_id: str) -> dict | None:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT * FROM pending_outcomes WHERE outcome_id = ?",
            (outcome_id,),
        ).fetchone()
    return dict(r) if r else None


def _fetch_action(db: Path, outcome_id: str) -> dict | None:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT * FROM action_outcomes WHERE outcome_id = ?",
            (outcome_id,),
        ).fetchone()
    return dict(r) if r else None


def _drain(db: Path) -> None:
    """Play the daemon's part (4.2.8): hooks hand events to the daemon's
    writer, spooled here because no daemon runs in these tests."""
    from superlocalmemory.server.hook_event_writer import HookEventWriter

    HookEventWriter(db, root=db.parent / "hook_spool").drain_once()


# ---------------------------------------------------------------------------
# post_tool_outcome_hook — 6 tests
# ---------------------------------------------------------------------------


def test_post_tool_hook_writes_signal_on_hmac_match(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """Valid HMAC marker in tool_response → register_signal recorded."""
    from superlocalmemory.hooks import post_tool_outcome_hook as h

    _seed_pending(
        memory_db,
        outcome_id="oid-ok-1",
        session_id="sess-A",
        fact_ids=["fact-42"],
    )
    marker = _make_marker("fact-42")
    payload = {
        "session_id": "sess-A",
        "tool_name": "Edit",
        "tool_response": f"some text {marker} more text",
    }

    rc, out = _invoke_hook(h.main, payload, monkeypatch)
    assert rc == 0
    assert out == "{}"
    assert _fetch_pending(memory_db, "oid-ok-1")["signals_json"] == '{"cite": true}', \
        "the hook itself never writes memory.db"
    _drain(memory_db)

    row = _fetch_pending(memory_db, "oid-ok-1")
    assert row is not None
    signals = json.loads(row["signals_json"])
    # Edit tool => 'edit' signal
    assert signals.get("edit") is True


def test_post_tool_hook_ignores_unvalidated_marker(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """Bare fact_id appearance (no HMAC) MUST NOT be treated as a hit (SEC-C-01)."""
    from superlocalmemory.hooks import post_tool_outcome_hook as h

    _seed_pending(
        memory_db,
        outcome_id="oid-bad",
        session_id="sess-A",
        fact_ids=["fact-evil"],
        signals_json="{}",   # this test asserts NOTHING is added
    )
    # Tool output has the fact_id and even a look-alike but wrong-HMAC marker.
    forged = "slm:fact:fact-evil:00000000"
    payload = {
        "session_id": "sess-A",
        "tool_name": "Edit",
        "tool_response": f"mention fact-evil {forged}",
    }

    rc, out = _invoke_hook(h.main, payload, monkeypatch)
    assert rc == 0
    assert out == "{}"
    _drain(memory_db)

    row = _fetch_pending(memory_db, "oid-bad")
    assert row is not None
    signals = json.loads(row["signals_json"])
    assert signals == {}  # Nothing registered


def test_post_tool_hook_bounded_100kb_response_scan(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """10 MB tool_response truncated to <=100KB before scan (perf bound)."""
    from superlocalmemory.hooks import post_tool_outcome_hook as h

    _seed_pending(
        memory_db,
        outcome_id="oid-big",
        session_id="sess-A",
        fact_ids=["fact-99"],
    )
    marker = _make_marker("fact-99")
    # Build a 2 MB prefix with the marker buried at the start (well within cap).
    big = ("x" * (2 * 1024 * 1024)) + marker
    payload = {
        "session_id": "sess-A",
        "tool_name": "Read",
        "tool_response": marker + big,  # marker is first → found within cap
    }

    t0 = time.monotonic()
    rc, out = _invoke_hook(h.main, payload, monkeypatch)
    elapsed = time.monotonic() - t0

    assert rc == 0
    assert out == "{}"
    # Bounded scan cost — well under 50 ms even on slow CI
    assert elapsed < 0.2, f"scan took {elapsed*1000:.1f}ms, expected <200ms"


def test_post_tool_hook_uses_safe_resolve_for_session_state(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """Malicious session_id (path traversal) MUST be rejected silently (SEC-C-02)."""
    from superlocalmemory.hooks import post_tool_outcome_hook as h

    # An attacker-controlled session_id trying to escape the session_state dir.
    payload = {
        "session_id": "../../../etc/passwd",
        "tool_name": "Read",
        "tool_response": "some text",
    }

    rc, out = _invoke_hook(h.main, payload, monkeypatch)
    assert rc == 0
    assert out == "{}"

    # Ensure no file was written outside the slm_home/session_state dir.
    passwd = Path("/etc/passwd.slm-session-state")
    assert not passwd.exists()
    # Session state dir should not contain the traversal identifier.
    ss_dir = slm_home / "session_state"
    if ss_dir.exists():
        for child in ss_dir.iterdir():
            assert ".." not in child.name
            assert "/" not in child.name


def test_post_tool_hook_crash_returns_0(
    memory_db, slm_home, monkeypatch
) -> None:
    """Any exception (import failure, delivery blown up) → exit 0, empty stdout."""
    from superlocalmemory.hooks import event_delivery
    from superlocalmemory.hooks import post_tool_outcome_hook as h

    def _boom(*_a, **_k):
        raise RuntimeError("delivery exploded")

    monkeypatch.setattr(event_delivery, "deliver", _boom)
    payload = {
        "session_id": "sess-X",
        "tool_name": "Edit",
        "tool_response": "anything",
    }

    rc, out = _invoke_hook(h.main, payload, monkeypatch)
    assert rc == 0
    assert out == "{}"


def test_post_tool_hook_under_10ms_p95(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """Hot-path median wall-clock cost stays small (I1 budget).

    No matching pending row → no DB write → pure read + early-return.
    That is the representative hot-path case (no recall happened).

    4.1.22 (W4): this used to assert a flat wall-clock ``p95 < 30ms``, which
    measured 49-197 ms under sibling-lane load on a shared machine (passes
    alone) -- a tail statistic over a modest sample is exactly what a
    minority of host-contention-delayed calls contaminates. The median
    ignores that minority while staying sensitive to a real regression,
    which adds cost to every call and so shifts the whole distribution (see
    test_post_tool_hook_timing_budget_catches_a_real_slowdown directly
    below, which proves this with an injected 50 ms stall).
    """
    from superlocalmemory.hooks import post_tool_outcome_hook as h
    from tests.test_hooks._timing import assert_median_under_budget

    payload = {
        "session_id": "sess-none",
        "tool_name": "Read",
        "tool_response": "no markers here",
    }

    # Warm imports / stat caches
    for _ in range(5):
        _invoke_hook(h.main, payload, monkeypatch)

    def _call_once() -> None:
        rc, _ = _invoke_hook(h.main, payload, monkeypatch)
        assert rc == 0

    # Generous ceiling for CI noise; typical median is 1-3 ms.
    assert_median_under_budget(
        _call_once, budget_ms=30.0, label="post_tool_hook",
    )


def test_post_tool_hook_timing_budget_catches_a_real_slowdown(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """The median-based timing check above must not be vacuous: a genuine
    50 ms stall added to the hook's hot path must still fail it.

    This is the test-only proof the fix comment above refers to -- it
    injects the stall via monkeypatch, never in production code.
    """
    from superlocalmemory.hooks import post_tool_outcome_hook as h
    from tests.test_hooks._timing import assert_median_under_budget

    payload = {
        "session_id": "sess-none",
        "tool_name": "Read",
        "tool_response": "no markers here",
    }

    for _ in range(5):
        _invoke_hook(h.main, payload, monkeypatch)

    real_inner_main = h._inner_main

    def _slow_inner_main(*args) -> str:
        time.sleep(0.05)
        return real_inner_main(*args)

    monkeypatch.setattr(h, "_inner_main", _slow_inner_main)

    def _call_once() -> None:
        rc, _ = _invoke_hook(h.main, payload, monkeypatch)
        assert rc == 0

    try:
        assert_median_under_budget(
            _call_once, budget_ms=30.0, iterations=20, label="post_tool_hook",
        )
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "timing budget did not catch an injected 50ms stall -- vacuous test"
        )


# ---------------------------------------------------------------------------
# user_prompt_rehash_hook — 2 tests
# ---------------------------------------------------------------------------


def test_user_prompt_rehash_writes_requery_on_dup_topic(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """Same topic signature within 60 s + prior outcome → requery signal."""
    from superlocalmemory.hooks import user_prompt_rehash_hook as h

    # Seed a prior pending outcome for this session.
    _seed_pending(
        memory_db,
        outcome_id="oid-rehash-1",
        session_id="sess-R",
        fact_ids=["fx"],
    )
    # Write prior state: same sig would match next prompt.
    from superlocalmemory.core.topic_signature import compute_topic_signature
    prompt = "How do I close the recall-outcome loop in SLM?"
    sig = compute_topic_signature(prompt)

    # Prime the session-state file so the rehash hook sees a prior prompt.
    ss_dir = slm_home / "session_state"
    ss_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "last_topic_sig": sig,
        "last_prompt_ts_ms": int(time.time() * 1000),
        "last_outcome_id": "oid-rehash-1",
    }
    (ss_dir / "sess-R.json").write_text(json.dumps(state), encoding="utf-8")

    payload = {"session_id": "sess-R", "prompt": prompt}
    rc, out = _invoke_hook(h.main, payload, monkeypatch)
    assert rc == 0
    assert out == "{}"
    _drain(memory_db)

    row = _fetch_pending(memory_db, "oid-rehash-1")
    assert row is not None
    signals = json.loads(row["signals_json"])
    assert signals.get("requery") is True


def test_user_prompt_rehash_ignores_stale_prior(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """Prior prompt >60 s old → no requery signal written."""
    from superlocalmemory.hooks import user_prompt_rehash_hook as h

    _seed_pending(
        memory_db,
        outcome_id="oid-stale",
        session_id="sess-S",
        fact_ids=["fy"],
    )
    from superlocalmemory.core.topic_signature import compute_topic_signature
    prompt = "different prompt but matching signature"
    sig = compute_topic_signature(prompt)

    ss_dir = slm_home / "session_state"
    ss_dir.mkdir(parents=True, exist_ok=True)
    stale_ts = int(time.time() * 1000) - (5 * 60 * 1000)  # 5 min ago
    state = {
        "last_topic_sig": sig,
        "last_prompt_ts_ms": stale_ts,
        "last_outcome_id": "oid-stale",
    }
    (ss_dir / "sess-S.json").write_text(json.dumps(state), encoding="utf-8")

    payload = {"session_id": "sess-S", "prompt": prompt}
    rc, out = _invoke_hook(h.main, payload, monkeypatch)
    assert rc == 0
    assert out == "{}"
    _drain(memory_db)

    row = _fetch_pending(memory_db, "oid-stale")
    assert row is not None
    signals = json.loads(row["signals_json"])
    assert "requery" not in signals


# ---------------------------------------------------------------------------
# stop_outcome_hook — 3 tests
# ---------------------------------------------------------------------------


def test_stop_hook_finalizes_all_pending_for_session(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """All pending outcomes for the session get finalized into action_outcomes."""
    from superlocalmemory.hooks import stop_outcome_hook as h

    _seed_pending(memory_db, outcome_id="oid-A", session_id="sess-F",
                  fact_ids=["f1"])
    _seed_pending(memory_db, outcome_id="oid-B", session_id="sess-F",
                  fact_ids=["f2"])
    # A different session — must NOT be touched.
    _seed_pending(memory_db, outcome_id="oid-OTHER", session_id="sess-OTHER",
                  fact_ids=["f3"])

    payload = {"session_id": "sess-F"}
    rc, out = _invoke_hook(h.main, payload, monkeypatch)
    assert rc == 0
    assert out == "{}"

    assert _fetch_action(memory_db, "oid-A") is None, "the hook never writes"
    _drain(memory_db)
    assert _fetch_action(memory_db, "oid-A") is not None
    assert _fetch_action(memory_db, "oid-B") is not None
    assert _fetch_action(memory_db, "oid-OTHER") is None

    # Pending rows for sess-F marked settled
    a = _fetch_pending(memory_db, "oid-A")
    b = _fetch_pending(memory_db, "oid-B")
    assert a["status"] == "settled"
    assert b["status"] == "settled"


def test_stop_hook_calls_finalize_outcome_with_outcome_id_only(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """Hook MUST call finalize_outcome(outcome_id=...) — never positional
    or legacy query_id. (LLD-00 §2 + Stage-5b CI gate)."""
    from superlocalmemory.hooks import stop_outcome_hook as h

    _seed_pending(memory_db, outcome_id="oid-K", session_id="sess-K",
                  fact_ids=["fk"])

    calls: list[dict] = []
    real_model_cls = None

    from superlocalmemory.learning import reward as reward_mod
    real_model_cls = reward_mod.EngagementRewardModel

    class _RecordingModel(real_model_cls):  # type: ignore[misc, valid-type]
        def finalize_outcome(self, *args, **kwargs):  # noqa: D401
            calls.append({"args": args, "kwargs": dict(kwargs)})
            return super().finalize_outcome(*args, **kwargs)

    monkeypatch.setattr(reward_mod, "EngagementRewardModel", _RecordingModel)

    payload = {"session_id": "sess-K"}
    rc, out = _invoke_hook(h.main, payload, monkeypatch)
    assert rc == 0
    assert out == "{}"
    _drain(memory_db)

    assert calls, "finalize_outcome was never called"
    for c in calls:
        assert c["args"] == (), \
            f"positional args forbidden, got {c['args']!r}"
        assert "outcome_id" in c["kwargs"], \
            f"missing outcome_id kwarg, got {c['kwargs']!r}"
        assert "query_id" not in c["kwargs"], \
            "legacy query_id kwarg is banned (LLD-00 §2)"


def test_stop_hook_action_outcomes_has_profile_id(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """Finalized action_outcomes row MUST carry profile_id (SEC-C-05)."""
    from superlocalmemory.hooks import stop_outcome_hook as h

    _seed_pending(memory_db, outcome_id="oid-P", session_id="sess-P",
                  profile_id="team-beta", fact_ids=["fp"])

    payload = {"session_id": "sess-P"}
    rc, _ = _invoke_hook(h.main, payload, monkeypatch)
    assert rc == 0
    _drain(memory_db)

    row = _fetch_action(memory_db, "oid-P")
    assert row is not None
    assert row["profile_id"] == "team-beta"


# ---------------------------------------------------------------------------
# Cross-hook properties — 2 tests
# ---------------------------------------------------------------------------


def test_all_hooks_exit_0_on_db_lock(
    memory_db, slm_home, install_token, monkeypatch, tmp_path
) -> None:
    """busy_timeout fast-fail: hooks still return 0 on SQLite OperationalError."""
    import sqlite3 as _sq

    from superlocalmemory.hooks import post_tool_outcome_hook as h_post
    from superlocalmemory.hooks import stop_outcome_hook as h_stop
    from superlocalmemory.hooks import user_prompt_rehash_hook as h_rehash

    # Force every sqlite3.connect from these modules to raise OperationalError.
    class _Boom:
        def __enter__(self): raise _sq.OperationalError("database is locked")
        def __exit__(self, *a): return False

    def _explode(*a, **kw):
        raise _sq.OperationalError("database is locked")

    monkeypatch.setattr(_sq, "connect", _explode)

    # post_tool
    rc, out = _invoke_hook(
        h_post.main,
        {"session_id": "s", "tool_name": "Read", "tool_response": "x"},
        monkeypatch,
    )
    assert rc == 0 and out == "{}"

    # user_prompt_rehash
    rc, out = _invoke_hook(
        h_rehash.main,
        {"session_id": "s", "prompt": "anything"},
        monkeypatch,
    )
    assert rc == 0 and out == "{}"

    # stop_outcome
    rc, out = _invoke_hook(
        h_stop.main,
        {"session_id": "s"},
        monkeypatch,
    )
    assert rc == 0 and out == "{}"


def test_hook_perf_log_shape(
    memory_db, slm_home, install_token, monkeypatch
) -> None:
    """hook-perf.log lines must be NDJSON with {ts, hook, duration_ms, outcome}."""
    from superlocalmemory.hooks import post_tool_outcome_hook as h

    payload = {
        "session_id": "sess-perf",
        "tool_name": "Read",
        "tool_response": "no markers",
    }
    _invoke_hook(h.main, payload, monkeypatch)

    log = slm_home / "logs" / "hook-perf.log"
    assert log.exists(), "hook-perf.log not written"
    line = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    obj = json.loads(line)
    for field in ("ts", "hook", "duration_ms", "outcome"):
        assert field in obj, f"missing {field!r} in perf log: {obj}"
    assert isinstance(obj["ts"], (int, float))
    assert isinstance(obj["hook"], str)
    assert isinstance(obj["duration_ms"], (int, float))
    assert isinstance(obj["outcome"], str)
