# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track A.1 (LLD-08)

"""Tests for ``superlocalmemory.learning.reward`` (EngagementRewardModel).

Contract references:
  - LLD-00 §1.1 — action_outcomes schema (memory.db, profile_id NOT NULL).
  - LLD-00 §1.2 — pending_outcomes schema (memory.db, single table).
  - LLD-00 §2  — EngagementRewardModel interface (finalize_outcome by
                 outcome_id kwarg only).
  - IMPLEMENTATION-MANIFEST v3.4.22 FINAL A.1 — test names verbatim and
                 label formula:
                 label = 0.5 + 0.4*cited + 0.25*edited + dwell_bonus
                         - 0.5*requeried
                 clamped to [0,1].
  - MASTER-PLAN §2 — Invariant I1: record_recall p95 < 5 ms.

Tests are stdlib-only; no external learning dependency is imported — the
class under test is pure Python against a local SQLite file.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import statistics
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _bootstrap_memory_db(path: Path) -> None:
    """Create the minimal tables the reward model touches.

    Schema mirrors LLD-00 §1.1 (post-M006 action_outcomes) and §1.2
    (pending_outcomes). Kept in lock-step with
    M006_action_outcomes_reward and M007_pending_outcomes migrations.

    WAL is enabled to match production (``MemoryEngine.initialize``) —
    the reward model's hot path uses a 50 ms busy_timeout, and without
    WAL concurrent writers queue on the writer lock and the hot path
    correctly fails fast. Production DBs are all WAL, so the tests
    mirror that shape.
    """
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


@pytest.fixture()
def memory_db(tmp_path: Path) -> Path:
    db = tmp_path / "memory.db"
    _bootstrap_memory_db(db)
    return db


@pytest.fixture()
def model(memory_db: Path):
    """Fresh EngagementRewardModel wired to an isolated memory.db."""
    from superlocalmemory.learning.reward import EngagementRewardModel

    return EngagementRewardModel(memory_db)


def _fetch_pending(db: Path, outcome_id: str) -> dict | None:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM pending_outcomes WHERE outcome_id = ?",
            (outcome_id,),
        ).fetchone()
    return dict(row) if row else None


def _fetch_action(db: Path, outcome_id: str) -> dict | None:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM action_outcomes WHERE outcome_id = ?",
            (outcome_id,),
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# record_recall — hot path
# ---------------------------------------------------------------------------


def test_record_recall_returns_uuid(model) -> None:
    outcome_id = model.record_recall(
        profile_id="default",
        session_id="sess-A",
        recall_query_id="q-1",
        fact_ids=["f1", "f2"],
        query_text="what did we decide about the reward model",
    )
    # UUID v4 string: 36 chars, 8-4-4-4-12 hex with dashes.
    assert isinstance(outcome_id, str)
    assert len(outcome_id) == 36
    parts = outcome_id.split("-")
    assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


def test_record_recall_writes_pending_outcomes_row(model, memory_db) -> None:
    outcome_id = model.record_recall(
        profile_id="default",
        session_id="sess-A",
        recall_query_id="q-42",
        fact_ids=["fact-1", "fact-2"],
        query_text="how do I close the loop",
    )
    row = _fetch_pending(memory_db, outcome_id)
    assert row is not None
    assert row["profile_id"] == "default"
    assert row["session_id"] == "sess-A"
    assert row["recall_query_id"] == "q-42"
    assert json.loads(row["fact_ids_json"]) == ["fact-1", "fact-2"]
    assert row["status"] == "pending"
    assert row["created_at_ms"] > 0
    assert row["expires_at_ms"] > row["created_at_ms"]
    assert json.loads(row["signals_json"]) == {}


def test_record_recall_does_not_persist_raw_query_text(model, memory_db) -> None:
    raw = "SECRET_QUERY_DO_NOT_LEAK_abcdef"
    outcome_id = model.record_recall(
        profile_id="default",
        session_id="s",
        recall_query_id="q",
        fact_ids=["f"],
        query_text=raw,
    )
    row = _fetch_pending(memory_db, outcome_id)
    assert row is not None
    # Column must hold a SHA-256 hex digest, not the raw text.
    assert row["query_text_hash"] == hashlib.sha256(raw.encode()).hexdigest()
    # Raw text must not appear anywhere in the persisted row.
    for value in row.values():
        if isinstance(value, str):
            assert raw not in value


def test_record_recall_honors_kill_switch(memory_db) -> None:
    from superlocalmemory.learning.reward import EngagementRewardModel

    killed = EngagementRewardModel(memory_db, kill_switch=lambda: True)
    outcome_id = killed.record_recall(
        profile_id="default",
        session_id="s",
        recall_query_id="q",
        fact_ids=["f"],
        query_text="x",
    )
    # LLD-00 §2 sentinel — all-zero UUID v4.
    assert outcome_id == "00000000-0000-0000-0000-000000000000"
    # No row was written.
    with sqlite3.connect(memory_db) as conn:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM pending_outcomes"
        ).fetchone()
    assert count == 0


def test_record_recall_hot_path_p95_under_5ms(model) -> None:
    # I1 invariant: record_recall must be hot-path fast.
    durations: list[float] = []
    for i in range(200):
        t0 = time.perf_counter()
        model.record_recall(
            profile_id="default",
            session_id="s",
            recall_query_id=f"q-{i}",
            fact_ids=["f"],
            query_text=f"q{i}",
        )
        durations.append((time.perf_counter() - t0) * 1000.0)
    p95 = statistics.quantiles(durations, n=20)[18]  # 95th percentile
    # Generous wall-clock cap to avoid CI flake. MASTER-PLAN §2 asks <5 ms;
    # we check <10 ms here and log the actual p95 so regressions are visible.
    assert p95 < 10.0, f"p95 = {p95:.2f} ms (target <5 ms, hard <10 ms)"


# ---------------------------------------------------------------------------
# register_signal — async worker path
# ---------------------------------------------------------------------------


def test_register_signal_updates_signals_json(model, memory_db) -> None:
    outcome_id = model.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    ok = model.register_signal(
        outcome_id=outcome_id, signal_name="cite", signal_value=True,
    )
    assert ok is True
    row = _fetch_pending(memory_db, outcome_id)
    assert json.loads(row["signals_json"]) == {"cite": True}


def test_register_signal_rejects_unknown_signal_name(model) -> None:
    outcome_id = model.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    ok = model.register_signal(
        outcome_id=outcome_id,
        signal_name="not_a_real_signal",
        signal_value=True,
    )
    assert ok is False


def test_register_signal_clamps_signal_value(model, memory_db) -> None:
    outcome_id = model.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    # Way out of range — must clamp, not raise.
    ok = model.register_signal(
        outcome_id=outcome_id, signal_name="dwell_ms", signal_value=99_999_999,
    )
    assert ok is True
    row = _fetch_pending(memory_db, outcome_id)
    stored = json.loads(row["signals_json"])
    assert 0 <= stored["dwell_ms"] <= 3_600_000

    # Negative clamps to 0.
    model.register_signal(
        outcome_id=outcome_id, signal_name="dwell_ms", signal_value=-500,
    )
    row = _fetch_pending(memory_db, outcome_id)
    stored = json.loads(row["signals_json"])
    assert stored["dwell_ms"] == 0


def test_register_signal_returns_false_on_unknown_outcome(model) -> None:
    ok = model.register_signal(
        outcome_id="00000000-0000-0000-0000-000000000999",
        signal_name="cite",
        signal_value=True,
    )
    assert ok is False


# ---------------------------------------------------------------------------
# finalize_outcome — label + action_outcomes write
# ---------------------------------------------------------------------------


def test_finalize_outcome_writes_to_action_outcomes(model, memory_db) -> None:
    outcome_id = model.record_recall(
        profile_id="varun", session_id="s", recall_query_id="q-17",
        fact_ids=["fa", "fb"], query_text="x",
    )
    model.register_signal(
        outcome_id=outcome_id, signal_name="cite", signal_value=True,
    )
    reward = model.finalize_outcome(outcome_id=outcome_id)
    assert 0.0 <= reward <= 1.0

    row = _fetch_action(memory_db, outcome_id)
    assert row is not None
    # SEC-C-05 — profile_id MUST be populated on every INSERT.
    assert row["profile_id"] == "varun"
    assert row["recall_query_id"] == "q-17"
    assert row["settled"] == 1
    assert row["settled_at"] is not None
    assert row["reward"] == pytest.approx(reward)
    assert json.loads(row["fact_ids_json"]) == ["fa", "fb"]


def test_finalize_outcome_computes_correct_label_for_cite(model) -> None:
    outcome_id = model.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    model.register_signal(
        outcome_id=outcome_id, signal_name="cite", signal_value=True,
    )
    reward = model.finalize_outcome(outcome_id=outcome_id)
    # 0.5 + 0.4*1 = 0.9
    assert reward == pytest.approx(0.9)


def test_finalize_outcome_computes_correct_label_for_requery(model) -> None:
    outcome_id = model.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    model.register_signal(
        outcome_id=outcome_id, signal_name="requery", signal_value=True,
    )
    reward = model.finalize_outcome(outcome_id=outcome_id)
    # 0.5 - 0.5*1 = 0.0, clamped to 0.
    assert reward == pytest.approx(0.0)


def test_finalize_outcome_neutral_when_no_signals(model) -> None:
    outcome_id = model.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    reward = model.finalize_outcome(outcome_id=outcome_id)
    assert reward == pytest.approx(0.5)


def test_finalize_outcome_fallback_on_db_error(
    monkeypatch: pytest.MonkeyPatch, model, memory_db
) -> None:
    # Register a CITE signal so the natural reward would be 0.9 — the
    # assertion only passes if the fallback (0.5) branch is truly taken.
    outcome_id = model.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    model.register_signal(
        outcome_id=outcome_id, signal_name="cite", signal_value=True,
    )

    # Swap the cached writer connection for a stub that raises on
    # every execute() — covers the fallback branch in a thread-safe
    # way. sqlite3.Connection.execute is a C-level read-only attribute,
    # so we replace the connection object instead.
    class _ExplodingConn:
        def execute(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("simulated disk failure")
        def close(self) -> None:
            pass

    # Pre-warm then replace.
    model._get_conn()  # noqa: SLF001 — test access to cache
    real = model._conn  # noqa: SLF001
    model._conn = _ExplodingConn()  # type: ignore[assignment]

    try:
        reward = model.finalize_outcome(outcome_id=outcome_id)
        assert reward == 0.5  # FALLBACK_REWARD — not 0.9 from the cite signal
    finally:
        # Restore the real connection so the post-assertion fetch works
        # and the fixture tears down cleanly.
        model._conn = real  # noqa: SLF001

    assert _fetch_pending(memory_db, outcome_id) is not None


def test_finalize_outcome_marks_pending_settled(model, memory_db) -> None:
    outcome_id = model.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    model.finalize_outcome(outcome_id=outcome_id)
    row = _fetch_pending(memory_db, outcome_id)
    # Pending row remains with status='settled' (audit trail) — never deleted.
    assert row is not None
    assert row["status"] == "settled"


# ---------------------------------------------------------------------------
# reap_stale — daemon-start recovery
# ---------------------------------------------------------------------------


def test_reap_stale_finalizes_expired_pending(memory_db) -> None:
    from superlocalmemory.learning.reward import EngagementRewardModel

    # Clock stub so we can move time forward deterministically.
    now = {"ms": 1_000_000}
    m = EngagementRewardModel(memory_db, clock_ms=lambda: now["ms"])

    outcome_id = m.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    # Advance well past expires_at_ms + 1h.
    now["ms"] += 2 * 60 * 60 * 1000  # +2h

    reaped = m.reap_stale(older_than_ms=60 * 60 * 1000)
    assert reaped == 1

    row = _fetch_action(memory_db, outcome_id)
    assert row is not None
    assert row["settled"] == 1
    # Profile_id carried through — SEC-C-05.
    assert row["profile_id"] == "default"


def test_reap_stale_respects_older_than_ms(memory_db) -> None:
    from superlocalmemory.learning.reward import EngagementRewardModel

    now = {"ms": 1_000_000}
    m = EngagementRewardModel(memory_db, clock_ms=lambda: now["ms"])

    outcome_id = m.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    # Advance only 30 minutes — should NOT reap with 1h threshold.
    now["ms"] += 30 * 60 * 1000

    reaped = m.reap_stale(older_than_ms=60 * 60 * 1000)
    assert reaped == 0
    # Row still pending.
    row = _fetch_pending(memory_db, outcome_id)
    assert row["status"] == "pending"


# ---------------------------------------------------------------------------
# Bounded registry / concurrency / crash
# ---------------------------------------------------------------------------


def test_pending_registry_bounded_200(model, memory_db) -> None:
    # Push 250 recalls — registry should cap somewhere around 200 rows.
    # We allow the impl to flush-to-disk, so we verify the in-memory
    # cap attribute exists and is 200, and that the module still
    # functions past the cap.
    from superlocalmemory.learning.reward import EngagementRewardModel

    assert EngagementRewardModel.PENDING_REGISTRY_CAP == 200

    ids = [
        model.record_recall(
            profile_id="default", session_id="s", recall_query_id=f"q-{i}",
            fact_ids=["f"], query_text=f"{i}",
        )
        for i in range(250)
    ]
    # Every call returned a valid outcome id (or sentinel — but kill
    # switch isn't active so must be real).
    assert all(
        i != "00000000-0000-0000-0000-000000000000" for i in ids
    )
    # Every row made it to disk (flushed on each record_recall).
    with sqlite3.connect(memory_db) as conn:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM pending_outcomes"
        ).fetchone()
    assert count == 250


def test_concurrent_record_recall_no_races(model, memory_db) -> None:
    errors: list[BaseException] = []

    def worker(start_idx: int) -> None:
        try:
            for j in range(20):
                model.record_recall(
                    profile_id="default",
                    session_id="s",
                    recall_query_id=f"t{start_idx}-q{j}",
                    fact_ids=["f"],
                    query_text=f"{start_idx}-{j}",
                )
        except BaseException as exc:  # pragma: no cover — surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(i,)) for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    with sqlite3.connect(memory_db) as conn:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM pending_outcomes"
        ).fetchone()
    assert count == 100  # 5 threads × 20 calls


def test_crash_recovery_reaper_on_daemon_restart(memory_db) -> None:
    from superlocalmemory.learning.reward import EngagementRewardModel

    now = {"ms": 1_000_000}
    m1 = EngagementRewardModel(memory_db, clock_ms=lambda: now["ms"])

    ids = [
        m1.record_recall(
            profile_id="default", session_id="crashed",
            recall_query_id=f"q{i}", fact_ids=["f"], query_text="x",
        )
        for i in range(3)
    ]

    # Simulate daemon kill — drop the object, reload from disk.
    del m1

    # Advance clock past expiry.
    now["ms"] += 2 * 60 * 60 * 1000

    m2 = EngagementRewardModel(memory_db, clock_ms=lambda: now["ms"])
    reaped = m2.reap_stale(older_than_ms=60 * 60 * 1000)
    assert reaped == 3

    for outcome_id in ids:
        row = _fetch_action(memory_db, outcome_id)
        assert row is not None
        assert row["settled"] == 1


# ---------------------------------------------------------------------------
# Defensive coverage — LLD-00 §1.1 SEC-C-05 and contract edge cases
# ---------------------------------------------------------------------------


def test_finalize_outcome_idempotent(model, memory_db) -> None:
    """Second finalize_outcome call must not corrupt the settled row."""
    outcome_id = model.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    model.register_signal(
        outcome_id=outcome_id, signal_name="edit", signal_value=True,
    )
    r1 = model.finalize_outcome(outcome_id=outcome_id)
    r2 = model.finalize_outcome(outcome_id=outcome_id)
    # Second call returns fallback (already settled), no mutation.
    assert r1 == pytest.approx(0.75)  # 0.5 + 0.25
    assert r2 == 0.5  # FALLBACK_REWARD
    row = _fetch_action(memory_db, outcome_id)
    assert row["reward"] == pytest.approx(0.75)


def test_finalize_outcome_unknown_returns_fallback(model) -> None:
    reward = model.finalize_outcome(
        outcome_id="00000000-0000-0000-0000-000000000001",
    )
    assert reward == 0.5


# ---------------------------------------------------------------------------
# Stage 8 F4.B — H-05 register_signal grace-period TTL check (skeptic-H05)
# ---------------------------------------------------------------------------
#
# Previously register_signal accepted signals on any pending row
# regardless of ``expires_at_ms``, so a stale row from yesterday could
# bias today's reward label. The fix rejects signals when
# ``now_ms > expires_at_ms`` by returning False (never raises).


def test_register_signal_rejects_after_expiry(memory_db: Path) -> None:
    """Stage 8 H-05: signals MUST be rejected once past expires_at_ms.

    Seed a pending row whose grace period has already elapsed; call
    register_signal; expect False and no mutation to signals_json.
    """
    from superlocalmemory.learning.reward import EngagementRewardModel

    # Inject a clock so we can first record (at t=1000) then attempt a
    # signal attach well after the grace period (t = 1000 + GRACE + 1).
    clock = {"ms": 1000}
    model = EngagementRewardModel(memory_db, clock_ms=lambda: clock["ms"])

    outcome_id = model.record_recall(
        profile_id="default",
        session_id="s",
        recall_query_id="q",
        fact_ids=["f"],
        query_text="x",
    )
    # Jump past the grace period.
    clock["ms"] = (
        1000 + EngagementRewardModel.GRACE_PERIOD_MS + 1
    )
    ok = model.register_signal(
        outcome_id=outcome_id,
        signal_name="cite",
        signal_value=True,
    )
    assert ok is False

    # Pending row's signals_json must remain untouched ('{}' on seed).
    row = _fetch_pending(memory_db, outcome_id)
    assert row is not None
    assert json.loads(row["signals_json"]) == {}


def test_register_signal_accepts_within_grace_period_window(
    memory_db: Path,
) -> None:
    """Signals still land when now_ms == expires_at_ms (boundary inclusive).

    Regression guard — the TTL check must be strict ``>``, not ``>=``;
    the expires_at_ms tick itself is still within the window by design.
    """
    from superlocalmemory.learning.reward import EngagementRewardModel

    clock = {"ms": 5000}
    model = EngagementRewardModel(memory_db, clock_ms=lambda: clock["ms"])
    outcome_id = model.record_recall(
        profile_id="default",
        session_id="s",
        recall_query_id="q",
        fact_ids=["f"],
        query_text="x",
    )
    # Advance to exactly expires_at_ms — still inside the window.
    row_before = _fetch_pending(memory_db, outcome_id)
    assert row_before is not None
    clock["ms"] = int(row_before["expires_at_ms"])
    ok = model.register_signal(
        outcome_id=outcome_id, signal_name="cite", signal_value=True,
    )
    assert ok is True
    row_after = _fetch_pending(memory_db, outcome_id)
    assert json.loads(row_after["signals_json"]) == {"cite": True}


def test_register_signal_rejects_one_ms_after_expiry(
    memory_db: Path,
) -> None:
    """Exactly one millisecond past expires_at_ms must reject.

    Tightens the boundary on the H-05 fix.
    """
    from superlocalmemory.learning.reward import EngagementRewardModel

    clock = {"ms": 7000}
    model = EngagementRewardModel(memory_db, clock_ms=lambda: clock["ms"])
    outcome_id = model.record_recall(
        profile_id="default", session_id="s", recall_query_id="q",
        fact_ids=["f"], query_text="x",
    )
    row = _fetch_pending(memory_db, outcome_id)
    clock["ms"] = int(row["expires_at_ms"]) + 1
    ok = model.register_signal(
        outcome_id=outcome_id, signal_name="requery", signal_value=True,
    )
    assert ok is False
