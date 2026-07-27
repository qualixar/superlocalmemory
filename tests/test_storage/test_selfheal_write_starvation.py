# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""TDD tests for SLM 3.8.4 self-heal write-starvation fix.

GATE BLOCKER
============
``_backfill_vector_store()`` in unified_daemon.py calls
``vs.rebuild_from_facts()`` which loops calling ``vs.upsert()`` — each upsert
opens its OWN ``sqlite3.connect()`` on ``memory.db``, bypassing
``DatabaseManager._lock``.

Concurrently, ``backfill_missing_embeddings()`` (called by the maintenance
scheduler) writes via ``db.execute()`` which:
  1. acquires ``db._lock``
  2. opens its own connection and tries to write
  3. the VectorStore bypass connection may already hold the SQLite write lock
  4. SQLite busy_timeout fires (waits up to 10 000 ms), then raises OperationalError
  5. ``_execute_one()`` retries — still holding ``db._lock`` — up to 5x = 50 s
  6. any concurrent user write that needs ``db._lock`` hangs for the full 50 s

Separately: the per-fact write loop in ``backfill_missing_embeddings()`` has no
cooperative yield, so on a large DB the tight write loop can hold ``db._lock``
continuously long enough to starve user writes through Python RLock unfairness.

Fix (two parts)
===============
A) ``embedding_migrator.py`` — add module-level constant::

    _SELFHEAL_WRITE_DELAY_S = float(
        os.environ.get("SLM_SELFHEAL_WRITE_DELAY_S", "0.005")
    )

   and insert ``time.sleep(_SELFHEAL_WRITE_DELAY_S)`` after each per-fact
   write pair in the write-back loop.

B) ``unified_daemon.py`` — replace::

    n = vs.rebuild_from_facts(with_emb)

   with a loop that wraps every ``vs.upsert()`` call inside ``with db._lock:``
   and sleeps ``SLM_SELFHEAL_WRITE_DELAY_S`` between upserts.

TDD contract
============
RED  (current code — no constant, no yield in write loop):
  * ``test_selfheal_write_delay_constant_exists`` FAILS (AttributeError)
  * ``test_backfill_yields_between_fact_writes`` FAILS (gaps < 2 ms, no yield)

GREEN  (after fix):
  * Both tests above PASS
  * Documentation tests show bypass mechanism and routed prevention
  * Regression tests confirm convergence is unaffected
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path) -> DatabaseManager:
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    return db


def _seed_null_embedding_facts(
    db: DatabaseManager, n: int, profile: str = "default"
) -> None:
    """Insert n atomic_facts rows with NULL embeddings, creating required FKs."""
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        (profile, profile),
    )
    for i in range(n):
        mem_id = uuid.uuid4().hex
        fact_id = uuid.uuid4().hex
        db.execute(
            "INSERT INTO memories (memory_id, profile_id, content, created_at)"
            " VALUES (?, ?, ?, '2026-01-01T00:00:00')",
            (mem_id, profile, f"memory content {i}"),
        )
        db.execute(
            "INSERT INTO atomic_facts"
            "  (fact_id, memory_id, profile_id, content, fact_type, created_at)"
            " VALUES (?, ?, ?, ?, 'semantic', '2026-01-01T00:00:00')",
            (fact_id, mem_id, profile, f"fact content {i}"),
        )


def _mock_config(profile: str = "default", dim: int = 16) -> Any:
    """Minimal duck-type SLMConfig for backfill tests."""
    return SimpleNamespace(
        active_profile=profile,
        embedding=SimpleNamespace(model_name="test-model", dimension=dim),
    )


def _mock_embedder(dim: int = 16) -> Any:
    """Embedder returning instant deterministic dim-d vectors."""
    e = MagicMock()
    e.embed_batch.side_effect = (
        lambda texts: [[float(i % 10) / 10] * dim for i in range(len(texts))]
    )
    return e


# ---------------------------------------------------------------------------
# Section 1 — Flip tests (RED on current code, GREEN after fix)
# ---------------------------------------------------------------------------


class TestFlipOnFix:
    """Tests that FAIL before the fix and PASS after it is applied.

    These are the canonical TDD RED->GREEN tests for this release blocker.
    """

    def test_selfheal_write_delay_constant_exists(self) -> None:
        """RED: embedding_migrator has no _SELFHEAL_WRITE_DELAY_S constant.

        GREEN: after fix, the constant is present, float, and >= 0.

        Fix path — near the top of embedding_migrator.py constants section:

            import os as _os
            _SELFHEAL_WRITE_DELAY_S = float(
                _os.environ.get("SLM_SELFHEAL_WRITE_DELAY_S", "0.005")
            )
        """
        import superlocalmemory.storage.embedding_migrator as em  # noqa: PLC0415

        assert hasattr(em, "_SELFHEAL_WRITE_DELAY_S"), (
            "_SELFHEAL_WRITE_DELAY_S is missing from embedding_migrator.\n"
            "The fix needs to add this constant so SLM operators can tune the\n"
            "cooperative yield delay without a code change.\n"
            "Add to embedding_migrator.py (constants section):\n"
            "  import os as _os\n"
            "  _SELFHEAL_WRITE_DELAY_S = float(\n"
            "      _os.environ.get('SLM_SELFHEAL_WRITE_DELAY_S', '0.005')\n"
            "  )"
        )
        assert isinstance(em._SELFHEAL_WRITE_DELAY_S, float), (
            f"_SELFHEAL_WRITE_DELAY_S must be float, got {type(em._SELFHEAL_WRITE_DELAY_S)}"
        )
        assert em._SELFHEAL_WRITE_DELAY_S >= 0, (
            f"_SELFHEAL_WRITE_DELAY_S must be >= 0, got {em._SELFHEAL_WRITE_DELAY_S}"
        )

    def test_backfill_yields_between_fact_writes(self, tmp_path: Path) -> None:
        """RED: the per-fact write loop has no inter-fact pause.

        The gap between fact[n]'s second write COMPLETING and fact[n+1]'s first
        write STARTING is pure Python bytecode overhead — typically < 0.2 ms.

        GREEN: after fix, time.sleep(_SELFHEAL_WRITE_DELAY_S) is inserted after
        'embedded += 1'.  With the 5 ms default, the inter-fact gap is
        consistently > 2 ms, giving user writes a guaranteed window each fact.

        Method: records both the END of each write and the START of the next one.
        The inter-fact gap is:

            gap = t_start(fact[n+1].UPDATE) - t_end(fact[n].INSERT)

        This excludes write execution time (open conn / execute / commit / close)
        and captures ONLY idle CPU time — where sleep() lives.

        Without yield:  gap = ~0.05 ms  (Python bytecodes, no sleep)
        With 5 ms yield: gap = ~5 ms   (sleep dominates)
        Threshold: 1 ms  — far above Python overhead, well below yield default.
        """
        from superlocalmemory.storage.embedding_migrator import (  # noqa: PLC0415
            backfill_missing_embeddings,
        )

        N_FACTS = 6
        db = _make_db(tmp_path)
        _seed_null_embedding_facts(db, N_FACTS)

        cfg = _mock_config()
        embedder = _mock_embedder()

        # Wrap db.execute to record BOTH the call start and end for DML.
        # We need call-START to mark the beginning of a write and call-END to
        # mark the boundary after which sleep() would run.
        call_starts: list[float] = []
        call_ends: list[float] = []
        orig_execute = db.execute

        def timed_execute(sql: str, params: tuple = ()) -> Any:  # type: ignore[override]
            first_word = sql.strip().upper().split(None, 1)[0] if sql.strip() else ""
            if first_word in ("INSERT", "UPDATE", "REPLACE", "UPSERT"):
                call_starts.append(time.monotonic())
                result = orig_execute(sql, params)
                call_ends.append(time.monotonic())
                return result
            return orig_execute(sql, params)

        db.execute = timed_execute  # type: ignore[method-assign]

        result = backfill_missing_embeddings(cfg, db, embedder, limit=None)

        assert result["embedded"] == N_FACTS, (
            f"Backfill embedded {result['embedded']}/{N_FACTS} — embedder issue?"
        )

        # Backfill writes exactly 2*N_FACTS DML calls (2 per fact).
        # Seed writes happened before the patch, so call_starts/ends hold
        # ONLY backfill DML calls.
        expected_calls = 2 * N_FACTS
        if len(call_ends) < expected_calls or len(call_starts) < expected_calls:
            pytest.skip(
                f"Only {len(call_ends)} DML calls captured; "
                "cannot measure inter-fact gaps."
            )

        backfill_starts = call_starts[-expected_calls:]
        backfill_ends = call_ends[-expected_calls:]

        # Each fact: index 2*i = UPDATE start, 2*i+1 = INSERT end
        # Inter-fact gap: backfill_ends[2*i + 1] → backfill_starts[2*(i+1)]
        inter_fact_gaps: list[float] = []
        for i in range(N_FACTS - 1):
            end_of_insert = backfill_ends[2 * i + 1]          # INSERT completed
            start_of_next_update = backfill_starts[2 * (i + 1)]  # next UPDATE starts
            inter_fact_gaps.append(start_of_next_update - end_of_insert)

        # Without yield: pure Python overhead, typically 0.01–0.05 ms.
        # With 5 ms yield: sleep dominates → gap >= 4 ms (minus jitter).
        MIN_YIELD_GAP_S = 0.001  # 1 ms — safely above Python overhead, below yield

        gaps_ms = [f"{g * 1000:.3f}" for g in inter_fact_gaps]
        assert all(g > MIN_YIELD_GAP_S for g in inter_fact_gaps), (
            f"Inter-fact write gaps: {gaps_ms} ms.\n"
            f"Expected all > {MIN_YIELD_GAP_S * 1000:.0f} ms (cooperative yield).\n"
            "WITHOUT the fix: gaps are < 0.1 ms (pure Python overhead, no sleep).\n"
            "Fix: add after 'embedded += 1' in backfill_missing_embeddings write loop:\n"
            "  if _SELFHEAL_WRITE_DELAY_S > 0:\n"
            "      time.sleep(_SELFHEAL_WRITE_DELAY_S)"
        )


# ---------------------------------------------------------------------------
# Section 2 — Bypass starvation mechanism (documentation tests)
#
# These always pass — they document both sides of the bypass starvation
# pattern.  They use a manual simulation and do NOT flip on the
# unified_daemon.py fix; that fix is best verified by integration tests.
# ---------------------------------------------------------------------------

@pytest.fixture()
def fast_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatabaseManager:
    """DatabaseManager with reduced retry timeouts so bypass tests finish fast."""
    import superlocalmemory.storage.database as db_mod  # noqa: PLC0415

    # 3 retries x 150 ms busy_timeout = 450 ms total starvation — well above threshold.
    monkeypatch.setattr(db_mod, "_BUSY_TIMEOUT_MS", 150)
    monkeypatch.setattr(db_mod, "_MAX_RETRIES", 3)
    monkeypatch.setattr(db_mod, "_RETRY_BASE_DELAY", 0.01)

    db = _make_db(tmp_path)
    return db


def _run_bypass_scenario(
    tmp_path: Path,
    db: DatabaseManager,
    *,
    bypass_routed: bool,
) -> list[str]:
    """Three-thread scenario: bypass writer + backfill writer + user writer.

    Args:
        bypass_routed: True = bypass uses db._lock (FIX simulation).
                       False = bypass uses independent sqlite3 connection (BUG).

    Returns:
        DB operations that entered while the bypass critical section was
        active.
    """
    user_completed = threading.Event()
    bypass_has_lock = threading.Event()
    bypass_released = threading.Event()
    contenders_ready = threading.Event()
    execute_one_entered = threading.Event()
    ready_count = 0
    ready_lock = threading.Lock()
    entered_during_bypass: list[str] = []
    synchronization_errors: list[str] = []
    original_execute_one = db._execute_one

    def observed_execute_one(sql: str, params: tuple[Any, ...]):
        if bypass_has_lock.is_set() and not bypass_released.is_set():
            entered_during_bypass.append(sql)
        # Publish entry only after its witness is durable. Otherwise the
        # bypass thread can release between Event.set() and list.append().
        execute_one_entered.set()
        return original_execute_one(sql, params)

    db._execute_one = observed_execute_one  # type: ignore[method-assign]

    def mark_contender_ready() -> None:
        nonlocal ready_count
        with ready_lock:
            ready_count += 1
            if ready_count == 2:
                contenders_ready.set()

    def bypass_writer() -> None:
        if bypass_routed:
            # FIX: hold db._lock while performing the write.
            with db._lock:
                bypass_has_lock.set()
                if not contenders_ready.wait(timeout=3.0):
                    synchronization_errors.append("routed contenders were not ready")
                if execute_one_entered.is_set():
                    synchronization_errors.append(
                        "routed operation entered before lock release"
                    )
                # Set while still holding the process-wide writer lock so no
                # routed operation can enter between release and observation.
                bypass_released.set()
        else:
            # BUG: independent connection grabs the SQLite write lock directly,
            # bypassing db._lock entirely.
            conn = sqlite3.connect(str(tmp_path / "memory.db"))
            try:
                conn.execute("BEGIN IMMEDIATE")
                bypass_has_lock.set()
                if not contenders_ready.wait(timeout=3.0):
                    synchronization_errors.append("bypass contenders were not ready")
                if not execute_one_entered.wait(timeout=3.0):
                    synchronization_errors.append(
                        "no DatabaseManager operation entered during bypass"
                    )
                conn.rollback()
                bypass_released.set()
            finally:
                conn.close()

    def backfill_writer() -> None:
        """Simulates backfill_missing_embeddings writing via db.execute().

        Bug path: grabs db._lock, then waits for the SQLite write lock held
        by the bypass connection.  The busy_timeout fires multiple times,
        holding db._lock for the entire retry loop (up to retries * timeout).
        """
        if not bypass_has_lock.wait(timeout=3.0):
            synchronization_errors.append("backfill did not observe bypass lock")
            return
        mark_contender_ready()
        try:
            db.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                ("backfill-thread", "backfill-thread"),
            )
        except Exception:
            pass  # SQLITE_BUSY after retries is expected in the bug path

    def user_writer() -> None:
        if not bypass_has_lock.wait(timeout=3.0):
            synchronization_errors.append("user did not observe bypass lock")
            return
        mark_contender_ready()
        try:
            db.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                ("user-thread", "user-thread"),
            )
        except Exception:
            pass
        user_completed.set()

    threads = [
        threading.Thread(target=bypass_writer, daemon=True),
        threading.Thread(target=backfill_writer, daemon=True),
        threading.Thread(target=user_writer, daemon=True),
    ]
    try:
        for t in threads:
            t.start()
        deadline = 6.0
        for t in threads:
            t.join(timeout=deadline)
        alive = [t.name for t in threads if t.is_alive()]
        assert alive == [], f"Scenario threads did not terminate: {alive}"
    finally:
        db._execute_one = original_execute_one  # type: ignore[method-assign]

    assert synchronization_errors == []
    assert user_completed.is_set(), "User write thread never completed within deadline"
    return entered_during_bypass


class TestBypassStarvationMechanism:
    """Documents the bypass starvation bug and confirms routed prevention.

    These tests do NOT flip on the unified_daemon.py fix — they permanently
    document both sides of the bypass pattern.
    """

    def test_bypass_connection_causes_starvation(
        self, fast_db: DatabaseManager, tmp_path: Path
    ) -> None:
        """BUG: VectorStore bypass connection holds the SQLite write lock.
        Concurrent backfill db.execute() hits SQLITE_BUSY retry loop while
        holding db._lock — user write is starved.

        The deterministic witness records a DatabaseManager operation entering
        SQLite while the independent connection still owns its write lock.
        """
        entered = _run_bypass_scenario(
            tmp_path,
            fast_db,
            bypass_routed=False,
        )
        assert entered, (
            "Expected a DatabaseManager operation to enter while the bypass "
            "held SQLite's lock."
        )

    def test_routed_write_prevents_starvation(
        self, fast_db: DatabaseManager, tmp_path: Path
    ) -> None:
        """FIX: bypass writes routed through db._lock eliminate SQLITE_BUSY.
        The deterministic witness proves no DatabaseManager operation can enter
        SQLite before the routed critical section releases its shared lock.
        """
        entered = _run_bypass_scenario(
            tmp_path,
            fast_db,
            bypass_routed=True,
        )
        assert entered == [], (
            "A routed DatabaseManager operation entered the SQLite layer "
            "during the protected critical section."
        )


# ---------------------------------------------------------------------------
# Section 3 — Regression guards (must pass before AND after fix)
# ---------------------------------------------------------------------------


class TestBackfillConvergence:
    """Cooperative yield must not break backfill convergence."""

    def test_backfill_embeds_all_null_facts(self, tmp_path: Path) -> None:
        """backfill_missing_embeddings embeds all N facts and reports
        remaining_null == 0.  Regression against yield breaking the loop.
        """
        from superlocalmemory.storage.embedding_migrator import (  # noqa: PLC0415
            backfill_missing_embeddings,
        )

        N_FACTS = 8
        db = _make_db(tmp_path)
        _seed_null_embedding_facts(db, N_FACTS)

        result = backfill_missing_embeddings(
            _mock_config(), db, _mock_embedder(), limit=None
        )

        assert result["embedded"] == N_FACTS, (
            f"Expected {N_FACTS} embedded, got {result['embedded']}."
        )
        assert result["remaining_null"] == 0

        null_count = db.execute(
            "SELECT COUNT(*) AS c FROM atomic_facts"
            " WHERE embedding IS NULL AND profile_id = 'default'",
        )
        assert int(null_count[0]["c"]) == 0, "Embeddings not persisted to DB."

    def test_backfill_is_idempotent(self, tmp_path: Path) -> None:
        """Running backfill twice on a fully-embedded DB: second pass embeds 0."""
        from superlocalmemory.storage.embedding_migrator import (  # noqa: PLC0415
            backfill_missing_embeddings,
        )

        N_FACTS = 4
        db = _make_db(tmp_path)
        _seed_null_embedding_facts(db, N_FACTS)

        cfg = _mock_config()
        embedder = _mock_embedder()

        first = backfill_missing_embeddings(cfg, db, embedder, limit=None)
        assert first["embedded"] == N_FACTS

        second = backfill_missing_embeddings(cfg, db, embedder, limit=None)
        assert second["embedded"] == 0, (
            f"Second pass re-embedded {second['embedded']} facts — not idempotent."
        )
        assert second["remaining_null"] == 0
