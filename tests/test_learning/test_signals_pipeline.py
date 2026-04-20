# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-02 §6.1

"""TDD tests for ``learning/signals.py`` — batch write + enqueue semantics."""

from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

from superlocalmemory.learning import signals as signals_mod
from superlocalmemory.learning.signals import (
    SignalBatch,
    SignalCandidate,
    enqueue,
    enqueue_shown_flip,
    record_signal_batch,
    _Q as _SIGNAL_QUEUE,
    _hash_query,
    get_counters,
    queue_size,
    reset_counters,
    _drain_queue_for_tests,
)
from tests.test_learning._signal_fixtures import (
    make_db_with_migrations,
    make_batch,
    open_conn,
)


@pytest.fixture(autouse=True)
def _clean_signal_state():
    """Reset module-level queue + counters between tests."""
    _drain_queue_for_tests()
    reset_counters()
    yield
    _drain_queue_for_tests()
    reset_counters()


# ---------------------------------------------------------------------------
# §6.1 test_record_signal_batch_writes_both_tables (S1)
# ---------------------------------------------------------------------------


def test_record_signal_batch_writes_both_tables(tmp_path):
    db = make_db_with_migrations(tmp_path)
    conn = open_conn(db)
    batch = make_batch(n_candidates=5)
    ids = record_signal_batch(conn, batch)
    assert len(ids) == 5
    assert all(isinstance(i, int) for i in ids)

    sig_rows = conn.execute(
        "SELECT id, fact_id, query_id, position, signal_type, query, "
        "       query_text_hash, channel_scores, cross_encoder "
        "FROM learning_signals ORDER BY id"
    ).fetchall()
    assert len(sig_rows) == 5
    for i, r in enumerate(sig_rows):
        d = dict(r)
        assert d["position"] == i
        assert d["signal_type"] == "candidate"

    feat_rows = conn.execute(
        "SELECT signal_id, features_json, is_synthetic "
        "FROM learning_features ORDER BY id"
    ).fetchall()
    assert len(feat_rows) == 5
    assert {dict(r)["signal_id"] for r in feat_rows} == set(ids)
    for r in feat_rows:
        parsed = json.loads(dict(r)["features_json"])
        # 20 features — FEATURE_DIM.
        assert len(parsed) == 20
        assert dict(r)["is_synthetic"] == 0
    conn.close()


# ---------------------------------------------------------------------------
# §6.1 test_query_text_never_stored (S2 — privacy rule)
# ---------------------------------------------------------------------------


def test_query_text_never_stored(tmp_path):
    db = make_db_with_migrations(tmp_path)
    conn = open_conn(db)
    secret = "AKIAsecretthatmustnotleak12345"
    batch = make_batch(query_text=secret, n_candidates=2)
    record_signal_batch(conn, batch)

    rows = conn.execute(
        "SELECT query, query_text_hash FROM learning_signals"
    ).fetchall()
    for r in rows:
        d = dict(r)
        assert d["query"] == ""
        assert len(d["query_text_hash"]) == 32
        # Must match the canonical hash computed by _hash_query.
        assert d["query_text_hash"] == _hash_query(secret)
        assert secret not in d["query_text_hash"]
    conn.close()


# ---------------------------------------------------------------------------
# §6.1 test_empty_candidates_zero_rows (S3)
# ---------------------------------------------------------------------------


def test_empty_candidates_zero_rows(tmp_path):
    db = make_db_with_migrations(tmp_path)
    conn = open_conn(db)
    batch = make_batch(n_candidates=0)
    result = record_signal_batch(conn, batch)
    assert result == []
    n = conn.execute("SELECT COUNT(*) FROM learning_signals").fetchone()[0]
    assert n == 0
    n = conn.execute("SELECT COUNT(*) FROM learning_features").fetchone()[0]
    assert n == 0
    conn.close()


# ---------------------------------------------------------------------------
# §6.1 test_write_is_atomic_on_feature_insert_failure (S1 atomicity)
# ---------------------------------------------------------------------------


class _FailingConnWrapper:
    """Delegate to a real sqlite3 Connection but blow up on features INSERT."""

    def __init__(self, inner: sqlite3.Connection):
        self._inner = inner

    def execute(self, sql, params=()):
        if "INSERT INTO learning_features" in sql:
            raise sqlite3.OperationalError("synthetic features INSERT failure")
        return self._inner.execute(sql, params)

    def commit(self):
        return self._inner.commit()

    def rollback(self):
        return self._inner.rollback()

    # ``with conn:`` context — mimic Connection: commit on success, rollback
    # on exception.
    def __enter__(self):
        self._inner.execute("BEGIN")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._inner.commit()
        else:
            self._inner.rollback()
        return False


def test_write_is_atomic_on_feature_insert_failure(tmp_path):
    db = make_db_with_migrations(tmp_path)
    real_conn = open_conn(db)
    wrapped = _FailingConnWrapper(real_conn)
    batch = make_batch(n_candidates=3)

    with pytest.raises(sqlite3.OperationalError):
        record_signal_batch(wrapped, batch)

    # All signals INSERTs should have rolled back (single TX).
    n = real_conn.execute("SELECT COUNT(*) FROM learning_signals").fetchone()[0]
    assert n == 0
    n = real_conn.execute("SELECT COUNT(*) FROM learning_features").fetchone()[0]
    assert n == 0
    real_conn.close()


# ---------------------------------------------------------------------------
# §6.1 test_signal_worker_never_blocks_hot_path (SW1)
# ---------------------------------------------------------------------------


def test_signal_worker_never_blocks_hot_path():
    # 1000 enqueues should finish in < 50 ms (plenty of headroom).
    start = time.monotonic()
    for i in range(1000):
        enqueue(make_batch(query_id=f"q{i}", n_candidates=0))
    elapsed = time.monotonic() - start
    assert elapsed < 0.05, f"enqueue too slow: {elapsed:.3f}s"
    assert queue_size() == 1000
    assert get_counters()["signal_enqueued_total"] == 1000


# ---------------------------------------------------------------------------
# §6.1 test_signal_worker_drop_policy_counts (SW2)
# ---------------------------------------------------------------------------


def test_signal_worker_drop_policy_counts(monkeypatch):
    # Shrink the module queue to force a drop.
    import queue as _queue

    small_q: "_queue.Queue[SignalBatch]" = _queue.Queue(maxsize=3)
    monkeypatch.setattr(signals_mod, "_Q", small_q)
    reset_counters()

    for i in range(10):
        enqueue(make_batch(query_id=f"q{i}", n_candidates=0))

    counters = get_counters()
    assert counters["signal_dropped_total"] == 7
    assert counters["signal_enqueued_total"] == 3


# ---------------------------------------------------------------------------
# §6.1 test_shown_flip_not_fake_positive (M1 honesty)
# ---------------------------------------------------------------------------


def test_shown_flip_not_fake_positive(tmp_path):
    db = make_db_with_migrations(tmp_path)
    conn = open_conn(db)
    batch = make_batch(n_candidates=3, query_id="q-honest")
    record_signal_batch(conn, batch)

    # Now emit shown-flip sentinels through the queue.
    shown_batch = SignalBatch(
        profile_id="",
        query_id="q-honest",
        query_text="",
        candidates=(),
        query_context={"_shown_flip": {"fact_id": "fact-000", "shown": True}},
    )
    record_signal_batch(conn, shown_batch)
    not_shown_batch = SignalBatch(
        profile_id="",
        query_id="q-honest",
        query_text="",
        candidates=(),
        query_context={"_shown_flip": {"fact_id": "fact-001", "shown": False}},
    )
    record_signal_batch(conn, not_shown_batch)

    rows = conn.execute(
        "SELECT fact_id, signal_type FROM learning_signals "
        "WHERE query_id = 'q-honest' ORDER BY position"
    ).fetchall()
    by_fid = {dict(r)["fact_id"]: dict(r)["signal_type"] for r in rows}
    assert by_fid["fact-000"] == "shown"
    assert by_fid["fact-001"] == "not_shown"
    # Third candidate untouched.
    assert by_fid["fact-002"] == "candidate"

    # Features rows: label must stay at 0.0 — shown flip does NOT forge
    # labels. Outcome columns remain absent from any row this LLD writes.
    feat = conn.execute(
        "SELECT label FROM learning_features"
    ).fetchall()
    assert all(dict(r)["label"] == 0.0 for r in feat)
    conn.close()


# ---------------------------------------------------------------------------
# §6.1 test_enqueue_swallows (RP1)
# ---------------------------------------------------------------------------


def test_enqueue_swallows(monkeypatch):
    class _ExplodingQueue:
        def put_nowait(self, _):
            raise RuntimeError("boom")

    monkeypatch.setattr(signals_mod, "_Q", _ExplodingQueue())
    reset_counters()
    # Must not raise even though the queue is broken.
    enqueue(make_batch(n_candidates=0))
    assert get_counters()["enqueue_failed_total"] == 1


def test_enqueue_rejects_non_batch():
    reset_counters()
    enqueue(None)  # type: ignore[arg-type]
    assert get_counters()["enqueue_failed_total"] == 1


def test_enqueue_shown_flip_dispatches_sentinel(tmp_path):
    reset_counters()
    enqueue_shown_flip("q-x", "fact-abc", shown=True)
    assert queue_size() == 1
    # Pull it off for inspection.
    batch = _SIGNAL_QUEUE.get_nowait()
    assert batch.query_context["_shown_flip"]["fact_id"] == "fact-abc"
    assert batch.query_context["_shown_flip"]["shown"] is True


# ---------------------------------------------------------------------------
# §6.2 test_recall_writes_20_features + names match (feature wiring)
# ---------------------------------------------------------------------------


def test_record_batch_writes_20_feature_names(tmp_path):
    from superlocalmemory.learning.features import FEATURE_NAMES

    db = make_db_with_migrations(tmp_path)
    conn = open_conn(db)
    batch = make_batch(n_candidates=1)
    record_signal_batch(conn, batch)
    row = conn.execute(
        "SELECT features_json FROM learning_features LIMIT 1"
    ).fetchone()
    parsed = json.loads(dict(row)["features_json"])
    assert set(parsed.keys()) == set(FEATURE_NAMES)
    assert len(parsed) == 20
    conn.close()
