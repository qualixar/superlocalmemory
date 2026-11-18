# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3 retention: old pull receipts go; causal evidence and what it needs stays."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.outcomes import receipt_retention as rr
from superlocalmemory.outcomes import receipts
from tests.test_outcomes.test_injection_receipts import _fact


@pytest.fixture
def store(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


NOW = datetime.now(UTC)
OLD = NOW - timedelta(days=rr.KEEP_PULL_DAYS + 5)


def _pull(store, tmp_path, at, *, session="conv-1", fact):
    assert receipts.record_pull(receipts.Pull(
        tmp_path, store.db_path, "default", session, "mcp_recall", "q", ((fact, 0.5),),
        "mcp", at=at))
    conn = sqlite3.connect(str(store.db_path))
    did = conn.execute("SELECT delivery_id FROM injection_deliveries ORDER BY rowid DESC"
                       " LIMIT 1").fetchone()[0]
    conn.execute("INSERT INTO injection_usage VALUES (?, ?, 'default', 'marker_edit', NULL, ?)",
                 (did, fact, at.isoformat()))
    conn.commit()
    conn.close()
    return did


def _randomised(store, did, session, at):
    conn = sqlite3.connect(str(store.db_path))
    conn.execute(
        "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, client, path,"
        " design, holdout_state, epsilon, eligible_count, withhold_draw, pick_draw, key_id,"
        " candidates, delivered_at) VALUES (?, 'default', ?, 'hook', 'session_context',"
        " 'one-out-v2', 'on', 0.1, 4, 0.5, 0.5, 'k1', 4, ?)", (did, session, at.isoformat()))
    conn.commit()
    conn.close()


def _ids(store, table="injection_deliveries", col="delivery_id"):
    conn = sqlite3.connect(str(store.db_path))
    out = {r[0] for r in conn.execute(f"SELECT {col} FROM {table}")}
    conn.close()
    return out


def test_old_pulls_go_with_their_receipts_and_usage(store, tmp_path) -> None:
    f = _fact(store, "the cache key includes the profile")
    old = _pull(store, tmp_path, OLD, fact=f)
    new = _pull(store, tmp_path, NOW - timedelta(days=10), fact=f)
    got = rr.prune(store.db_path, "default")
    assert got["pull_deliveries"] == 1
    assert _ids(store) == {new}
    assert _ids(store, "injection_receipts") == {new}
    assert _ids(store, "injection_usage") == {new}
    assert old not in _ids(store)


def test_randomised_deliveries_and_their_reask_pulls_stay(store, tmp_path) -> None:
    f = _fact(store, "the deploy runs on Fridays")
    _randomised(store, "rand-1", "conv-9", OLD)
    reask = _pull(store, tmp_path, OLD + timedelta(hours=2), session="conv-9", fact=f)
    late = _pull(store, tmp_path, OLD + timedelta(hours=30), session="conv-9", fact=f)
    other = _pull(store, tmp_path, OLD + timedelta(hours=2), session="conv-8", fact=f)
    rr.prune(store.db_path, "default")
    left = _ids(store)
    assert {"rand-1", reask} <= left
    assert late not in left and other not in left


def test_a_pull_with_a_logged_swap_stays(store, tmp_path) -> None:
    f = _fact(store, "the queue holds 256 items")
    did = _pull(store, tmp_path, OLD, fact=f)
    conn = sqlite3.connect(str(store.db_path))
    conn.execute("UPDATE injection_deliveries SET swap_draw = 0.01, swap_share = 0.05,"
                 " swap_at = 1 WHERE delivery_id = ?", (did,))
    conn.commit()
    conn.close()
    rr.prune(store.db_path, "default")
    assert did in _ids(store)


def test_deletes_run_in_batches_until_done(store, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rr, "BATCH", 2)
    f = _fact(store, "a fact")
    for i in range(5):
        _pull(store, tmp_path, OLD - timedelta(minutes=i), fact=f)
    assert rr.prune(store.db_path, "default")["pull_deliveries"] == 5
    assert _ids(store) == set()


def test_old_need_turns_go_and_other_profiles_are_untouched(store, tmp_path) -> None:
    from superlocalmemory.core import need_gate as ng

    for name, at, profile in (("old", OLD, "default"), ("new", NOW, "default"),
                              ("theirs", OLD, "work")):
        d = ng.Decision(f"turn_{name}", True, None, 0.5, None, False, 1.0, at)
        assert ng.log(store.db_path, d, profile_id=profile, session_id="conv-1",
                      prompt_digest=None)
    assert rr.prune(store.db_path, "default")["need_turns"] == 1
    assert _ids(store, "need_turns", "turn_id") == {"turn_new", "turn_theirs"}


def test_a_store_without_the_tables_is_left_alone(tmp_path) -> None:
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    assert rr.prune(db, "default") == {"pull_deliveries": 0, "need_turns": 0}
    assert rr.KEEP_PULL_DAYS == 180.0
