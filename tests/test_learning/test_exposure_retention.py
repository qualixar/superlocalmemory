# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-6c: exposure rows are kept only while a ranker can still learn from them."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.learning import exposure_retention as er
from superlocalmemory.storage.migrations import M075_injection_receipts as M075


def _ago(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat(timespec="seconds")


@pytest.fixture
def stores(tmp_path):
    learning = tmp_path / "learning.db"
    conn = sqlite3.connect(str(learning))
    conn.execute("CREATE TABLE learning_signals (id INTEGER PRIMARY KEY, profile_id TEXT,"
                 " query_id TEXT, fact_id TEXT, signal_type TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE learning_features (signal_id INTEGER, profile_id TEXT,"
                 " features_json TEXT)")
    conn.commit()
    conn.close()
    memory = tmp_path / "memory.db"
    conn = sqlite3.connect(str(memory))
    M075.apply(conn)
    conn.execute("CREATE TABLE action_outcomes (profile_id TEXT, recall_query_id TEXT)")
    conn.commit()
    conn.close()
    return learning, memory


def _exposure(learning, qid, hours, kind="candidate", profile="default") -> int:
    conn = sqlite3.connect(str(learning))
    cur = conn.execute("INSERT INTO learning_signals (profile_id, query_id, fact_id,"
                       " signal_type, created_at) VALUES (?, ?, 'f', ?, ?)",
                       (profile, qid, kind, _ago(hours)))
    conn.execute("INSERT INTO learning_features VALUES (?, ?, '{}')", (cur.lastrowid, profile))
    conn.commit()
    conn.close()
    return cur.lastrowid


def _named(memory, qid, *, outcome=False) -> None:
    conn = sqlite3.connect(str(memory))
    if outcome:
        conn.execute("INSERT INTO action_outcomes VALUES ('default', ?)", (qid,))
    else:
        conn.execute("INSERT INTO injection_deliveries (delivery_id, profile_id, path, design,"
                     " holdout_state, epsilon, eligible_count, candidates, delivered_at,"
                     " recall_query_id) VALUES (?, 'default', 'mcp_recall', 'x', 'off', 0, 0,"
                     " 0, ?, ?)", ("d-" + qid, _ago(50), qid))
    conn.commit()
    conn.close()


def _left(learning) -> set[str]:
    conn = sqlite3.connect(str(learning))
    sig = {r[0] for r in conn.execute("SELECT query_id FROM learning_signals")}
    orphans = conn.execute("SELECT COUNT(*) FROM learning_features WHERE signal_id NOT IN"
                           " (SELECT id FROM learning_signals)").fetchone()[0]
    conn.close()
    assert orphans == 0
    return sig


def test_only_old_exposures_nobody_names_are_deleted(stores) -> None:
    learning, memory = stores
    _exposure(learning, "orphan-old", 60)
    _exposure(learning, "orphan-young", 10)            # its window may still bring a label
    _exposure(learning, "delivered", 60)
    _exposure(learning, "reported", 60)
    _exposure(learning, "feedback", 60, kind="legacy_feedback")
    _exposure(learning, "elsewhere", 60, profile="work")
    _named(memory, "delivered")
    _named(memory, "reported", outcome=True)
    got = er.prune(learning, memory, "default")
    assert got["unlabellable"] == 1
    assert _left(learning) == {"orphan-young", "delivered", "reported", "feedback", "elsewhere"}


def test_without_a_label_source_nothing_is_judged_unlabellable(stores, tmp_path,
                                                              monkeypatch) -> None:
    learning, _memory = stores
    _exposure(learning, "old", 60)
    _exposure(learning, "older", 70)
    assert er.prune(learning, tmp_path / "missing.db", "default")["unlabellable"] == 0
    assert _left(learning) == {"old", "older"}
    monkeypatch.setattr(er, "KEEP_LABELLABLE", 1)   # the cap still runs in the same pass
    assert er.prune(learning, None, "default") == {"unlabellable": 0, "over_cap": 1}
    assert _left(learning) == {"old"}


def test_labellable_exposures_are_capped_newest_first(stores, monkeypatch) -> None:
    learning, memory = stores
    monkeypatch.setattr(er, "KEEP_LABELLABLE", 3)
    monkeypatch.setattr(er, "BATCH", 2)                 # several batches
    for i in range(7):
        _exposure(learning, f"q{i}", hours=7 - i)      # q6 newest
        _named(memory, f"q{i}")
    _exposure(learning, "feedback", 100, kind="legacy_feedback")
    assert er.prune(learning, memory, "default")["over_cap"] == 4
    assert _left(learning) == {"q4", "q5", "q6", "feedback"}


def test_an_old_learning_store_is_left_alone(tmp_path) -> None:
    path = tmp_path / "learning.db"
    sqlite3.connect(str(path)).execute("CREATE TABLE learning_signals (id INTEGER)").connection.close()
    assert er.prune(path, None, "default") == {"unlabellable": 0, "over_cap": 0}


def test_the_timer_prunes_before_it_decides_to_train(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from superlocalmemory.infra import memory_pressure
    from superlocalmemory.learning import retrain_trigger
    from superlocalmemory.server import consolidation_runner

    (tmp_path / "learning.db").touch()
    pruned = []
    monkeypatch.setattr(er, "prune", lambda *a: pruned.append(a) or {"unlabellable": 2})
    monkeypatch.setattr(memory_pressure, "pressed", lambda: False)
    monkeypatch.setattr(retrain_trigger, "should_retrain", lambda *a, **k: False)
    config = SimpleNamespace(base_dir=tmp_path, db_path=tmp_path / "memory.db")
    out = consolidation_runner.ranker_step("default", config=config)
    assert out == {"skipped": "not_due", "pruned": {"unlabellable": 2}} and len(pruned) == 1
