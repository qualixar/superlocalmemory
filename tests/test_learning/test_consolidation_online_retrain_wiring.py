# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track A.3 / Stage 8 SB-1

"""Stage 8 SB-1 — LLD-10 online retrain WIRING into ConsolidationWorker.

Before this fix cluster, ``_run_shadow_cycle`` and ``_should_retrain``
were defined and unit-tested in isolation but had zero callers in
``ConsolidationWorker.run()``. The legacy ``signal_count >= 200`` gate
was the only retrain path that ever fired.

These tests pin the wiring:

  * When ``_should_retrain(profile_id) is True`` — run() MUST call
    ``_run_shadow_cycle`` with the three kwargs (memory_db_path,
    learning_db_path, profile_id).
  * When ``_should_retrain is False`` but legacy signal_count >= 200 —
    run() MUST call ``_retrain_ranker`` (legacy cold-start path).
  * ``dry_run=True`` — NEITHER path fires.
  * Under concurrent retrain, the partial unique index on
    ``is_candidate=1`` is never violated.

Contract refs:
  - LLD-00 §1.3 — M009 partial unique indexes (idx_model_active_one,
                 idx_model_candidate_one).
  - LLD-10 §2   — trigger gate (≥50 outcomes OR ≥24h).
  - Stage 8 SB-1 (architect S8-ARC-C1 + skeptic C-01/C-02/H-07).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Minimal schemas shared with test_online_retrain (copy-paste avoids cross-file
# fixture coupling; gates the test even if that file changes)
# ---------------------------------------------------------------------------

_LEARNING_SCHEMA = """
CREATE TABLE learning_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    query TEXT NOT NULL DEFAULT '',
    query_id TEXT,
    fact_id TEXT NOT NULL,
    signal_type TEXT NOT NULL DEFAULT 'candidate',
    position INTEGER,
    value REAL DEFAULT 1.0,
    created_at TEXT NOT NULL
);
CREATE TABLE learning_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    signal_id INTEGER,
    query_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    features_json TEXT NOT NULL,
    is_synthetic INTEGER NOT NULL DEFAULT 0,
    label REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
);
CREATE TABLE learning_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    channel TEXT,
    signal_value REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE learning_model_state (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id        TEXT NOT NULL,
    model_version     TEXT NOT NULL DEFAULT '3.4.22',
    state_bytes       BLOB NOT NULL,
    bytes_sha256      TEXT NOT NULL DEFAULT '',
    trained_on_count  INTEGER NOT NULL DEFAULT 0,
    feature_names     TEXT NOT NULL DEFAULT '[]',
    metrics_json      TEXT NOT NULL DEFAULT '{}',
    is_active         INTEGER NOT NULL DEFAULT 0,
    trained_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    is_previous       INTEGER DEFAULT 0,
    is_rollback       INTEGER DEFAULT 0,
    is_candidate      INTEGER DEFAULT 0,
    shadow_results_json TEXT,
    promoted_at       TEXT,
    rollback_reason   TEXT,
    metadata_json     TEXT
);
CREATE UNIQUE INDEX idx_model_active_one
    ON learning_model_state(profile_id) WHERE is_active = 1;
CREATE UNIQUE INDEX idx_model_candidate_one
    ON learning_model_state(profile_id) WHERE is_candidate = 1;
CREATE TABLE migration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
"""

_MEMORY_SCHEMA = """
CREATE TABLE atomic_facts (
    fact_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    content TEXT NOT NULL,
    fact_type TEXT DEFAULT 'semantic',
    confidence REAL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    session_id TEXT,
    access_count INTEGER DEFAULT 0,
    lifecycle TEXT DEFAULT 'active',
    canonical_entities_json TEXT,
    archive_status TEXT DEFAULT 'active'
);
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
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_utc(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat() if dt.tzinfo is None \
        else dt.isoformat()


@pytest.fixture()
def learning_db(tmp_path: Path) -> Path:
    db = tmp_path / "learning.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_LEARNING_SCHEMA)
    return db


@pytest.fixture()
def memory_db(tmp_path: Path) -> Path:
    db = tmp_path / "memory.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_MEMORY_SCHEMA)
    return db


def _seed_active_model(
    db_path: Path,
    *,
    profile_id: str,
    new_outcomes: int = 0,
    last_retrain_at: datetime | None = None,
) -> int:
    meta = {
        "new_outcomes_since_last_retrain": new_outcomes,
        "last_retrain_at": _iso_utc(last_retrain_at)
        if last_retrain_at is not None else _now_iso(),
    }
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, bytes_sha256, is_active, "
            " trained_at, updated_at, metadata_json) "
            "VALUES (?, ?, ?, 1, ?, ?, ?)",
            (
                profile_id, b"seed", "0" * 64,
                _now_iso(), _now_iso(), json.dumps(meta),
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def _seed_learning_feedback(db_path: Path, *, profile_id: str, n: int) -> None:
    with sqlite3.connect(db_path) as conn:
        for _ in range(n):
            conn.execute(
                "INSERT INTO learning_feedback (profile_id, channel, signal_value) "
                "VALUES (?, 'semantic', 1.0)",
                (profile_id,),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# _should_retrain — pure trigger tests (no active model → False)
# ---------------------------------------------------------------------------


def test_should_retrain_false_when_no_outcomes(learning_db: Path) -> None:
    """With no active model row, _should_retrain is False — legacy
    cold-start owns first training."""
    from superlocalmemory.learning.consolidation_worker import ConsolidationWorker

    worker = ConsolidationWorker(
        memory_db=":memory:", learning_db=str(learning_db),
    )
    assert worker._should_retrain(profile_id="p") is False


def test_should_retrain_true_at_50_outcomes(learning_db: Path) -> None:
    """50 new outcomes trips the first online retrain trigger."""
    from superlocalmemory.learning.consolidation_worker import ConsolidationWorker

    _seed_active_model(
        learning_db, profile_id="p",
        last_retrain_at=datetime.now(timezone.utc) - timedelta(hours=1),
        new_outcomes=50,
    )
    worker = ConsolidationWorker(
        memory_db=":memory:", learning_db=str(learning_db),
    )
    assert worker._should_retrain(profile_id="p") is True


def test_should_retrain_true_at_24h_elapsed(learning_db: Path) -> None:
    """24h elapsed trips the second online retrain trigger."""
    from superlocalmemory.learning.consolidation_worker import ConsolidationWorker

    _seed_active_model(
        learning_db, profile_id="p",
        last_retrain_at=datetime.now(timezone.utc) - timedelta(hours=24),
        new_outcomes=0,
    )
    worker = ConsolidationWorker(
        memory_db=":memory:", learning_db=str(learning_db),
    )
    assert worker._should_retrain(profile_id="p") is True


# ---------------------------------------------------------------------------
# run() routing — online vs legacy vs neither
# ---------------------------------------------------------------------------


def test_run_routes_to_online_when_should_retrain_true(
    learning_db: Path, memory_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run() MUST dispatch ``_run_shadow_cycle`` when the trigger is True.

    Previously dead — run() only ever hit the legacy path.
    """
    from superlocalmemory.learning import consolidation_worker as cw_mod

    _seed_active_model(
        learning_db, profile_id="p",
        last_retrain_at=datetime.now(timezone.utc) - timedelta(hours=1),
        new_outcomes=60,
    )
    _seed_learning_feedback(learning_db, profile_id="p", n=300)

    captured: dict = {}

    def _fake_shadow_cycle(*, memory_db_path, learning_db_path, profile_id):
        captured["called"] = True
        captured["profile_id"] = profile_id
        captured["memory_db_path"] = memory_db_path
        captured["learning_db_path"] = learning_db_path
        return {
            "aborted": None, "candidate_persisted": True,
            "promoted": False, "metrics": {"mean_score": 0.5},
        }

    def _fake_legacy_retrain(profile_id, signal_count):
        captured["legacy_called"] = True
        return True

    monkeypatch.setattr(cw_mod, "_run_shadow_cycle", _fake_shadow_cycle)
    monkeypatch.setattr(
        cw_mod.ConsolidationWorker, "_retrain_ranker",
        lambda self, pid, sc: _fake_legacy_retrain(pid, sc),
    )

    worker = cw_mod.ConsolidationWorker(
        memory_db=str(memory_db), learning_db=str(learning_db),
    )
    stats = worker.run(profile_id="p", dry_run=False)

    assert captured.get("called") is True
    assert captured.get("legacy_called") is not True
    assert captured.get("profile_id") == "p"
    assert captured.get("learning_db_path") == str(learning_db)
    assert stats.get("online_retrain") is not None


def test_run_routes_to_legacy_when_outcomes_below_50_but_signals_above_200(
    learning_db: Path, memory_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy cold-start path remains reachable for profiles whose
    active model has not yet accumulated 50 outcomes."""
    from superlocalmemory.learning import consolidation_worker as cw_mod

    # NO active model seeded → _should_retrain returns False.
    _seed_learning_feedback(learning_db, profile_id="p", n=250)

    shadow_called = {"flag": False}
    legacy_called = {"flag": False}

    def _fake_shadow_cycle(**kw):
        shadow_called["flag"] = True
        return {}

    def _fake_legacy(self, pid, sc):
        legacy_called["flag"] = True
        return True

    monkeypatch.setattr(cw_mod, "_run_shadow_cycle", _fake_shadow_cycle)
    monkeypatch.setattr(cw_mod.ConsolidationWorker, "_retrain_ranker", _fake_legacy)

    worker = cw_mod.ConsolidationWorker(
        memory_db=str(memory_db), learning_db=str(learning_db),
    )
    stats = worker.run(profile_id="p", dry_run=False)

    assert shadow_called["flag"] is False
    assert legacy_called["flag"] is True
    assert stats.get("retrained") is True


def test_run_no_retrain_when_dry_run(
    learning_db: Path, memory_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dry_run=True is a hard gate: neither path fires."""
    from superlocalmemory.learning import consolidation_worker as cw_mod

    _seed_active_model(
        learning_db, profile_id="p",
        last_retrain_at=datetime.now(timezone.utc) - timedelta(hours=1),
        new_outcomes=60,
    )
    _seed_learning_feedback(learning_db, profile_id="p", n=300)

    calls = {"shadow": 0, "legacy": 0}

    def _fake_shadow(**kw):
        calls["shadow"] += 1
        return {}

    def _fake_legacy(self, pid, sc):
        calls["legacy"] += 1
        return True

    monkeypatch.setattr(cw_mod, "_run_shadow_cycle", _fake_shadow)
    monkeypatch.setattr(cw_mod.ConsolidationWorker, "_retrain_ranker", _fake_legacy)

    worker = cw_mod.ConsolidationWorker(
        memory_db=str(memory_db), learning_db=str(learning_db),
    )
    worker.run(profile_id="p", dry_run=True)

    assert calls["shadow"] == 0
    assert calls["legacy"] == 0


# ---------------------------------------------------------------------------
# Partial unique index integrity under legacy + online contention
# ---------------------------------------------------------------------------


def test_online_retrain_flips_lineage_atomically_under_contention(
    learning_db: Path, memory_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two threads try to run consolidation simultaneously. Only one
    candidate row is ever visible per profile (M009 partial unique
    index idx_model_candidate_one enforces)."""
    from superlocalmemory.learning import consolidation_worker as cw_mod

    _seed_active_model(
        learning_db, profile_id="p",
        last_retrain_at=datetime.now(timezone.utc) - timedelta(hours=1),
        new_outcomes=60,
    )
    _seed_learning_feedback(learning_db, profile_id="p", n=300)

    # Use real _run_shadow_cycle, but short-circuit training to keep it fast.
    class _FakeBooster:
        def model_to_string(self) -> str:
            return "tiny-model"
        def predict(self, X):
            import numpy as np
            return np.zeros(len(X))

    def _fake_train(lp, pid, *, training_rows, feature_names, prior_row):
        return _FakeBooster(), {"mean_score": 0.0}

    def _fake_fetch(lp, pid):
        return (
            [
                {"query_id": f"q{i // 3}", "fact_id": f"f{i}",
                 "position": i % 3,
                 "features": {n: 0.0 for n in cw_mod._feature_names()},
                 "outcome_reward": float(i % 2)}
                for i in range(30)
            ],
            ["cand"],
        )

    monkeypatch.setattr(cw_mod, "_train_booster", _fake_train)
    monkeypatch.setattr(cw_mod, "_fetch_training_rows", _fake_fetch)
    monkeypatch.setattr(cw_mod, "_measure_serialized_size", lambda b: 128)

    worker = cw_mod.ConsolidationWorker(
        memory_db=str(memory_db), learning_db=str(learning_db),
    )

    errors: list[BaseException] = []
    def _worker_run():
        try:
            worker.run(profile_id="p", dry_run=False)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=_worker_run) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    # Even if exceptions fired (candidate_one conflict), DB invariants hold.
    with sqlite3.connect(learning_db) as conn:
        n_cand = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_candidate=1",
        ).fetchone()[0]
        assert n_cand <= 1
        n_active = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()[0]
        assert n_active == 1


def test_legacy_retrain_cannot_double_activate_with_online(
    learning_db: Path, memory_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If legacy retrain AND online retrain both race on the same profile
    (both attempting to touch is_active=1), the partial unique index
    prevents two active rows."""
    from superlocalmemory.learning import consolidation_worker as cw_mod

    _seed_active_model(
        learning_db, profile_id="p",
        last_retrain_at=datetime.now(timezone.utc) - timedelta(hours=1),
        new_outcomes=60,
    )
    # Drive the online path to persist a candidate.
    class _FakeBooster:
        def model_to_string(self) -> str:
            return "tiny-model"
        def predict(self, X):
            import numpy as np
            return np.zeros(len(X))

    monkeypatch.setattr(
        cw_mod, "_train_booster",
        lambda *a, **kw: (_FakeBooster(), {"mean_score": 0.0}),
    )
    monkeypatch.setattr(
        cw_mod, "_fetch_training_rows",
        lambda lp, pid: (
            [
                {"query_id": "q0", "fact_id": f"f{i}", "position": i,
                 "features": {n: 0.0 for n in cw_mod._feature_names()},
                 "outcome_reward": 1.0}
                for i in range(30)
            ],
            ["cand"],
        ),
    )
    monkeypatch.setattr(cw_mod, "_measure_serialized_size", lambda b: 256)

    cw_mod._run_shadow_cycle(
        memory_db_path=str(memory_db),
        learning_db_path=str(learning_db),
        profile_id="p",
    )

    # Legacy INSERT attempting is_active=1 MUST collide with partial unique.
    with pytest.raises(sqlite3.IntegrityError):
        with sqlite3.connect(learning_db) as conn:
            conn.execute(
                "INSERT INTO learning_model_state "
                "(profile_id, state_bytes, bytes_sha256, is_active, "
                " trained_at, updated_at) "
                "VALUES ('p', ?, ?, 1, ?, ?)",
                (b"legacy-boom", "0" * 64, _now_iso(), _now_iso()),
            )
            conn.commit()
