# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track A.3 (LLD-10)

"""Tests for LLD-10 online retrain + model rollback.

Covers:
  * ``ConsolidationWorker._should_retrain`` — 50-outcomes OR 24h trigger.
  * ``_retrain_ranker_impl`` hyperparameter + size + wall-time caps.
  * Candidate NOT auto-promoted without shadow-test pass.
  * Promotion flips model lineage atomically under partial unique
    indexes (M009 §7 / LLD-00 §1.3).
  * ``ModelRollback`` — 200-recall watch window, 2% MRR regression,
    restore is_previous, 24h retrain disable, reason logged.

Contract references:
  - LLD-00 §1.3 — learning_model_state lineage columns (M009 installed).
  - LLD-10 §2   — trigger: ≥50 new outcomes OR ≥24h since last retrain.
  - LLD-10 §3.2 — hyperparameter caps (num_leaves ≤31, max_depth ≤7,
                 feature_fraction ≤0.8). Model size ≤10 MB. Wall-time
                 ≤30 s.
  - LLD-10 §5   — rollback execution (atomic BEGIN IMMEDIATE flip).
  - IMPLEMENTATION-MANIFEST v3.4.22 FINAL A.3 — 11 test names verbatim.
  - MANIFEST-DEVIATION — cap caps applied by *wrapper*, not runtime
                 booster flags (LightGBM enforces via params dict).

Stdlib-only fixtures; lightgbm is imported lazily inside the impl.
Tests that need a real booster either monkey-patch the lightgbm
module or skip cleanly if lightgbm is unavailable on the runner.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# Schema bootstrap — mirrors learning.db after M002+M009 and memory.db
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
CREATE INDEX idx_signals_profile ON learning_signals(profile_id);

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
CREATE INDEX idx_features_profile ON learning_features(profile_id);

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
CREATE INDEX idx_ao_profile ON action_outcomes(profile_id, settled_at);
"""


def _bootstrap_learning_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_LEARNING_SCHEMA)


def _bootstrap_memory_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_MEMORY_SCHEMA)


@pytest.fixture()
def learning_db(tmp_path: Path) -> Path:
    db = tmp_path / "learning.db"
    _bootstrap_learning_db(db)
    return db


@pytest.fixture()
def memory_db(tmp_path: Path) -> Path:
    db = tmp_path / "memory.db"
    _bootstrap_memory_db(db)
    return db


def _iso_utc(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_active_model(
    db_path: Path,
    *,
    profile_id: str,
    last_retrain_at: datetime | None = None,
    new_outcomes: int = 0,
    state_bytes: bytes = b"seed-state",
) -> int:
    """Insert an active model row with a populated metadata_json."""
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
                profile_id,
                state_bytes,
                "0" * 64,
                _now_iso(),
                _now_iso(),
                json.dumps(meta),
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def _seed_previous_model(
    db_path: Path, *, profile_id: str, state_bytes: bytes = b"prev-state",
) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, bytes_sha256, is_active, is_previous, "
            " trained_at, updated_at) "
            "VALUES (?, ?, ?, 0, 1, ?, ?)",
            (profile_id, state_bytes, "0" * 64, _now_iso(), _now_iso()),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


# ---------------------------------------------------------------------------
# Trigger tests — _should_retrain
# ---------------------------------------------------------------------------


def test_retrain_triggers_at_50_new_outcomes(learning_db: Path) -> None:
    """>=50 new outcomes since last retrain → _should_retrain is True."""
    from superlocalmemory.learning.consolidation_worker import ConsolidationWorker

    _seed_active_model(
        learning_db, profile_id="p",
        last_retrain_at=datetime.now(timezone.utc) - timedelta(hours=1),
        new_outcomes=50,
    )
    worker = ConsolidationWorker(memory_db=":memory:", learning_db=str(learning_db))
    assert worker._should_retrain(profile_id="p") is True

    # Below threshold → False
    with sqlite3.connect(learning_db) as conn:
        meta = json.dumps({
            "new_outcomes_since_last_retrain": 49,
            "last_retrain_at": _now_iso(),
        })
        conn.execute(
            "UPDATE learning_model_state SET metadata_json = ? "
            "WHERE profile_id = 'p' AND is_active = 1",
            (meta,),
        )
        conn.commit()
    assert worker._should_retrain(profile_id="p") is False


def test_retrain_triggers_at_24h_elapsed(learning_db: Path) -> None:
    """>=24h since last retrain AND <50 outcomes → _should_retrain True."""
    from superlocalmemory.learning.consolidation_worker import ConsolidationWorker

    _seed_active_model(
        learning_db, profile_id="p",
        last_retrain_at=datetime.now(timezone.utc) - timedelta(hours=25),
        new_outcomes=5,
    )
    worker = ConsolidationWorker(memory_db=":memory:", learning_db=str(learning_db))
    assert worker._should_retrain(profile_id="p") is True

    # 23h elapsed → False
    with sqlite3.connect(learning_db) as conn:
        meta = json.dumps({
            "new_outcomes_since_last_retrain": 5,
            "last_retrain_at": _iso_utc(
                datetime.now(timezone.utc) - timedelta(hours=23),
            ),
        })
        conn.execute(
            "UPDATE learning_model_state SET metadata_json = ? "
            "WHERE profile_id = 'p' AND is_active = 1",
            (meta,),
        )
        conn.commit()
    assert worker._should_retrain(profile_id="p") is False


# ---------------------------------------------------------------------------
# Hyperparameter + size + wall-time caps
# ---------------------------------------------------------------------------


def test_retrain_hyperparams_capped() -> None:
    """The retrain impl MUST cap num_leaves ≤ 31 and max_depth ≤ 7."""
    from superlocalmemory.learning import consolidation_worker as cw_mod

    caps = cw_mod.RETRAIN_HYPERPARAM_CAPS
    assert caps["num_leaves"] <= 31
    assert caps["max_depth"] <= 7
    assert caps["feature_fraction"] <= 0.8
    # Wall-time cap surfaced as a constant.
    assert cw_mod.RETRAIN_WALL_TIME_BUDGET_SEC == 30.0
    # Model-size cap exposed as constant (10 MB).
    assert cw_mod.RETRAIN_MODEL_SIZE_BYTES_CAP == 10 * 1024 * 1024


def test_retrain_model_size_capped_10mb(
    learning_db: Path, memory_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the trained booster serialises above 10 MB, retrain aborts
    without persisting a candidate or flipping active."""
    from superlocalmemory.learning import consolidation_worker as cw_mod

    _seed_active_model(learning_db, profile_id="p", new_outcomes=100)

    # Force the measure helper to report 11 MB.
    monkeypatch.setattr(
        cw_mod, "_measure_serialized_size",
        lambda booster: 11 * 1024 * 1024,
    )

    # Bypass real training path — return a sentinel booster + non-empty rows.
    class _FakeBooster:
        def model_to_string(self) -> str:
            return "x" * 10
        def predict(self, X):  # pragma: no cover
            import numpy as np
            return np.zeros(len(X))

    def _fake_train(
        learning_db_path, profile_id, *, training_rows, feature_names,
        prior_row,
    ):
        return _FakeBooster(), {"mean_score": 0.0}

    monkeypatch.setattr(cw_mod, "_train_booster", _fake_train)
    monkeypatch.setattr(
        cw_mod, "_fetch_training_rows",
        lambda db, pid: ([
            {"query_id": "q1", "fact_id": "f1", "position": 0,
             "features": {n: 0.0 for n in cw_mod._feature_names()},
             "outcome_reward": 1.0},
            {"query_id": "q1", "fact_id": "f2", "position": 1,
             "features": {n: 0.0 for n in cw_mod._feature_names()},
             "outcome_reward": 0.0},
        ] * 30, ["cand"]),
    )

    result = cw_mod._run_shadow_cycle(
        memory_db_path=str(memory_db),
        learning_db_path=str(learning_db),
        profile_id="p",
    )
    assert result["aborted"] == "model_too_large"
    # No candidate row was written.
    with sqlite3.connect(learning_db) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_candidate=1",
        ).fetchone()[0]
        assert n == 0
        # Active row still exactly 1.
        n_active = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()[0]
        assert n_active == 1


def test_retrain_wall_time_capped_30s(
    learning_db: Path, memory_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wall-time >= 30s aborts without promoting."""
    from superlocalmemory.learning import consolidation_worker as cw_mod

    _seed_active_model(learning_db, profile_id="p", new_outcomes=100)

    # Simulate a slow trainer via the _train_booster hook.
    def _slow_train(
        learning_db_path, profile_id, *, training_rows, feature_names,
        prior_row,
    ):
        raise cw_mod.RetrainWallTimeExceeded(elapsed_sec=31.0)

    monkeypatch.setattr(cw_mod, "_train_booster", _slow_train)
    monkeypatch.setattr(
        cw_mod, "_fetch_training_rows",
        lambda db, pid: (
            [
                {"query_id": "q1", "fact_id": f"f{i}", "position": i,
                 "features": {n: 0.0 for n in cw_mod._feature_names()},
                 "outcome_reward": float(i % 2)}
                for i in range(40)
            ],
            ["cand"],
        ),
    )
    result = cw_mod._run_shadow_cycle(
        memory_db_path=str(memory_db),
        learning_db_path=str(learning_db),
        profile_id="p",
    )
    assert result["aborted"] == "wall_time_exceeded"
    with sqlite3.connect(learning_db) as conn:
        n_active = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()[0]
        assert n_active == 1


# ---------------------------------------------------------------------------
# Candidate / promotion / lineage
# ---------------------------------------------------------------------------


def test_candidate_not_auto_promoted(
    learning_db: Path, memory_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful retrain writes a candidate row; the active row
    remains untouched until shadow-test decides ``promote``.
    """
    from superlocalmemory.learning import consolidation_worker as cw_mod

    _seed_active_model(learning_db, profile_id="p", new_outcomes=100)

    class _FakeBooster:
        def model_to_string(self) -> str:
            return "tiny-model"
        def predict(self, X):
            import numpy as np
            return np.arange(len(X), dtype=float)

    def _fake_train(
        learning_db_path, profile_id, *, training_rows, feature_names,
        prior_row,
    ):
        return _FakeBooster(), {"mean_score": 0.5}

    monkeypatch.setattr(cw_mod, "_train_booster", _fake_train)
    monkeypatch.setattr(cw_mod, "_measure_serialized_size", lambda b: 128)
    monkeypatch.setattr(
        cw_mod, "_fetch_training_rows",
        lambda db, pid: (
            [
                {"query_id": "q1", "fact_id": f"f{i}", "position": i,
                 "features": {n: 0.0 for n in cw_mod._feature_names()},
                 "outcome_reward": 1.0}
                for i in range(30)
            ],
            ["cand"],
        ),
    )
    result = cw_mod._run_shadow_cycle(
        memory_db_path=str(memory_db),
        learning_db_path=str(learning_db),
        profile_id="p",
    )
    assert result["candidate_persisted"] is True
    assert result.get("promoted") is False  # NOT auto-promoted.

    with sqlite3.connect(learning_db) as conn:
        n_cand = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_candidate=1",
        ).fetchone()[0]
        assert n_cand == 1
        n_active = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()[0]
        assert n_active == 1


def test_promotion_flips_lineage_atomically(learning_db: Path) -> None:
    """Promotion MUST flip (active→previous) + (candidate→active) in a
    single BEGIN IMMEDIATE transaction. The partial unique index
    ``idx_model_active_one`` prevents double-active; ``idx_model_candidate_one``
    prevents double-candidate."""
    from superlocalmemory.learning.consolidation_worker import _promote_candidate

    active_id = _seed_active_model(
        learning_db, profile_id="p", new_outcomes=0,
        state_bytes=b"active-state",
    )
    # Insert candidate row.
    with sqlite3.connect(learning_db) as conn:
        cur = conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, bytes_sha256, is_active, is_candidate,"
            " trained_at, updated_at) "
            "VALUES (?, ?, ?, 0, 1, ?, ?)",
            ("p", b"cand-state", "0" * 64, _now_iso(), _now_iso()),
        )
        conn.commit()
        cand_id = int(cur.lastrowid or 0)

    _promote_candidate(str(learning_db), profile_id="p", candidate_id=cand_id)

    with sqlite3.connect(learning_db) as conn:
        rows = list(conn.execute(
            "SELECT id, is_active, is_previous, is_candidate, promoted_at "
            "FROM learning_model_state WHERE profile_id='p' ORDER BY id",
        ))
        assert len(rows) == 2
        prev_row = [r for r in rows if r[0] == active_id][0]
        new_active = [r for r in rows if r[0] == cand_id][0]
        # Old active now previous, not active, not candidate.
        assert prev_row[1] == 0 and prev_row[2] == 1 and prev_row[3] == 0
        # New active flipped: active=1, previous=0, candidate=0, promoted_at set.
        assert new_active[1] == 1
        assert new_active[2] == 0
        assert new_active[3] == 0
        assert new_active[4] is not None

        # Partial unique indexes hold — exactly one active, zero candidates.
        n_active = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()[0]
        assert n_active == 1
        n_cand = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_candidate=1",
        ).fetchone()[0]
        assert n_cand == 0

    # Double-active attempt fails — partial unique index enforces.
    with pytest.raises(sqlite3.IntegrityError):
        with sqlite3.connect(learning_db) as conn:
            conn.execute(
                "INSERT INTO learning_model_state "
                "(profile_id, state_bytes, bytes_sha256, is_active, "
                " trained_at, updated_at) "
                "VALUES ('p', ?, ?, 1, ?, ?)",
                (b"boom", "0" * 64, _now_iso(), _now_iso()),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_fires_on_200_recall_regression(learning_db: Path) -> None:
    """After 200 post-promotion recalls with ≥2% MRR drop,
    ``ModelRollback.should_rollback`` is True.
    """
    from superlocalmemory.learning.model_rollback import ModelRollback

    rb = ModelRollback(
        learning_db_path=str(learning_db),
        profile_id="p",
        baseline_ndcg=0.50,
    )
    # Feed 200 recalls averaging 0.45 → 10% regression.
    for i in range(rb.WATCH_WINDOW):
        rb.record_post_promotion(query_id=f"q{i}", ndcg_at_10=0.45)
    assert rb.should_rollback() is True


# ---------------------------------------------------------------------------
# Stage 8 F4.B — H-04 ModelRollback baseline_ndcg ≤ 0 handling (skeptic-H03)
# ---------------------------------------------------------------------------
#
# Previously: should_rollback returned False whenever baseline_ndcg ≤ 0,
# which silently disarmed the watchdog when the pre-promotion shadow mean
# was exactly 0 (valid observation on sparse data — every query had zero
# relevance hits). A freshly-promoted-but-broken model could sit forever
# with no rollback trigger.
#
# Fix: on small / zero / negative baselines, use ABSOLUTE drop against a
# fixed floor (REGRESSION_THRESHOLD = 0.02). Ratio-based logic is still
# used when baseline ≥ 0.05 (where division by baseline is numerically
# safe AND semantically meaningful).


def test_rollback_fires_when_baseline_is_zero_and_current_is_negative(
    learning_db: Path,
) -> None:
    """Stage 8 H-04: baseline_ndcg=0 with negative observations must still
    trigger rollback. Previously the guard silently disarmed.
    """
    from superlocalmemory.learning.model_rollback import ModelRollback

    rb = ModelRollback(
        learning_db_path=str(learning_db),
        profile_id="p",
        baseline_ndcg=0.0,
    )
    # 200 observations averaging -0.04 — clear regression vs a zero baseline.
    for i in range(rb.WATCH_WINDOW):
        rb.record_post_promotion(query_id=f"q{i}", ndcg_at_10=-0.04)
    assert rb.should_rollback() is True, (
        "baseline=0, current=-0.04 must trigger rollback "
        "(absolute drop 0.04 > threshold 0.02)"
    )


def test_rollback_fires_when_baseline_is_zero_and_current_is_worse(
    learning_db: Path,
) -> None:
    """baseline=0, current drops to -0.02 — exactly at threshold, must fire."""
    from superlocalmemory.learning.model_rollback import ModelRollback

    rb = ModelRollback(
        learning_db_path=str(learning_db),
        profile_id="p",
        baseline_ndcg=0.0,
    )
    for i in range(rb.WATCH_WINDOW):
        rb.record_post_promotion(query_id=f"q{i}", ndcg_at_10=-0.02)
    assert rb.should_rollback() is True, (
        "baseline=0 with current=-0.02 (absolute drop = threshold) "
        "must trigger — the ratio path previously silently disarmed"
    )


def test_rollback_does_not_fire_when_baseline_zero_and_current_matches(
    learning_db: Path,
) -> None:
    """baseline=0, current≈0: no regression, no rollback. Regression guard
    must not over-fire on the fix path."""
    from superlocalmemory.learning.model_rollback import ModelRollback

    rb = ModelRollback(
        learning_db_path=str(learning_db),
        profile_id="p",
        baseline_ndcg=0.0,
    )
    for i in range(rb.WATCH_WINDOW):
        rb.record_post_promotion(query_id=f"q{i}", ndcg_at_10=0.0)
    assert rb.should_rollback() is False


def test_rollback_fires_on_tiny_positive_baseline_regression(
    learning_db: Path,
) -> None:
    """baseline=0.01 (below 0.05 ratio-safe floor), current drops by 0.03 —
    must fire via the absolute-drop fallback.
    """
    from superlocalmemory.learning.model_rollback import ModelRollback

    rb = ModelRollback(
        learning_db_path=str(learning_db),
        profile_id="p",
        baseline_ndcg=0.01,
    )
    for i in range(rb.WATCH_WINDOW):
        rb.record_post_promotion(query_id=f"q{i}", ndcg_at_10=-0.02)
    # Absolute drop = 0.03, above threshold 0.02 → fire.
    assert rb.should_rollback() is True


def test_rollback_respects_watch_window_200_observations(
    learning_db: Path,
) -> None:
    """Even with a visible regression, rollback doesn't fire before
    WATCH_WINDOW observations are collected. Ensures the fix didn't
    break the minimum-sample-size guard.
    """
    from superlocalmemory.learning.model_rollback import ModelRollback

    rb = ModelRollback(
        learning_db_path=str(learning_db),
        profile_id="p",
        baseline_ndcg=0.5,
    )
    # 199 observations — just under window.
    for i in range(rb.WATCH_WINDOW - 1):
        rb.record_post_promotion(query_id=f"q{i}", ndcg_at_10=0.1)
    assert rb.should_rollback() is False
    # One more → triggers.
    rb.record_post_promotion(query_id="q-final", ndcg_at_10=0.1)
    assert rb.should_rollback() is True


def test_rollback_fires_when_baseline_positive_ratio_path_unchanged(
    learning_db: Path,
) -> None:
    """When baseline ≥ 0.05, the ratio-based path keeps its existing
    semantics — 2% relative drop fires. Regression guard on the fix.
    """
    from superlocalmemory.learning.model_rollback import ModelRollback

    rb = ModelRollback(
        learning_db_path=str(learning_db),
        profile_id="p",
        baseline_ndcg=0.50,
    )
    # 2.1% relative drop = 0.489. Above 2% threshold → fire.
    for i in range(rb.WATCH_WINDOW):
        rb.record_post_promotion(query_id=f"q{i}", ndcg_at_10=0.489)
    assert rb.should_rollback() is True


def test_rollback_restores_is_previous(learning_db: Path) -> None:
    """Executing rollback flips current active → is_rollback and
    the is_previous row → is_active=1."""
    from superlocalmemory.learning.model_rollback import ModelRollback

    _seed_previous_model(learning_db, profile_id="p", state_bytes=b"old-good")
    active_id = _seed_active_model(
        learning_db, profile_id="p", new_outcomes=0,
        state_bytes=b"bad-new",
    )

    rb = ModelRollback(
        learning_db_path=str(learning_db),
        profile_id="p",
        baseline_ndcg=0.5,
    )
    rb.execute_rollback(reason="test_regression")

    with sqlite3.connect(learning_db) as conn:
        # The former active row is now is_rollback=1, is_active=0.
        bad = conn.execute(
            "SELECT is_active, is_rollback, rollback_reason "
            "FROM learning_model_state WHERE id=?",
            (active_id,),
        ).fetchone()
        assert bad[0] == 0
        assert bad[1] == 1
        assert bad[2] == "test_regression"

        # Exactly one active row remains, and its state_bytes == old-good.
        active = conn.execute(
            "SELECT state_bytes FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()
        assert active is not None
        assert bytes(active[0]) == b"old-good"


def test_rollback_disables_retrain_24h(learning_db: Path) -> None:
    """After rollback, metadata_json.retrain_disabled_until is set
    ≥24h in the future; ``_should_retrain`` returns False until then."""
    from superlocalmemory.learning.consolidation_worker import ConsolidationWorker
    from superlocalmemory.learning.model_rollback import ModelRollback

    _seed_previous_model(learning_db, profile_id="p")
    _seed_active_model(learning_db, profile_id="p", new_outcomes=500)

    rb = ModelRollback(
        learning_db_path=str(learning_db), profile_id="p",
        baseline_ndcg=0.5,
    )
    rb.execute_rollback(reason="regression")

    with sqlite3.connect(learning_db) as conn:
        meta_raw = conn.execute(
            "SELECT metadata_json FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()[0]
        meta = json.loads(meta_raw or "{}")
        disabled_until = meta.get("retrain_disabled_until")
        assert disabled_until is not None
        parsed = datetime.fromisoformat(disabled_until)
        now = datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta_h = (parsed - now).total_seconds() / 3600.0
        assert delta_h >= 23.5  # within clock skew tolerance

    worker = ConsolidationWorker(
        memory_db=":memory:", learning_db=str(learning_db),
    )
    # Even with 500 outcomes, trigger is suppressed while within the
    # disabled window.
    assert worker._should_retrain(profile_id="p") is False


def test_rollback_reason_logged(learning_db: Path, caplog) -> None:
    """execute_rollback writes rollback_reason on the rollback row AND
    logs a warning line carrying profile_id + reason."""
    import logging
    from superlocalmemory.learning.model_rollback import ModelRollback

    _seed_previous_model(learning_db, profile_id="p")
    _seed_active_model(learning_db, profile_id="p")

    rb = ModelRollback(
        learning_db_path=str(learning_db), profile_id="p",
        baseline_ndcg=0.5,
    )
    caplog.set_level(logging.WARNING, logger="superlocalmemory.learning.model_rollback")
    rb.execute_rollback(reason="bench_regression_v1")

    records = [r for r in caplog.records if "rollback" in r.getMessage().lower()]
    assert records, "rollback log line missing"
    msg = records[-1].getMessage()
    assert "p" in msg
    assert "bench_regression_v1" in msg

    with sqlite3.connect(learning_db) as conn:
        reason = conn.execute(
            "SELECT rollback_reason FROM learning_model_state "
            "WHERE profile_id='p' AND is_rollback=1",
        ).fetchone()[0]
        assert reason == "bench_regression_v1"
