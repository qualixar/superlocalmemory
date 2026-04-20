# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-02 §6.3

"""TDD tests for the LightGBM lambdarank training path + model_cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("numpy")

import lightgbm as lgb
import numpy as np

from superlocalmemory.learning import model_cache
from superlocalmemory.learning.consolidation_worker import (
    _build_training_matrix,
    _retrain_ranker_impl,
)
from superlocalmemory.learning.database import LearningDatabase
from superlocalmemory.learning.features import FEATURE_NAMES
from superlocalmemory.learning.labeler import label_gain
from superlocalmemory.learning.model_cache import (
    ActiveModel,
    drift_mode,
    load_active,
)
from superlocalmemory.learning.ranker import AdaptiveRanker
from superlocalmemory.learning.signals import record_signal_batch
from tests.test_learning._signal_fixtures import (
    make_db_with_migrations,
    make_batch,
    open_conn,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic training data generator
# ---------------------------------------------------------------------------


def _seed_training_rows(db: LearningDatabase, *, profile_id: str = "p1",
                         n_queries: int = 40, per_query: int = 10) -> int:
    conn = open_conn(db)
    for q in range(n_queries):
        batch = make_batch(
            profile_id=profile_id,
            query_id=f"q-train-{q:04d}",
            query_text=f"query number {q}",
            n_candidates=per_query,
        )
        record_signal_batch(conn, batch)
    conn.close()
    return n_queries * per_query


# ---------------------------------------------------------------------------
# §6.3 test_trains_with_lambdarank_int_labels (CR1 + CR2)
# ---------------------------------------------------------------------------


def test_trains_with_lambdarank_int_labels(tmp_path):
    db = make_db_with_migrations(tmp_path)
    _seed_training_rows(db, n_queries=40, per_query=10)
    ok = _retrain_ranker_impl(db._db_path, "p1")
    assert ok is True

    row = db.load_active_model("p1")
    assert row is not None
    # bytes_sha256 populated.
    assert row["bytes_sha256"] == hashlib.sha256(row["state_bytes"]).hexdigest()


# ---------------------------------------------------------------------------
# §6.3 test_group_param_required (CR3)
# ---------------------------------------------------------------------------


def test_group_param_required_returns_false_when_single_query(tmp_path):
    db = make_db_with_migrations(tmp_path)
    # Only one query_id across 250 rows → groups = [250], len<2 → False.
    conn = open_conn(db)
    for i in range(25):
        batch = make_batch(
            profile_id="p1",
            query_id="q-single",
            query_text="solo",
            n_candidates=10,
        )
        record_signal_batch(conn, batch)
    conn.close()
    ok = _retrain_ranker_impl(db._db_path, "p1")
    assert ok is False


# ---------------------------------------------------------------------------
# §6.3 test_model_roundtrip_via_model_to_string (CR5 + M2)
# ---------------------------------------------------------------------------


def test_model_roundtrip_via_model_to_string(tmp_path):
    db = make_db_with_migrations(tmp_path)
    _seed_training_rows(db)
    assert _retrain_ranker_impl(db._db_path, "p1")

    model = load_active(db, "p1", use_cache=False)
    assert model is not None
    # Predict works on a 20-dim matrix.
    X = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    preds = model.booster.predict(X)
    assert preds.shape == (3,)


# ---------------------------------------------------------------------------
# §6.3 test_persisted_bytes_sha256_matches_stored_digest (CR5)
# ---------------------------------------------------------------------------


def test_persisted_bytes_sha256_matches_stored_digest(tmp_path):
    db = make_db_with_migrations(tmp_path)
    _seed_training_rows(db)
    assert _retrain_ranker_impl(db._db_path, "p1")
    row = db.load_active_model("p1")
    assert row is not None
    assert hashlib.sha256(row["state_bytes"]).hexdigest() == row["bytes_sha256"]


# ---------------------------------------------------------------------------
# §6.3 test_phase3_requires_active_and_verified_model (M2 + phase truth)
# ---------------------------------------------------------------------------


def test_phase3_requires_active_and_verified_model(tmp_path):
    db = make_db_with_migrations(tmp_path)
    _seed_training_rows(db)

    # (a) No model yet → phase 2 (we seeded 400 signals).
    assert db.load_active_model("p1") is None
    sig_count = db.count_signals("p1")
    assert sig_count >= 200
    ranker = AdaptiveRanker(signal_count=sig_count, active_model=None)
    assert ranker.phase == 2

    # (b) Train → active model + verify → phase 3.
    assert _retrain_ranker_impl(db._db_path, "p1")
    model = load_active(db, "p1", use_cache=False)
    assert model is not None
    r3 = AdaptiveRanker(signal_count=sig_count, active_model=model)
    assert r3.phase == 3

    # (c) Tamper the bytes → SHA mismatch → None → phase 2.
    conn = sqlite3.connect(db._db_path)
    conn.execute(
        "UPDATE learning_model_state SET state_bytes = ? WHERE is_active=1",
        (b"corrupted", ),
    )
    conn.commit()
    conn.close()
    model_cache.invalidate("p1")
    tampered = load_active(db, "p1", use_cache=False)
    assert tampered is None


# ---------------------------------------------------------------------------
# §6.3 test_feature_name_drift_pads_subset + unknown refuse
# ---------------------------------------------------------------------------


def _make_fake_active(feature_names: tuple[str, ...]) -> ActiveModel:
    # Train a trivial booster on synthetic data so drift_mode can be tested
    # without touching the DB.
    X = np.random.rand(40, len(FEATURE_NAMES)).astype(np.float32)
    y = np.random.randint(0, 5, size=40).astype(np.int32)
    group = [10, 10, 10, 10]
    ds = lgb.Dataset(X, label=y, group=group,
                     feature_name=list(FEATURE_NAMES), free_raw_data=False)
    params = {
        "objective": "lambdarank", "metric": "ndcg",
        "label_gain": label_gain(), "verbosity": -1,
        "min_data_in_leaf": 1,
    }
    booster = lgb.train(params, ds, num_boost_round=3)
    return ActiveModel(
        profile_id="fake",
        booster=booster,
        feature_names=feature_names,
        trained_at="",
        sha256="0" * 64,
    )


def test_feature_name_drift_pads_subset():
    # Subset: only first 10 names — valid subset of current 20.
    model = _make_fake_active(tuple(FEATURE_NAMES[:10]))
    assert drift_mode(model) == "subset"


def test_feature_name_drift_unknown_name_refuses():
    model = _make_fake_active(tuple(FEATURE_NAMES) + ("mystery_feature",))
    assert drift_mode(model) == "unknown"


def test_drift_aligned_when_names_match():
    model = _make_fake_active(tuple(FEATURE_NAMES))
    assert drift_mode(model) == "aligned"


# ---------------------------------------------------------------------------
# §6.3 test_shadow_test_gate_2pct_ndcg (CR4) — integration via retrain path
# ---------------------------------------------------------------------------


def test_shadow_test_gate_keeps_old_when_not_better(tmp_path):
    db = make_db_with_migrations(tmp_path)
    _seed_training_rows(db)

    # First retrain — no prior → promote.
    assert _retrain_ranker_impl(db._db_path, "p1")
    row_a = db.load_active_model("p1")
    assert row_a is not None
    sha_a = row_a["bytes_sha256"]

    # Second retrain on the SAME data — should fail shadow (identical perf).
    # It's allowed to either keep old or promote if deterministic results
    # coincide; we simply assert the gate path runs without crash and that
    # the *active* row is still present + verifiable.
    _retrain_ranker_impl(db._db_path, "p1")
    row_b = db.load_active_model("p1")
    assert row_b is not None
    assert hashlib.sha256(row_b["state_bytes"]).hexdigest() == row_b["bytes_sha256"]
    # The shadow test predicate is deterministic per implementation — on
    # identical rows the improvement delta is ~0 so prior should win.
    assert row_b["bytes_sha256"] == sha_a


# ---------------------------------------------------------------------------
# §6.3 test_model_cache_lru_single_load (RP2) — concurrent loads share result
# ---------------------------------------------------------------------------


def test_model_cache_lru_single_load(tmp_path):
    db = make_db_with_migrations(tmp_path)
    _seed_training_rows(db)
    assert _retrain_ranker_impl(db._db_path, "p1")

    class _CountingDB:
        def __init__(self, inner):
            self._inner = inner
            self.calls = 0
            self._lock = threading.Lock()

        def load_active_model(self, profile_id):
            with self._lock:
                self.calls += 1
            return self._inner.load_active_model(profile_id)

    counting = _CountingDB(db)
    model_cache.invalidate("p1")
    results = []

    def worker():
        results.append(load_active(counting, "p1"))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r is not None for r in results)
    # All threads return the same shared ActiveModel.
    assert len({id(r) for r in results}) == 1
    # Only ONE DB SELECT happened across 10 concurrent loaders.
    assert counting.calls == 1


# ---------------------------------------------------------------------------
# Static CI guards (§7)
# ---------------------------------------------------------------------------


def _learning_dir() -> Path:
    return (Path(__file__).resolve().parent.parent.parent
            / "src" / "superlocalmemory" / "learning")


def test_pickle_loads_grep_guard():
    import re
    pat = re.compile(r"pickle\.loads\s*\(")
    for p in _learning_dir().rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert not pat.search(text), f"pickle.loads found in {p}"


def test_no_identity_mapping_regression():
    """§7: ``signal_value.*label`` pattern must not exist in learning/."""
    import re
    pat = re.compile(r"signal_value.*label")
    for p in _learning_dir().rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert not pat.search(text), f"identity mapping regression in {p}"


def test_no_legacy_model_bytes_column():
    import re
    pat_model = re.compile(r"model_bytes\s*=")
    pat_sha = re.compile(r"sha256_hex\s*=")
    for p in _learning_dir().rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert not pat_model.search(text), f"legacy model_bytes in {p}"
        assert not pat_sha.search(text), f"legacy sha256_hex in {p}"


def test_no_fstring_sql_in_lld02_new_modules():
    """f-string SQL is forbidden in the NEW modules introduced by LLD-02.

    Pre-existing f-string SQL with hard-coded values (e.g. iterating a tuple
    of known table names in ``database.reset()`` or ``IN``-clause
    placeholders in ``_generate_patterns``) is out of scope for LLD-02
    Wave 2 Stream B; the rule in §7 targets user-data interpolation.
    """
    import re
    new_modules = [
        "signals.py",
        "signal_worker.py",
        "model_cache.py",
        "labeler.py",
        "ranker.py",
    ]
    pattern = re.compile(
        r"""f["'][^"']*\b(SELECT|INSERT|UPDATE|DELETE)\b""",
    )
    base = _learning_dir()
    for name in new_modules:
        text = (base / name).read_text(encoding="utf-8")
        hits = pattern.findall(text)
        assert not hits, f"f-string SQL detected in {name}: {hits}"
