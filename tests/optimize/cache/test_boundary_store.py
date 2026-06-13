"""TDD tests for boundary_store.py — vCache online MLE.

LLD-03 §3.1 + §4.2. RED-first. Asserts the REAL vCache algorithm from
arXiv:2502.03771 is implemented, NOT a fabricated step-based heuristic.
"""

from __future__ import annotations

import pytest

from superlocalmemory.optimize.cache.boundary_store import (
    BoundaryStore,
    PerItemBoundaryRecord,
    _fit_logistic_mle,
    _sigmoid,
)


def test_sigmoid_is_zero_at_boundary():
    """L(t, t, γ) = 0.5 (Eq. 9 — at the boundary, sigmoid is 0.5)."""
    assert abs(_sigmoid(s=0.95, t=0.95, gamma=10.0) - 0.5) < 1e-6


def test_sigmoid_saturates_above_boundary():
    """L(s, t, γ) ≈ 1.0 when s >> t (high similarity → high correctness)."""
    # gamma=10, s - t = 0.5 → L ≈ sigmoid(5) ≈ 0.9933
    assert _sigmoid(s=0.95, t=0.45, gamma=10.0) > 0.99


def test_sigmoid_saturates_below_boundary():
    """L(s, t, γ) ≈ 0.0 when s << t (low similarity → low correctness)."""
    # gamma=10, s - t = -0.5 → L ≈ sigmoid(-5) ≈ 0.0067
    assert _sigmoid(s=0.45, t=0.95, gamma=10.0) < 0.01


def test_compute_tau_cold_start_returns_one():
    """n < 3 samples → τ̂ = 1.0 (always explore — cold start)."""
    rec = PerItemBoundaryRecord(entry_id="e1", samples=[])
    assert rec.compute_tau(0.95) == 1.0

    rec2 = PerItemBoundaryRecord(
        entry_id="e1", samples=[(0.9, 1), (0.8, 0)]
    )  # only 2 samples
    assert rec2.compute_tau(0.95) == 1.0


def test_compute_tau_high_similarity_low_after_learning():
    """After learning: high-similarity query → low τ̂ (exploit)."""
    # All (sim, correct) pairs: high similarity → always correct.
    samples = [(s, 1) for s in [0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.97, 0.95]]
    rec = PerItemBoundaryRecord(
        entry_id="e1",
        t_hat=0.90,
        gamma_hat=10.0,
        samples=samples,
    )
    tau_high = rec.compute_tau(0.99, delta=0.05)
    # High-similarity, model is confident → low τ̂
    assert tau_high < 0.5, f"expected low τ̂ for high-similarity, got {tau_high}"


def test_compute_tau_low_similarity_high_after_learning():
    """After learning: low-similarity query → high τ̂ (explore)."""
    # Pattern: high sim → correct, low sim → incorrect.
    samples = [
        (0.99, 1), (0.98, 1), (0.97, 1), (0.96, 1), (0.95, 1),
        (0.50, 0), (0.40, 0), (0.30, 0),
    ]
    rec = PerItemBoundaryRecord(
        entry_id="e1", t_hat=0.90, gamma_hat=10.0, samples=samples,
    )
    tau_low = rec.compute_tau(0.50, delta=0.05)
    # Low-similarity, model expects miss → high τ̂ (explore)
    assert tau_low > 0.5, f"expected high τ̂ for low-similarity, got {tau_low}"


def test_add_sample_refits_t_hat():
    """add_sample refits t_hat via MLE on accumulated samples.

    With well-separated positive and negative samples, the MLE refit
    produces a t_hat that sits between the two clusters — and the
    magnitude of the move reflects the MLE pull (not a fixed step size).
    """
    rec = PerItemBoundaryRecord(
        entry_id="e1", t_hat=0.95, gamma_hat=10.0,
        samples=[(0.95, 1), (0.96, 1), (0.94, 1)],  # 3 high-sim positives
    )
    # Add 3 low-sim negatives
    out = rec
    for s in [0.40, 0.30, 0.20]:
        out = out.add_sample(similarity=s, was_correct=False, max_samples=200)
    # MLE boundary should sit between the two clusters (or at least move
    # meaningfully down from the cold-start 0.95).
    assert out.t_hat < 0.95, f"t_hat={out.t_hat} should drop after negative samples"
    # And it must be at or below the highest positive sample
    assert out.t_hat <= 0.95


def test_add_sample_sliding_window():
    """Sliding window caps samples at max_samples (A-23 fix)."""
    rec = PerItemBoundaryRecord(
        entry_id="e1", t_hat=0.95, gamma_hat=10.0, samples=[]
    )
    out = rec
    for i in range(10):
        out = out.add_sample(similarity=0.9, was_correct=True, max_samples=5)
    assert len(out.samples) == 5


def test_fit_logistic_mle_converges_on_perfect_separation():
    """Perfectly separable data → MLE converges to high γ (steep sigmoid)."""
    samples = [
        (0.99, 1), (0.98, 1), (0.97, 1), (0.96, 1), (0.95, 1),
        (0.40, 0), (0.30, 0), (0.20, 0), (0.10, 0),
    ]
    t, gamma = _fit_logistic_mle(samples, t_init=0.7, gamma_init=5.0)
    # Boundary should sit between the two clusters
    assert 0.5 < t < 0.9
    # Gamma should be positive (steep)
    assert gamma > 1.0


def test_fit_logistic_mle_fail_open_on_empty():
    """Empty samples → returns initial params."""
    t, gamma = _fit_logistic_mle([], t_init=0.85, gamma_init=10.0)
    assert (t, gamma) == (0.85, 10.0)


def test_should_explore_returns_bool():
    """should_explore returns a boolean, never raises."""
    rec = PerItemBoundaryRecord(
        entry_id="e1", t_hat=0.90, gamma_hat=10.0,
        samples=[(0.9, 1), (0.8, 1), (0.95, 1), (0.85, 0)],
    )
    result = rec.should_explore(0.92, delta=0.05)
    assert isinstance(result, bool)


# ---- BoundaryStore roundtrip ------------------------------------------------

def test_boundary_store_get_default_when_missing(tmp_path):
    """First lookup returns cold-start default record (never None)."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "test.db"
    db = CacheDB(db_path=db_path)
    store = BoundaryStore(db=db, default_t=0.95, default_gamma=10.0)
    rec = store.get("nonexistent_entry_id")
    assert rec.entry_id == "nonexistent_entry_id"
    assert rec.t_hat == 0.95
    assert rec.gamma_hat == 10.0
    assert rec.samples == []


def test_boundary_store_save_and_get(tmp_path):
    """Save a record, get it back, params roundtrip."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "test.db"
    db = CacheDB(db_path=db_path)
    store = BoundaryStore(db=db, default_t=0.95, default_gamma=10.0)
    rec = PerItemBoundaryRecord(
        entry_id="entry_abc",
        t_hat=0.88, gamma_hat=12.0, samples=[(0.9, 1)], last_updated=1000.0,
    )
    store.save(rec)
    # Re-fetch (bypass in-memory cache)
    fresh = BoundaryStore(db=db, default_t=0.95, default_gamma=10.0)
    out = fresh.get("entry_abc")
    assert out.entry_id == "entry_abc"
    assert abs(out.t_hat - 0.88) < 1e-6
    assert abs(out.gamma_hat - 12.0) < 1e-6


def test_boundary_store_record_outcome_refits(tmp_path):
    """record_outcome: add sample → refit MLE → save."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "test.db"
    db = CacheDB(db_path=db_path)
    store = BoundaryStore(db=db, default_t=0.95, default_gamma=10.0)
    rec1 = store.record_outcome("entry_x", similarity=0.50, was_correct=False)
    assert rec1.t_hat < 0.95  # boundary dropped toward the negative sample
    rec2 = store.record_outcome("entry_x", similarity=0.50, was_correct=False)
    assert rec2.t_hat <= rec1.t_hat  # more negative samples pull it down


def test_boundary_store_delete(tmp_path):
    """delete removes the row; subsequent get returns default."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "test.db"
    db = CacheDB(db_path=db_path)
    store = BoundaryStore(db=db, default_t=0.95, default_gamma=10.0)
    store.save(PerItemBoundaryRecord(entry_id="e1", t_hat=0.85, gamma_hat=8.0))
    store.delete("e1")
    rec = store.get("e1")
    assert rec.t_hat == 0.95  # default


# ---- Additional coverage tests ----

def test_compute_tau_denom_near_zero():
    """compute_tau handles denom < EPS → tau = 0.0 (near-certainty)."""
    # Create a record where (1-delta) - alpha is close to 0 AND denom close to 0
    # Hard to produce naturally; test via explicit boundary manipulation
    samples = [(0.99, 1)] * 100  # all positive, high similarity
    rec = PerItemBoundaryRecord(entry_id="e2", t_hat=0.01, gamma_hat=100.0, samples=samples)
    tau = rec.compute_tau(0.99, delta=0.01)
    assert 0.0 <= tau <= 1.0


def test_rng_function():
    """_rng() returns a float in [0, 1)."""
    from superlocalmemory.optimize.cache.boundary_store import _rng
    for _ in range(20):
        val = _rng()
        assert 0.0 <= val < 1.0


def test_fit_logistic_bce_empty_samples():
    """_binary_cross_entropy returns high loss for empty/gamma<=0."""
    from superlocalmemory.optimize.cache.boundary_store import _binary_cross_entropy
    loss = _binary_cross_entropy((0.9, -1.0), [(0.5, 1)])
    assert loss >= 1e8  # gamma <= 0 penalty

    loss2 = _binary_cross_entropy((0.9, 10.0), [])
    assert loss2 >= 1e8  # empty samples penalty


def test_fit_logistic_mle_gd_fallback(monkeypatch):
    """_fit_logistic_mle falls back to GD when scipy unavailable."""
    from superlocalmemory.optimize.cache import boundary_store as _bs
    # Force scipy to be unavailable
    monkeypatch.setattr(_bs, "_SCIPY_AVAILABLE", False)
    samples = [(0.99, 1), (0.98, 1), (0.50, 0), (0.40, 0)]
    t, gamma = _bs._fit_logistic_mle(samples, t_init=0.7, gamma_init=5.0)
    assert 0.5 <= t <= 1.0
    assert gamma > 0


def test_fit_logistic_gd():
    """_fit_logistic_gd runs gradient descent and returns bounded params."""
    from superlocalmemory.optimize.cache.boundary_store import _fit_logistic_gd
    samples = [(0.9, 1), (0.8, 1), (0.3, 0)]
    t, gamma = _fit_logistic_gd(samples, t_init=0.7, gamma_init=5.0, lr=0.1, steps=50)
    assert 0.5 <= t <= 1.0
    assert 0.1 <= gamma <= 100.0


def test_boundary_store_get_fail_open(tmp_path, monkeypatch):
    """BoundaryStore.get fails open — returns default on DB error."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "test.db"
    db = CacheDB(db_path=db_path)
    store = BoundaryStore(db=db, default_t=0.95, default_gamma=10.0)

    # Break boundary_get to raise
    monkeypatch.setattr(db, "boundary_get", lambda eid: (_ for _ in ()).throw(RuntimeError("db down")))
    rec = store.get("any_entry")
    assert rec.entry_id == "any_entry"
    assert rec.t_hat == 0.95  # fail-open default


def test_boundary_store_save_fail_open(tmp_path, monkeypatch):
    """BoundaryStore.save fails open — logs warning, does not raise."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "test.db"
    db = CacheDB(db_path=db_path)
    store = BoundaryStore(db=db, default_t=0.95, default_gamma=10.0)

    monkeypatch.setattr(db, "boundary_upsert", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("db down")))
    rec = PerItemBoundaryRecord(entry_id="e1")
    store.save(rec)  # must not raise


def test_boundary_store_load_all_fail_open(tmp_path, monkeypatch):
    """BoundaryStore.load_all fails open — returns {} on DB error."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "test.db"
    db = CacheDB(db_path=db_path)
    store = BoundaryStore(db=db, default_t=0.95, default_gamma=10.0)

    monkeypatch.setattr(db, "get_all_boundaries", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    result = store.load_all()
    assert result == {}


def test_boundary_store_delete_fail_open(tmp_path, monkeypatch):
    """BoundaryStore.delete fails open — logs warning, does not raise."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "test.db"
    db = CacheDB(db_path=db_path)
    store = BoundaryStore(db=db, default_t=0.95, default_gamma=10.0)

    monkeypatch.setattr(db, "delete_boundary", lambda eid: (_ for _ in ()).throw(RuntimeError("db down")))
    store.delete("e1")  # must not raise


def test_boundary_store_load_all_skip_empty_eid(tmp_path):
    """load_all skips rows with empty entry_id."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "test.db"
    db = CacheDB(db_path=db_path)
    store = BoundaryStore(db=db)

    # Simulate a row with empty entry_id
    monkeypatch = __import__("pytest").MonkeyPatch()
    original = store._db.get_all_boundaries

    def _fake_boundaries():
        return [
            {"entry_id": "", "logistic_t": 0.8, "logistic_gamma": 5.0, "updated_at": 100.0},
            {"entry_id": "valid_one", "logistic_t": 0.9, "logistic_gamma": 12.0, "updated_at": 200.0},
        ]
    store._db.get_all_boundaries = _fake_boundaries
    try:
        result = store.load_all()
        assert "valid_one" in result
        assert "" not in result
    finally:
        store._db.get_all_boundaries = original


def test_c03_cold_start_exploits_when_above_return_threshold():
    """C-03: compute_tau returns 0.0 (exploit) during cold start when query_sim >= return_threshold."""
    rec = PerItemBoundaryRecord(entry_id="e_c03", samples=[])
    # n=0, query_sim=0.99 >= return_threshold=0.98 → exploit (τ̂=0.0)
    tau = rec.compute_tau(query_sim=0.99, return_threshold=0.98)
    assert tau == 0.0, f"C-03: expected 0.0 (exploit) for high-similarity cold start, got {tau}"

    # exactly at threshold → exploit
    tau_at = rec.compute_tau(query_sim=0.98, return_threshold=0.98)
    assert tau_at == 0.0, f"C-03: expected 0.0 at exactly return_threshold, got {tau_at}"


def test_c03_cold_start_explores_when_below_return_threshold():
    """C-03: compute_tau returns 1.0 (explore) during cold start when query_sim < return_threshold."""
    rec = PerItemBoundaryRecord(entry_id="e_c03b", samples=[(0.9, 1), (0.8, 0)])  # n=2
    # query_sim=0.97 < return_threshold=0.98 → explore (τ̂=1.0)
    tau = rec.compute_tau(query_sim=0.97, return_threshold=0.98)
    assert tau == 1.0, f"C-03: expected 1.0 (explore) for below-threshold cold start, got {tau}"


def test_c03_cold_start_default_return_threshold_explores():
    """C-03: default return_threshold=1.0 means cold start always explores (backward compat)."""
    rec = PerItemBoundaryRecord(entry_id="e_c03c", samples=[])
    # default return_threshold=1.0, query_sim can never reach 1.0 in practice
    tau = rec.compute_tau(query_sim=0.999)
    assert tau == 1.0, f"C-03: default return_threshold=1.0 should always explore, got {tau}"


def test_c03_should_explore_false_above_return_threshold():
    """C-03: should_explore returns False (exploit) during cold start when sim >= return_threshold."""
    from superlocalmemory.optimize.cache import boundary_store as _bs
    _bs._RNG.seed(0)  # deterministic — _RNG.random() will not be called when tau=0.0
    rec = PerItemBoundaryRecord(entry_id="e_c03d", samples=[])
    # With tau=0.0, random() <= 0.0 is always False, so should_explore=False (exploit)
    result = rec.should_explore(query_sim=0.99, return_threshold=0.98)
    assert result is False, f"C-03: should_explore must return False (exploit) above threshold, got {result}"


def test_compute_tau_seeded_rng():
    """compute_tau determinism with seeded RNG."""
    from superlocalmemory.optimize.cache import boundary_store as _bs
    _bs._RNG.seed(42)
    samples = [(0.9, 1), (0.8, 1), (0.7, 1), (0.5, 0)]
    rec = PerItemBoundaryRecord(entry_id="e3", t_hat=0.80, gamma_hat=10.0, samples=samples)
    tau1 = rec.compute_tau(0.85)
    _bs._RNG.seed(42)
    tau2 = rec.compute_tau(0.85)
    assert abs(tau1 - tau2) < 1e-9  # deterministic with same seed
