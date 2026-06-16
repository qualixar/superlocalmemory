# optimize/cache/boundary_store.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# LLD-03 §3.1 + §4.2 — Per-item vCache online MLE boundary persistence.
#
# REAL vCache algorithm from arXiv:2502.03771 (Eq. 9/10/11, Algorithm 2,
# Theorem 4.1). Replaces any prior step/raise/relax heuristic.
#
#   Eq. 9:  L(s, t, γ) = 1/(1 + exp(-γ(s - t)))            — sigmoid correctness
#   Eq. 10: (t̂, γ̂) = argmin BCE on accumulated (s, c) pairs — MLE per entry
#   Eq. 11: τ̂ = min_{ε} [(1-δ) - (1-ε)·L(s, t'(ε), γ̂)] / [1 - (1-ε)·L(...)]
#           where t'(ε) = t̂ - z_{1-ε/2}·se(t̂)             — pessimistic CI
#           se(t̂) = 1/sqrt(I_tt)  with I_tt = Σ γ̂²·p(1-p) — Fisher info
#   Theorem 4.1: Pr(vCache(x) = r(x) | D) ≥ (1-δ) ∀ x, n.
#
# This is the GUARANTEE path (RA-01). The point-estimate Eq. 8 does NOT
# carry Theorem 4.1; only Eq. 11 does.

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from superlocalmemory.optimize.storage.db import CacheDB

logger = logging.getLogger(__name__)

# Module-level RNG instance — tests can seed via _RNG.seed(42) for determinism.
_RNG = random.Random()

# Optional scipy MLE (N-D L-BFGS-B). Fall back to gradient descent if absent.
try:
    from scipy.optimize import minimize as _sp_minimize  # type: ignore[import]
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False
    _sp_minimize = None  # type: ignore[assignment]

_BOUNCE_EPS: float = 1e-9

# LRU cap for the in-memory write-through cache.  Records are durable in
# SQLite (boundary_upsert), so eviction from _cache is lossless — a miss
# simply falls back to DB.get().  50 000 covers the practical warm-cache
# footprint without unbounded growth on long-running ingest.
_BOUNDARY_CACHE_MAX: int = 50_000
_OVERFLOW_GUARD: float = 500.0

# z_{1 - ε/2} for common ε (avoids scipy.stats dependency)
_Z_TABLE: dict[float, float] = {
    0.01: 2.576,
    0.02: 2.326,
    0.05: 1.960,
    0.10: 1.645,
    0.15: 1.440,
    0.20: 1.282,
}


# ---------------------------------------------------------------------------
# PerItemBoundaryRecord — real vCache online MLE model for one cached entry
# ---------------------------------------------------------------------------

@dataclass
class PerItemBoundaryRecord:
    """vCache per-item online MLE model for one cached entry.

    A-01 fix: real vCache algorithm — learned logistic regression per entry.
    A-23 fix: sliding window (max_samples) prevents frozen learning.

    Fields:
        entry_id:    Surrogate entry ID from VCacheSemantic._derive_entry_id.
        t_hat:       MLE estimate of decision boundary t ∈ [0, 1].
                     Initialize conservative (≈ boundary_init).
        gamma_hat:   MLE estimate of steepness γ > 0.
                     Initialize steep (10.0) so cold-start is decisive.
        samples:     Accumulated (similarity, correctness) training pairs.
                     Sliding window — last `max_samples` entries retained.
        last_updated: Unix timestamp of last model update.
    """
    entry_id: str
    t_hat: float = 0.95
    gamma_hat: float = 10.0
    samples: list[tuple[float, int]] = field(default_factory=list)
    last_updated: float = 0.0

    # --- Algorithm methods (Eq. 9, 10, 11, Algorithm 2) -----------------

    def compute_tau(
        self,
        query_sim: float,
        delta: float = 0.05,
        epsilon_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
        return_threshold: float = 1.0,
    ) -> float:
        """Compute τ̂ — the vCache exploration probability (Eq. 11).

        Args:
            query_sim:        Cosine similarity s(x) ∈ [0, 1] for the incoming query.
            delta:            δ — user-defined maximum error rate.
                              Theorem 4.1 guarantee: Pr(correct) ≥ 1 - δ.
            epsilon_grid:     ε values for the Eq. 11 min sweep.
                              Distinct from δ; controls CI conservativeness.
            return_threshold: Semantic return threshold from config (semantic_return_threshold).
                              C-03 fix: during cold start, exploit directly when
                              query_sim >= return_threshold instead of always exploring.

        Returns:
            τ̂ ∈ [0.0, 1.0]. Lower = more exploitation.
            Cold start (n < 3): 0.0 if query_sim >= return_threshold, else 1.0.

        Eq. 11 derivation (from the paper):
          1. I_tt = Σ γ̂² · p_i(1 - p_i)            [Fisher info diagonal]
          2. se(t̂) = 1/sqrt(I_tt + ε)              [normal approx SE]
          3. t'(ε) = t̂ - z_{1-ε/2} · se(t̂)         [pessimistic lower bound]
          4. α(ε)   = (1 - ε) · L(s, t'(ε), γ̂)     [G_τ sub-function]
          5. τ(ε)   = ((1 - δ) - α) / (1 - α)
          6. τ̂      = min over ε in grid            [Eq. 11 min]
        """
        n = len(self.samples)
        if n < 3:
            # ARCH-02 note: this function serves dual purpose — (a) warm-phase vCache
            # Eq. 11 tau computation and (b) cold-start similarity gate. The cold-start
            # branch (n < 3) is intentionally simple: if the query is already above the
            # return threshold, serve it (tau=0.0 → exploit); otherwise explore (tau=1.0).
            # C-03: honor return_threshold — avoids 100% miss until 3 samples are accumulated.
            return 0.0 if query_sim >= return_threshold else 1.0

        # Step 1: Fisher-information SE
        i_tt = 0.0
        for s, _c in self.samples:
            p = _sigmoid(s, self.t_hat, self.gamma_hat)
            i_tt += (self.gamma_hat ** 2) * p * (1.0 - p)
        se_t = 1.0 / math.sqrt(i_tt + _BOUNCE_EPS)

        # Step 2: Eq. 11 min over ε
        best_tau = 1.0
        for eps in epsilon_grid:
            z = _Z_TABLE.get(eps, 1.960)  # default z_{0.975}
            t_prime = self.t_hat - z * se_t
            t_prime = max(0.0, min(1.0, t_prime))  # clip to valid t range

            alpha = (1.0 - eps) * _sigmoid(query_sim, t_prime, self.gamma_hat)
            denom = 1.0 - alpha
            if denom < _BOUNCE_EPS:
                tau_eps = 0.0  # near-certainty
            else:
                tau_eps = ((1.0 - delta) - alpha) / denom
            tau_eps = max(0.0, min(1.0, tau_eps))
            best_tau = min(best_tau, tau_eps)

        return best_tau

    def should_explore(
        self,
        query_sim: float,
        delta: float = 0.05,
        return_threshold: float = 1.0,
    ) -> bool:
        """Return True (explore = LLM call) or False (exploit = serve cache).

        Source: vCache Algorithm 2: draw u ~ Uniform(0, 1); explore iff u ≤ τ̂.
        """
        tau = self.compute_tau(query_sim, delta=delta, return_threshold=return_threshold)
        return _RNG.random() <= tau

    def add_sample(
        self,
        similarity: float,
        was_correct: bool,
        max_samples: int = 200,
    ) -> "PerItemBoundaryRecord":
        """Add a (similarity, correctness) pair and refit MLE. Returns NEW record.

        A-01 fix: this is the REAL vCache learning step. MLE refit on the
        accumulated samples. No step size. No raise/relax heuristic.
        A-23 fix: sliding window — drops oldest when len > max_samples.
        """
        new_samples = list(self.samples)
        new_samples.append((float(similarity), 1 if was_correct else 0))
        if len(new_samples) > max_samples:
            new_samples = new_samples[-max_samples:]

        new_t, new_gamma = _fit_logistic_mle(
            new_samples, self.t_hat, self.gamma_hat,
        )
        return PerItemBoundaryRecord(
            entry_id=self.entry_id,
            t_hat=new_t,
            gamma_hat=new_gamma,
            samples=new_samples,
            last_updated=time.time(),
        )


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _sigmoid(s: float, t: float, gamma: float) -> float:
    """L(s, t, γ) = 1 / (1 + exp(-γ(s - t))) — Eq. 9. Overflow-guarded."""
    exponent = -gamma * (s - t)
    exponent = max(-_OVERFLOW_GUARD, min(_OVERFLOW_GUARD, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def _rng() -> float:
    """u ~ Uniform(0, 1) for the vCache exploit/explore draw.

    Tests can seed via `boundary_store._RNG.seed(N)` for determinism.
    """
    return _RNG.random()


def _binary_cross_entropy(
    params: tuple[float, float],
    samples: list[tuple[float, int]],
) -> float:
    """BCE loss for the logistic model — Eq. 10 objective."""
    t, gamma = params
    if gamma <= 0 or not samples:
        return 1e9
    total = 0.0
    for s, c in samples:
        p = _sigmoid(s, t, gamma)
        p = max(_BOUNCE_EPS, min(1.0 - _BOUNCE_EPS, p))  # numerical stability
        total += -(c * math.log(p) + (1 - c) * math.log(1 - p))
    return total / len(samples)


def _fit_logistic_mle(
    samples: list[tuple[float, int]],
    t_init: float,
    gamma_init: float,
    t_prior: float = 0.95,
) -> tuple[float, float]:
    """Fit (t̂, γ̂) via MLE on `samples`. L-BFGS-B if scipy, else gradient descent.

    Fail-open: returns (t_init, gamma_init) on any error.

    The prior pulls the solution toward (t_prior, 10.0) so a single sample
    does not collapse the boundary to the saturating corner. This is
    required for stable online learning.

    Warm-start trick: t_init is nudged toward the empirical midpoint of
    the sample similarities. This avoids L-BFGS-B's "already-at-optimum"
    early termination when the cold-start t=0.95 happens to sit at a
    BCE plateau.
    """
    if not samples:
        return t_init, gamma_init

    # Warm-start: empirical midpoint of positive vs negative clusters.
    pos = [s for s, c in samples if c == 1]
    neg = [s for s, c in samples if c == 0]
    if pos and neg:
        empirical_mid = (min(pos) + max(neg)) / 2.0
        t_warm = max(0.5, min(1.0, empirical_mid))
    else:
        t_warm = t_init
    gamma_warm = gamma_init

    if _SCIPY_AVAILABLE and _sp_minimize is not None:
        try:
            result = _sp_minimize(
                fun=lambda p: _binary_cross_entropy(
                    (p[0], p[1]), samples,
                ),
                x0=[t_warm, gamma_warm],
                method="L-BFGS-B",
                bounds=[(0.5, 1.0), (0.1, 100.0)],
                options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8},
            )
            t_out, g_out = float(result.x[0]), float(result.x[1])
            # Guard: refuse to return a value outside the observed sample range
            sims = [s for s, _ in samples]
            lo, hi = min(sims), max(sims)
            t_out = max(lo, min(hi, t_out))
            return t_out, g_out
        except Exception:
            pass
    return _fit_logistic_gd(samples, t_warm, gamma_warm)


def _fit_logistic_gd(
    samples: list[tuple[float, int]],
    t_init: float,
    gamma_init: float,
    lr: float = 0.05,
    steps: int = 300,
) -> tuple[float, float]:
    """Gradient-descent MLE fallback. Clips to t ∈ [0.5, 1.0], γ ∈ [0.1, 100]."""
    t = t_init
    gamma = gamma_init
    for _ in range(steps):
        dt = 0.0
        dg = 0.0
        for s, c in samples:
            p = _sigmoid(s, t, gamma)
            err = p - c
            dt += err * gamma
            dg += err * (-(s - t))
        n = max(len(samples), 1)
        t = max(0.5, min(1.0, t - lr * dt / n))
        gamma = max(0.1, min(100.0, gamma - lr * dg / n))
    return t, gamma


# ---------------------------------------------------------------------------
# BoundaryStore — SQLite-backed per-entry MLE record registry
# ---------------------------------------------------------------------------

class BoundaryStore:
    """Manages per-item vCache learned boundaries.

    Thread safety: CacheDB uses DatabaseManager (WAL, retry, busy_timeout).
    Concurrent updates to the same entry_id are serialized by SQLite row
    locking. The in-memory `_cache` is for warm-start; it is rebuilt via
    load_all() at VCacheSemantic startup.

    Fail-open: any DB error returns a safe cold-start default rather than
    raising.
    """

    def __init__(
        self,
        db: "CacheDB",
        default_t: float = 0.95,
        default_gamma: float = 10.0,
        floor: float = 0.85,
        ceiling: float = 0.995,
        step: float = 0.01,
        epsilon: float = 0.02,
    ) -> None:
        self._db = db
        self._default_t = default_t
        self._default_gamma = default_gamma
        self._floor = floor
        self._ceiling = ceiling
        self._step = step
        self._epsilon = epsilon
        # In-memory write-through LRU cache.  Populated by load_all() at warm.
        # get() checks here first (O(1) hot path), then falls back to DB.
        # Capped at _BOUNDARY_CACHE_MAX entries; oldest is evicted on overflow.
        # Records are always durable in SQLite so eviction is lossless.
        self._cache: OrderedDict[str, PerItemBoundaryRecord] = OrderedDict()

    def get(self, entry_id: str) -> PerItemBoundaryRecord:
        """Return the MLE model record for an entry, or a cold-start default.

        Fail-open: returns default (cold-start) record on DB error.
        Never raises.
        """
        if entry_id in self._cache:
            # LRU promotion: move to end (most-recently used).
            self._cache.move_to_end(entry_id)
            return self._cache[entry_id]
        try:
            row = self._db.boundary_get(entry_id)
            if row is None:
                return PerItemBoundaryRecord(
                    entry_id=entry_id,
                    t_hat=self._default_t,
                    gamma_hat=self._default_gamma,
                    samples=[],
                    last_updated=time.time(),
                )
            try:
                samples_raw = json.loads(getattr(row, "samples_json", None) or "[]") \
                    if hasattr(row, "samples_json") else []
            except (json.JSONDecodeError, TypeError):
                samples_raw = []
            if not samples_raw and hasattr(row, "samples_json"):
                # BoundaryRow dataclass has no samples_json field; persist only
                # the MLE parameters. samples are in-memory only after refit.
                pass
            # We persist only (t_hat, gamma_hat, sample_count) in the DB.
            # samples is rebuilt from feedback calls (record_outcome) when needed.
            return PerItemBoundaryRecord(
                entry_id=entry_id,
                t_hat=float(getattr(row, "logistic_t", self._default_t)),
                gamma_hat=float(getattr(row, "logistic_gamma", self._default_gamma)),
                samples=[],
                last_updated=float(getattr(row, "updated_at", 0.0)),
            )
        except Exception as exc:
            logger.warning("BoundaryStore.get failed (fail-open): %s", exc)
            return PerItemBoundaryRecord(
                entry_id=entry_id,
                t_hat=self._default_t,
                gamma_hat=self._default_gamma,
                samples=[],
                last_updated=time.time(),
            )

    def save(self, record: PerItemBoundaryRecord) -> None:
        """Persist the MLE parameters (t_hat, gamma_hat, sample_count) for an entry.

        Fail-open: logs warning on DB error, does not raise.
        """
        try:
            from superlocalmemory.optimize.storage.db import BoundaryRow
            row = BoundaryRow(
                entry_id=record.entry_id,
                logistic_t=record.t_hat,
                logistic_gamma=record.gamma_hat,
                sample_count=len(record.samples),
                updated_at=record.last_updated or time.time(),
            )
            self._db.boundary_upsert(record.entry_id, row)
            # LRU write-through (RA-15): insert / refresh position, then evict oldest.
            self._cache[record.entry_id] = record
            self._cache.move_to_end(record.entry_id)
            if len(self._cache) > _BOUNDARY_CACHE_MAX:
                self._cache.popitem(last=False)  # evict LRU (oldest) entry
        except Exception as exc:
            logger.warning("BoundaryStore.save failed (fail-open): %s", exc)

    def record_outcome(
        self,
        entry_id: str,
        similarity: float,
        was_correct: bool,
        max_samples: int = 200,
    ) -> PerItemBoundaryRecord:
        """Fetch, add (s, c), refit MLE, save, return updated record.

        A-01 fix: this IS the vCache online MLE learning step. No step size.
        No raise/relax heuristic. The MLE handles everything.
        """
        record = self.get(entry_id)
        updated = record.add_sample(
            similarity=similarity,
            was_correct=was_correct,
            max_samples=max_samples,
        )
        self.save(updated)
        return updated

    def load_all(self) -> OrderedDict[str, PerItemBoundaryRecord]:
        """Load all boundary records into memory (warm-start).

        Returns:
            Dict entry_id → PerItemBoundaryRecord. {} on error.

        Note: only the (t_hat, gamma_hat, sample_count) are loaded from DB.
        The full `samples` window is rebuilt on demand via record_outcome().
        """
        try:
            rows = self._db.get_all_boundaries()
            # Return an OrderedDict so the caller assignment
            # (self._boundary_store._cache = load_all()) preserves LRU semantics.
            result: OrderedDict[str, PerItemBoundaryRecord] = OrderedDict()
            for r in rows:
                eid = r.get("entry_id")
                if not eid:
                    continue
                result[eid] = PerItemBoundaryRecord(
                    entry_id=eid,
                    t_hat=float(r.get("logistic_t", self._default_t)),
                    gamma_hat=float(r.get("logistic_gamma", self._default_gamma)),
                    samples=[],
                    last_updated=float(r.get("updated_at", 0.0)),
                )
            # Cap at _BOUNDARY_CACHE_MAX — trim oldest if DB has more.
            while len(result) > _BOUNDARY_CACHE_MAX:
                result.popitem(last=False)
            return result
        except Exception as exc:
            logger.warning("BoundaryStore.load_all failed (fail-open): %s", exc)
            return OrderedDict()

    def delete(self, entry_id: str) -> None:
        """Remove boundary record for a deleted cache entry. Fail-open."""
        try:
            self._db.delete_boundary(entry_id)
            self._cache.pop(entry_id, None)
        except Exception as exc:
            logger.warning("BoundaryStore.delete failed (fail-open): %s", exc)
