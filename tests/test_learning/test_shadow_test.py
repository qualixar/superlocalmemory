# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — Track A.3 (LLD-10 / LLD-00 §8)

"""Tests for ``superlocalmemory.learning.shadow_test`` (ShadowTest).

Contract references:
  - LLD-00 §8   — two-phase shadow test (Phase A n=100 early-stop,
                 Phase B n=885 full validation; α=0.05 paired t-test;
                 promotion criterion: MRR lift ≥ +0.02 with p<0.05).
  - LLD-10 §4.1 — deterministic A/B routing via SHA-256(query_id)
                 first 8 hex chars → 2-bucket modulo split.
  - LLD-10 §4.5 — promotion decision (paired t-test, fallback to
                 hardcoded t>2.0 if scipy absent).
  - IMPLEMENTATION-MANIFEST v3.4.21 FINAL A.3 — 5 test names verbatim.

Stdlib-only tests; no scipy/lightgbm required. The ShadowTest class is
pure bookkeeping + paired-t math — it does not touch the DB or any
learning artifacts. Tests exercise the state machine (Phase A→B→decide)
and the deterministic routing invariant.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable

import pytest

from superlocalmemory.learning.shadow_test import ShadowTest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_pairs(
    st: ShadowTest,
    *,
    pairs: Iterable[tuple[float, float]],
    start_query_num: int = 0,
) -> int:
    """Record ``(active_score, candidate_score)`` pairs through ``st``.

    S9-STAT-08: each paired observation MUST use the same query_id on
    both arms. The production router enforces one-arm-per-query, so
    this helper simulates the paired-scoring counter-factual by writing
    the SAME query_id to both arms. The route-exclusivity verifier
    (H-P-12) is bypassed in test mode by writing the candidate arm
    first (in the live system each qid only ever lands on one arm).
    To keep the bypass explicit, the helper uses two distinct qid
    spaces per sample but records them both through the ShadowTest
    internals directly, ordered so the candidate write never triggers
    the exclusivity guard.

    Returns number of pairs recorded.
    """
    count = 0
    for i, (active, candidate) in enumerate(pairs):
        qid = f"q-{start_query_num + i}"
        # Write both arms for the SAME qid by temporarily disabling
        # the exclusivity guard — paired test data is the one
        # legitimate case where both arms see identical qids.
        st._active.append(float(active))
        st._active_by_qid[qid] = float(active)
        st._candidate.append(float(candidate))
        st._candidate_by_qid[qid] = float(candidate)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Manifest-locked tests
# ---------------------------------------------------------------------------


def test_shadow_phase_a_early_stop_on_strong_signal() -> None:
    """Phase A (n=100) early-stops when |effect| > 0.08 AND p<0.01.

    Seed a strong, low-variance lift; after Phase A's 100 pairs the
    decision should be ``promote`` without waiting for Phase B.
    """
    st = ShadowTest(profile_id="p", candidate_model_id="cand-1")
    rng = random.Random(0xA3)
    pairs: list[tuple[float, float]] = []
    for _ in range(st.PHASE_A_N):
        a = 0.40 + rng.gauss(0, 0.01)
        c = a + 0.12 + rng.gauss(0, 0.01)
        pairs.append((a, c))
    _record_pairs(st, pairs=pairs)
    decision, stats = st.decide()
    assert decision == "promote", f"expected promote, got {decision} ({stats})"
    assert stats.get("phase") == "A"
    assert abs(stats.get("effect", 0.0)) > 0.08


def test_shadow_phase_b_requires_885_pairs() -> None:
    """When Phase A is inconclusive but Phase B criterion met, promote.

    Set up a marginal effect (~+0.025) that does NOT trip Phase A's
    0.08 early-stop. Provide 885 paired recalls. Phase B must
    conclude promote.
    """
    st = ShadowTest(profile_id="p", candidate_model_id="cand-2")
    rng = random.Random(0xB5)
    pairs: list[tuple[float, float]] = []
    for _ in range(st.PHASE_B_N):
        a = 0.45 + rng.gauss(0, 0.15)
        c = a + 0.025 + rng.gauss(0, 0.015)
        pairs.append((a, c))
    _record_pairs(st, pairs=pairs)
    decision, stats = st.decide()
    assert decision == "promote", f"expected promote, got {decision} ({stats})"
    assert stats.get("phase") == "B"
    assert stats.get("n_pairs", 0) >= st.PHASE_B_N


def test_shadow_rejects_no_improvement() -> None:
    """Zero-effect (candidate == active) → reject after Phase B."""
    st = ShadowTest(profile_id="p", candidate_model_id="cand-3")
    rng = random.Random(0xCC)
    pairs: list[tuple[float, float]] = []
    for _ in range(st.PHASE_B_N):
        base = 0.5 + rng.gauss(0, 0.1)
        pairs.append((base, base + rng.gauss(0, 0.001)))
    _record_pairs(st, pairs=pairs)
    decision, stats = st.decide()
    assert decision == "reject", f"expected reject, got {decision} ({stats})"


def test_shadow_continues_when_inconclusive() -> None:
    """Below PHASE_B_N with marginal effect → decide returns 'continue'."""
    st = ShadowTest(profile_id="p", candidate_model_id="cand-4")
    rng = random.Random(0xEE)
    # Only Phase A (100) pairs with noisy marginal lift — not enough
    # for early-stop, not enough pairs for Phase B.
    pairs: list[tuple[float, float]] = []
    for _ in range(st.PHASE_A_N):
        a = 0.5 + rng.gauss(0, 0.2)
        c = a + 0.02 + rng.gauss(0, 0.2)
        pairs.append((a, c))
    _record_pairs(st, pairs=pairs)
    decision, stats = st.decide()
    assert decision == "continue", f"expected continue, got {decision} ({stats})"


def test_shadow_deterministic_ab_route_by_query_hash() -> None:
    """ShadowTest.route_query deterministically buckets via SHA-256.

    Same query_id MUST always bucket the same way. The split is defined
    by LLD-10 §4.1: ``int(sha256(qid)[:8], 16) % 2`` → 0='active',
    1='candidate'.
    """
    st = ShadowTest(profile_id="p", candidate_model_id="cand-5")

    def _expected(qid: str) -> str:
        h = hashlib.sha256(qid.encode("utf-8")).hexdigest()[:8]
        return "candidate" if int(h, 16) % 2 == 1 else "active"

    for qid in ("query-1", "query-2", "abc", "01HE...ULID", "x" * 40):
        a1 = st.route_query(qid)
        a2 = st.route_query(qid)
        assert a1 == a2, "route must be deterministic per query_id"
        assert a1 == _expected(qid), (
            f"route for {qid} ({a1}) must match SHA-256 bucket "
            f"({_expected(qid)})"
        )

    # Roughly balanced over a large sample (sanity — not a strict bound).
    counts = {"active": 0, "candidate": 0}
    for i in range(1000):
        counts[st.route_query(f"q-{i}")] += 1
    assert 400 <= counts["active"] <= 600
    assert 400 <= counts["candidate"] <= 600


# ---------------------------------------------------------------------------
# Stage 8 F4.B — H-02 Phase A t-table correctness (skeptic-H01 fix)
# ---------------------------------------------------------------------------
#
# _critical_t(df, alpha) previously returned the critical-t at the next
# tabled df >= requested df, which UNDER-shoots the true value when the
# requested df falls between rows (e.g. df=99 returned the df=120 value
# 2.617 instead of the true ~2.626). That makes Phase A early-stop more
# permissive than advertised (promotes on weaker evidence). The fix
# returns a value within ±0.01 of the scipy reference for all df values
# Phase A actually uses (99 in production; 9/19/49 for defense-in-depth
# below Phase A threshold — hit when external callers reuse the helper).
#
# Tests compare against scipy.stats.t.ppf where scipy is available, else
# against hand-computed reference values.


def _scipy_ref_critical_t(df: int, alpha: float) -> float:
    """Return the two-tailed critical-t from scipy; hand-table fallback."""
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(1.0 - alpha / 2.0, df))
    except Exception:  # pragma: no cover — scipy always present in CI
        # Hand table — only used when scipy is absent. Values verified
        # against Abramowitz & Stegun / scipy at module import time.
        _HAND = {
            (9, 0.05): 2.262, (9, 0.01): 3.250,
            (19, 0.05): 2.093, (19, 0.01): 2.861,
            (49, 0.05): 2.010, (49, 0.01): 2.680,
            (99, 0.05): 1.984, (99, 0.01): 2.626,
        }
        return _HAND[(df, alpha)]


@pytest.mark.parametrize(
    "df,alpha",
    [
        (9, 0.05), (9, 0.01),
        (19, 0.05), (19, 0.01),
        (49, 0.05), (49, 0.01),
        (99, 0.05), (99, 0.01),
    ],
)
def test_critical_t_matches_scipy_reference(df: int, alpha: float) -> None:
    """Stage 8 H-02: _critical_t must match scipy within ±0.01.

    Previously _critical_t(99, 0.01) returned 2.617 (the df=120 row).
    True value per scipy is 2.626 — current code was 0.009 too low,
    making Phase A strong-signal early-stop more permissive than the
    contract says (α=0.01 two-tailed). Fix tightens the lookup so ALL
    table gaps return within ±0.01 of the scipy reference.
    """
    from superlocalmemory.learning.shadow_test import _critical_t

    got = _critical_t(df, alpha=alpha)
    want = _scipy_ref_critical_t(df, alpha)
    assert abs(got - want) <= 0.01, (
        f"_critical_t(df={df}, alpha={alpha}) = {got:.4f}, "
        f"scipy reference = {want:.4f}, diff = {abs(got - want):.4f} "
        f"(must be ≤ 0.01)"
    )


def test_critical_t_phase_a_production_df_99() -> None:
    """Phase A strong-signal early-stop uses df=99 (n=100 pairs).

    The production value MUST be within ±0.01 of scipy's 2.626. This is
    the single most important df for H-02 because it's the only value
    _critical_t(_, 0.01) is actually called with in Phase A production.
    """
    from superlocalmemory.learning.shadow_test import _critical_t

    got = _critical_t(99, alpha=0.01)
    assert 2.616 <= got <= 2.636, (
        f"_critical_t(99, 0.01) = {got:.4f} — out of ±0.01 band around "
        f"scipy 2.626. Phase A strong-signal early-stop is mis-calibrated."
    )


def test_critical_t_does_not_become_trivial_for_tiny_df() -> None:
    """Sanity: critical-t must stay large for very small df (no off-by-one
    that collapses df=1 to the df=10k row). Regression guard.
    """
    from superlocalmemory.learning.shadow_test import _critical_t

    # df=1, α=0.05: scipy 12.706. Must be > 10 (nowhere near 1.96).
    assert _critical_t(1, alpha=0.05) > 10.0
    # df=1, α=0.01: scipy 63.657. Must be > 50.
    assert _critical_t(1, alpha=0.01) > 50.0


# ---------------------------------------------------------------------------
# Phase A small-n boundary — early-stop fires on true-strong signal at
# the minimum Phase A sample size (100), NOT below it.
# ---------------------------------------------------------------------------


def test_phase_a_does_not_early_stop_below_phase_a_n() -> None:
    """Phase A requires n_pairs ≥ PHASE_A_N (100). At n=50 (well below),
    even a huge lift returns 'continue' — accumulating more data, not
    firing on noise.
    """
    st = ShadowTest(profile_id="p", candidate_model_id="cand-small-n")
    # Strong signal but only 50 pairs — below Phase A threshold.
    rng = random.Random(0xA7)
    pairs = []
    for _ in range(50):
        a = 0.4 + rng.gauss(0, 0.01)
        c = a + 0.12 + rng.gauss(0, 0.01)
        pairs.append((a, c))
    _record_pairs(st, pairs=pairs)
    decision, stats = st.decide()
    # Below PHASE_A_N must return 'continue', not 'promote'.
    assert decision == "continue"
    assert stats["phase"] == "A"
