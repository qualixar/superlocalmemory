# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""
Tests for the query-type-conditioned recency amplifier in RetrievalEngine._build_results.

The amplifier multiplies the existing boosted_score by a factor that depends on:
  - query type: "recency" or "temporal" — other types receive exactly 1.0 (no change)
  - fact age: exponential decay from a type-specific maximum amplitude
  - recency_prior_strength: amplitude scalar in RetrievalConfig (0.0 = no effect)

Tests 1 and 4 are RED before the implementation is in place and GREEN after.
Tests 2 and 3 are invariant (pass both ways) and exist to prevent regressions
on factual queries and on the strength=0.0 identity guarantee.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.retrieval.fusion import FusionResult
from superlocalmemory.retrieval.strategy import QueryStrategy
from superlocalmemory.storage.models import AtomicFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_db() -> MagicMock:
    db = MagicMock()
    db.get_invalidated_fact_ids.return_value = set()
    db.get_nonapplied_correction_successor_ids.return_value = set()
    db.get_strict_temporal_excluded_fact_ids.return_value = set()
    return db


def _engine(recency_prior_strength: float = 0.5) -> RetrievalEngine:
    """Minimal engine — no channels, no embedder, no trust scorer."""
    cfg = RetrievalConfig(recency_prior_strength=recency_prior_strength)
    return RetrievalEngine(db=_mock_db(), config=cfg, channels={})


def _fact(fact_id: str, age_days: float) -> AtomicFact:
    """AtomicFact with a controlled created_at and enough content for quality=1.0."""
    created = datetime.now(UTC) - timedelta(days=age_days)
    return AtomicFact(
        fact_id=fact_id,
        memory_id="m0",
        profile_id="default",
        content="A sufficiently long content string for the quality gate to pass cleanly",
        confidence=0.9,
        access_count=0,
        created_at=created.isoformat(),
    )


def _fused(fact_id: str, score: float) -> FusionResult:
    return FusionResult(fact_id=fact_id, fused_score=score)


def _strat(query_type: str) -> QueryStrategy:
    return QueryStrategy(query_type=query_type, weights={}, confidence=0.8)


def _build(
    engine: RetrievalEngine,
    fused_results: list[FusionResult],
    facts: list[AtomicFact],
    query_type: str,
) -> list:
    fact_map = {f.fact_id: f for f in facts}
    return engine._build_results(fused_results, fact_map, _strat(query_type))


# Ebbinghaus baseline boost as implemented in engine.py (S_base=30, access=0)
def _ebbinghaus(age_days: float) -> float:
    S = 30.0
    return 0.8 + 0.3 * math.exp(-(math.log(2) / S) * age_days)


# Expected cond_boost formula (mirrors the implementation under test)
def _expected_cond(query_type: str, age_days: float, strength: float) -> float:
    if strength <= 0.0 or query_type not in ("recency", "temporal"):
        return 1.0
    half_life = 7.0 if query_type == "recency" else 30.0
    # Both query types use max_amp=1.5. The temporal type previously used 1.2,
    # which caused the prior to be inert for the first ~39 days (both 0d and
    # 30d raw values clamp to 1.2). The cap was raised to 1.5 so decay is visible.
    max_amp = 1.5
    raw = 1.0 + strength * math.exp(-(math.log(2) / half_life) * age_days)
    return min(raw, max_amp)


# ---------------------------------------------------------------------------
# Test 1 — RED before change, GREEN after
# Recency query must promote a fresh fact over a same-topic older fact even
# when the older fact has a higher raw fused_score from the channels.
# ---------------------------------------------------------------------------

class TestRecencyQueryPromotesFreshFact:
    """
    Setup: two facts with identical content quality and trust.
      old_fact  — 7 days old,   fused_score=0.45  (higher raw score)
      fresh_fact — 0 days old,  fused_score=0.40  (lower raw score)

    Without the conditioned amplifier (before change):
      old  → 0.45 * eb(7)  ≈ 0.475,  fresh → 0.40 * eb(0) = 0.44  → old wins

    With the amplifier at strength=0.5 for a "recency" query:
      old  → 0.475 * cond(7)  = 0.475 * 1.25  = 0.594
      fresh → 0.44  * cond(0) = 0.44  * 1.5   = 0.66   → fresh wins
    """

    OLD_FUSED = 0.45
    FRESH_FUSED = 0.40
    OLD_AGE = 7.0    # days
    FRESH_AGE = 0.0  # essentially now (will be tiny float due to timedelta)

    def _run(self, strength: float = 0.5) -> tuple[str, str]:
        """Return (first_fact_id, second_fact_id) from _build_results."""
        engine = _engine(recency_prior_strength=strength)
        old_fact = _fact("old", self.OLD_AGE)
        fresh_fact = _fact("fresh", self.FRESH_AGE)
        fused = [
            _fused("old", self.OLD_FUSED),
            _fused("fresh", self.FRESH_FUSED),
        ]
        results = _build(engine, fused, [old_fact, fresh_fact], "recency")
        # No sort here — asserts the order engine.recall() actually returns.
        # If this fails, _build_results is not sorting by ranking_score.
        return results[0].fact.fact_id, results[1].fact.fact_id

    def test_recency_query_fresh_wins_with_default_strength(self) -> None:
        """Fresh fact must rank first on a recency query. RED before change."""
        first, _ = self._run(strength=0.5)
        assert first == "fresh", (
            f"Expected fresh fact to rank first on recency query, got {first!r}. "
            "The query-type-conditioned amplifier may not be implemented yet."
        )

    def test_old_fact_wins_without_amplifier_on_factual_query(self) -> None:
        """Factual query must NOT reorder facts — old keeps its higher raw score."""
        engine = _engine(recency_prior_strength=0.5)
        old_fact = _fact("old", self.OLD_AGE)
        fresh_fact = _fact("fresh", self.FRESH_AGE)
        fused = [
            _fused("old", self.OLD_FUSED),
            _fused("fresh", self.FRESH_FUSED),
        ]
        results = _build(engine, fused, [old_fact, fresh_fact], "factual")
        # No sort here — asserts the order engine.recall() actually returns.
        first = results[0].fact.fact_id
        assert first == "old", (
            f"Expected old fact to rank first on factual query, got {first!r}. "
            "The amplifier must not affect factual queries."
        )


# ---------------------------------------------------------------------------
# Test 2 — Invariant (passes both before and after)
# A factual query's ordered fact IDs must be identical before and after the
# change is in place. This test should never be broken by the amplifier.
# ---------------------------------------------------------------------------

class TestFactualQueryOrderingUnchanged:
    """
    With three facts at different ages and fused scores, a factual query must
    produce the SAME ordering regardless of whether the amplifier is present
    (strength=0.5) or zeroed out (strength=0.0). The two orderings must be
    byte-identical in fact_id sequence.
    """

    def _ordered_ids(self, query_type: str, strength: float) -> list[str]:
        engine = _engine(recency_prior_strength=strength)
        facts = [
            _fact("f_old",  30.0),
            _fact("f_mid",   7.0),
            _fact("f_new",   0.0),
        ]
        fused = [
            _fused("f_old",  0.60),
            _fused("f_mid",  0.50),
            _fused("f_new",  0.30),
        ]
        results = _build(engine, fused, facts, query_type)
        # No sort here — asserts the order engine.recall() actually returns.
        return [r.fact.fact_id for r in results]

    def test_factual_ordering_identical_at_strength_0_and_05(self) -> None:
        """Ordering under factual query must not change with strength 0.0 vs 0.5."""
        order_00 = self._ordered_ids("factual", strength=0.0)
        order_05 = self._ordered_ids("factual", strength=0.5)
        assert order_00 == order_05, (
            f"Factual ordering changed when strength was turned on: "
            f"strength=0.0 gave {order_00}, strength=0.5 gave {order_05}"
        )

    def test_factual_ordering_identical_for_any_query_type_pair(self) -> None:
        """Factual ordering must equal the reference ordering built at strength=0."""
        reference = self._ordered_ids("factual", strength=0.0)
        with_amplifier = self._ordered_ids("factual", strength=0.5)
        assert reference == with_amplifier


# ---------------------------------------------------------------------------
# Test 3 — Identity guarantee (invariant)
# recency_prior_strength = 0.0 must reproduce the previous ranking exactly.
# Both before and after the change, this is satisfied; the guard ensures it.
# ---------------------------------------------------------------------------

class TestStrengthZeroIdentity:
    """
    With strength=0.0, a recency query must produce the SAME ordering as a
    factual query would produce (no amplification at all). This directly
    proves the previous ranking is reproduced when the knob is off.
    """

    def _run(self, query_type: str, strength: float) -> list[str]:
        engine = _engine(recency_prior_strength=strength)
        facts = [
            _fact("f_old",   7.0),
            _fact("f_fresh",  0.0),
        ]
        fused = [
            _fused("f_old",   0.45),
            _fused("f_fresh", 0.40),
        ]
        results = _build(engine, fused, facts, query_type)
        # No sort here — asserts the order engine.recall() actually returns.
        return [r.fact.fact_id for r in results]

    def test_strength_zero_recency_matches_factual_ordering(self) -> None:
        """
        At strength=0.0, a recency query must order facts identically to a
        factual query (since the amplifier is a no-op). Both must show old first
        because old has a higher raw fused_score.
        """
        recency_order = self._run("recency", strength=0.0)
        factual_order = self._run("factual", strength=0.5)
        assert recency_order == factual_order, (
            f"strength=0.0 recency gave {recency_order}, factual gave {factual_order}. "
            "Setting strength=0.0 must restore the previous ranking."
        )

    def test_strength_zero_gives_old_first(self) -> None:
        """The old fact must rank first when the amplifier is disabled."""
        order = self._run("recency", strength=0.0)
        assert order[0] == "f_old", (
            f"Expected f_old first with strength=0.0, got {order[0]!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Decay validation. RED before change, GREEN after.
# The factor must equal the formula value at known points in time.
# Tolerance: 1 % relative (floating-point rounding only).
# ---------------------------------------------------------------------------

class TestDecayFactors:
    """
    Verify the cond_boost formula at specific ages by extracting the implied
    factor from ranking_score:
        cond_boost = ranking_score / (fused_score * ebbinghaus(age))
    where quality=1.0 (long content) and trust_weight=1.0 (no trust scorer).
    """

    FUSED = 0.5

    def _implied_cond(self, query_type: str, age_days: float) -> float:
        engine = _engine(recency_prior_strength=0.5)
        fact = _fact("f", age_days)
        fused = [_fused("f", self.FUSED)]
        results = _build(engine, fused, [fact], query_type)
        assert results, "Expected at least one result"
        rs = results[0].ranking_score
        base = self.FUSED * _ebbinghaus(age_days)
        return rs / base

    def test_recency_at_age_0_days_hits_cap(self) -> None:
        """At 0 days, recency cond_boost must equal max_amp=1.5 (capped)."""
        factor = self._implied_cond("recency", 0.0)
        expected = 1.5  # min(1.0 + 0.5 * 1.0, 1.5) = 1.5
        assert factor == pytest.approx(expected, rel=0.01), (
            f"recency 0d: got {factor:.4f}, expected {expected:.4f}. "
            "The cond_boost may not be applied yet."
        )

    def test_recency_at_age_1_day_is_below_cap(self) -> None:
        """At 1 day, recency cond_boost must be between 1.4 and 1.5."""
        factor = self._implied_cond("recency", 1.0)
        expected = _expected_cond("recency", 1.0, 0.5)
        assert factor == pytest.approx(expected, rel=0.01)
        # Also assert it is strictly below the cap and above 1.0
        assert 1.0 < factor < 1.5

    def test_recency_at_half_life_7_days(self) -> None:
        """At 7 days (the half-life), recency cond_boost must equal 1.25."""
        factor = self._implied_cond("recency", 7.0)
        expected = 1.25  # 1.0 + 0.5 * exp(-ln2) = 1.0 + 0.5 * 0.5 = 1.25
        assert factor == pytest.approx(expected, rel=0.01), (
            f"recency 7d (half-life): got {factor:.4f}, expected {expected:.4f}"
        )

    def test_recency_at_30_days_strongly_decayed(self) -> None:
        """At 30 days (~4 half-lives), recency cond_boost must be close to 1.0."""
        factor = self._implied_cond("recency", 30.0)
        expected = _expected_cond("recency", 30.0, 0.5)
        assert factor == pytest.approx(expected, rel=0.01)
        # Factor must have decayed significantly from 1.5; should be near 1.0
        assert factor < 1.1, (
            f"recency 30d: got {factor:.4f}, expected near 1.026. Decay not working."
        )

    def test_temporal_at_0_days_hits_cap(self) -> None:
        """At 0 days, temporal cond_boost must equal max_amp=1.5.

        The previous cap was 1.2. With half_life=30 and strength=0.5:
          raw at 0d = 1.0 + 0.5 * 1.0 = 1.5

        Both 0d and 30d (raw=1.25) were clamped to 1.2, making the prior
        produce the same value for the first ~39 days — completely inert.
        The cap was raised to 1.5 so the decay is visible across that range:
        0d → 1.5, 30d → 1.25, older → approaches 1.0.
        This changes ranking: a 2-day-old fact and a 30-day-old fact now
        receive different boosts. This is a correction, not a measured gain.
        """
        factor = self._implied_cond("temporal", 0.0)
        expected = 1.5  # min(1.0 + 0.5 * 1.0, 1.5) = 1.5
        assert factor == pytest.approx(expected, rel=0.01), (
            f"temporal 0d: got {factor:.4f}, expected {expected:.4f}"
        )

    def test_temporal_at_30_days_below_cap(self) -> None:
        """At 30 days (the half-life), temporal cond_boost must equal 1.25.

        With max_amp=1.5, half_life=30, strength=0.5:
          raw at 30d = 1.0 + 0.5 * exp(-ln2) = 1.0 + 0.25 = 1.25
        This is below the 1.5 cap, so the decay is now visible — a 30d fact
        receives less boost than a 0d fact, as intended.
        """
        factor = self._implied_cond("temporal", 30.0)
        expected = 1.25  # 1.0 + 0.5 * 0.5 = 1.25 (no longer capped)
        assert factor == pytest.approx(expected, rel=0.01), (
            f"temporal 30d: got {factor:.4f}, expected {expected:.4f}"
        )

    def test_factual_at_any_age_gives_exactly_1(self) -> None:
        """Factual query must give cond_boost=1.0 at every age (no amplifier)."""
        for age in [0.0, 1.0, 7.0, 30.0]:
            factor = self._implied_cond("factual", age)
            assert factor == pytest.approx(1.0, abs=1e-9), (
                f"factual age={age}d: got {factor:.6f}, expected exactly 1.0"
            )

    def test_other_query_types_give_exactly_1(self) -> None:
        """Entity, multi_hop, and general queries must not be amplified."""
        for qt in ("entity", "multi_hop", "general", "aggregation"):
            factor = self._implied_cond(qt, 0.0)
            assert factor == pytest.approx(1.0, abs=1e-9), (
                f"query_type={qt!r}: got {factor:.6f}, expected exactly 1.0"
            )


# ---------------------------------------------------------------------------
# The time-based prior must discriminate within the first 30 days
# ---------------------------------------------------------------------------

class TestTemporalPriorVaries:
    """The temporal cond_boost must produce different values at different ages.

    The formula is min(1.0 + strength * exp(-ln2/half_life * age), max_amp).
    With max_amp=1.2, half_life=30, strength=0.5:
      at 0 days:  raw = 1.5 → clamped to 1.2
      at 30 days: raw = 1.25 → clamped to 1.2
    Both clamp to the same value; the prior does not discriminate at all
    within the first ~39 days. A question about 2 days ago and a question
    about a month ago receive the same boost, making the prior inert over
    the entire range it was designed for.

    The fix raises max_amp to 1.5 for temporal queries, matching the recency
    setting. This allows the formula to vary from 1.5 (fresh) down through
    1.25 (30 days) toward 1.0 (old), giving the prior its intended behaviour.

    This changes ranking: facts from 2 days ago and 30 days ago now receive
    different boosts. The change is a reasoned correction to a clamp that made
    the prior inert, not a measured performance gain.
    """

    def _implied_cond(self, query_type: str, age_days: float) -> float:
        """Extract the implied cond_boost from the engine's ranking output."""
        engine = _engine(recency_prior_strength=0.5)
        fact = _fact("f", age_days)
        fused = [_fused("f", 0.5)]
        results = _build(engine, fused, [fact], query_type)
        assert results, "Expected at least one result"
        rs = results[0].ranking_score
        base = 0.5 * _ebbinghaus(age_days)
        return rs / base

    def test_temporal_boost_at_0_days_differs_from_30_days(self) -> None:
        """Temporal cond_boost at 0 days must be strictly greater than at 30 days.

        This fails before the fix because both ages clamp to 1.2, making the
        prior identical (and inert) over that entire range.
        """
        boost_0 = self._implied_cond("temporal", 0.0)
        boost_30 = self._implied_cond("temporal", 30.0)
        assert boost_0 > boost_30, (
            f"Temporal prior is identical at 0d ({boost_0:.4f}) and 30d ({boost_30:.4f}). "
            "Both ages are clamped to max_amp=1.2, making the prior inert over "
            "the first ~39 days. Raise max_amp for temporal so the decay is visible."
        )

    def test_temporal_boost_at_0_days_is_above_1_2(self) -> None:
        """At age 0, temporal cond_boost must exceed the old inert cap of 1.2."""
        boost = self._implied_cond("temporal", 0.0)
        assert boost > 1.2, (
            f"Temporal boost at 0 days is {boost:.4f} — equal to the cap. "
            "This means the cap is still suppressing the fresh-fact signal."
        )
