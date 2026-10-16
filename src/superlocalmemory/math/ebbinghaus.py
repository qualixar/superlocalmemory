# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Ebbinghaus forgetting curve — memory strength and retention.

Implements the classic Ebbinghaus retention formula:

    R(t) = e^(-t/S)

where:
    R = retention score in [0, 1]
    t = hours since last access
    S = memory strength (composite of access, importance, confirmations, emotion)

Memory strength formula (ACT-R inspired):

    S = alpha * log(1 + access_count)
      + beta  * importance
      + gamma * confirmation_count
      + delta * emotional_salience

Lifecycle zones are derived from retention score:
    active   (R > 0.8)  -> weight 1.0
    warm     (R > 0.5)  -> weight 0.7
    cold     (R > 0.2)  -> weight 0.3
    archive  (R > 0.05) -> weight 0.0
    forgotten (R <= 0.05) -> weight 0.0

References:
    Ebbinghaus H (1885). Memory: A Contribution to Experimental Psychology.
    Anderson J R & Lebiere C (1998). The Atomic Components of Thought. ACT-R.
    Zhong W et al. (2024). MemoryBank: Enhancing Large Language Models with
        Long-Term Memory. AAAI 2024.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    pass

from superlocalmemory.core.config import ForgettingConfig
from superlocalmemory.core.keep_worthiness import cap_retention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryStrength:
    """Computed strength for a single memory."""

    fact_id: str
    strength: float  # S(m) in [S_MIN, S_MAX]
    access_component: float  # alpha * log(1 + access_count)
    importance_component: float  # beta * importance
    confirmation_component: float  # gamma * confirmation_count
    emotional_component: float  # delta * emotional_salience


class FactRetentionInput(TypedDict):
    """Input dict for batch retention computation. All keys are required."""

    fact_id: str
    access_count: int
    importance: float  # PageRank score from fact_importance
    confirmation_count: int  # Mapped from atomic_facts.evidence_count
    emotional_salience: float  # Mapped from atomic_facts.emotional_valence
    last_accessed_at: str  # ISO 8601 datetime string
    trust_score: float  # Source trust in [0, 1]. Default 1.0.
    #: The memory's text. When present, junk is held at the keep threshold
    #: (``core/keep_worthiness.py``); absent, the score is the curve's alone.
    content: NotRequired[str]


# ---------------------------------------------------------------------------
# Lifecycle zone weights
# ---------------------------------------------------------------------------

_ZONE_WEIGHTS: dict[str, float] = {
    "active": 1.0,
    "warm": 0.7,
    "cold": 0.3,
    "archive": 0.0,
    "forgotten": 0.0,
}


# ---------------------------------------------------------------------------
# EbbinghausCurve
# ---------------------------------------------------------------------------

#: Retention a baseline memory has after ``ARCHIVE_AFTER_DAYS`` of neglect.
#: Anchors the decay CLOCK. Deliberately a constant rather than
#: ``ForgettingConfig.archive_threshold`` — see ``store_scaled_strength``.
_ARCHIVE_REFERENCE_RETENTION: float = 0.2


class EbbinghausCurve:
    """Ebbinghaus forgetting curve with configurable strength formula.

    Provides retention computation, memory strength calculation,
    spaced repetition updates, lifecycle zone classification, and
    batch processing for the forgetting scheduler.
    """

    __slots__ = ("_config",)

    def __init__(self, config: ForgettingConfig) -> None:
        self._config = config

    def retention(self, hours_since_access: float, strength: float) -> float:
        """Compute Ebbinghaus retention R(t) = e^(-t/S).

        Args:
            hours_since_access: Hours since last access. Negative treated as fresh.
            strength: Memory strength S. Floored at min_strength.

        Returns:
            Retention score in [0.0, 1.0]. Never NaN, never Inf.
        """
        # HR-02: Validate negative time
        if hours_since_access < 0:
            return 1.0

        # HR-03: Floor strength
        s = max(self._config.min_strength, strength)

        # Compute decay rate and retention
        lambda_ = 1.0 / s
        r = math.exp(-lambda_ * hours_since_access)

        # NaN/Inf guard (A-MED-02) — MUST come before clamp
        if math.isnan(r) or math.isinf(r):
            logger.warning(
                "retention(): NaN/Inf detected (lambda_=%f, t=%f), returning 0.0",
                lambda_,
                hours_since_access,
            )
            return 0.0

        # HR-02: Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, r))

    def store_scaled_strength(self, strength: float) -> float:
        """Rescale an hours-tuned strength onto the store's own timescale.

        WHY THIS EXISTS. This curve models working memory: ``max_strength`` is
        100, i.e. about four days. Fed real elapsed time from a persistent
        store, a memory a month old computes ``R ~ 0`` and is filed
        ``forgotten``. Measured on a real store, that put 5,546 of 5,561
        memories in ``forgotten`` -- and via the mirror, 93.6% of the store
        showed as archived. GitHub #136.

        The curve's SHAPE is right; only its time constant was wrong. That
        constant is taken from the ladder the product already documents in
        ``core/tier_manager.py``: a baseline memory left alone for
        ``ARCHIVE_AFTER_DAYS`` must have decayed to ``archive_threshold``.
        Nothing new is invented, and this is the ONE place the conversion
        lives -- the seed path and the decay path both call it, so they cannot
        drift apart the way they did in 4.1.15.

        Returns S in HOURS, so every existing caller works unchanged.
        """
        from superlocalmemory.core.tier_manager import ARCHIVE_AFTER_DAYS

        cfg = self._config
        baseline = self.memory_strength(
            access_count=0,
            importance=0.5,
            confirmation_count=0,
            emotional_salience=0.0,
        )
        # The reference retention is FIXED, not read from config. Deriving the
        # clock from the configured ``archive_threshold`` would mean a caller
        # tightening a CLASSIFICATION boundary silently rescaled TIME: a test
        # setting ``archive_threshold=0.999`` to force quick forgetting got
        # ``-log(0.999)`` and a thousand-year time constant, i.e. the exact
        # opposite of what it asked for. Thresholds decide which band a
        # retention lands in; they do not decide how fast retention falls.
        decades = -math.log(_ARCHIVE_REFERENCE_RETENTION)
        baseline_hours = (ARCHIVE_AFTER_DAYS * 24.0) / decades
        scaled = baseline_hours * (max(strength, cfg.min_strength) / max(baseline, 1e-6))
        return min(scaled, baseline_hours * 1000.0)

    def trust_modulated_retention(
        self,
        hours_since_access: float,
        strength: float,
        trust_score: float = 1.0,
    ) -> float:
        """Compute trust-weighted Ebbinghaus retention.

        lambda_eff = lambda * (1 + kappa * (1 - trust))

        Low-trust memories decay faster. When trust=1.0, identical to
        standard retention. When trust=0.0, decay rate is (1+kappa)x faster.

        Paper 3, Section 5.5: Trust-Weighted Forgetting.

        Args:
            hours_since_access: Hours since last access.
            strength: Memory strength S.
            trust_score: Source trust in [0, 1]. Default 1.0 (fully trusted).

        Returns:
            Retention score in [0.0, 1.0].
        """
        if hours_since_access < 0:
            return 1.0

        s = max(self._config.min_strength, strength)
        tau = max(0.0, min(1.0, trust_score))
        kappa = self._config.trust_kappa

        # Trust-modulated decay rate
        lambda_base = 1.0 / s
        lambda_eff = lambda_base * (1.0 + kappa * (1.0 - tau))

        r = math.exp(-lambda_eff * hours_since_access)

        if math.isnan(r) or math.isinf(r):
            return 0.0

        return max(0.0, min(1.0, r))

    def memory_strength(
        self,
        access_count: int,
        importance: float,
        confirmation_count: int,
        emotional_salience: float,
    ) -> float:
        """Compute composite memory strength S(m).

        S = alpha * log(1 + access_count)
          + beta  * importance
          + gamma * confirmation_count
          + delta * emotional_salience

        Args:
            access_count: Number of times memory was accessed.
            importance: PageRank importance score.
            confirmation_count: Number of confirmations/evidence.
            emotional_salience: Emotional strength [-1, 1].

        Returns:
            Clamped strength in [min_strength, max_strength].
        """
        cfg = self._config
        a = cfg.alpha * math.log(1.0 + access_count)
        b = cfg.beta * importance
        c = cfg.gamma * confirmation_count
        d = cfg.delta * emotional_salience
        s = a + b + c + d

        # HR-03: Clamp to bounds
        return max(cfg.min_strength, min(cfg.max_strength, s))

    def compute_strength(
        self,
        fact_id: str,
        access_count: int,
        importance: float,
        confirmation_count: int,
        emotional_salience: float,
    ) -> MemoryStrength:
        """Compute MemoryStrength dataclass for a single fact.

        Returns:
            MemoryStrength with all component scores.
        """
        cfg = self._config
        a = cfg.alpha * math.log(1.0 + access_count)
        b = cfg.beta * importance
        c = cfg.gamma * confirmation_count
        d = cfg.delta * emotional_salience
        s = a + b + c + d
        s = max(cfg.min_strength, min(cfg.max_strength, s))

        return MemoryStrength(
            fact_id=fact_id,
            strength=s,
            access_component=a,
            importance_component=b,
            confirmation_component=c,
            emotional_component=d,
        )

    def spaced_repetition_update(
        self,
        current_strength: float,
        hours_since_last_access: float,
    ) -> float:
        """Boost strength on re-access (spaced repetition effect).

        Longer gaps between accesses produce larger boosts — this is the
        spacing effect from cognitive science.

        HR-07: Only INCREASES strength, never decreases.

        Args:
            current_strength: Current memory strength.
            hours_since_last_access: Hours since last access.

        Returns:
            Updated strength, clamped to [min_strength, max_strength].
        """
        cfg = self._config
        interval = math.log(1.0 + hours_since_last_access / 24.0)
        boost = cfg.learning_rate * interval
        s_new = current_strength + boost

        # HR-03: Clamp
        return max(cfg.min_strength, min(cfg.max_strength, s_new))

    def lifecycle_zone(self, retention_score: float) -> str:
        """Classify retention score into lifecycle zone.

        Args:
            retention_score: R in [0, 1].

        Returns:
            One of: 'active', 'warm', 'cold', 'archive', 'forgotten'.
        """
        if retention_score > 0.8:
            return "active"
        if retention_score > 0.5:
            return "warm"
        if retention_score > self._config.archive_threshold:
            return "cold"
        if retention_score > self._config.forget_threshold:
            return "archive"
        return "forgotten"

    def lifecycle_weight(self, zone: str) -> float:
        """Get retrieval weight for a lifecycle zone.

        Args:
            zone: Lifecycle zone name.

        Returns:
            Weight in [0.0, 1.0].
        """
        return _ZONE_WEIGHTS.get(zone, 0.0)

    def batch_compute_retention(
        self,
        facts: list[FactRetentionInput],
    ) -> list[dict]:
        """Compute trust-weighted decay while retaining learned repetition strength."""
        now = datetime.now(UTC)
        results: list[dict] = []

        for fact in facts:
            access_count = fact["access_count"]
            importance = fact["importance"]
            confirmation_count = fact["confirmation_count"]
            emotional_salience = fact["emotional_salience"]
            last_accessed_at = fact["last_accessed_at"]

            hours_since = _hours_since(last_accessed_at, now)

            strength = self.memory_strength(
                access_count,
                importance,
                confirmation_count,
                emotional_salience,
            )
            strength = max(strength, float(fact.get("memory_strength", 0.0)))
            trust = fact.get("trust_score", 1.0)
            # Scaled to the store's timescale before decay is applied. Without
            # this, anything older than a few days computes R ~ 0 and is filed
            # `forgotten` regardless of how often it is used. GitHub #136.
            ret = self.trust_modulated_retention(
                hours_since,
                self.store_scaled_strength(strength),
                trust,
            )
            if "content" in fact:
                # The curve measures use, not worth: a fresh "2024-01-15 File
                # was modified" computes 1.0. Content decides whether it may.
                ret = cap_retention(ret, fact["content"])
            zone = self.lifecycle_zone(ret)

            results.append(
                {
                    "fact_id": fact["fact_id"],
                    "retention": ret,
                    "strength": strength,
                    "zone": zone,
                    "access_count": access_count,
                    "last_accessed_at": last_accessed_at,
                }
            )

        return results


def _hours_since(value, now):
    try:
        last = datetime.fromisoformat(value)
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        return (now - last).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return 0.0
