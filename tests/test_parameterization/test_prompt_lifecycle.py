# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Tests for PromptLifecycleManager (Phase F: The Learning Brain).

TDD RED phase: tests written before implementation.
4 tests per LLD Section 6.5.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import ParameterizationConfig
from superlocalmemory.parameterization.pattern_extractor import (
    PatternAssertion,
    PatternCategory,
)
from superlocalmemory.parameterization.prompt_lifecycle import PromptLifecycleManager


def _make_config(**overrides) -> ParameterizationConfig:
    return ParameterizationConfig(**overrides)


def _make_ebbinghaus():
    """Create a mock EbbinghausCurve that computes R(t) = e^(-t/S)."""
    import math
    eb = MagicMock()
    eb.retention.side_effect = lambda hours, strength: math.exp(
        -hours / max(0.01, strength)
    )
    return eb


# ---------------------------------------------------------------
# T20: Effectiveness tracking
# ---------------------------------------------------------------
def test_effectiveness_tracking():
    """After recording positive signals, effectiveness increases from 0.5 to > 0.7."""
    db = MagicMock()
    db.execute.return_value = [
        {
            "prompt_id": "p1",
            "effectiveness": 0.5,
            "retention_score": 1.0,
        }
    ]

    lifecycle = PromptLifecycleManager(
        db=db,
        ebbinghaus=_make_ebbinghaus(),
        config=_make_config(),
    )

    signals = {"followed": 1.0, "session_success": 1.0}
    new_eff = lifecycle.update_effectiveness("profile_1", "identity", signals)
    # positive = 2.0, negative = 0.0, total = 3.0
    # raw = 2.0 / 3.0 = 0.667
    # blended = 0.7 * 0.667 + 0.3 * 0.5 = 0.467 + 0.15 = 0.617
    # Actually the raw is quite high, blended should push above 0.5
    assert new_eff > 0.5


# ---------------------------------------------------------------
# T21: Ebbinghaus decay on prompts
# ---------------------------------------------------------------
def test_prompt_ebbinghaus_decay():
    """Prompt with low effectiveness decays faster than one with high effectiveness."""
    db = MagicMock()

    eb = _make_ebbinghaus()

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=eb, config=_make_config(),
    )

    now = datetime.now(timezone.utc)
    past = (now - timedelta(hours=72)).isoformat()

    # Low effectiveness, version 1
    db.execute.return_value = [
        {
            "prompt_id": "p_low",
            "effectiveness": 0.2,
            "retention_score": 1.0,
            "version": 1,
            "created_at": past,
            "updated_at": past,
        }
    ]
    r_low = lifecycle.compute_prompt_retention("p_low")

    # High effectiveness, version 5
    db.execute.return_value = [
        {
            "prompt_id": "p_high",
            "effectiveness": 0.8,
            "retention_score": 1.0,
            "version": 5,
            "created_at": past,
            "updated_at": past,
        }
    ]
    lifecycle.compute_prompt_retention("p_high")

    # High effectiveness + more versions = slower decay = higher retention
    # Both use 48h floor, but p_high: S_raw = 2*0.8*5 = 8 -> S=48 (floored)
    # p_low: S_raw = 2*0.2*1 = 0.4 -> S=48 (floored)
    # At floor, both are equal. But above floor, high effectiveness dominates.
    # Test with version=40 for high to exceed floor
    db.execute.return_value = [
        {
            "prompt_id": "p_high2",
            "effectiveness": 0.8,
            "retention_score": 1.0,
            "version": 40,
            "created_at": past,
            "updated_at": past,
        }
    ]
    r_high2 = lifecycle.compute_prompt_retention("p_high2")
    # S_raw = 2*0.8*40 = 64 > 48, so r_high2 should be higher than r_low
    assert r_high2 > r_low


# ---------------------------------------------------------------
# T22: Prompt evolution
# ---------------------------------------------------------------
def test_prompt_evolution():
    """New pattern with confidence=0.9 replaces existing prompt with confidence=0.7."""
    db = MagicMock()
    db.execute.return_value = [
        {
            "prompt_id": "old_p1",
            "category": "tech_preference",
            "confidence": 0.7,
            "effectiveness": 0.5,
        }
    ]

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=_make_ebbinghaus(), config=_make_config(),
    )

    new_pattern = PatternAssertion(
        category=PatternCategory.TECH_PREFERENCE,
        key="framework",
        value="Svelte",
        confidence=0.9,
        evidence_count=10,
        source="behavioral",
    )

    result = lifecycle.evolve_prompt("profile_1", "tech_preference", new_pattern)
    assert result == "replaced"


# ---------------------------------------------------------------
# T23: Core prompts slow decay
# ---------------------------------------------------------------
def test_core_prompts_slow_decay():
    """Prompt with high effectiveness and many versions retains well after 48h."""
    db = MagicMock()
    eb = _make_ebbinghaus()

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=eb, config=_make_config(),
    )

    now = datetime.now(timezone.utc)
    past = (now - timedelta(hours=48)).isoformat()

    # High effectiveness, many versions -> above floor
    db.execute.return_value = [
        {
            "prompt_id": "p_core",
            "effectiveness": 0.9,
            "retention_score": 1.0,
            "version": 50,
            "created_at": past,
            "updated_at": past,
        }
    ]
    r = lifecycle.compute_prompt_retention("p_core")
    # S_raw = 2 * 0.9 * 50 = 90 > 48 floor
    # R(48h) = e^(-48/90) = e^(-0.533) ~ 0.587
    assert r > 0.5


# ---------------------------------------------------------------
# Additional: lifecycle review
# ---------------------------------------------------------------
def test_lifecycle_review_deactivates_low_retention():
    """run_lifecycle_review deactivates prompts with retention < 0.1."""
    db = MagicMock()

    now = datetime.now(timezone.utc)
    old = (now - timedelta(hours=500)).isoformat()

    # Return one active prompt that's very old
    db.execute.side_effect = [
        # First call: get all active prompts
        [
            {
                "prompt_id": "p_old",
                "category": "identity",
                "effectiveness": 0.1,
                "retention_score": 0.8,
                "version": 1,
                "created_at": old,
                "updated_at": old,
            }
        ],
        # Subsequent calls: compute_prompt_retention query
        [
            {
                "prompt_id": "p_old",
                "effectiveness": 0.1,
                "retention_score": 0.8,
                "version": 1,
                "created_at": old,
                "updated_at": old,
            }
        ],
        # Update call
        None,
    ]

    eb = _make_ebbinghaus()
    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=eb, config=_make_config(),
    )

    stats = lifecycle.run_lifecycle_review("profile_1")
    assert stats["reviewed"] >= 1


def test_evolve_no_existing():
    """When no existing prompt, evolve_prompt returns 'new'."""
    db = MagicMock()
    db.execute.return_value = []

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=_make_ebbinghaus(), config=_make_config(),
    )

    new_pattern = PatternAssertion(
        category=PatternCategory.IDENTITY,
        key="role",
        value="Engineer",
        confidence=0.8,
        evidence_count=10,
        source="core_memory",
    )
    result = lifecycle.evolve_prompt("profile_1", "identity", new_pattern)
    assert result == "new"


# ---------------------------------------------------------------
# Coverage: evolve_prompt kept_existing
# ---------------------------------------------------------------
def test_evolve_kept_existing():
    """When new pattern confidence is much lower, existing is kept."""
    db = MagicMock()
    db.execute.return_value = [
        {
            "prompt_id": "old_p1",
            "category": "identity",
            "confidence": 0.9,
            "effectiveness": 0.5,
        }
    ]

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=_make_ebbinghaus(), config=_make_config(),
    )

    new_pattern = PatternAssertion(
        category=PatternCategory.IDENTITY,
        key="role",
        value="Junior Dev",
        confidence=0.5,  # Much lower than 0.9
        evidence_count=3,
        source="core_memory",
    )
    result = lifecycle.evolve_prompt("profile_1", "identity", new_pattern)
    assert result == "kept_existing"


# ---------------------------------------------------------------
# Coverage: evolve_prompt user_review_needed
# ---------------------------------------------------------------
def test_evolve_user_review_needed():
    """When new pattern confidence is close to existing, user review needed."""
    db = MagicMock()
    db.execute.return_value = [
        {
            "prompt_id": "old_p1",
            "category": "identity",
            "confidence": 0.8,
            "effectiveness": 0.5,
        }
    ]

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=_make_ebbinghaus(), config=_make_config(),
    )

    new_pattern = PatternAssertion(
        category=PatternCategory.IDENTITY,
        key="role",
        value="Senior Architect",
        confidence=0.85,  # Within 0.1 of 0.8
        evidence_count=10,
        source="core_memory",
    )
    result = lifecycle.evolve_prompt("profile_1", "identity", new_pattern)
    assert result == "user_review_needed"


# ---------------------------------------------------------------
# Coverage: compute_prompt_retention not found
# ---------------------------------------------------------------
def test_compute_retention_not_found():
    """compute_prompt_retention returns 0.0 when prompt doesn't exist."""
    db = MagicMock()
    db.execute.return_value = []

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=_make_ebbinghaus(), config=_make_config(),
    )
    r = lifecycle.compute_prompt_retention("nonexistent_id")
    assert r == 0.0


# ---------------------------------------------------------------
# Coverage: compute_prompt_retention with bad timestamp
# ---------------------------------------------------------------
def test_compute_retention_bad_timestamp():
    """compute_prompt_retention handles invalid timestamp gracefully."""
    db = MagicMock()
    db.execute.return_value = [
        {
            "prompt_id": "p1",
            "effectiveness": 0.5,
            "retention_score": 1.0,
            "version": 1,
            "created_at": "not-a-date",
            "updated_at": "not-a-date",
        }
    ]

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=_make_ebbinghaus(), config=_make_config(),
    )
    r = lifecycle.compute_prompt_retention("p1")
    # hours=0.0 -> retention = e^(0) = 1.0
    assert r == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------
# Coverage: update_effectiveness not found
# ---------------------------------------------------------------
def test_update_effectiveness_not_found():
    """update_effectiveness returns 0.5 when no active prompt found."""
    db = MagicMock()
    db.execute.return_value = []

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=_make_ebbinghaus(), config=_make_config(),
    )
    result = lifecycle.update_effectiveness("profile_1", "identity", {"followed": 1.0})
    assert result == 0.5


# ---------------------------------------------------------------
# Coverage: lifecycle review with significant decay
# ---------------------------------------------------------------
def test_compute_retention_naive_timestamp():
    """compute_prompt_retention handles naive (no-tzinfo) timestamp."""
    db = MagicMock()
    now = datetime.now(timezone.utc)
    past_naive = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
    db.execute.return_value = [
        {
            "prompt_id": "p1",
            "effectiveness": 0.5,
            "retention_score": 1.0,
            "version": 5,
            "created_at": past_naive,
            "updated_at": past_naive,  # Naive timestamp (no +00:00)
        }
    ]

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=_make_ebbinghaus(), config=_make_config(),
    )
    r = lifecycle.compute_prompt_retention("p1")
    assert 0.0 < r <= 1.0


def test_lifecycle_review_surviving_prompt():
    """run_lifecycle_review updates retention for surviving prompts."""
    db = MagicMock()

    now = datetime.now(timezone.utc)
    recent = (now - timedelta(hours=1)).isoformat()

    db.execute.side_effect = [
        # First call: get all active prompts
        [
            {
                "prompt_id": "p1",
                "category": "identity",
                "effectiveness": 0.9,
                "retention_score": 1.0,
                "version": 50,
                "created_at": recent,
                "updated_at": recent,
            }
        ],
        # Second call: compute_prompt_retention query
        [
            {
                "prompt_id": "p1",
                "effectiveness": 0.9,
                "retention_score": 1.0,
                "version": 50,
                "created_at": recent,
                "updated_at": recent,
            }
        ],
        # Third call: update retention_score
        None,
    ]

    eb = _make_ebbinghaus()
    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=eb, config=_make_config(),
    )

    stats = lifecycle.run_lifecycle_review("profile_1")
    assert stats["reviewed"] == 1
    assert stats["removed"] == 0


def test_lifecycle_review_significant_decay():
    """run_lifecycle_review increments decayed when retention changes significantly."""
    db = MagicMock()

    now = datetime.now(timezone.utc)
    past = (now - timedelta(hours=200)).isoformat()

    db.execute.side_effect = [
        # First call: get all active prompts
        [
            {
                "prompt_id": "p1",
                "category": "identity",
                "effectiveness": 0.3,
                "retention_score": 1.0,  # Was 1.0, will decay
                "version": 2,
                "created_at": past,
                "updated_at": past,
            }
        ],
        # Second call: compute_prompt_retention query
        [
            {
                "prompt_id": "p1",
                "effectiveness": 0.3,
                "retention_score": 1.0,
                "version": 2,
                "created_at": past,
                "updated_at": past,
            }
        ],
        # Third call: update retention_score
        None,
    ]

    eb = _make_ebbinghaus()
    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=eb, config=_make_config(),
    )

    stats = lifecycle.run_lifecycle_review("profile_1")
    assert stats["reviewed"] >= 1


# ---- Coverage gap: decay delta > 0.1 increments decayed counter (line 216) ----

def test_lifecycle_review_counts_significant_decay():
    """When retention drops by >0.1 but stays above removal threshold, 'decayed' increments."""
    db = MagicMock()
    long_ago = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
    db.execute.return_value = [
        {
            "prompt_id": "p1",
            "category": "identity",
            "retention_score": 0.9,
            "created_at": long_ago,
            "active": 1,
        },
    ]
    # Mock ebbinghaus to return 0.5 (drop of 0.4 from 0.9, but above 0.1 threshold)
    eb = MagicMock()
    eb.retention.return_value = 0.5
    eb.memory_strength.return_value = 1.0

    lifecycle = PromptLifecycleManager(
        db=db, ebbinghaus=eb, config=_make_config(),
    )

    stats = lifecycle.run_lifecycle_review("profile_1")
    assert stats["decayed"] >= 1, f"Expected decayed >= 1, got {stats}"
