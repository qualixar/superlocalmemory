# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Tests for PatternExtractor (Phase F: The Learning Brain).

TDD RED phase: tests written before implementation.
7 tests per LLD Section 6.1.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.config import ParameterizationConfig
from superlocalmemory.parameterization.pattern_extractor import (
    PatternAssertion,
    PatternCategory,
    PatternExtractor,
)


def _make_config(**overrides) -> ParameterizationConfig:
    defaults = {
        "min_confidence": 0.7,
        "min_evidence": 5,
        "cross_project_boost": 1.2,
    }
    defaults.update(overrides)
    return ParameterizationConfig(**defaults)


def _make_extractor(
    db=None,
    behavioral_store=None,
    cross_project=None,
    workflow_miner=None,
    config=None,
):
    return PatternExtractor(
        db=db or MagicMock(),
        behavioral_store=behavioral_store or MagicMock(),
        cross_project=cross_project or MagicMock(),
        workflow_miner=workflow_miner or MagicMock(),
        config=config or _make_config(),
    )


# ---------------------------------------------------------------
# T1: Extract from core memory
# ---------------------------------------------------------------
def test_extract_from_core_memory():
    """Given a user_profile block with identity content and 10 source_fact_ids,
    extract produces PatternAssertion with category=IDENTITY, confidence >= 0.7."""
    db = MagicMock()
    source_ids = json.dumps([f"fact_{i}" for i in range(10)])
    db.execute.return_value = [
        {
            "block_id": "block_1",
            "block_type": "user_profile",
            "content": "- Senior Architect at NovaTech, expertise in AI and cloud",
            "source_fact_ids": source_ids,
        }
    ]
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")

    assert len(patterns) >= 1
    identity_patterns = [p for p in patterns if p.category == PatternCategory.IDENTITY]
    assert len(identity_patterns) >= 1
    assert identity_patterns[0].confidence >= 0.7
    assert identity_patterns[0].source == "core_memory"


# ---------------------------------------------------------------
# T2: Extract from behavioral
# ---------------------------------------------------------------
def test_extract_from_behavioral():
    """Given behavioral pattern entity_pref='typescript' with evidence_count=8,
    confidence=0.8, produces PatternAssertion with category=TECH_PREFERENCE."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = [
        {
            "pattern_id": 1,
            "pattern_type": "entity_pref",
            "pattern_key": "typescript",
            "pattern_value": "typescript",
            "confidence": 0.8,
            "evidence_count": 8,
        }
    ]
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")

    tech_patterns = [p for p in patterns if p.category == PatternCategory.TECH_PREFERENCE]
    assert len(tech_patterns) >= 1
    assert tech_patterns[0].source == "behavioral"


# ---------------------------------------------------------------
# T3: Extract from cross-project with confidence boost
# ---------------------------------------------------------------
def test_extract_from_cross_project():
    """Given cross-project preference language='python' with confidence=0.75,
    extract produces PatternAssertion with cross_project_validated=True
    and boosted confidence (0.75 * 1.2 = 0.9)."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {
        "language": {
            "value": "python",
            "confidence": 0.75,
            "evidence_count": 10,
        }
    }
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")

    cross_patterns = [p for p in patterns if p.cross_project_validated]
    assert len(cross_patterns) >= 1
    assert cross_patterns[0].confidence == pytest.approx(0.9, abs=0.01)
    assert cross_patterns[0].source == "cross_project"


# ---------------------------------------------------------------
# T4: Confidence filter
# ---------------------------------------------------------------
def test_confidence_filter():
    """Given patterns with confidence 0.3, 0.5, 0.8, only the 0.8 pattern
    survives when min_confidence=0.7."""
    db = MagicMock()
    # Return blocks that will produce patterns at different confidences
    db.execute.return_value = [
        {
            "block_id": "b1",
            "block_type": "user_profile",
            "content": "- Senior role in tech company",
            "source_fact_ids": json.dumps(["f1", "f2", "f3"]),  # 3 -> conf=0.3
        },
        {
            "block_id": "b2",
            "block_type": "learned_preferences",
            "content": "- Prefers typescript style code",
            "source_fact_ids": json.dumps([f"f{i}" for i in range(5)]),  # 5 -> conf=0.5
        },
        {
            "block_id": "b3",
            "block_type": "behavioral_patterns",
            "content": "- Uses React framework extensively",
            "source_fact_ids": json.dumps([f"f{i}" for i in range(10)]),  # 10 -> conf=1.0
        },
    ]
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    config = _make_config(min_confidence=0.7)
    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
        config=config,
    )
    patterns = ext.extract("profile_1")

    # Only patterns with confidence >= 0.7 should survive
    for p in patterns:
        assert p.confidence >= 0.7


# ---------------------------------------------------------------
# T5: Deduplication with independence assumption
# ---------------------------------------------------------------
def test_deduplication():
    """Given two patterns with same (category, key) from different sources,
    merge produces single pattern with confidence = 1 - (1-0.7)(1-0.8) = 0.94."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    patterns = [
        PatternAssertion(
            category=PatternCategory.TECH_PREFERENCE,
            key="language",
            value="TypeScript",
            confidence=0.7,
            evidence_count=5,
            source="core_memory",
            source_ids=("b1",),
        ),
        PatternAssertion(
            category=PatternCategory.TECH_PREFERENCE,
            key="language",
            value="TypeScript",
            confidence=0.8,
            evidence_count=8,
            source="behavioral",
            source_ids=("p1",),
        ),
    ]

    deduped = ext._deduplicate(patterns)
    assert len(deduped) == 1
    # 1 - (1-0.7)(1-0.8) = 1 - 0.3*0.2 = 0.94
    assert deduped[0].confidence == pytest.approx(0.94, abs=0.01)
    assert deduped[0].evidence_count == 13


# ---------------------------------------------------------------
# T6: Contradiction detection (temporal ordering)
# ---------------------------------------------------------------
def test_contradiction_detection():
    """Given two patterns for same key 'preferred_framework': value='React' (older)
    and value='Svelte' (newer), only 'Svelte' survives."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    patterns = [
        PatternAssertion(
            category=PatternCategory.TECH_PREFERENCE,
            key="preferred_framework",
            value="React",
            confidence=0.7,
            evidence_count=5,
            source="core_memory",
            created_at="2026-01-01T00:00:00+00:00",
        ),
        PatternAssertion(
            category=PatternCategory.TECH_PREFERENCE,
            key="preferred_framework",
            value="Svelte",
            confidence=0.8,
            evidence_count=5,
            source="behavioral",
            created_at="2026-03-01T00:00:00+00:00",
        ),
    ]

    resolved = ext._check_contradictions(patterns, "profile_1")
    values = [p.value for p in resolved]
    assert "Svelte" in values
    # React should be gone (older)
    assert "React" not in values


# ---------------------------------------------------------------
# T7: Contradiction close confidence (both survive as alternate)
# ---------------------------------------------------------------
def test_contradiction_close_confidence():
    """Given two patterns for same key: value='React' (conf=0.85) and
    value='Svelte' (conf=0.82), both survive with alternate key marking."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    patterns = [
        PatternAssertion(
            category=PatternCategory.TECH_PREFERENCE,
            key="framework",
            value="React",
            confidence=0.85,
            evidence_count=10,
            source="core_memory",
            created_at="2026-03-01T00:00:00+00:00",
        ),
        PatternAssertion(
            category=PatternCategory.TECH_PREFERENCE,
            key="framework",
            value="Svelte",
            confidence=0.82,
            evidence_count=8,
            source="behavioral",
            created_at="2026-03-01T00:00:00+00:00",
        ),
    ]

    resolved = ext._check_contradictions(patterns, "profile_1")
    # Both should survive -- one with _alternate suffix
    assert len(resolved) == 2
    keys = [p.key for p in resolved]
    assert "framework" in keys
    assert "framework_alternate" in keys


# ---------------------------------------------------------------
# Additional: cold start (empty data)
# ---------------------------------------------------------------
def test_extract_cold_start():
    """When all sources return empty, extract returns empty list."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")
    assert patterns == []


def test_config_validation():
    """min_confidence outside [0.3, 1.0] raises ValueError."""
    with pytest.raises(ValueError):
        _make_extractor(config=_make_config(min_confidence=0.1))


def test_config_validation_above_one():
    """min_confidence > 1.0 raises ValueError."""
    with pytest.raises(ValueError):
        _make_extractor(config=_make_config(min_confidence=1.5))


# ---------------------------------------------------------------
# Coverage: BehavioralPattern objects (non-dict path)
# ---------------------------------------------------------------
def test_extract_from_behavioral_object():
    """Behavioral patterns as objects (not dicts) are handled."""
    db = MagicMock()
    db.execute.return_value = []

    bp_obj = MagicMock()
    bp_obj.evidence_count = 8
    bp_obj.pattern_type = "entity_pref"
    bp_obj.pattern_key = "react"
    bp_obj.pattern_value = "react"
    bp_obj.confidence = 0.85
    bp_obj.pattern_id = 42
    # Make isinstance(bp, dict) return False
    type(bp_obj).__instancecheck__ = lambda cls, inst: False

    behavioral = MagicMock()
    behavioral.get_patterns.return_value = [bp_obj]
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")
    # bp_obj is a MagicMock — isinstance(MagicMock(), dict) is False
    # so it takes the else branch
    assert len(patterns) >= 1
    assert patterns[0].source == "behavioral"


def test_extract_from_behavioral_low_evidence():
    """Behavioral patterns with low evidence are filtered out."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = [
        {
            "pattern_id": 1,
            "pattern_type": "entity_pref",
            "pattern_key": "vue",
            "pattern_value": "vue",
            "confidence": 0.9,
            "evidence_count": 2,  # Below min_evidence=5
        }
    ]
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")
    assert len(patterns) == 0


# ---------------------------------------------------------------
# Coverage: workflow extraction
# ---------------------------------------------------------------
def test_extract_from_workflows():
    """Workflow miner patterns with sufficient support are extracted."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = [
        {
            "sequence": ["search", "read", "implement"],
            "support": 0.8,
            "count": 10,
        }
    ]

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")
    wf_patterns = [p for p in patterns if p.category == PatternCategory.WORKFLOW_PATTERN]
    assert len(wf_patterns) >= 1
    assert "search -> read -> implement" in wf_patterns[0].value


def test_extract_from_workflows_low_confidence():
    """Workflow patterns with low support are filtered."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = [
        {
            "sequence": ["search", "read"],
            "support": 0.3,  # Below min_confidence=0.7
            "count": 10,
        }
    ]

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")
    assert len(patterns) == 0


def test_extract_from_workflows_low_count():
    """Workflow patterns with low count are filtered."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = [
        {
            "sequence": ["search", "read"],
            "support": 0.8,
            "count": 2,  # Below min_evidence=5
        }
    ]

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")
    assert len(patterns) == 0


# ---------------------------------------------------------------
# Coverage: malformed source_fact_ids
# ---------------------------------------------------------------
def test_extract_core_memory_malformed_ids():
    """Malformed source_fact_ids JSON is handled gracefully."""
    db = MagicMock()
    db.execute.return_value = [
        {
            "block_id": "b1",
            "block_type": "user_profile",
            "content": "- Senior Architect at a major firm",
            "source_fact_ids": "not json{{{",  # malformed
        }
    ]
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    # Should not raise, evidence=0, confidence=0.0 < 0.7 -> no patterns
    patterns = ext.extract("profile_1")
    assert patterns == []


def test_extract_core_memory_empty_ids():
    """Empty source_fact_ids is handled gracefully."""
    db = MagicMock()
    db.execute.return_value = [
        {
            "block_id": "b1",
            "block_type": "user_profile",
            "content": "- Senior Architect",
            "source_fact_ids": "",  # empty
        }
    ]
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")
    assert patterns == []  # 0 evidence -> confidence 0.0 < 0.7


# ---------------------------------------------------------------
# Coverage: classification heuristics
# ---------------------------------------------------------------
def test_classify_avoidance():
    """Text with 'avoid' keyword is classified as AVOIDANCE."""
    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor
    assert PatternExtractor._classify_text("avoid using jQuery") == PatternCategory.AVOIDANCE


def test_classify_decision():
    """Text with 'decided' keyword is classified as DECISION_HISTORY."""
    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor
    assert PatternExtractor._classify_text("decided to use React") == PatternCategory.DECISION_HISTORY


def test_classify_communication_style():
    """Text with 'prefer' keyword is classified as COMMUNICATION_STYLE."""
    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor
    assert PatternExtractor._classify_text("prefer concise answers") == PatternCategory.COMMUNICATION_STYLE


def test_classify_tech():
    """Text with 'typescript' keyword is classified as TECH_PREFERENCE."""
    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor
    assert PatternExtractor._classify_text("uses typescript daily") == PatternCategory.TECH_PREFERENCE


def test_classify_custom():
    """Unrecognized text defaults to CUSTOM."""
    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor
    assert PatternExtractor._classify_text("some random text") == PatternCategory.CUSTOM


# ---------------------------------------------------------------
# Coverage: contradiction with no timestamps
# ---------------------------------------------------------------
def test_contradiction_no_timestamps():
    """When contradicting patterns have no timestamps, higher confidence wins."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    patterns = [
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="company",
            value="NovaTech",
            confidence=0.9,
            evidence_count=5,
            source="core_memory",
            created_at="",  # no timestamp
        ),
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="company",
            value="Google",
            confidence=0.7,
            evidence_count=3,
            source="behavioral",
            created_at="",  # no timestamp
        ),
    ]

    resolved = ext._check_contradictions(patterns, "profile_1")
    assert len(resolved) == 1
    assert resolved[0].value == "NovaTech"  # higher confidence wins


# ---------------------------------------------------------------
# Coverage: contradiction same timestamp, confidence wins
# ---------------------------------------------------------------
def test_contradiction_same_timestamp():
    """When timestamps are equal, higher confidence wins."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    ts = "2026-03-01T00:00:00+00:00"
    patterns = [
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="company",
            value="NovaTech",
            confidence=0.9,
            evidence_count=5,
            source="core_memory",
            created_at=ts,
        ),
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="company",
            value="Google",
            confidence=0.7,
            evidence_count=3,
            source="behavioral",
            created_at=ts,
        ),
    ]

    resolved = ext._check_contradictions(patterns, "profile_1")
    assert len(resolved) == 1
    assert resolved[0].value == "NovaTech"


# ---------------------------------------------------------------
# Coverage: split_assertions
# ---------------------------------------------------------------
def test_split_assertions_multiple():
    """Split content handles multiple bullet formats."""
    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor
    content = "- First item\n- Second item\n* Third item"
    parts = PatternExtractor._split_assertions(content)
    assert len(parts) >= 3


def test_extract_key():
    """_extract_key extracts first words as key."""
    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor
    key = PatternExtractor._extract_key("Senior Architect at NovaTech")
    assert key == "senior_architect_at_novatech"


def test_extract_key_empty():
    """_extract_key handles empty input."""
    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor
    key = PatternExtractor._extract_key("")
    assert key == "unknown"


def test_extract_from_behavioral_object_low_evidence():
    """BehavioralPattern objects with low evidence are filtered out."""
    db = MagicMock()
    db.execute.return_value = []

    bp_obj = MagicMock()
    bp_obj.evidence_count = 2  # Below min_evidence=5
    bp_obj.pattern_type = "entity_pref"
    bp_obj.pattern_key = "angular"
    bp_obj.pattern_value = "angular"
    bp_obj.confidence = 0.85
    bp_obj.pattern_id = 99

    behavioral = MagicMock()
    behavioral.get_patterns.return_value = [bp_obj]
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )
    patterns = ext.extract("profile_1")
    assert len(patterns) == 0


# ---------------------------------------------------------------
# Coverage: contradiction where lower confidence pattern is in position i
# ---------------------------------------------------------------
def test_contradiction_lower_first():
    """Temporal resolution when lower-confidence pattern is at index 0."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    patterns = [
        PatternAssertion(
            category=PatternCategory.TECH_PREFERENCE,
            key="database",
            value="MySQL",
            confidence=0.7,
            evidence_count=5,
            source="core_memory",
            created_at="2026-03-15T00:00:00+00:00",
        ),
        PatternAssertion(
            category=PatternCategory.TECH_PREFERENCE,
            key="database",
            value="PostgreSQL",
            confidence=0.8,
            evidence_count=8,
            source="behavioral",
            created_at="2026-01-01T00:00:00+00:00",
        ),
    ]

    resolved = ext._check_contradictions(patterns, "profile_1")
    # MySQL is newer, should win
    assert len(resolved) == 1
    assert resolved[0].value == "MySQL"


# ---------------------------------------------------------------
# Coverage: close-confidence with lower in position i (alternate on i)
# ---------------------------------------------------------------
def test_contradiction_three_patterns_skip_removed():
    """With 3 contradicting patterns, once one is removed, pair check skips it."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    patterns = [
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="company",
            value="OldCorp",
            confidence=0.7,
            evidence_count=3,
            source="core_memory",
            created_at="2025-01-01T00:00:00+00:00",
        ),
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="company",
            value="MidCorp",
            confidence=0.75,
            evidence_count=5,
            source="behavioral",
            created_at="2025-06-01T00:00:00+00:00",
        ),
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="company",
            value="NewCorp",
            confidence=0.8,
            evidence_count=8,
            source="cross_project",
            created_at="2026-01-01T00:00:00+00:00",
        ),
    ]

    resolved = ext._check_contradictions(patterns, "profile_1")
    # NewCorp is newest, should be only survivor
    values = [p.value for p in resolved]
    assert "NewCorp" in values


def test_contradiction_same_value_not_contradiction():
    """Patterns with same key and same value are not contradictions."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    patterns = [
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="company",
            value="NovaTech",
            confidence=0.7,
            evidence_count=3,
            source="core_memory",
        ),
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="company",
            value="NovaTech",
            confidence=0.8,
            evidence_count=5,
            source="behavioral",
        ),
    ]

    resolved = ext._check_contradictions(patterns, "profile_1")
    # Same value -> no contradiction -> both survive
    assert len(resolved) == 2


def test_contradiction_same_ts_lower_first():
    """Same timestamp, lower-confidence in position i: confidence wins, i removed."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    ts = "2026-03-01T00:00:00+00:00"
    patterns = [
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="role",
            value="Junior Dev",
            confidence=0.7,  # Lower
            evidence_count=3,
            source="core_memory",
            created_at=ts,
        ),
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="role",
            value="Senior Architect",
            confidence=0.9,  # Higher
            evidence_count=10,
            source="behavioral",
            created_at=ts,
        ),
    ]

    resolved = ext._check_contradictions(patterns, "profile_1")
    assert len(resolved) == 1
    assert resolved[0].value == "Senior Architect"


def test_contradiction_no_ts_lower_first():
    """No timestamps, lower-confidence in position i: higher confidence wins."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    patterns = [
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="role",
            value="Junior Dev",
            confidence=0.7,  # Lower, position i
            evidence_count=3,
            source="core_memory",
        ),
        PatternAssertion(
            category=PatternCategory.IDENTITY,
            key="role",
            value="Senior Architect",
            confidence=0.9,  # Higher, position j
            evidence_count=10,
            source="behavioral",
        ),
    ]

    resolved = ext._check_contradictions(patterns, "profile_1")
    assert len(resolved) == 1
    assert resolved[0].value == "Senior Architect"


def test_contradiction_close_confidence_lower_first():
    """When lower-confidence pattern is at index i, it gets _alternate suffix."""
    db = MagicMock()
    db.execute.return_value = []
    behavioral = MagicMock()
    behavioral.get_patterns.return_value = []
    cross_proj = MagicMock()
    cross_proj.get_preferences.return_value = {}
    workflow = MagicMock()
    workflow.mine.return_value = []

    ext = _make_extractor(
        db=db,
        behavioral_store=behavioral,
        cross_project=cross_proj,
        workflow_miner=workflow,
    )

    patterns = [
        PatternAssertion(
            category=PatternCategory.TECH_PREFERENCE,
            key="framework",
            value="Angular",
            confidence=0.82,  # Lower
            evidence_count=8,
            source="behavioral",
            created_at="2026-03-01T00:00:00+00:00",
        ),
        PatternAssertion(
            category=PatternCategory.TECH_PREFERENCE,
            key="framework",
            value="Vue",
            confidence=0.85,  # Higher
            evidence_count=10,
            source="core_memory",
            created_at="2026-03-01T00:00:00+00:00",
        ),
    ]

    resolved = ext._check_contradictions(patterns, "profile_1")
    assert len(resolved) == 2
    keys = [p.key for p in resolved]
    assert "framework_alternate" in keys
    assert "framework" in keys
