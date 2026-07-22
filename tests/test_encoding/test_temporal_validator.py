# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for temporal intelligence -- 4-timestamp bi-temporal model + contradiction.

Covers:
  - Temporal validity CRUD (store, get, invalidate)
  - Bi-temporal integrity (BOTH valid_until + system_expired_at set atomically)
  - Idempotent double invalidation
  - is_temporally_valid() with correct signature
  - Event-time and transaction-time queries
  - Contradiction detection (sheaf Mode A)
  - Trust penalty application
  - Validate-and-invalidate end-to-end
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import TemporalValidatorConfig
from superlocalmemory.encoding.temporal_validator import TemporalValidator
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """DatabaseManager wired to a temp directory with full schema."""
    db_path = tmp_path / "test_temporal.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def _seed_profile(db: DatabaseManager) -> None:
    """Ensure default profile exists."""
    # Schema init creates 'default' profile.


@pytest.fixture()
def _seed_facts(db: DatabaseManager) -> tuple[str, str, str]:
    """Seed 3 facts with a parent memory for FK satisfaction."""
    record = MemoryRecord(
        profile_id="default", content="test content", session_id="s1",
    )
    db.store_memory(record)

    facts = []
    for i in range(3):
        fact = AtomicFact(
            profile_id="default",
            memory_id=record.memory_id,
            content=f"Fact number {i}",
        )
        db.store_fact(fact)
        facts.append(fact.fact_id)
    return tuple(facts)  # type: ignore[return-value]


@pytest.fixture()
def config() -> TemporalValidatorConfig:
    return TemporalValidatorConfig(enabled=True, mode="a")


@pytest.fixture()
def validator(db: DatabaseManager, config: TemporalValidatorConfig) -> TemporalValidator:
    return TemporalValidator(db=db, config=config)


# ---------------------------------------------------------------------------
# TestTemporalValidity -- 4-timestamp bi-temporal model
# ---------------------------------------------------------------------------

class TestTemporalValidity:
    """Tests for the 4-timestamp bi-temporal model [L8 fix]."""

    def test_store_temporal_creates_record(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """Storing temporal validity creates a record with all 4 timestamps."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default", valid_from="2026-01-01")
        tv = db.get_temporal_validity(fid)
        assert tv is not None
        assert tv["fact_id"] == fid
        assert tv["profile_id"] == "default"
        assert tv["valid_from"] == "2026-01-01"
        assert tv["valid_until"] is None
        assert tv["system_created_at"] is not None  # auto-populated
        assert tv["system_expired_at"] is None

    def test_system_created_at_auto_populated(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """system_created_at defaults to datetime('now')."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default")
        tv = db.get_temporal_validity(fid)
        assert tv is not None
        assert tv["system_created_at"] is not None
        assert len(tv["system_created_at"]) > 10  # ISO format

    def test_valid_from_nullable(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """valid_from can be NULL (fact always been true)."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default", valid_from=None)
        tv = db.get_temporal_validity(fid)
        assert tv is not None
        assert tv["valid_from"] is None

    def test_valid_until_null_means_current(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """valid_until NULL = fact is still true."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default")
        tv = db.get_temporal_validity(fid)
        assert tv is not None
        assert tv["valid_until"] is None

    def test_system_expired_at_null_means_active(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """system_expired_at NULL = system still considers fact current."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default")
        tv = db.get_temporal_validity(fid)
        assert tv is not None
        assert tv["system_expired_at"] is None

    def test_invalidation_sets_both_timestamps(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """invalidate_fact_temporal() sets BOTH valid_until AND system_expired_at.
        BI-TEMPORAL INTEGRITY: both set in same operation."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default")
        db.invalidate_fact_temporal(fid, "new-fact-123", "contradiction detected")
        tv = db.get_temporal_validity(fid)
        assert tv is not None
        assert tv["valid_until"] is not None
        assert tv["system_expired_at"] is not None

    def test_invalidation_records_invalidated_by(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """invalidated_by field links to the contradicting fact."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default")
        db.invalidate_fact_temporal(fid, "new-fact-ABC", "reason")
        tv = db.get_temporal_validity(fid)
        assert tv is not None
        assert tv["invalidated_by"] == "new-fact-ABC"

    def test_invalidation_records_reason(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """invalidation_reason provides human-readable explanation."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default")
        db.invalidate_fact_temporal(fid, "x", "Sheaf severity 0.72")
        tv = db.get_temporal_validity(fid)
        assert tv is not None
        assert tv["invalidation_reason"] == "Sheaf severity 0.72"

    def test_is_temporally_valid_true_for_new_fact(
        self,
        validator: TemporalValidator,
        _seed_facts: tuple[str, str, str],
    ) -> None:
        """New fact with no temporal record is valid."""
        assert validator.is_temporally_valid(_seed_facts[0]) is True

    def test_is_temporally_valid_false_after_invalidation(
        self,
        db: DatabaseManager,
        validator: TemporalValidator,
        _seed_facts: tuple[str, str, str],
    ) -> None:
        """Invalidated fact returns False."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default")
        db.invalidate_fact_temporal(fid, "x", "reason")
        assert validator.is_temporally_valid(fid, "default") is False

    def test_double_invalidation_is_idempotent(
        self,
        db: DatabaseManager,
        validator: TemporalValidator,
        _seed_facts: tuple[str, str, str],
    ) -> None:
        """Invalidating an already-invalidated fact is a no-op."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default")
        db.invalidate_fact_temporal(fid, "x", "first")
        tv1 = db.get_temporal_validity(fid)

        # Second invalidation via the validator (which checks idempotency)
        validator.invalidate_fact(fid, "y", "second")
        tv2 = db.get_temporal_validity(fid)

        # Timestamps should be unchanged (first invalidation sticks)
        assert tv1["valid_until"] == tv2["valid_until"]
        assert tv1["system_expired_at"] == tv2["system_expired_at"]
        assert tv2["invalidated_by"] == "x"  # Still first invalidator

    def test_delete_temporal_validity(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """delete_temporal_validity removes the record (testing only)."""
        fid = _seed_facts[0]
        db.store_temporal_validity(fid, "default")
        assert db.get_temporal_validity(fid) is not None
        db.delete_temporal_validity(fid)
        assert db.get_temporal_validity(fid) is None


# ---------------------------------------------------------------------------
# TestTemporalQueries -- event-time and transaction-time queries
# ---------------------------------------------------------------------------

class TestTemporalQueries:
    def test_get_valid_facts_excludes_expired(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """get_valid_facts() returns only facts with NULL valid_until."""
        fid0, fid1, fid2 = _seed_facts
        db.store_temporal_validity(fid0, "default")
        db.store_temporal_validity(fid1, "default")
        db.invalidate_fact_temporal(fid1, "x", "expired")

        valid = db.get_valid_facts("default")
        assert fid0 in valid
        assert fid1 not in valid
        # fid2 has no temporal record -> assumed valid
        assert fid2 in valid

    def test_get_facts_valid_at_event_time(
        self,
        db: DatabaseManager,
        validator: TemporalValidator,
        _seed_facts: tuple[str, str, str],
    ) -> None:
        """Event time query returns facts valid at that time."""
        fid0 = _seed_facts[0]
        db.store_temporal_validity(fid0, "default", valid_from="2026-01-01")
        results = validator.get_facts_valid_at("default", "2026-06-01")
        assert fid0 in results

    def test_get_system_knowledge_at_transaction_time(
        self,
        db: DatabaseManager,
        validator: TemporalValidator,
        _seed_facts: tuple[str, str, str],
    ) -> None:
        """Transaction time query returns what system knew at that time."""
        fid0 = _seed_facts[0]
        db.store_temporal_validity(fid0, "default")
        # Query at a future time -- should include this fact
        future = "2099-12-31T23:59:59"
        results = validator.get_system_knowledge_at("default", future)
        assert fid0 in results

    def test_facts_without_temporal_record_are_valid(
        self,
        db: DatabaseManager,
        validator: TemporalValidator,
        _seed_facts: tuple[str, str, str],
    ) -> None:
        """Facts with no temporal_validity row are assumed valid."""
        fid0 = _seed_facts[0]
        # No store_temporal_validity call
        assert validator.is_temporally_valid(fid0) is True
        valid = db.get_valid_facts("default")
        assert fid0 in valid


# ---------------------------------------------------------------------------
# TestContradictionDetection -- sheaf Mode A
# ---------------------------------------------------------------------------

class TestContradictionDetection:
    """Mode A: sheaf consistency (pure math, no LLM)."""

    def test_no_sheaf_checker_returns_empty(
        self, db: DatabaseManager,
    ) -> None:
        """Graceful degradation when sheaf_checker is None."""
        config = TemporalValidatorConfig(enabled=True, mode="a")
        tv = TemporalValidator(db=db, sheaf_checker=None, config=config)
        fact = AtomicFact(
            profile_id="default", memory_id="m1", content="test",
        )
        result = tv.detect_contradiction(fact, "default")
        assert result == []

    def test_sheaf_contradiction_above_threshold(
        self, db: DatabaseManager,
    ) -> None:
        """Contradictions above threshold are returned."""
        mock_result = MagicMock()
        mock_result.fact_id_a = "new"
        mock_result.fact_id_b = "old"
        mock_result.severity = 0.7  # Above 0.45 threshold
        mock_result.edge_type = "entity"
        mock_result.description = "High disagreement"

        sheaf = MagicMock()
        sheaf.check_consistency.return_value = [mock_result]

        config = TemporalValidatorConfig(
            enabled=True, mode="a", contradiction_threshold=0.45,
        )
        tv = TemporalValidator(db=db, sheaf_checker=sheaf, config=config)
        fact = AtomicFact(
            profile_id="default", memory_id="m1", content="test",
        )
        result = tv.detect_contradiction(fact, "default")
        assert len(result) == 1
        assert result[0]["severity"] == 0.7
        assert result[0]["fact_id_b"] == "old"

    def test_sheaf_contradiction_below_threshold(
        self, db: DatabaseManager,
    ) -> None:
        """Contradictions below threshold are filtered out."""
        mock_result = MagicMock()
        mock_result.severity = 0.2  # Below 0.45

        sheaf = MagicMock()
        sheaf.check_consistency.return_value = [mock_result]

        config = TemporalValidatorConfig(
            enabled=True, mode="a", contradiction_threshold=0.45,
        )
        tv = TemporalValidator(db=db, sheaf_checker=sheaf, config=config)
        fact = AtomicFact(
            profile_id="default", memory_id="m1", content="test",
        )
        result = tv.detect_contradiction(fact, "default")
        assert len(result) == 0

    def test_threshold_configurable(
        self, db: DatabaseManager,
    ) -> None:
        """Lower threshold catches more contradictions."""
        mock_result = MagicMock()
        mock_result.fact_id_a = "new"
        mock_result.fact_id_b = "old"
        mock_result.severity = 0.3
        mock_result.edge_type = "semantic"
        mock_result.description = "mild"

        sheaf = MagicMock()
        sheaf.check_consistency.return_value = [mock_result]

        # Low threshold: catches severity=0.3
        config = TemporalValidatorConfig(
            enabled=True, mode="a", contradiction_threshold=0.2,
        )
        tv = TemporalValidator(db=db, sheaf_checker=sheaf, config=config)
        fact = AtomicFact(
            profile_id="default", memory_id="m1", content="test",
        )
        result = tv.detect_contradiction(fact, "default")
        assert len(result) == 1

    def test_sheaf_checker_exception_returns_empty(
        self, db: DatabaseManager,
    ) -> None:
        """Sheaf checker exception -> empty results (Rule 19)."""
        sheaf = MagicMock()
        sheaf.check_consistency.side_effect = RuntimeError("boom")

        config = TemporalValidatorConfig(enabled=True, mode="a")
        tv = TemporalValidator(db=db, sheaf_checker=sheaf, config=config)
        fact = AtomicFact(
            profile_id="default", memory_id="m1", content="test",
        )
        result = tv.detect_contradiction(fact, "default")
        assert result == []


class TestContradictionInvalidation:
    """End-to-end: detect contradiction -> invalidate old fact."""

    def test_contradiction_invalidates_old_fact(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """New fact contradicting old fact sets old fact's valid_until."""
        fid0 = _seed_facts[0]  # old fact
        db.store_temporal_validity(fid0, "default")

        # Mock sheaf to report contradiction
        mock_result = MagicMock()
        mock_result.fact_id_a = "new-fact"
        mock_result.fact_id_b = fid0
        mock_result.severity = 0.8
        mock_result.edge_type = "entity"
        mock_result.description = "Direct contradiction"

        sheaf = MagicMock()
        sheaf.check_consistency.return_value = [mock_result]

        config = TemporalValidatorConfig(enabled=True, mode="a")
        tv = TemporalValidator(db=db, sheaf_checker=sheaf, config=config)

        new_fact = AtomicFact(
            profile_id="default", memory_id="m1", content="new info",
        )
        actions = tv.validate_and_invalidate(new_fact, "default")

        assert len(actions) == 1
        assert actions[0]["old_fact_id"] == fid0

        # Verify bi-temporal integrity
        record = db.get_temporal_validity(fid0)
        assert record is not None
        assert record["valid_until"] is not None
        assert record["system_expired_at"] is not None

    def test_contradiction_applies_trust_penalty(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """Invalidated fact gets trust penalty via update_on_contradiction()."""
        fid0 = _seed_facts[0]
        db.store_temporal_validity(fid0, "default")

        mock_result = MagicMock()
        mock_result.fact_id_a = "new"
        mock_result.fact_id_b = fid0
        mock_result.severity = 0.9
        mock_result.edge_type = "entity"
        mock_result.description = "contradiction"

        sheaf = MagicMock()
        sheaf.check_consistency.return_value = [mock_result]

        trust_scorer = MagicMock()
        trust_scorer.update_on_contradiction.return_value = 0.2

        config = TemporalValidatorConfig(enabled=True, mode="a")
        tv = TemporalValidator(
            db=db, sheaf_checker=sheaf,
            trust_scorer=trust_scorer, config=config,
        )

        new_fact = AtomicFact(
            profile_id="default", memory_id="m1", content="new",
        )
        tv.validate_and_invalidate(new_fact, "default")

        trust_scorer.update_on_contradiction.assert_called_once_with(
            target_type="fact",
            target_id=fid0,
            profile_id="default",
        )

    def test_new_fact_remains_valid(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """The new (contradicting) fact remains temporally valid."""
        fid0 = _seed_facts[0]
        new_fid = _seed_facts[1]
        db.store_temporal_validity(fid0, "default")
        db.store_temporal_validity(new_fid, "default")

        mock_result = MagicMock()
        mock_result.fact_id_a = new_fid
        mock_result.fact_id_b = fid0
        mock_result.severity = 0.8
        mock_result.edge_type = "entity"
        mock_result.description = "contradiction"

        sheaf = MagicMock()
        sheaf.check_consistency.return_value = [mock_result]

        config = TemporalValidatorConfig(enabled=True, mode="a")
        tv = TemporalValidator(db=db, sheaf_checker=sheaf, config=config)

        new_fact = AtomicFact(
            fact_id=new_fid,
            profile_id="default", memory_id="m1", content="new",
        )
        tv.validate_and_invalidate(new_fact, "default")

        # New fact should remain valid
        new_tv = db.get_temporal_validity(new_fid)
        assert new_tv is not None
        assert new_tv["valid_until"] is None
        assert new_tv["system_expired_at"] is None

    def test_invalidation_reason_recorded(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """Invalidation reason contains sheaf description."""
        fid0 = _seed_facts[0]
        db.store_temporal_validity(fid0, "default")

        mock_result = MagicMock()
        mock_result.fact_id_a = "new"
        mock_result.fact_id_b = fid0
        mock_result.severity = 0.75
        mock_result.edge_type = "entity"
        mock_result.description = "Sheaf severity 0.750"

        sheaf = MagicMock()
        sheaf.check_consistency.return_value = [mock_result]

        config = TemporalValidatorConfig(enabled=True, mode="a")
        tv = TemporalValidator(db=db, sheaf_checker=sheaf, config=config)

        new_fact = AtomicFact(
            profile_id="default", memory_id="m1", content="new",
        )
        tv.validate_and_invalidate(new_fact, "default")

        record = db.get_temporal_validity(fid0)
        assert record is not None
        assert record["invalidation_reason"] == "Sheaf severity 0.750"


class TestTemporalAnchorGuard:
    """T2 guard: distinct-era historical facts are not superseded (no over-fire)."""

    @staticmethod
    def _mock_sheaf(old_fid: str) -> MagicMock:
        m = MagicMock()
        m.fact_id_a = "new"
        m.fact_id_b = old_fid
        m.severity = 0.9
        m.edge_type = "entity"
        m.description = "contradiction"
        s = MagicMock()
        s.check_consistency.return_value = [m]
        return s

    @staticmethod
    def _store(db: DatabaseManager, content: str, referenced_date: str | None) -> str:
        rec = MemoryRecord(profile_id="default", content="parent", session_id="s")
        db.store_memory(rec)
        f = AtomicFact(
            profile_id="default", memory_id=rec.memory_id,
            content=content, referenced_date=referenced_date,
        )
        db.store_fact(f)
        db.store_temporal_validity(f.fact_id, "default", valid_from=referenced_date)
        return f.fact_id

    def _validator(self, db: DatabaseManager, old_fid: str) -> TemporalValidator:
        return TemporalValidator(
            db=db, sheaf_checker=self._mock_sheaf(old_fid),
            config=TemporalValidatorConfig(enabled=True, mode="a"),
        )

    def test_distinct_era_not_superseded(self, db: DatabaseManager) -> None:
        old_fid = self._store(db, "lived in Delhi", "2020-01-01")
        tv = self._validator(db, old_fid)
        new_fact = AtomicFact(
            profile_id="default", memory_id="m1",
            content="lives in Mumbai", referenced_date="2024-01-01",
        )
        actions = tv.validate_and_invalidate(new_fact, "default")
        assert actions == []  # guard skipped supersession
        rec = db.get_temporal_validity(old_fid)
        assert rec["system_expired_at"] is None       # never invalidated
        # T1 keeps the historical fact recallable.
        assert db.get_invalidated_fact_ids([old_fid], "default") == set()

    def test_same_day_anchor_still_supersedes(self, db: DatabaseManager) -> None:
        old_fid = self._store(db, "status A", "2024-06-01")
        tv = self._validator(db, old_fid)
        new_fact = AtomicFact(
            profile_id="default", memory_id="m1",
            content="status B", referenced_date="2024-06-01",
        )
        actions = tv.validate_and_invalidate(new_fact, "default")
        assert len(actions) == 1  # same day => genuine contradiction
        assert db.get_temporal_validity(old_fid)["system_expired_at"] is not None

    def test_undated_new_fact_still_supersedes(self, db: DatabaseManager) -> None:
        old_fid = self._store(db, "old dated fact", "2020-01-01")
        tv = self._validator(db, old_fid)
        new_fact = AtomicFact(
            profile_id="default", memory_id="m1", content="new undated",
        )  # no anchor on the new fact => guard does not fire
        actions = tv.validate_and_invalidate(new_fact, "default")
        assert len(actions) == 1


class TestT2T1ClosedLoop:
    """T2 supersession -> T1 exclusion, end to end."""

    def test_supersession_excludes_from_t1(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        old_fid = _seed_facts[0]
        db.store_temporal_validity(old_fid, "default")
        m = MagicMock()
        m.fact_id_a, m.fact_id_b = "new", old_fid
        m.severity, m.edge_type, m.description = 0.9, "entity", "contradiction"
        sheaf = MagicMock()
        sheaf.check_consistency.return_value = [m]
        tv = TemporalValidator(
            db=db, sheaf_checker=sheaf,
            config=TemporalValidatorConfig(enabled=True, mode="a"),
        )
        new_fact = AtomicFact(profile_id="default", memory_id="m1", content="new")
        tv.validate_and_invalidate(new_fact, "default")
        # T2 set system_expired_at; T1's admission filter must now exclude it.
        assert db.get_invalidated_fact_ids([old_fid], "default") == {old_fid}
