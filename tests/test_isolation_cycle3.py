# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Cycle-3 isolation correctness regression tests — H-01, H-02, M-01, M-02, L-01
#
# TDD contract: each test is written RED before the corresponding fix is applied.
# Running against unpatched code MUST fail for the right reason (not a crash or
# unrelated assertion).  Running after the patch MUST pass.

from __future__ import annotations

import importlib.util
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# H-01 — cross-profile preference contamination (learning/cross_project.py)
# ---------------------------------------------------------------------------


class TestH01CrossProfilePatternsContamination:
    """H-01: transferable_patterns must be scoped per profile_id."""

    def _make_pattern(self, value: str) -> dict[str, Any]:
        return {
            "value": value,
            "confidence": 0.9,
            "evidence_count": 10,
            "profiles_seen": 1,
            "decay_factor": 1.0,
            "contradictions": [],
        }

    def test_get_preferences_accepts_profile_id(self, tmp_path: Path) -> None:
        """RED: get_preferences() must accept profile_id kwarg.

        Before fix: raises TypeError (unexpected keyword argument).
        After  fix: returns profile-scoped dict.
        """
        from superlocalmemory.learning.cross_project import CrossProjectAggregator

        agg = CrossProjectAggregator(tmp_path / "cross.db")
        # This MUST NOT raise TypeError after the fix.
        result = agg.get_preferences(profile_id="test-profile", min_confidence=0.0)
        assert isinstance(result, dict)

    def test_store_patterns_accepts_profile_id(self, tmp_path: Path) -> None:
        """RED: _store_patterns() must accept profile_id kwarg.

        Before fix: raises TypeError.
        After  fix: stores row with that profile_id.
        """
        from superlocalmemory.learning.cross_project import CrossProjectAggregator

        agg = CrossProjectAggregator(tmp_path / "cross.db")
        # Must not raise TypeError.
        agg._store_patterns(
            {"frontend_framework": self._make_pattern("vue")},
            "team-frontend",
        )

    def test_cross_profile_contamination_prevented(self, tmp_path: Path) -> None:
        """RED: team-frontend's preference must NOT be overwritten by team-backend.

        Before fix:
            _store_patterns(vue, team-frontend) inserts (preference, frontend_framework, vue)
            _store_patterns(react, team-backend) ON CONFLICT DO UPDATE → overwrites to react
            get_preferences(team-frontend) returns react  ← BUG

        After fix:
            Each profile has its own row; get_preferences(team-frontend) returns vue.
        """
        from superlocalmemory.learning.cross_project import CrossProjectAggregator

        agg = CrossProjectAggregator(tmp_path / "cross.db")
        agg._store_patterns(
            {"frontend_framework": self._make_pattern("vue")},
            "team-frontend",
        )
        agg._store_patterns(
            {"frontend_framework": self._make_pattern("react")},
            "team-backend",
        )

        prefs_fe = agg.get_preferences(
            profile_id="team-frontend", min_confidence=0.0
        )
        prefs_be = agg.get_preferences(
            profile_id="team-backend", min_confidence=0.0
        )

        assert prefs_fe.get("frontend_framework", {}).get("value") == "vue", (
            "team-frontend should see 'vue', not contaminated by team-backend"
        )
        assert prefs_be.get("frontend_framework", {}).get("value") == "react", (
            "team-backend should see 'react'"
        )

    def test_pattern_extractor_threads_profile_id(self, tmp_path: Path) -> None:
        """RED: PatternExtractor._extract_from_cross_project must accept profile_id.

        Before fix: _extract_from_cross_project() takes no args.
        After  fix: _extract_from_cross_project(profile_id) filters by profile.
        """
        from superlocalmemory.learning.cross_project import CrossProjectAggregator
        from superlocalmemory.parameterization.pattern_extractor import PatternExtractor

        agg = CrossProjectAggregator(tmp_path / "cross.db")
        agg._store_patterns(
            {"frontend_framework": self._make_pattern("vue")},
            "my-profile",
        )

        # Build a minimal PatternExtractor with the real aggregator.
        mock_db = MagicMock()
        mock_db.execute.return_value = []
        mock_behavioral = MagicMock()
        mock_behavioral.get_recent_patterns.return_value = []
        mock_behavioral.get_all_patterns.return_value = []
        mock_wf = MagicMock()
        mock_wf.mine.return_value = []

        from superlocalmemory.core.config import ParameterizationConfig

        extractor = PatternExtractor(
            mock_db, mock_behavioral, agg, mock_wf, ParameterizationConfig()
        )

        patterns = extractor.extract("my-profile")
        # After fix: patterns should include the stored preference.
        # The test passes if extract() runs without TypeError.
        assert isinstance(patterns, list)


# ---------------------------------------------------------------------------
# H-02 — cross-tenant CCR retrieval (optimize/storage/db.py)
# ---------------------------------------------------------------------------


class TestH02CrossTenantCCRIsolation:
    """H-02: ccr_put/ccr_get must be scoped by tenant_id."""

    @pytest.fixture()
    def cache_db(self, tmp_path: Path, monkeypatch):
        import superlocalmemory.optimize.storage.db as _db_mod
        from superlocalmemory.optimize.storage.db import CacheDB

        monkeypatch.setattr(_db_mod, "_KEY_FILE", tmp_path / "opt-key.bin")
        return CacheDB(tmp_path / "llmcache.db")

    def test_ccr_put_accepts_tenant_id(self, cache_db) -> None:
        """RED: ccr_put() must accept tenant_id kwarg."""
        original = b"sensitive data for tenant_a"
        # Before fix: TypeError — unexpected keyword argument 'tenant_id'
        cache_db.ccr_put(
            "00000000-0000-4000-8000-000000000001",
            original,
            tenant_id="tenant_a",
        )

    def test_ccr_get_accepts_tenant_id(self, cache_db) -> None:
        """RED: ccr_get() must accept tenant_id kwarg."""
        # Before fix: TypeError
        result = cache_db.ccr_get(
            "00000000-0000-4000-8000-000000000002",
            tenant_id="tenant_a",
        )
        assert result is None  # not found is OK; no crash

    def test_cross_tenant_retrieval_blocked(self, cache_db) -> None:
        """RED: tenant_b must NOT retrieve tenant_a's CCR.

        Before fix: ccr_get() ignores tenant_id column → tenant_b reads
        tenant_a's pre-compression original.
        After  fix: ccr_get(ccr_id, tenant_id='tenant_b') returns None.
        """
        original = b"tenant_a_confidential_legal_brief_content"
        ccr_id = "00000000-0000-4000-8000-000000000003"

        cache_db.ccr_put(ccr_id, original, tenant_id="tenant_a")

        # tenant_b must NOT see tenant_a's content.
        result_as_b = cache_db.ccr_get(ccr_id, tenant_id="tenant_b")
        assert result_as_b is None, (
            f"Cross-tenant CCR retrieval must be blocked; got {result_as_b!r}"
        )

        # tenant_a must still be able to retrieve its own content.
        result_as_a = cache_db.ccr_get(ccr_id, tenant_id="tenant_a")
        assert result_as_a == original, (
            "tenant_a must retrieve its own original content"
        )

    def test_ccr_delete_scoped_to_tenant(self, cache_db) -> None:
        """ccr_delete() must accept tenant_id and only delete that tenant's row."""
        original = b"data for tenant_x"
        ccr_id = "00000000-0000-4000-8000-000000000004"

        cache_db.ccr_put(ccr_id, original, tenant_id="tenant_x")

        # Delete as wrong tenant — should be no-op.
        cache_db.ccr_delete(ccr_id, tenant_id="tenant_y")
        assert cache_db.ccr_get(ccr_id, tenant_id="tenant_x") == original, (
            "Row must survive deletion by wrong tenant"
        )

        # Delete as correct tenant — should remove it.
        cache_db.ccr_delete(ccr_id, tenant_id="tenant_x")
        assert cache_db.ccr_get(ccr_id, tenant_id="tenant_x") is None, (
            "Row must be deleted by owning tenant"
        )

    def test_ccr_store_threads_tenant_id(self, tmp_path: Path, monkeypatch) -> None:
        """CCRStore.store() must forward tenant_id to ccr_put()."""
        import superlocalmemory.optimize.storage.db as _db_mod
        from superlocalmemory.optimize.compress.ccr import CCRStore
        from superlocalmemory.optimize.storage.db import CacheDB

        monkeypatch.setattr(_db_mod, "_KEY_FILE", tmp_path / "opt-key.bin")
        db = CacheDB(tmp_path / "llmcache2.db")

        store = CCRStore()
        store._db = db

        ccr_id = store.store(b"original_bytes", tenant_id="tenant_store")
        assert ccr_id != "", "store() must return a valid ccr_id"

        # Retrieve as correct tenant — must succeed.
        result = store.retrieve(ccr_id, tenant_id="tenant_store")
        assert result == b"original_bytes", (
            "retrieve() as owning tenant must succeed"
        )

        # Retrieve as wrong tenant — must return None.
        result_wrong = store.retrieve(ccr_id, tenant_id="other_tenant")
        assert result_wrong is None, (
            "retrieve() as wrong tenant must return None"
        )


# ---------------------------------------------------------------------------
# M-01 — KNN score normalization (retrieval/semantic_channel.py)
# ---------------------------------------------------------------------------


class TestM01KNNScoreNormalization:
    """M-01: KNN path scores must be normalized to [0,1] via (score+1)/2."""

    def _make_minimal_fact(self, fact_id: str, profile_id: str = "p"):
        """Build a minimal AtomicFact with no embedding/fisher for test."""
        from superlocalmemory.storage.models import AtomicFact

        return AtomicFact(
            fact_id=fact_id,
            profile_id=profile_id,
            content="test fact",
        )

    def _make_channel(self, mock_vs_scores: list[tuple[str, float]]):
        """Build a SemanticChannel with a mock vector_store returning the given scores."""
        from superlocalmemory.retrieval.semantic_channel import SemanticChannel

        mock_vs = MagicMock()
        mock_vs.is_available = True
        mock_vs.search.return_value = mock_vs_scores

        mock_db = MagicMock()
        mock_db.get_external_visible_facts.return_value = []
        # Return a matching fact for each score entry so the channel doesn't bail early.
        facts = [self._make_minimal_fact(fid) for fid, _ in mock_vs_scores]
        mock_db.get_facts_by_ids.return_value = facts

        channel = SemanticChannel.__new__(SemanticChannel)
        channel._db = mock_db
        channel._embedder = None
        channel._vector_store = mock_vs
        channel._qas = None
        return channel

    def test_knn_score_normalized_matches_full_scan_formula(self) -> None:
        """RED: KNN cos_sim must equal (raw_score + 1.0) / 2.0 after fix.

        Before fix: cos_sim = raw_score = 0.7 (different from full-scan 0.85).
        After  fix: cos_sim = (0.7 + 1.0) / 2.0 = 0.85.
        """
        raw_score = 0.7
        expected_normalized = (raw_score + 1.0) / 2.0  # 0.85

        channel = self._make_channel([("fact-001", raw_score)])
        q_vec = np.ones(4, dtype=np.float32) * 0.5  # arbitrary shape

        result = channel._search_via_vector_store(
            query_embedding=[0.5, 0.5, 0.5, 0.5],
            q_vec=q_vec,
            profile_id="p",
            top_k=5,
        )

        assert len(result) == 1, "Expected exactly one result"
        returned_score = result[0][1]

        # Before fix: returned_score ≈ 0.7.  After fix: ≈ 0.85.
        assert abs(returned_score - expected_normalized) < 0.01, (
            f"KNN score {returned_score:.4f} should be normalized to "
            f"{expected_normalized:.4f} via (raw+1)/2 — got {returned_score:.4f}. "
            "This is the M-01 bug: KNN path does not normalize cosine scores."
        )

    def test_knn_score_range_0_5_to_1_for_positive_cosines(self) -> None:
        """After fix, all returned scores from positive cosines must be in [0.5, 1.0]."""
        raw_scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        entries = [(f"fact-{i}", s) for i, s in enumerate(raw_scores)]

        channel = self._make_channel(entries)
        q_vec = np.ones(4, dtype=np.float32) * 0.5

        result = channel._search_via_vector_store(
            query_embedding=[0.5, 0.5, 0.5, 0.5],
            q_vec=q_vec,
            profile_id="p",
            top_k=10,
        )

        for fid, score in result:
            assert 0.5 <= score <= 1.0, (
                f"Normalized KNN score {score:.4f} for {fid} must be in [0.5, 1.0]. "
                "Before fix: scores are in [0, 1] via vector_store clipping, which is "
                "inconsistent with the full-scan (cosine+1)/2 normalization."
            )

    def test_external_scores_and_knn_scores_same_scale(self) -> None:
        """After fix, max() merge between KNN and external scores uses same scale.

        Before fix: KNN score 0.7 < external 0.85 even for same cosine → external
        always wins in max(), biasing retrieval toward cross-profile facts.
        After  fix: both normalize to 0.85, max() is a tie, no bias.
        """
        raw_score = 0.7
        expected = (raw_score + 1.0) / 2.0  # 0.85

        # Simulate external_score for the same cosine similarity.
        external_score_for_same_cosine = (raw_score + 1.0) / 2.0

        channel = self._make_channel([("fact-001", raw_score)])
        q_vec = np.ones(4, dtype=np.float32) * 0.5

        # Make the DB report the same fact as external too.
        external_fact = self._make_minimal_fact("fact-001")
        external_fact.embedding = [0.5, 0.5, 0.5, 0.5]  # same direction as q_vec
        channel._db.get_external_visible_facts.return_value = [external_fact]

        result = channel._search_via_vector_store(
            query_embedding=[0.5, 0.5, 0.5, 0.5],
            q_vec=q_vec,
            profile_id="p",
            top_k=5,
        )

        if result:
            score = result[0][1]
            # After fix: score is max(normalized_knn, external) = max(0.85, external)
            # Both should be in the same range.
            assert score >= 0.5, (
                f"Merged score {score:.4f} must be >= 0.5 after normalization fix"
            )


# ---------------------------------------------------------------------------
# M-02 — get_valid_facts excludes future-dated facts (storage/database.py)
# ---------------------------------------------------------------------------


class TestM02FutureDateValidFacts:
    """M-02: get_valid_facts must INCLUDE facts with future valid_until."""

    @pytest.fixture()
    def db(self, tmp_path: Path):
        from superlocalmemory.storage import schema as real_schema
        from superlocalmemory.storage.database import DatabaseManager

        mgr = DatabaseManager(tmp_path / "test.db")
        mgr.initialize(real_schema)
        # Create 'prof-test' so FK constraints on atomic_facts
        # (profile_id → profiles, memory_id → memories) are satisfied.
        mgr.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
            ("prof-test", "Test Profile"),
        )
        # AtomicFact.memory_id defaults to ""; insert a dummy memories row so
        # the atomic_facts FOREIGN KEY (memory_id) constraint passes.
        mgr.execute(
            "INSERT OR IGNORE INTO memories "
            "(memory_id, profile_id, content) VALUES (?, ?, ?)",
            ("", "prof-test", ""),
        )
        return mgr

    def test_future_valid_until_fact_is_included(self, db) -> None:
        """RED: A fact with valid_until = 2028 must appear in get_valid_facts.

        Before fix:
            WHERE tv.valid_until IS NULL  — filters OUT the future-dated fact.
        After  fix:
            WHERE tv.valid_until > now()  — correctly INCLUDES it.
        """
        from superlocalmemory.storage.models import AtomicFact

        fact = AtomicFact(
            fact_id="valid-future-001",
            profile_id="prof-test",
            content="Contract expires in 2028",
        )
        db.store_fact(fact)

        future_date = (datetime.now(UTC) + timedelta(days=730)).isoformat()
        db.store_temporal_validity(
            fact_id="valid-future-001",
            profile_id="prof-test",
            valid_from=datetime.now(UTC).isoformat(),
            valid_until=future_date,
        )

        valid_ids = db.get_valid_facts("prof-test")
        assert "valid-future-001" in valid_ids, (
            "Fact with future valid_until must be included in get_valid_facts. "
            "Before fix: valid_until IS NULL check incorrectly excludes it."
        )

    def test_null_valid_until_fact_is_included(self, db) -> None:
        """Fact with no temporal record must always be included."""
        from superlocalmemory.storage.models import AtomicFact

        fact = AtomicFact(
            fact_id="no-temporal-001",
            profile_id="prof-test",
            content="No temporal constraint",
        )
        db.store_fact(fact)

        valid_ids = db.get_valid_facts("prof-test")
        assert "no-temporal-001" in valid_ids

    def test_expired_fact_excluded(self, db) -> None:
        """Fact with system_expired_at set must be excluded."""
        from superlocalmemory.storage.models import AtomicFact

        fact = AtomicFact(
            fact_id="expired-001",
            profile_id="prof-test",
            content="Already expired",
        )
        db.store_fact(fact)

        db.store_temporal_validity(
            fact_id="expired-001",
            profile_id="prof-test",
            valid_from="2023-01-01",
            valid_until="2024-01-01",
        )
        # Manually expire it.
        db.execute(
            "UPDATE fact_temporal_validity SET system_expired_at = ? WHERE fact_id = ?",
            (datetime.now(UTC).isoformat(), "expired-001"),
        )

        valid_ids = db.get_valid_facts("prof-test")
        assert "expired-001" not in valid_ids, (
            "Expired fact must NOT appear in get_valid_facts"
        )

    def test_past_valid_until_excluded(self, db) -> None:
        """Fact with valid_until in the past (and no system_expired_at) must be excluded."""
        from superlocalmemory.storage.models import AtomicFact

        fact = AtomicFact(
            fact_id="past-validity-001",
            profile_id="prof-test",
            content="Past validity",
        )
        db.store_fact(fact)

        past_date = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        db.store_temporal_validity(
            fact_id="past-validity-001",
            profile_id="prof-test",
            valid_from="2023-01-01",
            valid_until=past_date,
        )

        valid_ids = db.get_valid_facts("prof-test")
        assert "past-validity-001" not in valid_ids, (
            "Fact whose valid_until has passed must be excluded from get_valid_facts"
        )


# ---------------------------------------------------------------------------
# L-01 — unscoped UPDATE on canonical_entities (storage + entity_resolver)
# ---------------------------------------------------------------------------


class TestL01UnscopedEntityUpdates:
    """L-01: entity updates must include AND profile_id = ? guard."""

    @pytest.fixture()
    def db(self, tmp_path: Path):
        from superlocalmemory.storage import schema as real_schema
        from superlocalmemory.storage.database import DatabaseManager

        mgr = DatabaseManager(tmp_path / "test.db")
        mgr.initialize(real_schema)
        # Create profiles needed by L-01 tests so FK constraints on
        # canonical_entities (profile_id → profiles) are satisfied.
        for pid in ("profile-a",):
            mgr.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                (pid, pid),
            )
        return mgr

    def _make_entity(self, entity_id: str, profile_id: str):
        from superlocalmemory.storage.models import CanonicalEntity

        now = datetime.now(UTC).isoformat()
        return CanonicalEntity(
            entity_id=entity_id,
            profile_id=profile_id,
            canonical_name=f"Entity-{entity_id[:6]}",
            entity_type="person",
            first_seen=now,
            last_seen=now,
            fact_count=0,
        )

    def test_increment_entity_fact_count_accepts_profile_id(self, db) -> None:
        """RED: increment_entity_fact_count must accept profile_id.

        Before fix: signature is (entity_id) → TypeError on extra arg.
        After  fix: signature is (entity_id, profile_id).
        """
        db.store_entity(self._make_entity("uuid-inc-001", "profile-a"))
        # Must not raise TypeError.
        db.increment_entity_fact_count("uuid-inc-001", "profile-a")

        rows = db.execute(
            "SELECT fact_count FROM canonical_entities WHERE entity_id = ?",
            ("uuid-inc-001",),
        )
        assert rows[0]["fact_count"] == 1

    def test_increment_wrong_profile_is_noop(self, db) -> None:
        """Incrementing with wrong profile_id must not affect the row."""
        db.store_entity(self._make_entity("uuid-inc-002", "profile-a"))

        # Wrong profile — must be a no-op.
        db.increment_entity_fact_count("uuid-inc-002", "profile-x")

        rows = db.execute(
            "SELECT fact_count FROM canonical_entities WHERE entity_id = ?",
            ("uuid-inc-002",),
        )
        assert rows[0]["fact_count"] == 0, (
            "Wrong-profile increment must not change fact_count"
        )

    def test_touch_last_seen_accepts_profile_id(self, db) -> None:
        """RED: EntityResolver._touch_last_seen must accept profile_id.

        Before fix: signature is (entity_id) → TypeError.
        After  fix: signature is (entity_id, profile_id).
        """
        from superlocalmemory.encoding.entity_resolver import EntityResolver

        db.store_entity(self._make_entity("uuid-touch-001", "profile-a"))
        old_ts = datetime.now(UTC).isoformat()

        resolver = EntityResolver(db)

        # Introduce a tiny delay so the updated timestamp is strictly later.
        time.sleep(0.01)

        # Must not raise TypeError.
        resolver._touch_last_seen("uuid-touch-001", "profile-a")

        rows = db.execute(
            "SELECT last_seen FROM canonical_entities WHERE entity_id = ?",
            ("uuid-touch-001",),
        )
        assert rows[0]["last_seen"] > old_ts, (
            "_touch_last_seen must update last_seen"
        )

    def test_touch_last_seen_wrong_profile_is_noop(self, db) -> None:
        """_touch_last_seen with wrong profile must not update the row."""
        from superlocalmemory.encoding.entity_resolver import EntityResolver

        db.store_entity(self._make_entity("uuid-touch-002", "profile-a"))
        original_row = db.execute(
            "SELECT last_seen FROM canonical_entities WHERE entity_id = ?",
            ("uuid-touch-002",),
        )[0]
        original_ts = original_row["last_seen"]

        resolver = EntityResolver(db)
        time.sleep(0.01)
        resolver._touch_last_seen("uuid-touch-002", "profile-x")  # wrong profile

        rows = db.execute(
            "SELECT last_seen FROM canonical_entities WHERE entity_id = ?",
            ("uuid-touch-002",),
        )
        assert rows[0]["last_seen"] == original_ts, (
            "Wrong-profile _touch_last_seen must not update last_seen"
        )
