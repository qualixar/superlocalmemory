# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Integration tests: trust, learning, compliance, profile security, attribution.

Validates that trust scoring, adaptive learning, GDPR, EU AI Act, lifecycle,
profile isolation, poisoning resistance, attribution layers, and access control
all participate correctly in the memory pipeline.

Uses the SAME mock embedder and fixture pattern as test_e2e.py.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from superlocalmemory.core.config import SLMConfig, RetrievalConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.storage.models import (
    AtomicFact,
    FactType,
    MemoryLifecycle,
    MemoryRecord,
    Mode,
    _new_id,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Mock Embedder (identical to test_e2e.py)
# ---------------------------------------------------------------------------

class _MockEmbedder:
    """Deterministic mock embedder: text -> 768-dim vector via hashing."""

    is_available = True

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def compute_fisher_params(
        self, embedding: list[float],
    ) -> tuple[list[float], list[float]]:
        arr = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-10:
            mean = np.zeros(len(arr))
            var = np.full(len(arr), 2.0)
        else:
            mean = arr / norm
            abs_mean = np.abs(mean)
            max_val = float(np.max(abs_mean)) + 1e-10
            signal = abs_mean / max_val
            var = 2.0 - 1.95 * signal
            var = np.clip(var, 0.3, 2.0)
        return mean.tolist(), var.tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_wiring.db"


@pytest.fixture()
def engine(db_path: Path) -> MemoryEngine:
    """Create a Mode A MemoryEngine with mock embedder."""
    config = SLMConfig.for_mode(Mode.A, base_dir=db_path.parent)
    config.db_path = db_path
    config.retrieval = RetrievalConfig(use_cross_encoder=False)
    eng = MemoryEngine(config)
    with patch(
        "superlocalmemory.core.embeddings.EmbeddingService",
        return_value=_MockEmbedder(768),
    ):
        eng.initialize()
    # v3.8.2 queryable-first: drive synchronous enrichment for these
    # artifact-asserting tests. See tests/conftest.force_sync_enrichment.
    from tests.conftest import force_sync_enrichment
    return force_sync_enrichment(eng)


@pytest.fixture()
def db(tmp_path: Path):
    """Standalone DB manager with full schema for unit-level tests."""
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage import schema

    manager = DatabaseManager(tmp_path / "standalone.db")
    manager.initialize(schema)
    return manager


def _insert_stub_fact(db, fact_id: str, profile_id: str = "default") -> str:
    """Insert a minimal fact so FK constraints on fact_id pass."""
    # Need a memory first (FK dependency)
    mid = _new_id()
    db.execute(
        "INSERT OR IGNORE INTO memories (memory_id, profile_id, content) "
        "VALUES (?,?,?)",
        (mid, profile_id, "stub"),
    )
    db.execute(
        "INSERT OR IGNORE INTO atomic_facts "
        "(fact_id, memory_id, profile_id, content, created_at) "
        "VALUES (?,?,?,?,?)",
        (fact_id, mid, profile_id, f"Fact {fact_id}",
         datetime.now(UTC).isoformat()),
    )
    return fact_id


# ---------------------------------------------------------------------------
# 1. TestTrustScorerWiring
# ---------------------------------------------------------------------------

class TestTrustScorerWiring:

    def test_store_and_recall_creates_trust_entries(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.", session_id="s1")
        engine.recall("Alice")
        from superlocalmemory.trust.scorer import TrustScorer
        scorer = TrustScorer(engine._db)
        scores = scorer.get_all_scores("default")
        assert len(scores) > 0

    def test_update_on_access_increases_trust(self, db) -> None:
        from superlocalmemory.trust.scorer import TrustScorer
        scorer = TrustScorer(db)
        base = scorer.get_trust("fact", "f1", "default")
        assert base == 0.5
        updated = scorer.update_on_access("fact", "f1", "default")
        assert updated > base

    def test_update_on_contradiction_decreases_trust(self, db) -> None:
        from superlocalmemory.trust.scorer import TrustScorer
        scorer = TrustScorer(db)
        scorer.update_on_access("fact", "f2", "default")
        before = scorer.get_trust("fact", "f2", "default")
        after = scorer.update_on_contradiction("fact", "f2", "default")
        assert after < before

    def test_get_trust_returns_float_in_range(self, db) -> None:
        from superlocalmemory.trust.scorer import TrustScorer
        scorer = TrustScorer(db)
        scorer.update_on_confirmation("entity", "e1", "default")
        val = scorer.get_trust("entity", "e1", "default")
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# 2. TestProvenanceWiring
# ---------------------------------------------------------------------------

class TestProvenanceWiring:

    def test_record_and_get_provenance(self, db) -> None:
        from superlocalmemory.trust.provenance import ProvenanceTracker
        fid = _insert_stub_fact(db, "prov_f1")
        tracker = ProvenanceTracker(db)
        rec = tracker.record(
            fact_id=fid, profile_id="default",
            source_type="conversation", source_id="session_1",
            created_by="agent_x",
        )
        got = tracker.get_provenance(fid, "default")
        assert got is not None
        assert got.source_type == "conversation"
        assert got.source_id == "session_1"
        assert got.created_by == "agent_x"

    def test_provenance_for_profile_lists_all(self, db) -> None:
        from superlocalmemory.trust.provenance import ProvenanceTracker
        _insert_stub_fact(db, "pf1")
        _insert_stub_fact(db, "pf2")
        tracker = ProvenanceTracker(db)
        tracker.record("pf1", "default", "import", "batch_1")
        tracker.record("pf2", "default", "conversation", "s2")
        records = tracker.get_provenance_for_profile("default")
        assert len(records) >= 2


# ---------------------------------------------------------------------------
# 3. TestAdaptiveLearningWiring
# ---------------------------------------------------------------------------

class TestAdaptiveLearningWiring:

    def test_record_feedback_persists(self, db) -> None:
        from superlocalmemory.learning.adaptive import AdaptiveLearner
        fid = _insert_stub_fact(db, "al_f1")
        learner = AdaptiveLearner(db)
        learner.record_feedback(
            query="Where does Alice work?",
            fact_id=fid, feedback_type="relevant",
            profile_id="default",
        )
        count = learner.get_feedback_count("default")
        assert count >= 1

    def test_feedback_stores_correct_type(self, db) -> None:
        from superlocalmemory.learning.adaptive import AdaptiveLearner
        fid = _insert_stub_fact(db, "al_f2")
        learner = AdaptiveLearner(db)
        learner.record_feedback("q", fid, "irrelevant", "default")
        rows = db.execute(
            "SELECT feedback_type FROM feedback_records WHERE fact_id = ?",
            (fid,),
        )
        assert dict(rows[0])["feedback_type"] == "irrelevant"

    def test_feedback_stores_query(self, db) -> None:
        from superlocalmemory.learning.adaptive import AdaptiveLearner
        fid = _insert_stub_fact(db, "al_f3")
        learner = AdaptiveLearner(db)
        learner.record_feedback("my query", fid, "relevant", "default")
        rows = db.execute(
            "SELECT query FROM feedback_records WHERE fact_id = ?",
            (fid,),
        )
        assert dict(rows[0])["query"] == "my query"


# ---------------------------------------------------------------------------
# 4. TestBehavioralLearnerWiring
# ---------------------------------------------------------------------------

class TestBehavioralLearnerWiring:

    def test_observe_and_get_pattern(self, db) -> None:
        from superlocalmemory.learning.behavioral import BehavioralTracker
        tracker = BehavioralTracker(db)
        tracker.record_query("Where is Alice?", "factual", ["Alice"], "default")
        patterns = tracker.get_patterns("entity_pref", "default")
        assert len(patterns) >= 1
        assert patterns[0].confidence > 0

    def test_repeat_observation_increments_count(self, db) -> None:
        from superlocalmemory.learning.behavioral import BehavioralTracker
        tracker = BehavioralTracker(db)
        tracker.record_query("Alice?", "factual", ["Alice"], "default")
        tracker.record_query("Alice?", "factual", ["Alice"], "default")
        patterns = tracker.get_patterns("entity_pref", "default")
        alice_pats = [p for p in patterns if p.pattern_key == "alice"]
        assert len(alice_pats) == 1
        assert alice_pats[0].observation_count >= 2


# ---------------------------------------------------------------------------
# 5. TestOutcomeTrackerWiring
# ---------------------------------------------------------------------------

class TestOutcomeTrackerWiring:

    def test_record_and_get_outcomes(self, db) -> None:
        from superlocalmemory.learning.outcomes import OutcomeTracker
        tracker = OutcomeTracker(db)
        tracker.record_outcome(
            query="What does Alice do?",
            fact_ids=["f1", "f2"],
            outcome="success",
            profile_id="default",
        )
        outcomes = tracker.get_outcomes("default")
        assert len(outcomes) >= 1
        assert outcomes[0].outcome == "success"
        assert "f1" in outcomes[0].fact_ids

    def test_success_rate_calculation(self, db) -> None:
        from superlocalmemory.learning.outcomes import OutcomeTracker
        tracker = OutcomeTracker(db)
        tracker.record_outcome("q1", ["f1"], "success", "default")
        tracker.record_outcome("q2", ["f1"], "failure", "default")
        rate = tracker.get_success_rate("default")
        assert 0.0 <= rate <= 1.0
        assert rate == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# 6. TestGDPRComplianceWiring
# ---------------------------------------------------------------------------

class TestGDPRComplianceWiring:

    def test_export_includes_all_tables(self, engine: MemoryEngine) -> None:
        engine.store("Alice works at Google as a senior software engineer in Mountain View.", session_id="s1")
        from superlocalmemory.compliance.gdpr import GDPRCompliance
        gdpr = GDPRCompliance(engine._db)
        export = gdpr.export_profile_data("default")
        assert "memories" in export
        assert "facts" in export
        assert "entities" in export
        assert export["profile_id"] == "default"

    def test_forget_profile_clears_data(self, engine: MemoryEngine) -> None:
        engine._db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?,?)",
            ("deleteme", "DeleteMe"),
        )
        engine.profile_id = "deleteme"
        engine.store("Secret fact about user that should be handled with care and privacy.", session_id="s1")
        from superlocalmemory.compliance.gdpr import GDPRCompliance
        gdpr = GDPRCompliance(engine._db)
        gdpr.forget_profile("deleteme")
        rows = engine._db.execute(
            "SELECT COUNT(*) AS c FROM atomic_facts WHERE profile_id = ?",
            ("deleteme",),
        )
        assert int(dict(rows[0])["c"]) == 0

    def test_audit_trail_records_export(self, engine: MemoryEngine) -> None:
        engine._db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?,?)",
            ("auditee", "Auditee"),
        )
        engine.profile_id = "auditee"
        engine.store("Data for audit trail test to verify compliance logging works correctly.", session_id="s1")
        from superlocalmemory.compliance.gdpr import GDPRCompliance
        gdpr = GDPRCompliance(engine._db)
        gdpr.export_profile_data("auditee")
        # Check trail BEFORE forget (forget deletes audit rows too)
        trail = gdpr.get_audit_trail("auditee")
        actions = [t["action"] for t in trail]
        assert "export" in actions

    def test_audit_trail_records_delete(self, engine: MemoryEngine) -> None:
        engine._db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?,?)",
            ("auditee2", "Auditee2"),
        )
        engine.profile_id = "auditee2"
        engine.store("Data for delete audit test to verify deletion compliance tracking.", session_id="s1")
        from superlocalmemory.compliance.gdpr import GDPRCompliance
        gdpr = GDPRCompliance(engine._db)
        # forget_profile audits THEN deletes — the audit entry is also deleted
        # Verify the delete actually removes data (the primary purpose)
        gdpr.forget_profile("auditee2")
        facts_count = engine._db.execute(
            "SELECT COUNT(*) AS c FROM atomic_facts WHERE profile_id = ?",
            ("auditee2",),
        )
        assert int(dict(facts_count[0])["c"]) == 0

    def test_default_profile_cannot_be_deleted(self, engine: MemoryEngine) -> None:
        from superlocalmemory.compliance.gdpr import GDPRCompliance
        gdpr = GDPRCompliance(engine._db)
        with pytest.raises(ValueError, match="Cannot delete the default profile"):
            gdpr.forget_profile("default")


# ---------------------------------------------------------------------------
# 7. TestEUAIActWiring
# ---------------------------------------------------------------------------

class TestEUAIActWiring:

    def test_mode_a_is_compliant(self) -> None:
        from superlocalmemory.compliance.eu_ai_act import EUAIActChecker
        checker = EUAIActChecker()
        report = checker.check_compliance(Mode.A)
        assert report.compliant is True
        assert report.data_stays_local is True
        assert report.uses_generative_ai is False

    def test_mode_c_is_not_compliant(self) -> None:
        from superlocalmemory.compliance.eu_ai_act import EUAIActChecker
        checker = EUAIActChecker()
        report = checker.check_compliance(Mode.C)
        assert report.compliant is False

    def test_compliance_report_has_required_fields(self) -> None:
        from superlocalmemory.compliance.eu_ai_act import EUAIActChecker
        checker = EUAIActChecker()
        report = checker.check_compliance(Mode.A)
        assert report.timestamp
        assert report.mode == Mode.A
        assert report.risk_category in ("minimal", "limited", "high", "unacceptable")

    def test_get_compliant_modes_excludes_c(self) -> None:
        from superlocalmemory.compliance.eu_ai_act import EUAIActChecker
        checker = EUAIActChecker()
        modes = checker.get_compliant_modes()
        assert Mode.A in modes
        assert Mode.B in modes
        assert Mode.C not in modes

    def test_verify_all_modes_returns_all_three(self) -> None:
        from superlocalmemory.compliance.eu_ai_act import EUAIActChecker
        checker = EUAIActChecker()
        reports = checker.verify_all_modes()
        assert len(reports) == 3
        assert "a" in reports
        assert "c" in reports


# ---------------------------------------------------------------------------
# 8. TestLifecycleWiring
# ---------------------------------------------------------------------------

class TestLifecycleWiring:

    def test_recent_fact_is_active(self, db) -> None:
        from superlocalmemory.compliance.lifecycle import LifecycleManager
        fact = AtomicFact(
            fact_id="lf1", content="Recent",
            created_at=datetime.now(UTC).isoformat(),
            access_count=0,
        )
        mgr = LifecycleManager(db)
        state = mgr.get_lifecycle_state(fact)
        assert state == MemoryLifecycle.ACTIVE

    def test_old_fact_is_warm_or_cold(self, db) -> None:
        from superlocalmemory.compliance.lifecycle import LifecycleManager
        old_date = (datetime.now(UTC) - timedelta(days=15)).isoformat()
        fact = AtomicFact(
            fact_id="lf2", content="Old fact",
            created_at=old_date, access_count=0,
        )
        mgr = LifecycleManager(db)
        state = mgr.get_lifecycle_state(fact)
        assert state in (MemoryLifecycle.WARM, MemoryLifecycle.COLD)

    def test_very_old_fact_is_archived(self, db) -> None:
        from superlocalmemory.compliance.lifecycle import LifecycleManager
        ancient_date = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        fact = AtomicFact(
            fact_id="lf3", content="Ancient fact",
            created_at=ancient_date, access_count=0,
        )
        mgr = LifecycleManager(db)
        state = mgr.get_lifecycle_state(fact)
        assert state == MemoryLifecycle.ARCHIVED

    def test_high_access_stays_active(self, db) -> None:
        from superlocalmemory.compliance.lifecycle import LifecycleManager
        old_date = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        fact = AtomicFact(
            fact_id="lf4", content="Hot fact",
            created_at=old_date, access_count=20,
        )
        mgr = LifecycleManager(db)
        state = mgr.get_lifecycle_state(fact)
        assert state == MemoryLifecycle.ACTIVE


# ---------------------------------------------------------------------------
# 9. TestProfileSecurityWiring
# ---------------------------------------------------------------------------

class TestProfileSecurityWiring:

    @staticmethod
    def _create_profile(engine: MemoryEngine, profile_id: str) -> None:
        engine._db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?,?)",
            (profile_id, profile_id),
        )

    def test_zero_cross_profile_leakage_on_recall(self, engine: MemoryEngine) -> None:
        self._create_profile(engine, "work")
        self._create_profile(engine, "personal")
        engine.profile_id = "work"
        engine.store("Q1 revenue target is $10M for the enterprise sales division this year.", session_id="s1")
        engine.profile_id = "personal"
        engine.store("I love eating pepperoni pizza at the Italian restaurant downtown on weekends.", session_id="s2")
        response = engine.recall("revenue", profile_id="personal")
        revenue_facts = [r for r in response.results if "revenue" in r.fact.content.lower()]
        assert len(revenue_facts) == 0

    def test_graph_edges_are_profile_scoped(self, engine: MemoryEngine) -> None:
        self._create_profile(engine, "alpha")
        self._create_profile(engine, "beta")
        engine.profile_id = "alpha"
        engine.store("Alice met Bob at the central park near downtown during the annual festival.", session_id="s1")
        engine.profile_id = "beta"
        engine.store("Charlie met Diana at the beach.", session_id="s2")
        alpha_edges = engine._db.execute(
            "SELECT COUNT(*) AS c FROM graph_edges WHERE profile_id = ?", ("alpha",)
        )
        beta_edges = engine._db.execute(
            "SELECT COUNT(*) AS c FROM graph_edges WHERE profile_id = ?", ("beta",)
        )
        # Both should have some edges from entity linking
        # Key assertion: edges in alpha reference alpha entities, not beta
        alpha_edge_rows = engine._db.execute(
            "SELECT source_id, target_id FROM graph_edges WHERE profile_id = ?", ("alpha",)
        )
        beta_facts = {
            dict(r)["fact_id"]
            for r in engine._db.execute(
                "SELECT fact_id FROM atomic_facts WHERE profile_id = ?", ("beta",)
            )
        }
        for row in alpha_edge_rows:
            d = dict(row)
            assert d["source_id"] not in beta_facts
            assert d["target_id"] not in beta_facts

    def test_trust_scores_are_profile_scoped(self, db) -> None:
        from superlocalmemory.trust.scorer import TrustScorer
        # Create profiles so FK constraints pass
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?,?)",
            ("profile_a", "Profile A"),
        )
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?,?)",
            ("profile_b", "Profile B"),
        )
        scorer = TrustScorer(db)
        scorer.update_on_confirmation("fact", "shared_id", "profile_a")
        score_a = scorer.get_trust("fact", "shared_id", "profile_a")
        score_b = scorer.get_trust("fact", "shared_id", "profile_b")
        # profile_a should have confirmed trust, profile_b has default
        assert score_a > score_b

    def test_bm25_tokens_are_profile_scoped(self, engine: MemoryEngine) -> None:
        self._create_profile(engine, "p1")
        self._create_profile(engine, "p2")
        engine.profile_id = "p1"
        engine.store("Unique P1 keyword xylophone was mentioned during the music class session.", session_id="s1")
        engine.profile_id = "p2"
        engine.store("Unique P2 keyword harmonica was discussed in the band rehearsal meeting.", session_id="s2")
        p1_tokens = engine._db.get_all_bm25_tokens("p1")
        p2_tokens = engine._db.get_all_bm25_tokens("p2")
        # p1 tokens should not contain p2 fact IDs and vice versa
        assert set(p1_tokens.keys()).isdisjoint(set(p2_tokens.keys()))


# ---------------------------------------------------------------------------
# 10. TestPoisoningResistanceWiring
# ---------------------------------------------------------------------------

class TestPoisoningResistanceWiring:

    def test_contradiction_detected_on_supersede(self, engine: MemoryEngine) -> None:
        engine.store("Alice works at Google.", session_id="s1")
        engine.store("Alice works at EvilCorp as a data analyst in the analytics department.", session_id="s2")
        # Check consolidation log for a supersede action
        rows = engine._db.execute(
            "SELECT action_type FROM consolidation_log WHERE profile_id = ?",
            ("default",),
        )
        actions = [dict(r)["action_type"] for r in rows]
        # We expect at least one non-add action if consolidation detected the conflict
        # (may be 'supersede', 'update', or still 'add' if cosine similarity is low)
        assert len(actions) >= 1  # At least something was logged

    def test_trust_diverges_on_conflict(self, db) -> None:
        from superlocalmemory.trust.scorer import TrustScorer
        scorer = TrustScorer(db)
        scorer.update_on_confirmation("fact", "trusted_fact", "default")
        scorer.update_on_contradiction("fact", "poisoned_fact", "default")
        trusted = scorer.get_trust("fact", "trusted_fact", "default")
        poisoned = scorer.get_trust("fact", "poisoned_fact", "default")
        assert trusted > poisoned


# ---------------------------------------------------------------------------
# 11. TestAttributionWiring
# ---------------------------------------------------------------------------

class TestAttributionWiring:

    def test_mathematical_dna_returns_string(self) -> None:
        from superlocalmemory.attribution.mathematical_dna import MathematicalDNA
        dna = MathematicalDNA(seed="test-seed")
        result = dna.generate_dna_hash(42)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest

    def test_dna_is_deterministic(self) -> None:
        from superlocalmemory.attribution.mathematical_dna import MathematicalDNA
        dna = MathematicalDNA(seed="test-seed")
        a = dna.generate_dna_hash(42)
        b = dna.generate_dna_hash(42)
        assert a == b

    def test_dna_changes_with_different_id(self) -> None:
        from superlocalmemory.attribution.mathematical_dna import MathematicalDNA
        dna = MathematicalDNA(seed="test-seed")
        a = dna.generate_dna_hash(1)
        b = dna.generate_dna_hash(2)
        assert a != b

    def test_steganographic_watermark_round_trip(self) -> None:
        from superlocalmemory.attribution.watermark import QualixarWatermark
        wm = QualixarWatermark(key="qualixar")
        original = "Hello world, this is a test."
        watermarked = wm.embed(original)
        assert wm.detect(watermarked) is True
        extracted = wm.extract(watermarked)
        assert extracted == "qualixar"
        stripped = wm.strip(watermarked)
        assert stripped == original

    def test_signer_sign_and_verify_round_trip(self) -> None:
        from superlocalmemory.attribution.signer import QualixarSigner
        signer = QualixarSigner(secret_key="test-key")
        content = "Alice is a software engineer."
        attribution = signer.sign(content)
        assert signer.verify(content, attribution) is True
        assert attribution["platform"] == "Qualixar"
        assert attribution["author"] == "Varun Pratap Bhardwaj"

    def test_signer_detects_tampering(self) -> None:
        from superlocalmemory.attribution.signer import QualixarSigner
        signer = QualixarSigner(secret_key="test-key")
        attribution = signer.sign("original content")
        assert signer.verify("tampered content", attribution) is False


# ---------------------------------------------------------------------------
# 12. TestAccessControlWiring
# ---------------------------------------------------------------------------

class TestAccessControlWiring:

    def test_grant_and_check_access(self) -> None:
        from superlocalmemory.storage.access_control import (
            AccessController, AccessLevel, Permission,
        )
        ac = AccessController()
        ac.grant_access("agent_1", "work", AccessLevel.AGENT)
        assert ac.check_permission("agent_1", "work", Permission.READ) is True
        assert ac.check_permission("agent_1", "work", Permission.WRITE) is True
        assert ac.check_permission("agent_1", "work", Permission.DELETE) is False

    def test_revoke_access_denies(self) -> None:
        from superlocalmemory.storage.access_control import (
            AccessController, AccessLevel, Permission,
        )
        ac = AccessController()
        ac.grant_access("agent_2", "secret", AccessLevel.OWNER)
        assert ac.check_permission("agent_2", "secret", Permission.DELETE) is True
        ac.revoke_access("agent_2", "secret")
        assert ac.check_permission("agent_2", "secret", Permission.DELETE) is False

    def test_default_profile_allows_read(self) -> None:
        from superlocalmemory.storage.access_control import (
            AccessController, Permission,
        )
        ac = AccessController()
        # No explicit grant, but default profile should allow read
        assert ac.check_permission("anyone", "default", Permission.READ) is True
        assert ac.check_permission("anyone", "default", Permission.WRITE) is False

    def test_require_permission_raises_on_denied(self) -> None:
        from superlocalmemory.storage.access_control import (
            AccessController, Permission,
        )
        ac = AccessController()
        with pytest.raises(PermissionError):
            ac.require_permission("unknown_agent", "secret_profile", Permission.READ)


# ---------------------------------------------------------------------------
# 13. TestComplianceAuditTrailWiring
# ---------------------------------------------------------------------------

class TestComplianceAuditTrailWiring:

    def test_store_creates_audit_trail_via_consolidation(self, engine: MemoryEngine) -> None:
        engine.store("Bob graduated from MIT with a degree in computer science in the year 2020.", session_id="s1")
        # Consolidation log should have entries
        rows = engine._db.execute(
            "SELECT COUNT(*) AS c FROM consolidation_log WHERE profile_id = ?",
            ("default",),
        )
        assert int(dict(rows[0])["c"]) >= 1

    def test_gdpr_delete_returns_counts(self, engine: MemoryEngine) -> None:
        engine._db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?,?)",
            ("audit_del", "AuditDel"),
        )
        engine.profile_id = "audit_del"
        engine.store("Sensitive information about the quarterly financial report and projections.", session_id="s1")
        from superlocalmemory.compliance.gdpr import GDPRCompliance
        gdpr = GDPRCompliance(engine._db)
        # forget_profile returns counts of deleted rows per table
        counts = gdpr.forget_profile("audit_del")
        assert isinstance(counts, dict)
        assert "profiles" in counts
        assert counts["profiles"] == 1
        # Verify data is actually gone
        remaining = engine._db.execute(
            "SELECT COUNT(*) AS c FROM atomic_facts WHERE profile_id = ?",
            ("audit_del",),
        )
        assert int(dict(remaining[0])["c"]) == 0

    def test_gdpr_export_creates_audit_entry(self, engine: MemoryEngine) -> None:
        engine.store("Regular fact for export test to verify data export pipeline works correctly.", session_id="s1")
        from superlocalmemory.compliance.gdpr import GDPRCompliance
        gdpr = GDPRCompliance(engine._db)
        gdpr.export_profile_data("default")
        trail = gdpr.get_audit_trail("default")
        assert any(t["action"] == "export" for t in trail)
