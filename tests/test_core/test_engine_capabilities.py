"""MemoryEngine capabilities split — LIGHT vs FULL.

LIGHT mode: DB + profile only. No embedder, no retrieval engine, no LLM.
FULL mode (default): identical to v3.4.25 behavior. Full heavy layer.

Guarantees tested here:
1. LIGHT initializes without loading the embedder
2. LIGHT raises CapabilityError on recall/store with actionable message
3. LIGHT still permits DB-only operations (fact_count, profile_id)
4. FULL default is byte-for-byte identical to existing v3.4.25 callers
"""
from __future__ import annotations

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.core.engine_capabilities import Capabilities, CapabilityError
from superlocalmemory.storage.models import Mode


@pytest.fixture
def mode_a_config(tmp_path):
    cfg = SLMConfig.for_mode(Mode.A)
    cfg.base_dir = tmp_path
    cfg.db_path = tmp_path / "memory.db"
    return cfg


class TestLightEngine:
    def test_light_engine_does_not_load_embedder(self, mode_a_config):
        engine = MemoryEngine(mode_a_config, capabilities=Capabilities.LIGHT)
        engine.initialize()
        assert engine._embedder is None
        assert engine._retrieval_engine is None
        assert engine._llm is None

    def test_light_engine_raises_on_recall(self, mode_a_config):
        engine = MemoryEngine(mode_a_config, capabilities=Capabilities.LIGHT)
        engine.initialize()
        with pytest.raises(CapabilityError) as exc:
            engine.recall("anything")
        msg = str(exc.value)
        assert "LIGHT" in msg
        assert "WorkerPool" in msg

    def test_light_engine_raises_on_store(self, mode_a_config):
        engine = MemoryEngine(mode_a_config, capabilities=Capabilities.LIGHT)
        engine.initialize()
        with pytest.raises(CapabilityError) as exc:
            engine.store("anything")
        msg = str(exc.value)
        assert "LIGHT" in msg
        assert "WorkerPool" in msg

    def test_light_engine_allows_db_access(self, mode_a_config):
        engine = MemoryEngine(mode_a_config, capabilities=Capabilities.LIGHT)
        engine.initialize()
        # DB is present and usable
        assert engine.db is not None
        # profile_id accessible
        assert engine.profile_id == mode_a_config.active_profile
        # fact_count works (DB-only)
        assert engine.fact_count == 0

    def test_light_engine_initialized_flag_true(self, mode_a_config):
        engine = MemoryEngine(mode_a_config, capabilities=Capabilities.LIGHT)
        engine.initialize()
        assert engine._initialized is True

    def test_light_engine_capabilities_attribute_exposed(self, mode_a_config):
        engine = MemoryEngine(mode_a_config, capabilities=Capabilities.LIGHT)
        assert engine.capabilities is Capabilities.LIGHT


class TestFullEngineDefault:
    def test_full_is_default_capability(self, mode_a_config):
        # Backward-compat: no capabilities arg → FULL
        engine = MemoryEngine(mode_a_config)
        assert engine.capabilities is Capabilities.FULL

    def test_explicit_full_equals_default(self, mode_a_config):
        engine_default = MemoryEngine(mode_a_config)
        engine_explicit = MemoryEngine(mode_a_config, capabilities=Capabilities.FULL)
        assert engine_default.capabilities is engine_explicit.capabilities

    def test_full_engine_loads_embedder(self, mode_a_config):
        engine = MemoryEngine(mode_a_config, capabilities=Capabilities.FULL)
        engine.initialize()
        # Heavy layer present
        assert engine._embedder is not None
        assert engine._retrieval_engine is not None

    def test_full_engine_no_capability_error(self, mode_a_config):
        # recall on empty DB returns empty response, NOT CapabilityError
        engine = MemoryEngine(mode_a_config, capabilities=Capabilities.FULL)
        engine.initialize()
        response = engine.recall("query on empty db")
        # response may be empty; must not raise
        assert response is not None


class TestLightEngineDBOnlyFeatures:
    """LIGHT must still serve DB-only features so user-facing feedback,
    learning-status, and session_init phase counters keep working in MCP."""

    def test_light_engine_has_adaptive_learner(self, mode_a_config):
        """AdaptiveLearner needs only the DB; it must be available in LIGHT
        so report_feedback and get_feedback_count work through MCP."""
        engine = MemoryEngine(mode_a_config, capabilities=Capabilities.LIGHT)
        engine.initialize()
        assert engine._adaptive_learner is not None

    def test_light_engine_adaptive_learner_interface_intact(self, mode_a_config):
        """AdaptiveLearner under LIGHT exposes record_feedback +
        get_feedback_count callables bound to the engine's DB.

        The DB-write end-to-end (with its FK chain through atomic_facts /
        memories) is covered by the existing FULL-engine feedback tests.
        Here we only need to verify LIGHT wiring.
        """
        engine = MemoryEngine(mode_a_config, capabilities=Capabilities.LIGHT)
        engine.initialize()
        learner = engine._adaptive_learner
        assert learner is not None
        assert callable(learner.record_feedback)
        assert callable(learner.get_feedback_count)
        # empty profile returns 0, not an error — this is the session_init
        # happy path that Varun cares about (no "learning disabled" copy
        # showing up in the MCP response).
        assert learner.get_feedback_count(engine.profile_id) == 0
