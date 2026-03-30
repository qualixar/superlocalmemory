# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for MemoryEngine session lifecycle — init, close, profile switching.

Covers:
  - __init__ sets _config, _caps, _initialized=False
  - initialize() transitions _initialized to True
  - initialize() is idempotent (second call is no-op)
  - _ensure_init() triggers initialize() on first call
  - close_session() creates temporal_events for session facts
  - close_session() returns 0 when no facts exist for session
  - close() resets _initialized to False
  - profile_id getter returns _profile_id
  - profile_id setter updates _profile_id
  - fact_count delegates to _db.get_fact_count
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.core.modes import get_capabilities
from superlocalmemory.storage.models import AtomicFact, Mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(
    fact_id: str,
    session_id: str = "s1",
    content: str = "",
    canonical_entities: list[str] | None = None,
    observation_date: str | None = None,
) -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id,
        memory_id="m0",
        content=content or f"fact {fact_id}",
        session_id=session_id,
        canonical_entities=canonical_entities or [],
        observation_date=observation_date,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    """Verify __init__, initialize(), and lazy init behavior."""

    def test_init_sets_config_and_caps(self, mode_a_config: SLMConfig) -> None:
        """__init__ sets _config and _caps from the supplied config."""
        engine = MemoryEngine(mode_a_config)
        assert engine._config is mode_a_config
        expected_caps = get_capabilities(Mode.A)
        assert engine._caps.mode == expected_caps.mode

    def test_init_not_initialized_by_default(self, mode_a_config: SLMConfig) -> None:
        """Engine is NOT initialized right after __init__."""
        engine = MemoryEngine(mode_a_config)
        assert engine._initialized is False

    def test_initialize_sets_initialized_true(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """After initialize(), _initialized is True."""
        assert engine_with_mock_deps._initialized is True

    def test_initialize_idempotent(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """Calling initialize() a second time is a no-op (early return)."""
        # Record current DB ref
        db_ref = engine_with_mock_deps._db
        engine_with_mock_deps.initialize()
        # DB reference should be unchanged (no re-init)
        assert engine_with_mock_deps._db is db_ref
        assert engine_with_mock_deps._initialized is True

    def test_ensure_init_triggers_initialize(self, mode_a_config: SLMConfig, mock_embedder: MagicMock) -> None:
        """_ensure_init() calls initialize() when _initialized is False."""
        engine = MemoryEngine(mode_a_config)
        assert engine._initialized is False
        with patch('superlocalmemory.core.engine_wiring.init_embedder', return_value=mock_embedder):
            engine._ensure_init()
            engine._embedder = mock_embedder
        assert engine._initialized is True


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    """Verify close_session() and close() behavior."""

    def test_close_session_creates_temporal_events(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """close_session() creates temporal_events rows grouped by entity."""
        from superlocalmemory.storage.models import MemoryRecord

        db = engine_with_mock_deps._db
        pid = engine_with_mock_deps._profile_id

        # Ensure the profile row exists (FK: canonical_entities, temporal_events)
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
            (pid, pid),
        )

        # Create canonical entity (FK: temporal_events.entity_id)
        db.execute(
            "INSERT OR IGNORE INTO canonical_entities "
            "(entity_id, profile_id, canonical_name) VALUES (?, ?, ?)",
            ("ent-alice", pid, "Alice"),
        )

        # Create parent memory record (FK: atomic_facts.memory_id)
        record = MemoryRecord(
            memory_id="m-sess1", profile_id=pid,
            content="Alice went to Paris", session_id="sess-1",
        )
        db.store_memory(record)

        # Store a fact with canonical entity so close_session can find it
        fact = _make_fact(
            "f1", session_id="sess-1",
            content="Alice went to Paris",
            canonical_entities=["ent-alice"],
            observation_date="2026-03-01",
        )
        fact.memory_id = "m-sess1"
        fact.profile_id = pid
        db.store_fact(fact)

        count = engine_with_mock_deps.close_session("sess-1")
        assert count >= 1

    def test_close_session_no_facts_returns_zero(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """close_session() returns 0 when no facts exist for the session."""
        count = engine_with_mock_deps.close_session("nonexistent-session")
        assert count == 0

    def test_close_resets_initialized(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """close() sets _initialized to False."""
        assert engine_with_mock_deps._initialized is True
        engine_with_mock_deps.close()
        assert engine_with_mock_deps._initialized is False


# ---------------------------------------------------------------------------
# Profile management
# ---------------------------------------------------------------------------

class TestProfileManagement:
    """Verify profile_id property and fact_count."""

    def test_profile_id_getter(self, mode_a_config: SLMConfig) -> None:
        """profile_id property returns _profile_id."""
        engine = MemoryEngine(mode_a_config)
        assert engine.profile_id == mode_a_config.active_profile

    def test_profile_id_setter(self, mode_a_config: SLMConfig) -> None:
        """profile_id setter updates _profile_id."""
        engine = MemoryEngine(mode_a_config)
        engine.profile_id = "custom-profile"
        assert engine.profile_id == "custom-profile"
        assert engine._profile_id == "custom-profile"

    def test_fact_count_queries_db(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """fact_count property delegates to _db.get_fact_count()."""
        # Initially 0 facts
        count = engine_with_mock_deps.fact_count
        assert isinstance(count, int)
        assert count >= 0
