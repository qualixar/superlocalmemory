# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Integration tests for the Unified Daemon (Phase A).

Tests: single engine, daemon routes, backward compat, config, dual-port.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_slm_home(tmp_path):
    """Create a temporary SLM home directory."""
    slm_home = tmp_path / ".superlocalmemory"
    slm_home.mkdir()
    (slm_home / "logs").mkdir()
    return slm_home


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database with base schema."""
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    # Minimal schema for testing
    conn.execute("CREATE TABLE IF NOT EXISTS atomic_facts (fact_id TEXT, content TEXT, "
                 "confidence REAL, created_at TEXT, profile_id TEXT, "
                 "canonical_entities_json TEXT DEFAULT '[]', fact_type TEXT DEFAULT 'fact', "
                 "memory_id TEXT DEFAULT '')")
    conn.execute("CREATE TABLE IF NOT EXISTS canonical_entities (entity_id TEXT, "
                 "profile_id TEXT, canonical_name TEXT, entity_type TEXT DEFAULT 'unknown', "
                 "first_seen TEXT, last_seen TEXT, fact_count INTEGER DEFAULT 0)")
    conn.execute("CREATE TABLE IF NOT EXISTS fact_importance (fact_id TEXT PRIMARY KEY, "
                 "profile_id TEXT, pagerank_score REAL, community_id INTEGER, "
                 "degree_centrality REAL, computed_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS entity_profiles (profile_entry_id TEXT PRIMARY KEY, "
                 "entity_id TEXT, profile_id TEXT DEFAULT 'default', "
                 "knowledge_summary TEXT DEFAULT '', fact_ids_json TEXT DEFAULT '[]', "
                 "last_updated TEXT DEFAULT '')")
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Schema Migration Tests
# ---------------------------------------------------------------------------

class TestSchemaMigration:
    """Tests for schema_v343.py migration."""

    def test_migration_creates_mesh_tables(self, tmp_db):
        from superlocalmemory.storage.schema_v343 import apply_v343_schema
        result = apply_v343_schema(str(tmp_db))
        assert "mesh tables" in str(result["applied"])

        conn = sqlite3.connect(str(tmp_db))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()

        assert "mesh_peers" in tables
        assert "mesh_messages" in tables
        assert "mesh_state" in tables
        assert "mesh_locks" in tables
        assert "mesh_events" in tables

    def test_migration_creates_ingestion_log(self, tmp_db):
        from superlocalmemory.storage.schema_v343 import apply_v343_schema
        result = apply_v343_schema(str(tmp_db))

        conn = sqlite3.connect(str(tmp_db))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "ingestion_log" in tables

    def test_migration_alters_entity_profiles(self, tmp_db):
        from superlocalmemory.storage.schema_v343 import apply_v343_schema
        apply_v343_schema(str(tmp_db))

        conn = sqlite3.connect(str(tmp_db))
        cols = [r[1] for r in conn.execute("PRAGMA table_info(entity_profiles)").fetchall()]
        conn.close()

        assert "compiled_truth" in cols
        assert "timeline" in cols
        assert "project_name" in cols
        assert "compilation_confidence" in cols
        assert "last_compiled_at" in cols

    def test_migration_idempotent(self, tmp_db):
        from superlocalmemory.storage.schema_v343 import apply_v343_schema
        r1 = apply_v343_schema(str(tmp_db))
        r2 = apply_v343_schema(str(tmp_db))
        # Second call should skip
        assert "already applied" in str(r2.get("skipped", []))

    def test_migration_version_recorded(self, tmp_db):
        from superlocalmemory.storage.schema_v343 import apply_v343_schema
        apply_v343_schema(str(tmp_db))

        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute("SELECT version FROM schema_version WHERE version='3.4.3'").fetchone()
        conn.close()
        assert row is not None


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestConfigBackwardCompat:
    """Tests for config.py v3.4.3 additions."""

    def test_new_fields_have_defaults(self):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode
        config = SLMConfig.for_mode(Mode.A)
        assert config.daemon_idle_timeout == 0  # 24/7 default
        assert config.daemon_port == 8765
        assert config.daemon_legacy_port == 8767
        assert config.mesh_enabled is True
        assert config.entity_compilation_enabled is True
        assert config.entity_compilation_retrieval_boost == 1.0  # Off by default

    def test_old_config_loads_with_defaults(self, tmp_path):
        """A pre-3.4.3 config.json should load without error."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mode": "a", "active_profile": "default"}))

        from superlocalmemory.core.config import SLMConfig
        config = SLMConfig.load(config_file)
        assert config.daemon_idle_timeout == 0
        assert config.mesh_enabled is True


# ---------------------------------------------------------------------------
# Observe Buffer Tests
# ---------------------------------------------------------------------------

class TestObserveBuffer:
    """Tests for debounce observation buffer."""

    def test_buffer_deduplicates(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from superlocalmemory.server.unified_daemon import ObserveBuffer
        buf = ObserveBuffer(debounce_sec=60)  # Long debounce so flush doesn't auto-fire
        buf.set_engine(SimpleNamespace(_config=SimpleNamespace(scope=None), _profile_id="default"))
        decision = SimpleNamespace(capture=False, category="none", confidence=0.0, reason="not matched")
        with patch("superlocalmemory.hooks.auto_capture.AutoCapture") as auto:
            auto.return_value.evaluate.return_value = decision
            r1 = buf.enqueue("hello world")
            r2 = buf.enqueue("hello world")  # Duplicate
        assert r1["captured"] is False
        assert r1["reason"] == "not matched"
        assert r2["captured"] is False
        assert r2["reason"] == "duplicate within debounce window"

    def test_buffer_accepts_different_content(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from superlocalmemory.server.unified_daemon import ObserveBuffer
        buf = ObserveBuffer(debounce_sec=60)
        buf.set_engine(SimpleNamespace(_config=SimpleNamespace(scope=None), _profile_id="default"))
        decision = SimpleNamespace(capture=False, category="none", confidence=0.0, reason="not matched")
        with patch("superlocalmemory.hooks.auto_capture.AutoCapture") as auto:
            auto.return_value.evaluate.return_value = decision
            r1 = buf.enqueue("fact one")
            r2 = buf.enqueue("fact two")
        assert r1["reason"] == "not matched"
        assert r2["reason"] == "not matched"
