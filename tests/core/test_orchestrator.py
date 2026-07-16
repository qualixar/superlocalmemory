# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for BackendOrchestrator — Sprint 4."""

from __future__ import annotations

import pytest
import sqlite3
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, PropertyMock

from superlocalmemory.core.backend_orchestrator import (
    BackendOrchestrator,
    get_orchestrator,
    set_orchestrator,
)


class MockConfig:
    def __init__(self, **kwargs):
        self.data_dir = kwargs.get("data_dir", "/tmp/test_slm")
        self._cfg = kwargs

    def get(self, key, default=None):
        return self._cfg.get(key, default)


class MockDB:
    """Faithful double of DatabaseManager's surface used by BackendOrchestrator.

    The real DatabaseManager has NO `.conn` attribute — it exposes execute()
    and raw_connection(). The old mock exposed `.conn` (and nothing else),
    which is exactly why issue #47 escaped: the test validated the buggy
    `self._db.conn` access that fails in production. This mock now mirrors the
    real contract.
    """

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS backend_status (
                backend_name TEXT PRIMARY KEY,
                status TEXT DEFAULT 'not_initialized',
                record_count INTEGER DEFAULT 0,
                last_sync_at TEXT,
                error_message TEXT DEFAULT ''
            )
        """)
        self._conn.commit()

    def execute(self, sql, params=()):
        rows = self._conn.execute(sql, params).fetchall()
        self._conn.commit()
        return rows

    @contextmanager
    def raw_connection(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self):
        self._conn.close()


@pytest.fixture
def orch():
    config = MockConfig(data_dir="/tmp/test_slm_orch")
    db = MockDB()
    orch = BackendOrchestrator(config, db)
    set_orchestrator(orch)
    yield orch
    db.close()


class TestLifecycle:
    def test_orchestrator_singleton(self, orch):
        assert get_orchestrator() is orch

    def test_daemon_start_no_backends(self, orch):
        """Daemon starts without CozoDB or LanceDB installed."""
        with patch.object(orch, "_detect_cozo", return_value=False):
            with patch.object(orch, "_detect_lancedb", return_value=False):
                orch.on_daemon_start()
                assert orch.get_graph_backend() is None
                assert orch.get_vector_backend() is None

    def test_health_check_always_works(self, orch):
        """Health check returns valid structure even without backends."""
        result = orch.health_check()
        assert "sqlite" in result
        assert "cozo" in result
        assert "lancedb" in result
        assert "tiers" in result
        assert "warnings" in result
        assert result["cozo"]["status"] == "not_available"

    def test_status_tracking(self, orch):
        """Backend status is tracked in cache and SQLite."""
        orch._update_status("cozo", "active", 100)
        assert orch._cozo_status() == "active"

        rows = orch._db.execute(
            "SELECT status, record_count FROM backend_status WHERE backend_name = 'cozo'"
        )
        assert rows[0][0] == "active"
        assert rows[0][1] == 100


class TestIncrementalSync:
    def test_sync_new_fact_no_backends(self, orch):
        """sync_new_fact is safe when no backends are active."""
        fact = MagicMock()
        fact.fact_id = "test-1"
        fact.lifecycle = "active"
        orch.sync_new_fact(fact)  # Should not raise

    def test_sync_new_fact_with_cozo(self, orch):
        """sync_new_fact calls CozoDB when active."""
        orch._cozo = MagicMock()
        orch._update_status("cozo", "active")
        orch._lancedb = MagicMock()
        orch._update_status("lancedb", "active")

        fact = MagicMock()
        fact.fact_id = "test-2"
        fact.lifecycle = "active"
        fact.canonical_entities = ["e1", "e2"]
        fact.embedding = [0.1] * 768

        orch.sync_new_fact(fact)
        # CozoDB add_entity called for both entities
        assert orch._cozo.add_entity.call_count == 2


class TestBackendRouting:
    def test_graph_backend_none_when_no_cozo(self, orch):
        assert orch.get_graph_backend() is None

    def test_graph_backend_active(self, orch):
        orch._cozo = MagicMock()
        orch._update_status("cozo", "active")
        assert orch.get_graph_backend() is orch._cozo

    def test_graph_retrieval_is_held_until_id_space_parity_is_proven(self, orch):
        orch._cozo = MagicMock()
        orch._update_status("cozo", "active")
        assert orch.graph_retrieval_ready() is False

    def test_graph_backend_not_active_during_migration(self, orch):
        orch._cozo = MagicMock()
        orch._update_status("cozo", "migrating")
        assert orch.get_graph_backend() is None  # Not returned until "active"
