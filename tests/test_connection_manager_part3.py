#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V2 - Connection Manager Tests

Hardcore test suite for db_connection_manager.py and the memory_store_v2.py refactor.
Covers: unit tests, regression, edge cases, security, concurrency, new-user/Docker scenarios.
"""
import sqlite3
import sys
import os
import json
import tempfile
import shutil
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Import from repo source
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db_connection_manager import DbConnectionManager, DEFAULT_BUSY_TIMEOUT_MS


class TestBackwardCompatibilityFallback(unittest.TestCase):
    """Test that everything works when DbConnectionManager is NOT available."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "fallback.db"
        self.vectors_path = Path(self.tmpdir) / "vectors"
        self.vectors_path.mkdir()

        import memory_store_v2
        self._orig_db = memory_store_v2.DB_PATH
        self._orig_mem = memory_store_v2.MEMORY_DIR
        self._orig_vec = memory_store_v2.VECTORS_PATH
        self._orig_flag = memory_store_v2.USE_CONNECTION_MANAGER
        memory_store_v2.DB_PATH = self.db_path
        memory_store_v2.MEMORY_DIR = Path(self.tmpdir)
        memory_store_v2.VECTORS_PATH = self.vectors_path
        memory_store_v2.USE_CONNECTION_MANAGER = False

        from memory_store_v2 import MemoryStoreV2
        self.store = MemoryStoreV2(self.db_path)

    def tearDown(self):
        import memory_store_v2
        memory_store_v2.DB_PATH = self._orig_db
        memory_store_v2.MEMORY_DIR = self._orig_mem
        memory_store_v2.VECTORS_PATH = self._orig_vec
        memory_store_v2.USE_CONNECTION_MANAGER = self._orig_flag
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fallback_no_db_mgr(self):
        """With USE_CONNECTION_MANAGER=False, _db_mgr must be None."""
        self.assertIsNone(self.store._db_mgr)

    def test_fallback_add_memory(self):
        """add_memory must work in fallback mode."""
        mid = self.store.add_memory("Fallback test")
        self.assertGreater(mid, 0)

    def test_fallback_search(self):
        """search must work in fallback mode."""
        self.store.add_memory("Fallback search content")
        results = self.store.search("Fallback", limit=5)
        self.assertTrue(len(results) > 0)

    def test_fallback_stats(self):
        """get_stats must work in fallback mode."""
        self.store.add_memory("Stat in fallback")
        stats = self.store.get_stats()
        self.assertEqual(stats["total_memories"], 1)

    def test_fallback_delete(self):
        """delete must work in fallback mode."""
        mid = self.store.add_memory("Delete in fallback")
        self.assertTrue(self.store.delete_memory(mid))


class TestNewUserDockerScenario(unittest.TestCase):
    """
    Simulates a brand new user on Docker/Windows:
    - No existing database
    - No profiles.json
    - No vectors directory
    - Fresh install from scratch
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "memory.db"
        self.vectors_path = Path(self.tmpdir) / "vectors"
        # Deliberately do NOT create vectors dir — install should handle it
        # Deliberately do NOT create profiles.json

        import memory_store_v2
        self._orig_db = memory_store_v2.DB_PATH
        self._orig_mem = memory_store_v2.MEMORY_DIR
        self._orig_vec = memory_store_v2.VECTORS_PATH
        memory_store_v2.DB_PATH = self.db_path
        memory_store_v2.MEMORY_DIR = Path(self.tmpdir)
        memory_store_v2.VECTORS_PATH = self.vectors_path

    def tearDown(self):
        import memory_store_v2
        memory_store_v2.DB_PATH = self._orig_db
        memory_store_v2.MEMORY_DIR = self._orig_mem
        memory_store_v2.VECTORS_PATH = self._orig_vec
        DbConnectionManager.reset_instance(self.db_path)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fresh_install_creates_database(self):
        """First-ever instantiation must create database from scratch."""
        from memory_store_v2 import MemoryStoreV2
        # Database does NOT exist yet
        self.assertFalse(self.db_path.exists())

        store = MemoryStoreV2(self.db_path)
        self.assertTrue(self.db_path.exists())

    def test_fresh_install_add_first_memory(self):
        """New user's first memory must succeed."""
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mid = store.add_memory("My first memory!")
        self.assertEqual(mid, 1)

    def test_fresh_install_search_empty(self):
        """Search on empty fresh database must not crash."""
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        results = store.search("anything")
        self.assertEqual(results, [])

    def test_fresh_install_no_profiles_json(self):
        """Missing profiles.json must fall back to 'default' profile."""
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        profile = store._get_active_profile()
        self.assertEqual(profile, "default")

    def test_fresh_install_stats(self):
        """Stats on fresh database must return valid zeros."""
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        stats = store.get_stats()
        self.assertEqual(stats["total_memories"], 0)
        self.assertEqual(stats["total_clusters"], 0)

    def test_fresh_install_attribution_present(self):
        """Attribution must be embedded in fresh database."""
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        attr = store.get_attribution()
        self.assertEqual(attr["creator_name"], "Varun Pratap Bhardwaj")

    def test_wal_mode_on_fresh_database(self):
        """WAL mode must be active on newly created database."""
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        if store._db_mgr:
            diag = store._db_mgr.get_diagnostics()
            self.assertEqual(diag["journal_mode"], "wal")


class TestMcpServerSingleton(unittest.TestCase):
    """Test the singleton accessor pattern used in mcp_server.py."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "mcp_test.db"
        self.vectors_path = Path(self.tmpdir) / "vectors"
        self.vectors_path.mkdir()

        import memory_store_v2
        self._orig_db = memory_store_v2.DB_PATH
        self._orig_mem = memory_store_v2.MEMORY_DIR
        self._orig_vec = memory_store_v2.VECTORS_PATH
        memory_store_v2.DB_PATH = self.db_path
        memory_store_v2.MEMORY_DIR = Path(self.tmpdir)
        memory_store_v2.VECTORS_PATH = self.vectors_path

    def tearDown(self):
        import memory_store_v2
        memory_store_v2.DB_PATH = self._orig_db
        memory_store_v2.MEMORY_DIR = self._orig_mem
        memory_store_v2.VECTORS_PATH = self._orig_vec
        DbConnectionManager.reset_instance(self.db_path)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_singleton_returns_same_instance(self):
        """get_store() pattern must return the same object every time."""
        from memory_store_v2 import MemoryStoreV2

        _store = None

        def get_store():
            nonlocal _store
            if _store is None:
                _store = MemoryStoreV2(self.db_path)
            return _store

        s1 = get_store()
        s2 = get_store()
        s3 = get_store()
        self.assertIs(s1, s2)
        self.assertIs(s2, s3)

    def test_singleton_shared_db_mgr(self):
        """Singleton must share one DbConnectionManager across all calls."""
        from memory_store_v2 import MemoryStoreV2

        store = MemoryStoreV2(self.db_path)
        self.assertIsNotNone(store._db_mgr)

        # Simulating what mcp_server.py does — multiple tool handlers use same store
        mid = store.add_memory("MCP call 1")
        results = store.search("MCP", limit=5)
        stats = store.get_stats()
        self.assertGreater(mid, 0)
        self.assertTrue(len(results) > 0)
        self.assertEqual(stats["total_memories"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
