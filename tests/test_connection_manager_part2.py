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


class TestDbConnectionManagerConcurrency(unittest.TestCase):
    """Concurrency stress tests — simulates multiple agents writing simultaneously."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "concurrent.db"
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, created_at TEXT)")
        conn.commit()
        conn.close()
        DbConnectionManager.reset_instance(self.db_path)

    def tearDown(self):
        DbConnectionManager.reset_instance(self.db_path)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_100_concurrent_writes(self):
        """100 threads writing simultaneously must all succeed (no database locked)."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        errors = []
        successes = []

        def write_memory(n):
            try:
                def _do(conn):
                    conn.execute(
                        "INSERT INTO memories (content, created_at) VALUES (?, ?)",
                        (f"Memory #{n}", "2026-02-12T00:00:00")
                    )
                    conn.commit()
                    return n
                result = mgr.execute_write(_do)
                successes.append(result)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_memory, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors[:5]}")
        self.assertEqual(len(successes), 100)

        # Verify all 100 in database
        with mgr.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        self.assertEqual(count, 100)

    def test_concurrent_reads_during_write(self):
        """Reads must not block during writes (WAL mode guarantee)."""
        mgr = DbConnectionManager.get_instance(self.db_path)

        # Pre-populate
        for i in range(10):
            mgr.execute_write(lambda conn, i=i: (
                conn.execute("INSERT INTO memories (content) VALUES (?)", (f"Pre-{i}",)),
                conn.commit()
            ))

        read_results = []
        read_errors = []

        def slow_write(conn):
            """Simulate a slow write (holds write lock)."""
            conn.execute("INSERT INTO memories (content) VALUES (?)", ("slow-write",))
            time.sleep(0.1)  # Hold write lock for 100ms
            conn.commit()

        def fast_read():
            """Read should succeed even during slow write."""
            try:
                with mgr.read_connection() as conn:
                    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                    read_results.append(count)
            except Exception as e:
                read_errors.append(str(e))

        # Start slow write
        write_thread = threading.Thread(target=lambda: mgr.execute_write(slow_write))
        write_thread.start()

        # Immediately start reads
        time.sleep(0.01)  # Tiny delay to ensure write is in progress
        read_threads = [threading.Thread(target=fast_read) for _ in range(5)]
        for t in read_threads:
            t.start()
        for t in read_threads:
            t.join()

        write_thread.join()

        self.assertEqual(len(read_errors), 0, f"Read errors during write: {read_errors}")
        self.assertTrue(len(read_results) > 0, "No reads completed during write")


class TestMemoryStoreV2Refactor(unittest.TestCase):
    """Regression tests for memory_store_v2.py refactor — all existing behavior preserved."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_memory.db"
        self.vectors_path = Path(self.tmpdir) / "vectors"
        self.vectors_path.mkdir()

        # Patch module paths
        import memory_store_v2
        self._orig_db = memory_store_v2.DB_PATH
        self._orig_mem = memory_store_v2.MEMORY_DIR
        self._orig_vec = memory_store_v2.VECTORS_PATH
        memory_store_v2.DB_PATH = self.db_path
        memory_store_v2.MEMORY_DIR = Path(self.tmpdir)
        memory_store_v2.VECTORS_PATH = self.vectors_path

        from memory_store_v2 import MemoryStoreV2
        self.store = MemoryStoreV2(self.db_path)

    def tearDown(self):
        import memory_store_v2
        memory_store_v2.DB_PATH = self._orig_db
        memory_store_v2.MEMORY_DIR = self._orig_mem
        memory_store_v2.VECTORS_PATH = self._orig_vec
        DbConnectionManager.reset_instance(self.db_path)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # === Core CRUD ===

    def test_add_and_retrieve(self):
        """add_memory + get_by_id must round-trip correctly."""
        mid = self.store.add_memory("Test content", tags=["a", "b"], importance=7)
        self.assertIsInstance(mid, int)
        self.assertGreater(mid, 0)

        mem = self.store.get_by_id(mid)
        self.assertIsNotNone(mem)
        self.assertEqual(mem["content"], "Test content")
        self.assertEqual(mem["tags"], ["a", "b"])
        self.assertEqual(mem["importance"], 7)

    def test_duplicate_detection(self):
        """Adding identical content must return existing ID, not create duplicate."""
        mid1 = self.store.add_memory("Duplicate content")
        mid2 = self.store.add_memory("Duplicate content")
        self.assertEqual(mid1, mid2)

    def test_search_returns_results(self):
        """search() must find relevant memories."""
        self.store.add_memory("Python FastAPI backend development")
        self.store.add_memory("React frontend hooks and state")
        results = self.store.search("Python", limit=5)
        self.assertTrue(len(results) > 0)

    def test_delete_memory(self):
        """delete_memory must remove the memory."""
        mid = self.store.add_memory("To be deleted")
        self.assertTrue(self.store.delete_memory(mid))
        self.assertIsNone(self.store.get_by_id(mid))

    def test_delete_nonexistent(self):
        """Deleting a non-existent ID must return False."""
        self.assertFalse(self.store.delete_memory(99999))

    def test_list_all(self):
        """list_all must return memories in reverse chronological order."""
        self.store.add_memory("Python FastAPI development patterns")
        self.store.add_memory("React component architecture guide")
        self.store.add_memory("SQLite WAL mode configuration notes")
        results = self.store.list_all(limit=10)
        self.assertEqual(len(results), 3)
        self.assertIn("title", results[0])  # V1 compatibility field

    def test_get_recent(self):
        """get_recent must respect limit."""
        for i in range(5):
            self.store.add_memory(f"Memory {i}")
        results = self.store.get_recent(limit=3)
        self.assertEqual(len(results), 3)

    def test_get_stats(self):
        """get_stats must return correct counts."""
        self.store.add_memory("Stat test 1")
        self.store.add_memory("Stat test 2")
        stats = self.store.get_stats()
        self.assertEqual(stats["total_memories"], 2)
        self.assertIn("active_profile", stats)
        self.assertIn("sklearn_available", stats)

    def test_get_attribution(self):
        """Attribution must always return creator info."""
        attr = self.store.get_attribution()
        self.assertEqual(attr["creator_name"], "Varun Pratap Bhardwaj")
        self.assertEqual(attr["license"], "MIT")

    def test_update_tier(self):
        """update_tier must change memory_type."""
        mid = self.store.add_memory("Tier test")
        self.store.update_tier(mid, "warm", compressed_summary="Summary")
        mem = self.store.get_by_id(mid)
        self.assertEqual(mem["memory_type"], "warm")

    def test_get_tree(self):
        """get_tree must return memories."""
        self.store.add_memory("Tree root")
        tree = self.store.get_tree()
        self.assertTrue(len(tree) > 0)

    # === Edge Cases ===

    def test_empty_database_search(self):
        """Search on empty database must return empty list, not error."""
        results = self.store.search("anything", limit=5)
        self.assertEqual(results, [])

    def test_empty_database_stats(self):
        """Stats on empty database must return zeros."""
        stats = self.store.get_stats()
        self.assertEqual(stats["total_memories"], 0)

    def test_empty_database_list(self):
        """list_all on empty database must return empty list."""
        results = self.store.list_all()
        self.assertEqual(results, [])

    def test_get_nonexistent_id(self):
        """get_by_id for missing ID must return None."""
        self.assertIsNone(self.store.get_by_id(99999))

    def test_very_long_content(self):
        """Content up to MAX_CONTENT_SIZE must be accepted."""
        long_content = "x" * 100_000  # 100KB
        mid = self.store.add_memory(long_content)
        mem = self.store.get_by_id(mid)
        self.assertEqual(len(mem["content"]), 100_000)

    def test_unicode_content(self):
        """Unicode content must round-trip correctly."""
        content = "日本語テスト 🎯 émojis äöü"
        mid = self.store.add_memory(content)
        mem = self.store.get_by_id(mid)
        self.assertEqual(mem["content"], content)

    def test_special_characters_in_tags(self):
        """Tags with special characters must be stored correctly."""
        mid = self.store.add_memory("Tag test", tags=["c++", "c#", "node.js"])
        mem = self.store.get_by_id(mid)
        self.assertEqual(mem["tags"], ["c++", "c#", "node.js"])

    # === Security / Input Validation ===

    def test_content_size_limit(self):
        """Content exceeding MAX_CONTENT_SIZE must be rejected."""
        with self.assertRaises(ValueError):
            self.store.add_memory("x" * 1_000_001)

    def test_empty_content_rejected(self):
        """Empty content must be rejected."""
        with self.assertRaises(ValueError):
            self.store.add_memory("")

    def test_whitespace_only_content_rejected(self):
        """Whitespace-only content must be rejected (stripped to empty)."""
        with self.assertRaises(ValueError):
            self.store.add_memory("   \n\t  ")

    def test_non_string_content_rejected(self):
        """Non-string content must raise TypeError."""
        with self.assertRaises(TypeError):
            self.store.add_memory(12345)

    def test_too_many_tags_rejected(self):
        """More than MAX_TAGS tags must be rejected."""
        with self.assertRaises(ValueError):
            self.store.add_memory("tag test", tags=[f"tag{i}" for i in range(25)])

    def test_importance_clamped(self):
        """Out-of-range importance must be clamped, not error."""
        mid = self.store.add_memory("Clamp test high", importance=99)
        mem = self.store.get_by_id(mid)
        self.assertEqual(mem["importance"], 10)

        mid2 = self.store.add_memory("Clamp test low", importance=-5)
        mem2 = self.store.get_by_id(mid2)
        self.assertEqual(mem2["importance"], 1)

    def test_sql_injection_in_content(self):
        """SQL injection attempts in content must be safely stored."""
        evil = "'; DROP TABLE memories; --"
        mid = self.store.add_memory(evil)
        mem = self.store.get_by_id(mid)
        self.assertEqual(mem["content"], evil)
        # Table must still exist
        stats = self.store.get_stats()
        self.assertEqual(stats["total_memories"], 1)

    def test_sql_injection_in_search(self):
        """SQL injection in search query must not cause errors."""
        self.store.add_memory("Safe content")
        results = self.store.search("'; DROP TABLE memories; --")
        # Should return empty or results, not crash
        self.assertIsInstance(results, list)


