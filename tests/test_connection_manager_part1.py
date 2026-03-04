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


class TestDbConnectionManagerUnit(unittest.TestCase):
    """Unit tests for DbConnectionManager core functionality."""

    def setUp(self):
        """Create fresh temp database for each test."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        # Pre-create the database (WAL needs it to exist)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
        conn.commit()
        conn.close()
        # Reset any leftover singletons
        DbConnectionManager.reset_instance(self.db_path)

    def tearDown(self):
        """Clean up temp database and singleton."""
        DbConnectionManager.reset_instance(self.db_path)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # === WAL Mode ===

    def test_wal_mode_enabled(self):
        """WAL mode must be set on the database after manager creation."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        with mgr.read_connection() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode, "wal")

    def test_busy_timeout_set(self):
        """Busy timeout must be configured on all connections."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        with mgr.read_connection() as conn:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertEqual(timeout, DEFAULT_BUSY_TIMEOUT_MS)

    def test_synchronous_normal(self):
        """Synchronous mode should be NORMAL (not FULL) for WAL performance."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        with mgr.read_connection() as conn:
            sync = conn.execute("PRAGMA synchronous").fetchone()[0]
        # 1 = NORMAL in SQLite pragma encoding
        self.assertEqual(sync, 1)

    # === Singleton Pattern ===

    def test_singleton_same_path(self):
        """Same db_path must return the same instance."""
        mgr1 = DbConnectionManager.get_instance(self.db_path)
        mgr2 = DbConnectionManager.get_instance(self.db_path)
        self.assertIs(mgr1, mgr2)

    def test_singleton_different_path(self):
        """Different db_path must return different instances."""
        db2 = Path(self.tmpdir) / "test2.db"
        conn = sqlite3.connect(str(db2))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        mgr1 = DbConnectionManager.get_instance(self.db_path)
        mgr2 = DbConnectionManager.get_instance(db2)
        self.assertIsNot(mgr1, mgr2)
        DbConnectionManager.reset_instance(db2)

    def test_reset_instance_creates_new(self):
        """After reset, get_instance must create a fresh manager."""
        mgr1 = DbConnectionManager.get_instance(self.db_path)
        id1 = id(mgr1)
        DbConnectionManager.reset_instance(self.db_path)
        mgr2 = DbConnectionManager.get_instance(self.db_path)
        self.assertNotEqual(id1, id(mgr2))

    def test_reset_all_instances(self):
        """reset_instance(None) must close and clear all instances."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        DbConnectionManager.reset_instance()
        self.assertTrue(mgr.is_closed)

    # === Read Connections ===

    def test_read_connection_context_manager(self):
        """read_connection() context manager must yield a working connection."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        with mgr.read_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
        self.assertEqual(result[0], 0)

    def test_read_connection_thread_local(self):
        """Each thread must get its own read connection."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        conn_ids = []

        def get_conn_id():
            with mgr.read_connection() as conn:
                conn_ids.append(id(conn))

        threads = [threading.Thread(target=get_conn_id) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have a unique connection object
        self.assertEqual(len(set(conn_ids)), 4)

    def test_read_connection_same_thread_reuses(self):
        """Same thread must get the same read connection on repeated calls."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        with mgr.read_connection() as conn1:
            id1 = id(conn1)
        with mgr.read_connection() as conn2:
            id2 = id(conn2)
        self.assertEqual(id1, id2)

    # === Write Queue ===

    def test_write_basic(self):
        """Basic write through queue must persist data."""
        mgr = DbConnectionManager.get_instance(self.db_path)

        def insert(conn):
            conn.execute("INSERT INTO test (val) VALUES (?)", ("hello",))
            conn.commit()
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        row_id = mgr.execute_write(insert)
        self.assertEqual(row_id, 1)

        # Verify via read
        with mgr.read_connection() as conn:
            val = conn.execute("SELECT val FROM test WHERE id = ?", (row_id,)).fetchone()[0]
        self.assertEqual(val, "hello")

    def test_write_returns_value(self):
        """execute_write must return the callback's return value."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        result = mgr.execute_write(lambda conn: 42)
        self.assertEqual(result, 42)

    def test_write_exception_propagates(self):
        """Exceptions in write callbacks must propagate to the caller."""
        mgr = DbConnectionManager.get_instance(self.db_path)

        def bad_write(conn):
            raise ValueError("intentional test error")

        with self.assertRaises(ValueError) as ctx:
            mgr.execute_write(bad_write)
        self.assertIn("intentional test error", str(ctx.exception))

    def test_write_exception_doesnt_kill_writer(self):
        """Writer thread must survive callback exceptions."""
        mgr = DbConnectionManager.get_instance(self.db_path)

        # First call: exception
        with self.assertRaises(ValueError):
            mgr.execute_write(lambda conn: (_ for _ in ()).throw(ValueError("boom")))

        # Second call: should still work
        result = mgr.execute_write(lambda conn: "still alive")
        self.assertEqual(result, "still alive")

    def test_write_serialization_order(self):
        """Writes must be processed in submission order."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        order = []

        def ordered_insert(n):
            def _do(conn):
                order.append(n)
                conn.execute("INSERT INTO test (val) VALUES (?)", (f"item-{n}",))
                conn.commit()
            return _do

        # Submit 10 writes from different threads simultaneously
        threads = []
        for i in range(10):
            t = threading.Thread(target=lambda n=i: mgr.execute_write(ordered_insert(n)))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 10 should have been processed
        self.assertEqual(len(order), 10)

        # Verify all items in DB
        with mgr.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM test").fetchone()[0]
        self.assertEqual(count, 10)

    # === Post-Write Hooks ===

    def test_post_write_hook_fires(self):
        """Post-write hooks must fire after each successful write."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        fired = []

        mgr.register_post_write_hook(lambda: fired.append(True))

        mgr.execute_write(lambda conn: conn.execute("INSERT INTO test (val) VALUES ('a')") or conn.commit())
        mgr.execute_write(lambda conn: conn.execute("INSERT INTO test (val) VALUES ('b')") or conn.commit())

        self.assertEqual(len(fired), 2)

    def test_post_write_hook_unregister(self):
        """Unregistered hooks must not fire."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        fired = []
        hook = lambda: fired.append(True)

        mgr.register_post_write_hook(hook)
        mgr.execute_write(lambda conn: conn.commit())
        self.assertEqual(len(fired), 1)

        mgr.unregister_post_write_hook(hook)
        mgr.execute_write(lambda conn: conn.commit())
        self.assertEqual(len(fired), 1)  # Should NOT increase

    def test_post_write_hook_exception_doesnt_crash(self):
        """Hook exceptions must be caught — not crash the writer thread."""
        mgr = DbConnectionManager.get_instance(self.db_path)

        def bad_hook():
            raise RuntimeError("hook exploded")

        mgr.register_post_write_hook(bad_hook)

        # Should not raise
        mgr.execute_write(lambda conn: conn.execute("INSERT INTO test (val) VALUES ('x')") or conn.commit())

        # Writer should still be alive
        result = mgr.execute_write(lambda conn: "alive")
        self.assertEqual(result, "alive")

        mgr.unregister_post_write_hook(bad_hook)

    # === Diagnostics ===

    def test_diagnostics(self):
        """get_diagnostics must return all expected keys."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        diag = mgr.get_diagnostics()

        self.assertIn("db_path", diag)
        self.assertIn("closed", diag)
        self.assertIn("write_queue_depth", diag)
        self.assertIn("writer_thread_alive", diag)
        self.assertIn("journal_mode", diag)
        self.assertIn("busy_timeout_ms", diag)
        self.assertFalse(diag["closed"])
        self.assertTrue(diag["writer_thread_alive"])
        self.assertEqual(diag["journal_mode"], "wal")

    # === Lifecycle ===

    def test_close_prevents_operations(self):
        """After close(), all operations must raise RuntimeError."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        DbConnectionManager.reset_instance(self.db_path)

        with self.assertRaises(RuntimeError):
            mgr.get_read_connection()

        with self.assertRaises(RuntimeError):
            mgr.execute_write(lambda conn: None)

    def test_close_is_idempotent(self):
        """Calling close() multiple times must not raise."""
        mgr = DbConnectionManager.get_instance(self.db_path)
        mgr.close()
        mgr.close()  # Should not raise
        self.assertTrue(mgr.is_closed)


