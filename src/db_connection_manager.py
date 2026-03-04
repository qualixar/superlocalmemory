#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
DbConnectionManager — Thread-safe SQLite connection management with WAL mode.

Solves the "database is locked" bug when multiple agents (CLI + MCP from Claude +
MCP from Cursor + API) try to write or recall simultaneously.

Architecture:
    - WAL mode (Write-Ahead Logging): Concurrent reads OK during writes
    - Busy timeout (5s): Connections wait instead of failing immediately
    - Connection pool: Reusable read connections via thread-local storage
    - Write queue: Single dedicated writer thread serializes all writes
    - Singleton: One manager per database path per process

This is a PREREQUISITE for the Event Bus (v2.5). The write queue guarantees:
    1. Every write succeeds (queued, not dropped)
    2. Events fire after commit, not before (consistency)
    3. Events arrive in correct order (queue preserves order)
    4. Multiple agents can write simultaneously without conflict

Usage:
    from db_connection_manager import DbConnectionManager

    mgr = DbConnectionManager.get_instance(db_path)

    # Reads — concurrent, non-blocking
    conn = mgr.get_read_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT ...")
        rows = cursor.fetchall()
    finally:
        mgr.release_read_connection(conn)

    # Writes — serialized through queue
    def do_insert(conn):
        cursor = conn.cursor()
        cursor.execute("INSERT INTO ...", (...))
        conn.commit()
        return cursor.lastrowid

    result = mgr.execute_write(do_insert)

    # Context manager for reads (preferred)
    with mgr.read_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ...")
"""

import sqlite3
import threading
import logging
from pathlib import Path
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager
from queue import Queue

logger = logging.getLogger("superlocalmemory.db")

# Default configuration
DEFAULT_BUSY_TIMEOUT_MS = 5000
DEFAULT_READ_POOL_SIZE = 4
WRITE_QUEUE_SENTINEL = None  # Signals the writer thread to stop
MAX_READ_CONNECTIONS = 50  # Maximum concurrent read connections


class DbConnectionManager:
    """
    Thread-safe SQLite connection manager with WAL mode and serialized writes.

    Singleton per database path — all callers in the same process share one manager.
    This prevents the "database is locked" errors that occur when multiple agents
    (CLI, MCP, API, Dashboard) write simultaneously.

    Key features:
        - WAL mode: Multiple readers + one writer concurrently
        - Busy timeout: Wait up to 5s instead of failing immediately
        - Read pool: Thread-local read connections, reused across calls
        - Write queue: All writes serialized through a single thread
        - Post-write hooks: Callback after each successful commit (for Event Bus)
    """

    # Singleton registry: db_path -> instance
    _instances: Dict[str, "DbConnectionManager"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "DbConnectionManager":
        """
        Get or create the singleton DbConnectionManager for a database path.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.claude-memory/memory.db

        Returns:
            Shared DbConnectionManager instance for this db_path
        """
        if db_path is None:
            db_path = Path.home() / ".claude-memory" / "memory.db"

        key = str(db_path)

        with cls._instances_lock:
            if key not in cls._instances:
                instance = cls(db_path)
                cls._instances[key] = instance
            return cls._instances[key]

    @classmethod
    def reset_instance(cls, db_path: Optional[Path] = None) -> None:
        """
        Remove and close a singleton instance. Used for testing and cleanup.

        Args:
            db_path: Path to the database. If None, resets all instances.
        """
        with cls._instances_lock:
            if db_path is None:
                # Reset all
                for instance in cls._instances.values():
                    instance.close()
                cls._instances.clear()
            else:
                key = str(db_path)
                if key in cls._instances:
                    cls._instances[key].close()
                    del cls._instances[key]

    def __init__(self, db_path: Path):
        """
        Initialize the connection manager. Do NOT call directly — use get_instance().

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._closed = False

        # Thread-local storage for read connections
        self._local = threading.local()

        # Read pool tracking (for cleanup)
        self._read_connections: list = []
        self._read_connections_lock = threading.Lock()

        # Write queue and dedicated writer thread
        self._write_queue: Queue = Queue(maxsize=1000)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="slm-db-writer",
            daemon=True  # Dies when main process exits
        )
        self._write_connection: Optional[sqlite3.Connection] = None

        # Post-write hooks (for Event Bus integration)
        self._post_write_hooks: list = []
        self._post_write_hooks_lock = threading.Lock()

        # Initialize WAL mode and start writer
        self._init_wal_mode()
        self._writer_thread.start()

        logger.info(
            "DbConnectionManager initialized: db=%s, WAL=enabled, busy_timeout=%dms",
            self.db_path, DEFAULT_BUSY_TIMEOUT_MS
        )

    def _init_wal_mode(self):
        """
        Enable WAL mode and set pragmas on a temporary connection.

        WAL (Write-Ahead Logging) allows concurrent reads during writes.
        This is the single most impactful fix for the "database is locked" bug.
        Once set, WAL mode persists across all connections to this database.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
            # Sync mode NORMAL is safe with WAL and faster than FULL
            conn.execute("PRAGMA synchronous=NORMAL")
            # Incremental auto-vacuum reclaims space without full rebuild (v2.6)
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            conn.close()
        except Exception:
            conn.close()
            raise

    def _create_connection(self, readonly: bool = False) -> sqlite3.Connection:
        """
        Create a new SQLite connection with proper pragmas.

        Args:
            readonly: If True, opens in read-only mode (SQLite URI)

        Returns:
            Configured sqlite3.Connection
        """
        if readonly:
            # URI mode for read-only — prevents accidental writes from read connections
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

        # Apply pragmas to every connection
        conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
        # WAL mode is database-level (persists), but set it here for safety
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        return conn

    # =========================================================================
    # Read connections — thread-local pool, concurrent access OK
    # =========================================================================

    def get_read_connection(self) -> sqlite3.Connection:
        """
        Get a read connection for the current thread.

        Uses thread-local storage so each thread reuses its own connection.
        Read connections are safe to use concurrently with WAL mode.

        Returns:
            sqlite3.Connection configured for reading

        Raises:
            RuntimeError: If the manager has been closed
        """
        if self._closed:
            raise RuntimeError("DbConnectionManager is closed")

        # Check thread-local for existing connection
        conn = getattr(self._local, 'read_conn', None)
        if conn is not None:
            try:
                # Verify connection is still alive
                conn.execute("SELECT 1")
                return conn
            except sqlite3.Error:
                # Connection is dead, create a new one
                self._remove_from_pool(conn)
                conn = None

        # Create new read connection for this thread (with pool limit)
        with self._read_connections_lock:
            if len(self._read_connections) >= MAX_READ_CONNECTIONS:
                logger.warning(
                    "Read connection pool at capacity (%d). Reusing oldest connection.",
                    MAX_READ_CONNECTIONS
                )
                # Reuse the least recently used connection
                conn = self._read_connections[0]
                self._local.read_conn = conn
                return conn

        conn = self._create_connection(readonly=True)
        self._local.read_conn = conn

        with self._read_connections_lock:
            self._read_connections.append(conn)

        return conn

    def release_read_connection(self, conn: sqlite3.Connection) -> None:
        """
        Release a read connection back to the pool.

        With thread-local storage, this is a no-op — the connection stays
        assigned to the thread. Call this for API compatibility and future
        pool expansion.

        Args:
            conn: The connection to release
        """
        # No-op with thread-local strategy — connection stays with thread.
        # Explicit close happens in close() or when thread dies.
        pass

    @contextmanager
    def read_connection(self):
        """
        Context manager for read connections. Preferred API.

        Usage:
            with mgr.read_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
                rows = cursor.fetchall()
        """
        conn = self.get_read_connection()
        try:
            yield conn
        finally:
            self.release_read_connection(conn)

    def _remove_from_pool(self, conn: sqlite3.Connection):
        """Remove a dead connection from the tracking list."""
        with self._read_connections_lock:
            try:
                self._read_connections.remove(conn)
            except ValueError:
                pass
        try:
            conn.close()
        except Exception:
            pass

    # =========================================================================
    # Write connection — single writer thread, serialized queue
    # =========================================================================

    def execute_write(self, callback: Callable[[sqlite3.Connection], Any]) -> Any:
        """
        Execute a write operation through the serialized write queue.

        The callback receives a sqlite3.Connection and should perform its
        write operations (INSERT/UPDATE/DELETE) and call conn.commit().
        The callback runs on the dedicated writer thread — never on the caller's thread.

        Args:
            callback: Function that takes a sqlite3.Connection and returns a result.
                     The callback MUST call conn.commit() for changes to persist.

        Returns:
            Whatever the callback returns

        Raises:
            RuntimeError: If the manager is closed
            Exception: Re-raises any exception from the callback
        """
        if self._closed:
            raise RuntimeError("DbConnectionManager is closed")

        # Create a future-like result holder
        result_event = threading.Event()
        result_holder = {"value": None, "error": None}

        def wrapped_callback(conn):
            try:
                result_holder["value"] = callback(conn)
            except Exception as e:
                result_holder["error"] = e

            result_event.set()

        # Enqueue the work
        self._write_queue.put(wrapped_callback)

        # Wait for completion
        result_event.wait()

        # Re-raise if callback failed
        if result_holder["error"] is not None:
            raise result_holder["error"]

        return result_holder["value"]

    def _writer_loop(self):
        """
        Writer thread main loop. Processes write callbacks sequentially.

        This is the heart of the concurrency fix. All writes go through this
        single thread, eliminating write-write collisions entirely.
        """
        # Create the dedicated write connection
        self._write_connection = self._create_connection(readonly=False)

        while True:
            callback = self._write_queue.get()

            # Sentinel value means shutdown
            if callback is WRITE_QUEUE_SENTINEL:
                self._write_queue.task_done()
                break

            try:
                callback(self._write_connection)

                # Fire post-write hooks (for Event Bus)
                self._fire_post_write_hooks()

            except Exception as e:
                # Log but don't crash the writer thread
                logger.error("Write callback failed: %s", e)

            self._write_queue.task_done()

        # Cleanup
        if self._write_connection:
            try:
                self._write_connection.close()
            except Exception:
                pass
            self._write_connection = None

    # =========================================================================
    # Post-write hooks (Event Bus integration point)
    # =========================================================================

    def register_post_write_hook(self, hook: Callable[[], None]) -> None:
        """
        Register a callback that fires after every successful write commit.

        This is the integration point for the Event Bus (Phase A).
        Hooks run on the writer thread, so they should be fast and non-blocking.
        For heavy work, hooks should enqueue to their own async queue.

        Args:
            hook: Zero-argument callable invoked after each write
        """
        with self._post_write_hooks_lock:
            self._post_write_hooks.append(hook)

    def unregister_post_write_hook(self, hook: Callable[[], None]) -> None:
        """
        Remove a previously registered post-write hook.

        Args:
            hook: The hook to remove
        """
        with self._post_write_hooks_lock:
            try:
                self._post_write_hooks.remove(hook)
            except ValueError:
                pass

    def _fire_post_write_hooks(self):
        """Fire all registered post-write hooks. Errors are logged, not raised."""
        with self._post_write_hooks_lock:
            hooks = list(self._post_write_hooks)

        for hook in hooks:
            try:
                hook()
            except Exception as e:
                logger.error("Post-write hook failed: %s", e)

    # =========================================================================
    # Direct write connection access (for DDL / schema init)
    # =========================================================================

    def execute_ddl(self, callback: Callable[[sqlite3.Connection], Any]) -> Any:
        """
        Execute DDL (CREATE TABLE, ALTER TABLE, etc.) through the write queue.

        Identical to execute_write() but named separately for clarity.
        DDL operations like schema initialization must go through the writer
        to prevent "database is locked" during table creation.

        Args:
            callback: Function that takes a sqlite3.Connection and performs DDL

        Returns:
            Whatever the callback returns
        """
        return self.execute_write(callback)

    # =========================================================================
    # Lifecycle management
    # =========================================================================

    def close(self) -> None:
        """
        Shut down the connection manager. Drains the write queue and closes
        all connections.

        Safe to call multiple times. After close(), all operations raise RuntimeError.
        """
        if self._closed:
            return

        self._closed = True

        # Signal writer thread to stop
        self._write_queue.put(WRITE_QUEUE_SENTINEL)

        # Wait for writer to finish (with timeout to prevent hanging)
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=10)

        # Close all read connections
        with self._read_connections_lock:
            for conn in self._read_connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._read_connections.clear()

        logger.info("DbConnectionManager closed: db=%s", self.db_path)

    @property
    def is_closed(self) -> bool:
        """Check if the manager has been shut down."""
        return self._closed

    @property
    def write_queue_size(self) -> int:
        """Current number of pending write operations. Useful for monitoring."""
        return self._write_queue.qsize()

    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information about the connection manager state.

        Returns:
            Dictionary with connection pool stats, queue depth, WAL status
        """
        diagnostics = {
            "db_path": str(self.db_path),
            "closed": self._closed,
            "write_queue_depth": self._write_queue.qsize(),
            "writer_thread_alive": self._writer_thread.is_alive(),
            "read_connections_count": len(self._read_connections),
            "post_write_hooks_count": len(self._post_write_hooks),
        }

        # Check WAL mode
        try:
            with self.read_connection() as conn:
                cursor = conn.execute("PRAGMA journal_mode")
                diagnostics["journal_mode"] = cursor.fetchone()[0]
                cursor = conn.execute("PRAGMA busy_timeout")
                diagnostics["busy_timeout_ms"] = cursor.fetchone()[0]
        except Exception as e:
            diagnostics["pragma_check_error"] = str(e)

        return diagnostics

    def __repr__(self) -> str:
        status = "closed" if self._closed else "active"
        return f"<DbConnectionManager db={self.db_path} status={status}>"
