#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
EventBus — Real-time event broadcasting for memory operations.

Transforms SuperLocalMemory from passive storage (filing cabinet) to active
coordination layer (nervous system). Every memory write, update, delete, or
recall triggers an event that subscribed agents and the dashboard receive.

Architecture:
    memory_store_v2.py (write) → EventBus.emit()
                                    ├── SQLite memory_events table (persistence)
                                    ├── In-memory listeners (real-time delivery)
                                    │   ├── SSE endpoint (dashboard, MCP clients)
                                    │   ├── WebSocket (real-time agents)
                                    │   └── Webhook dispatcher (external services)
                                    └── Tiered retention (hot → warm → cold → archive)

Event Types:
    memory.created    — New memory written
    memory.updated    — Existing memory modified
    memory.deleted    — Memory removed
    memory.recalled   — Memory retrieved by an agent
    graph.updated     — Knowledge graph rebuilt
    pattern.learned   — New pattern detected
    agent.connected   — New agent connects
    agent.disconnected — Agent disconnects

Retention Tiers:
    Hot  (0-48h, configurable)  — Full events, fully queryable
    Warm (2-14d, configurable)  — Key events only (importance >= 5)
    Cold (14-30d, configurable) — Daily aggregates only
    Archive (30d+)              — Pruned, stats in pattern_learner
"""

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

logger = logging.getLogger("superlocalmemory.events")

# Default retention windows (in hours)
DEFAULT_HOT_HOURS = 48
DEFAULT_WARM_HOURS = 14 * 24   # 14 days
DEFAULT_COLD_HOURS = 30 * 24   # 30 days

# In-memory buffer size for real-time delivery
EVENT_BUFFER_SIZE = 200

# Valid event types
VALID_EVENT_TYPES = frozenset([
    "memory.created",
    "memory.updated",
    "memory.deleted",
    "memory.recalled",
    "graph.updated",
    "pattern.learned",
    "agent.connected",
    "agent.disconnected",
])


class EventBus:
    """
    Central event bus for SuperLocalMemory.

    Singleton per database path. Emits events to persistent storage and
    in-memory listeners simultaneously.

    Thread-safe: emit() can be called from any thread.
    Listener callbacks run on the emitter's thread — keep them fast.
    For heavy work, listeners should enqueue to their own async queue.
    """

    _instances: Dict[str, "EventBus"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "EventBus":
        """Get or create the singleton EventBus for a database path."""
        if db_path is None:
            db_path = Path.home() / ".claude-memory" / "memory.db"

        key = str(db_path)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(db_path)
            return cls._instances[key]

    @classmethod
    def reset_instance(cls, db_path: Optional[Path] = None) -> None:
        """Remove and close a singleton instance. Used for testing."""
        with cls._instances_lock:
            if db_path is None:
                cls._instances.clear()
            else:
                key = str(db_path)
                if key in cls._instances:
                    del cls._instances[key]

    def __init__(self, db_path: Path):
        """Initialize EventBus. Use get_instance() instead of calling directly."""
        self.db_path = Path(db_path)

        # In-memory event buffer for real-time delivery (thread-safe deque)
        self._buffer: deque = deque(maxlen=EVENT_BUFFER_SIZE)
        self._buffer_lock = threading.Lock()

        # Event counter (monotonic, reset on restart — DB id is authoritative)
        self._event_counter = 0
        self._counter_lock = threading.Lock()

        # Listeners: list of callbacks called on every event
        # Signature: callback(event: dict) -> None
        self._listeners: List[Callable[[dict], None]] = []
        self._listeners_lock = threading.Lock()

        # Auto-prune tracking: lightweight heuristic trigger
        self._write_count = 0
        self._last_prune = datetime.now()

        # Initialize schema
        self._init_schema()

        logger.info("EventBus initialized: db=%s", self.db_path)

    def _init_schema(self):
        """Create the memory_events table if it doesn't exist."""
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _create_table(conn):
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memory_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        memory_id INTEGER,
                        source_agent TEXT DEFAULT 'user',
                        source_protocol TEXT DEFAULT 'internal',
                        payload TEXT,
                        importance INTEGER DEFAULT 5,
                        tier TEXT DEFAULT 'hot',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                # Index for efficient querying and pruning
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_events_type
                    ON memory_events(event_type)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_events_created
                    ON memory_events(created_at)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_events_tier
                    ON memory_events(tier)
                ''')
                conn.commit()

            mgr.execute_write(_create_table)
        except ImportError:
            # Fallback: direct connection
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memory_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        memory_id INTEGER,
                        source_agent TEXT DEFAULT 'user',
                        source_protocol TEXT DEFAULT 'internal',
                        payload TEXT,
                        importance INTEGER DEFAULT 5,
                        tier TEXT DEFAULT 'hot',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON memory_events(event_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_created ON memory_events(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_tier ON memory_events(tier)')
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Event Emission
    # =========================================================================

    def emit(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        memory_id: Optional[int] = None,
        source_agent: str = "user",
        source_protocol: str = "internal",
        importance: int = 5,
    ) -> Optional[int]:
        """
        Emit an event to all subscribers and persist to database.

        Args:
            event_type: One of VALID_EVENT_TYPES (e.g., "memory.created")
            payload: Event-specific data (dict, serialized to JSON)
            memory_id: Associated memory ID (if applicable)
            source_agent: Agent that triggered the event
            source_protocol: Protocol used (mcp, cli, rest, python)
            importance: Event importance 1-10 (affects retention)

        Returns:
            Event ID from database, or None if persistence failed

        Raises:
            ValueError: If event_type is not in VALID_EVENT_TYPES
        """
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event type: {event_type}. "
                f"Valid types: {', '.join(sorted(VALID_EVENT_TYPES))}"
            )

        # Clamp importance
        importance = max(1, min(10, importance))

        # Build event dict
        now = datetime.now().isoformat()
        with self._counter_lock:
            self._event_counter += 1
            seq = self._event_counter

        event = {
            "seq": seq,
            "event_type": event_type,
            "memory_id": memory_id,
            "source_agent": source_agent,
            "source_protocol": source_protocol,
            "payload": payload or {},
            "importance": importance,
            "timestamp": now,
        }

        # 1. Persist to database (non-blocking if it fails)
        event_id = self._persist_event(event)
        if event_id:
            event["id"] = event_id

        # 2. Add to in-memory buffer
        with self._buffer_lock:
            self._buffer.append(event)

        # 3. Notify all listeners
        self._notify_listeners(event)

        logger.debug("Event emitted: type=%s, id=%s, memory_id=%s", event_type, event_id, memory_id)

        # Auto-prune every 100 events or every 24 hours, whichever comes first
        self._write_count += 1
        if self._write_count >= 100 or (datetime.now() - self._last_prune).total_seconds() > 86400:
            try:
                self.prune_events()
                self._write_count = 0
                self._last_prune = datetime.now()
            except Exception:
                pass  # Don't let prune failures block event emission

        return event_id

    def _persist_event(self, event: dict) -> Optional[int]:
        """Persist event to memory_events table. Returns event ID or None."""
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _insert(conn):
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO memory_events
                        (event_type, memory_id, source_agent, source_protocol,
                         payload, importance, tier, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'hot', ?)
                ''', (
                    event["event_type"],
                    event.get("memory_id"),
                    event["source_agent"],
                    event["source_protocol"],
                    json.dumps(event["payload"]),
                    event["importance"],
                    event["timestamp"],
                ))
                conn.commit()
                return cursor.lastrowid

            return mgr.execute_write(_insert)

        except Exception as e:
            # Event persistence failure must NEVER break core operations
            logger.error("Failed to persist event: %s", e)
            return None

    # =========================================================================
    # Listener Management
    # =========================================================================

    def add_listener(self, callback: Callable[[dict], None]) -> None:
        """
        Register a listener that receives every emitted event.

        Callbacks run on the emitter's thread — keep them fast and non-blocking.
        For async/heavy work, the callback should enqueue to its own queue.

        Args:
            callback: Function(event_dict) called on every emit()
        """
        with self._listeners_lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[dict], None]) -> None:
        """Remove a previously registered listener."""
        with self._listeners_lock:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

    def _notify_listeners(self, event: dict):
        """Call all registered listeners. Errors are logged, not raised."""
        with self._listeners_lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error("Event listener failed: %s", e)

    # =========================================================================
    # Event Retrieval (for replay, SSE, polling)
    # =========================================================================

    def get_recent_events(
        self,
        since_id: Optional[int] = None,
        limit: int = 50,
        event_type: Optional[str] = None,
    ) -> List[dict]:
        """
        Get recent events from the database.

        Used for:
        - SSE replay on reconnect (client sends Last-Event-ID)
        - Dashboard polling
        - Subscription replay (durable subscribers reconnecting)

        Args:
            since_id: Return events with ID greater than this (for replay)
            limit: Maximum events to return (default 50, max 200)
            event_type: Filter by event type (optional)

        Returns:
            List of event dicts, ordered by ID ascending
        """
        limit = min(limit, 200)

        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()

                query = "SELECT id, event_type, memory_id, source_agent, source_protocol, payload, importance, tier, created_at FROM memory_events WHERE 1=1"
                params = []

                if since_id is not None:
                    query += " AND id > ?"
                    params.append(since_id)

                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)

                query += " ORDER BY id ASC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

            events = []
            for row in rows:
                payload = row[5]
                try:
                    payload = json.loads(payload) if payload else {}
                except (json.JSONDecodeError, TypeError):
                    payload = {}

                events.append({
                    "id": row[0],
                    "event_type": row[1],
                    "memory_id": row[2],
                    "source_agent": row[3],
                    "source_protocol": row[4],
                    "payload": payload,
                    "importance": row[6],
                    "tier": row[7],
                    "timestamp": row[8],
                })

            return events

        except Exception as e:
            logger.error("Failed to get recent events: %s", e)
            return []

    def get_buffered_events(self, since_seq: int = 0) -> List[dict]:
        """
        Get events from the in-memory buffer (fast, no DB hit).

        Used for real-time SSE/WebSocket delivery.

        Args:
            since_seq: Return events with seq > this value

        Returns:
            List of event dicts from the buffer
        """
        with self._buffer_lock:
            return [e for e in self._buffer if e.get("seq", 0) > since_seq]

    def get_event_stats(self) -> dict:
        """Get event system statistics."""
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM memory_events")
                total = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT event_type, COUNT(*) as count
                    FROM memory_events
                    GROUP BY event_type
                    ORDER BY count DESC
                """)
                by_type = dict(cursor.fetchall())

                cursor.execute("""
                    SELECT tier, COUNT(*) as count
                    FROM memory_events
                    GROUP BY tier
                """)
                by_tier = dict(cursor.fetchall())

                cursor.execute("""
                    SELECT COUNT(*) FROM memory_events
                    WHERE created_at >= datetime('now', '-24 hours')
                """)
                last_24h = cursor.fetchone()[0]

            return {
                "total_events": total,
                "events_last_24h": last_24h,
                "by_type": by_type,
                "by_tier": by_tier,
                "buffer_size": len(self._buffer),
                "listener_count": len(self._listeners),
            }

        except Exception as e:
            logger.error("Failed to get event stats: %s", e)
            return {"total_events": 0, "error": str(e)}

    # =========================================================================
    # Tiered Retention (pruning)
    # =========================================================================

    def prune_events(
        self,
        hot_hours: int = DEFAULT_HOT_HOURS,
        warm_hours: int = DEFAULT_WARM_HOURS,
        cold_hours: int = DEFAULT_COLD_HOURS,
    ) -> dict:
        """
        Apply tiered retention policy to events.

        Tiers:
            Hot  (0-48h)  — Keep all events
            Warm (2-14d)  — Keep events with importance >= 5
            Cold (14-30d) — Keep daily aggregates only
            Archive (30d+) — Delete (stats preserved in pattern_learner)

        Args:
            hot_hours: Hours to keep all events (default 48)
            warm_hours: Hours before warm→cold transition (default 336 / 14 days)
            cold_hours: Hours before cold→archive (default 720 / 30 days)

        Returns:
            Dict with counts of events transitioned/pruned per tier
        """
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            stats = {"hot_to_warm": 0, "warm_to_cold": 0, "archived": 0}

            def _do_prune(conn):
                cursor = conn.cursor()
                now = datetime.now()

                # Hot → Warm: events older than hot_hours with importance < 5
                warm_cutoff = (now - timedelta(hours=hot_hours)).isoformat()
                cursor.execute("""
                    UPDATE memory_events
                    SET tier = 'warm'
                    WHERE tier = 'hot'
                      AND created_at < ?
                      AND importance < 5
                """, (warm_cutoff,))
                stats["hot_to_warm"] = cursor.rowcount

                # Warm → Cold: events older than warm_hours
                cold_cutoff = (now - timedelta(hours=warm_hours)).isoformat()
                cursor.execute("""
                    DELETE FROM memory_events
                    WHERE tier = 'warm'
                      AND created_at < ?
                """, (cold_cutoff,))
                stats["warm_to_cold"] = cursor.rowcount

                # Archive: delete events older than cold_hours
                archive_cutoff = (now - timedelta(hours=cold_hours)).isoformat()
                cursor.execute("""
                    DELETE FROM memory_events
                    WHERE created_at < ?
                """, (archive_cutoff,))
                stats["archived"] = cursor.rowcount

                conn.commit()

            mgr.execute_write(_do_prune)

            logger.info(
                "Event pruning complete: hot→warm=%d, warm→cold=%d, archived=%d",
                stats["hot_to_warm"], stats["warm_to_cold"], stats["archived"]
            )
            return stats

        except Exception as e:
            logger.error("Event pruning failed: %s", e)
            return {"error": str(e)}
