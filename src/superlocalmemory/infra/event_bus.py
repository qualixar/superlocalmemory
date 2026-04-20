# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""EventBus -- Real-time event broadcasting for memory operations.

Thread-safe singleton per DB path. In-memory deque buffer + SQLite persistence.
Listener callbacks run on the emitter's thread.
"""

import json
import logging
import sqlite3
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("superlocalmemory.events")

# Default retention windows (hours)
DEFAULT_HOT_HOURS = 48
DEFAULT_WARM_HOURS = 14 * 24  # 14 days
DEFAULT_COLD_HOURS = 30 * 24  # 30 days

# In-memory buffer size for real-time delivery
EVENT_BUFFER_SIZE = 200

# Valid event types (V3 superset)
VALID_EVENT_TYPES = frozenset([
    "memory.stored",        # New memory written (was memory.created in V2)
    "memory.updated",       # Existing memory modified
    "memory.deleted",       # Memory removed
    "memory.recalled",      # Memory retrieved by an agent
    "graph.updated",        # Knowledge graph rebuilt
    "pattern.learned",      # New pattern detected
    "agent.connected",      # New agent connects
    "agent.disconnected",   # Agent disconnects
    "trust.signal",         # V3: trust score change
    "compliance.audit",     # V3: compliance event logged
    "learning.feedback",    # V3: learning feedback received
    # CodeGraph events (v3.4) — NOTE: "graph.updated" is SLM entity graph, "code_graph.*" is AST code graph
    "code_graph.built",          # Full code graph build completed
    "code_graph.updated",        # Incremental code graph update completed
    "code_graph.node_changed",   # Function/class signature or body changed
    "code_graph.node_deleted",   # Function/class/file removed from codebase
])


class EventBus:
    """
    Central event bus for SuperLocalMemory V3.

    Singleton per database path. Emits events to persistent storage and
    in-memory listeners simultaneously. Thread-safe.
    """

    _instances: Dict[str, "EventBus"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "EventBus":
        """Get or create the singleton EventBus for a database path."""
        if db_path is None:
            db_path = Path.home() / ".superlocalmemory" / "memory.db"

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

    def __init__(self, db_path: Path) -> None:
        """Initialize EventBus. Prefer get_instance() over direct construction."""
        self.db_path = Path(db_path)
        self._buffer: deque = deque(maxlen=EVENT_BUFFER_SIZE)
        self._buffer_lock = threading.Lock()
        self._event_counter = 0
        self._counter_lock = threading.Lock()
        self._listeners: List[Callable[[dict], None]] = []
        self._listeners_lock = threading.Lock()
        self._write_count = 0
        self._last_prune = datetime.now()
        self._init_schema()
        logger.info("EventBus initialized: db=%s", self.db_path)

    def _init_schema(self) -> None:
        """Create the memory_events table if it does not exist."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.cursor()
            cur.execute("""
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
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON memory_events(event_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_created ON memory_events(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_tier ON memory_events(tier)")
            conn.commit()
        finally:
            conn.close()

    def emit(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        memory_id: Optional[int] = None,
        source_agent: str = "user",
        source_protocol: str = "internal",
        importance: int = 5,
    ) -> Optional[int]:
        """Emit an event to all subscribers and persist to database."""
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event type: {event_type}. "
                f"Valid: {', '.join(sorted(VALID_EVENT_TYPES))}"
            )

        importance = max(1, min(10, importance))

        now = datetime.now(timezone.utc).isoformat()
        with self._counter_lock:
            self._event_counter += 1
            seq = self._event_counter

        event: Dict[str, Any] = {
            "seq": seq,
            "event_type": event_type,
            "memory_id": memory_id,
            "source_agent": source_agent,
            "source_protocol": source_protocol,
            "payload": payload or {},
            "importance": importance,
            "timestamp": now,
        }

        # 1. Persist
        event_id = self._persist_event(event)
        if event_id is not None:
            event["id"] = event_id

        # 2. Buffer
        with self._buffer_lock:
            self._buffer.append(event)

        # 3. Notify
        self._notify_listeners(event)

        logger.debug(
            "Event emitted: type=%s id=%s memory_id=%s",
            event_type, event_id, memory_id,
        )

        # Auto-prune heuristic
        self._write_count += 1
        if (
            self._write_count >= 100
            or (datetime.now() - self._last_prune).total_seconds() > 86400
        ):
            try:
                self.prune_events()
            except Exception:
                pass
            self._write_count = 0
            self._last_prune = datetime.now()

        return event_id

    # Alias for V3 compatibility
    publish = emit

    def _persist_event(self, event: dict) -> Optional[int]:
        """Persist event to the memory_events table. Returns row id or None."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO memory_events (event_type, memory_id, source_agent,"
                    " source_protocol, payload, importance, tier, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, 'hot', ?)",
                    (event["event_type"], event.get("memory_id"),
                     event["source_agent"], event["source_protocol"],
                     json.dumps(event["payload"]), event["importance"],
                     event["timestamp"]),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()
        except Exception as exc:
            logger.error("Failed to persist event: %s", exc)
            return None

    def add_listener(self, callback: Callable[[dict], None]) -> None:
        """Register a listener that receives every emitted event."""
        with self._listeners_lock:
            self._listeners.append(callback)

    # Alias
    subscribe = add_listener

    def remove_listener(self, callback: Callable[[dict], None]) -> None:
        """Remove a previously registered listener."""
        with self._listeners_lock:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

    # Alias
    unsubscribe = remove_listener

    def _notify_listeners(self, event: dict) -> None:
        """Call all registered listeners. Errors are logged, not raised."""
        with self._listeners_lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:
                logger.error("Event listener failed: %s", exc)

    def get_recent_events(
        self,
        since_id: Optional[int] = None,
        limit: int = 50,
        event_type: Optional[str] = None,
    ) -> List[dict]:
        """Get recent events from the database."""
        limit = min(limit, 200)

        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cur = conn.cursor()

                query = ("SELECT id, event_type, memory_id, source_agent,"
                         " source_protocol, payload, importance, tier, created_at"
                         " FROM memory_events WHERE 1=1")
                params: List[Any] = []

                if since_id is not None:
                    query += " AND id > ?"
                    params.append(since_id)

                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)

                query += " ORDER BY id ASC LIMIT ?"
                params.append(limit)

                cur.execute(query, params)
                rows = cur.fetchall()
            finally:
                conn.close()

            events: List[dict] = []
            for row in rows:
                try:
                    parsed = json.loads(row[5]) if row[5] else {}
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                events.append({
                    "id": row[0], "event_type": row[1], "memory_id": row[2],
                    "source_agent": row[3], "source_protocol": row[4],
                    "payload": parsed, "importance": row[6],
                    "tier": row[7], "timestamp": row[8],
                })
            return events

        except Exception as exc:
            logger.error("Failed to get recent events: %s", exc)
            return []

    def get_buffered_events(self, since_seq: int = 0) -> List[dict]:
        """Get events from the in-memory buffer (no DB hit)."""
        with self._buffer_lock:
            return [e for e in self._buffer if e.get("seq", 0) > since_seq]

    def get_event_stats(self) -> dict:
        """Get event system statistics."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cur = conn.cursor()

                total = cur.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
                cur.execute("SELECT event_type, COUNT(*) FROM memory_events GROUP BY event_type")
                by_type = dict(cur.fetchall())
                cur.execute("SELECT tier, COUNT(*) FROM memory_events GROUP BY tier")
                by_tier = dict(cur.fetchall())
                cur.execute("SELECT COUNT(*) FROM memory_events WHERE created_at >= datetime('now', '-24 hours')")
                last_24h = cur.fetchone()[0]
            finally:
                conn.close()

            return {
                **by_type,
                "total_events": total,
                "events_last_24h": last_24h,
                "by_type": by_type,
                "by_tier": by_tier,
                "buffer_size": len(self._buffer),
                "listener_count": len(self._listeners),
            }

        except Exception as exc:
            logger.error("Failed to get event stats: %s", exc)
            return {"total_events": 0, "error": str(exc)}

    def prune_events(
        self,
        hot_hours: int = DEFAULT_HOT_HOURS,
        warm_hours: int = DEFAULT_WARM_HOURS,
        cold_hours: int = DEFAULT_COLD_HOURS,
    ) -> dict:
        """Apply tiered retention policy to persisted events."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cur = conn.cursor()
                now = datetime.now()
                stats = {"hot_to_warm": 0, "warm_to_cold": 0, "archived": 0}

                # Hot -> Warm: older than hot_hours, importance < 5
                warm_cutoff = (now - timedelta(hours=hot_hours)).isoformat()
                cur.execute(
                    "UPDATE memory_events SET tier = 'warm' "
                    "WHERE tier = 'hot' AND created_at < ? AND importance < 5",
                    (warm_cutoff,),
                )
                stats["hot_to_warm"] = cur.rowcount

                # Warm -> Cold: delete warm events older than warm_hours
                cold_cutoff = (now - timedelta(hours=warm_hours)).isoformat()
                cur.execute(
                    "DELETE FROM memory_events "
                    "WHERE tier = 'warm' AND created_at < ?",
                    (cold_cutoff,),
                )
                stats["warm_to_cold"] = cur.rowcount

                # Archive: delete everything older than cold_hours
                archive_cutoff = (now - timedelta(hours=cold_hours)).isoformat()
                cur.execute(
                    "DELETE FROM memory_events WHERE created_at < ?",
                    (archive_cutoff,),
                )
                stats["archived"] = cur.rowcount

                conn.commit()
            finally:
                conn.close()

            logger.info(
                "Prune complete: hot->warm=%d warm->cold=%d archived=%d",
                stats["hot_to_warm"], stats["warm_to_cold"], stats["archived"],
            )
            return stats

        except Exception as exc:
            logger.error("Event pruning failed: %s", exc)
            return {"error": str(exc)}
