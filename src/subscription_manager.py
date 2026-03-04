#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
SubscriptionManager — Manages durable and ephemeral event subscriptions.

Subscribers register interest in specific event types and receive matching
events via their chosen channel (SSE, WebSocket, Webhook).

Subscription Types:
    Durable (default)  — Persisted to DB, survives disconnect, auto-replay on reconnect
    Ephemeral (opt-in) — In-memory only, dies on disconnect

Filter Syntax:
    {
        "event_types": ["memory.created", "memory.deleted"],  // null = all types
        "min_importance": 5,                                    // null = no filter
        "source_protocols": ["mcp", "cli"],                     // null = all protocols
        "projects": ["myapp"]                                   // null = all projects
    }
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger("superlocalmemory.subscriptions")


class SubscriptionManager:
    """
    Manages event subscriptions for the Event Bus.

    Thread-safe. Durable subscriptions persist to SQLite. Ephemeral
    subscriptions are in-memory only.
    """

    _instances: Dict[str, "SubscriptionManager"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "SubscriptionManager":
        """Get or create the singleton SubscriptionManager."""
        if db_path is None:
            db_path = Path.home() / ".claude-memory" / "memory.db"
        key = str(db_path)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(db_path)
            return cls._instances[key]

    @classmethod
    def reset_instance(cls, db_path: Optional[Path] = None) -> None:
        """Remove singleton. Used for testing."""
        with cls._instances_lock:
            if db_path is None:
                cls._instances.clear()
            else:
                key = str(db_path)
                if key in cls._instances:
                    del cls._instances[key]

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

        # Ephemeral subscriptions (in-memory only)
        self._ephemeral: Dict[str, dict] = {}
        self._ephemeral_lock = threading.Lock()

        self._init_schema()
        logger.info("SubscriptionManager initialized: db=%s", self.db_path)

    def _init_schema(self):
        """Create subscriptions table if it doesn't exist."""
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _create(conn):
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS subscriptions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subscriber_id TEXT NOT NULL UNIQUE,
                        channel TEXT NOT NULL,
                        filter TEXT NOT NULL DEFAULT '{}',
                        webhook_url TEXT,
                        durable INTEGER DEFAULT 1,
                        last_event_id INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_subs_channel
                    ON subscriptions(channel)
                ''')
                conn.commit()

            mgr.execute_write(_create)
        except ImportError:
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS subscriptions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subscriber_id TEXT NOT NULL UNIQUE,
                        channel TEXT NOT NULL,
                        filter TEXT NOT NULL DEFAULT '{}',
                        webhook_url TEXT,
                        durable INTEGER DEFAULT 1,
                        last_event_id INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_subs_channel ON subscriptions(channel)')
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Subscribe / Unsubscribe
    # =========================================================================

    def subscribe(
        self,
        subscriber_id: str,
        channel: str = "sse",
        filter_obj: Optional[dict] = None,
        webhook_url: Optional[str] = None,
        durable: bool = True,
    ) -> dict:
        """
        Register a subscription.

        Args:
            subscriber_id: Unique identifier for the subscriber
            channel: Delivery channel — 'sse', 'websocket', 'webhook'
            filter_obj: Event filter (see module docstring for syntax)
            webhook_url: URL for webhook channel (required if channel='webhook')
            durable: If True, persists to DB; if False, in-memory only

        Returns:
            Subscription dict with id and details

        Raises:
            ValueError: If channel is invalid or webhook_url missing for webhook channel
        """
        if channel not in ("sse", "websocket", "webhook"):
            raise ValueError(f"Invalid channel: {channel}. Must be sse, websocket, or webhook")

        if channel == "webhook" and not webhook_url:
            raise ValueError("webhook_url is required for webhook channel")

        # Validate webhook URL format
        if webhook_url and not (webhook_url.startswith("http://") or webhook_url.startswith("https://")):
            raise ValueError("webhook_url must start with http:// or https://")

        filter_json = json.dumps(filter_obj or {})
        now = datetime.now().isoformat()

        sub = {
            "subscriber_id": subscriber_id,
            "channel": channel,
            "filter": filter_obj or {},
            "webhook_url": webhook_url,
            "durable": durable,
            "last_event_id": 0,
            "created_at": now,
        }

        if durable:
            self._persist_subscription(sub, filter_json)
        else:
            with self._ephemeral_lock:
                self._ephemeral[subscriber_id] = sub

        logger.info("Subscription created: id=%s, channel=%s, durable=%s", subscriber_id, channel, durable)
        return sub

    def _persist_subscription(self, sub: dict, filter_json: str):
        """Save durable subscription to database."""
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _upsert(conn):
                conn.execute('''
                    INSERT INTO subscriptions (subscriber_id, channel, filter, webhook_url, durable, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(subscriber_id) DO UPDATE SET
                        channel = excluded.channel,
                        filter = excluded.filter,
                        webhook_url = excluded.webhook_url,
                        updated_at = excluded.updated_at
                ''', (
                    sub["subscriber_id"],
                    sub["channel"],
                    filter_json,
                    sub.get("webhook_url"),
                    sub["created_at"],
                    sub["created_at"],
                ))
                conn.commit()

            mgr.execute_write(_upsert)
        except Exception as e:
            logger.error("Failed to persist subscription: %s", e)

    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Remove a subscription (durable or ephemeral).

        Args:
            subscriber_id: ID of the subscription to remove

        Returns:
            True if subscription was found and removed
        """
        removed = False

        # Remove ephemeral
        with self._ephemeral_lock:
            if subscriber_id in self._ephemeral:
                del self._ephemeral[subscriber_id]
                removed = True

        # Remove durable
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _delete(conn):
                conn.execute("DELETE FROM subscriptions WHERE subscriber_id = ?", (subscriber_id,))
                conn.commit()
                return conn.total_changes > 0

            if mgr.execute_write(_delete):
                removed = True
        except Exception as e:
            logger.error("Failed to delete subscription: %s", e)

        return removed

    def update_last_event_id(self, subscriber_id: str, event_id: int) -> None:
        """Update the last event ID received by a durable subscriber (for replay)."""
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _update(conn):
                conn.execute(
                    "UPDATE subscriptions SET last_event_id = ?, updated_at = ? WHERE subscriber_id = ?",
                    (event_id, datetime.now().isoformat(), subscriber_id)
                )
                conn.commit()

            mgr.execute_write(_update)
        except Exception as e:
            logger.error("Failed to update last_event_id: %s", e)

    # =========================================================================
    # Query Subscriptions
    # =========================================================================

    def get_matching_subscribers(self, event: dict) -> List[dict]:
        """
        Get all subscriptions that match a given event.

        Applies filter logic: event_types, min_importance, source_protocols.

        Args:
            event: Event dict with event_type, importance, source_protocol, etc.

        Returns:
            List of matching subscription dicts
        """
        all_subs = self.list_subscriptions()
        matching = []

        for sub in all_subs:
            if self._matches_filter(sub.get("filter", {}), event):
                matching.append(sub)

        return matching

    def _matches_filter(self, filter_obj: dict, event: dict) -> bool:
        """Check if an event matches a subscription filter."""
        if not filter_obj:
            return True  # No filter = match all

        # Event type filter
        allowed_types = filter_obj.get("event_types")
        if allowed_types and event.get("event_type") not in allowed_types:
            return False

        # Importance filter
        min_importance = filter_obj.get("min_importance")
        if min_importance and (event.get("importance", 0) < min_importance):
            return False

        # Protocol filter
        allowed_protocols = filter_obj.get("source_protocols")
        if allowed_protocols and event.get("source_protocol") not in allowed_protocols:
            return False

        return True

    def list_subscriptions(self) -> List[dict]:
        """Get all active subscriptions (durable + ephemeral)."""
        subs = []

        # Ephemeral
        with self._ephemeral_lock:
            subs.extend(list(self._ephemeral.values()))

        # Durable (from DB)
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT subscriber_id, channel, filter, webhook_url, durable,
                           last_event_id, created_at, updated_at
                    FROM subscriptions
                """)
                for row in cursor.fetchall():
                    filter_obj = {}
                    try:
                        filter_obj = json.loads(row[2]) if row[2] else {}
                    except (json.JSONDecodeError, TypeError):
                        pass

                    subs.append({
                        "subscriber_id": row[0],
                        "channel": row[1],
                        "filter": filter_obj,
                        "webhook_url": row[3],
                        "durable": bool(row[4]),
                        "last_event_id": row[5],
                        "created_at": row[6],
                        "updated_at": row[7],
                    })
        except Exception as e:
            logger.error("Failed to list durable subscriptions: %s", e)

        return subs

    def get_subscription(self, subscriber_id: str) -> Optional[dict]:
        """Get a specific subscription by ID."""
        # Check ephemeral first
        with self._ephemeral_lock:
            if subscriber_id in self._ephemeral:
                return self._ephemeral[subscriber_id]

        # Check durable
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT subscriber_id, channel, filter, webhook_url, durable, last_event_id FROM subscriptions WHERE subscriber_id = ?",
                    (subscriber_id,)
                )
                row = cursor.fetchone()
                if row:
                    filter_obj = {}
                    try:
                        filter_obj = json.loads(row[2]) if row[2] else {}
                    except (json.JSONDecodeError, TypeError):
                        pass
                    return {
                        "subscriber_id": row[0],
                        "channel": row[1],
                        "filter": filter_obj,
                        "webhook_url": row[3],
                        "durable": bool(row[4]),
                        "last_event_id": row[5],
                    }
        except Exception as e:
            logger.error("Failed to get subscription: %s", e)

        return None
