#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
AgentRegistry — Tracks which AI agents connect to SuperLocalMemory,
what they write, when, and via which protocol.

Every MCP client (Claude, Cursor, Windsurf), CLI call, and REST API request
gets registered here. This powers:
    - Dashboard "Connected Agents" panel
    - Trust scoring input (v2.5 silent collection)
    - Provenance tracking (who created which memory)
    - Usage analytics

Agent Identity:
    Each agent gets a unique agent_id derived from its protocol + name.
    Example: "mcp:claude-desktop", "cli:terminal", "rest:api-client"

Protocols:
    mcp     — Model Context Protocol (Claude Desktop, Cursor, Windsurf, etc.)
    cli     — Command-line interface (slm command, bin/ scripts)
    rest    — REST API (api_server.py)
    python  — Direct Python import
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger("superlocalmemory.agents")


class AgentRegistry:
    """
    Registry of all agents that interact with SuperLocalMemory.

    Singleton per database path. Thread-safe.
    """

    _instances: Dict[str, "AgentRegistry"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "AgentRegistry":
        """Get or create the singleton AgentRegistry."""
        if db_path is None:
            db_path = Path.home() / ".claude-memory" / "memory.db"
        key = str(db_path)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(db_path)
            return cls._instances[key]

    @classmethod
    def reset_instance(cls, db_path: Optional[Path] = None):
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
        self._init_schema()
        logger.info("AgentRegistry initialized: db=%s", self.db_path)

    def _init_schema(self):
        """Create agent_registry table if it doesn't exist."""
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _create(conn):
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS agent_registry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id TEXT NOT NULL UNIQUE,
                        agent_name TEXT,
                        protocol TEXT NOT NULL,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        memories_written INTEGER DEFAULT 0,
                        memories_recalled INTEGER DEFAULT 0,
                        trust_score REAL DEFAULT 0.667,
                        metadata TEXT DEFAULT '{}'
                    )
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_agent_protocol
                    ON agent_registry(protocol)
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_agent_last_seen
                    ON agent_registry(last_seen)
                ''')
                conn.commit()

            mgr.execute_write(_create)
        except ImportError:
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS agent_registry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id TEXT NOT NULL UNIQUE,
                        agent_name TEXT,
                        protocol TEXT NOT NULL,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        memories_written INTEGER DEFAULT 0,
                        memories_recalled INTEGER DEFAULT 0,
                        trust_score REAL DEFAULT 0.667,
                        metadata TEXT DEFAULT '{}'
                    )
                ''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_agent_protocol ON agent_registry(protocol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_agent_last_seen ON agent_registry(last_seen)')
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # Agent Registration
    # =========================================================================

    def register_agent(
        self,
        agent_id: str,
        agent_name: Optional[str] = None,
        protocol: str = "cli",
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Register or update an agent in the registry.

        If the agent already exists, updates last_seen and metadata.
        If new, creates the entry with trust_score=0.667 (Beta(2,1) prior).

        Args:
            agent_id: Unique identifier (e.g., "mcp:claude-desktop")
            agent_name: Human-readable name (e.g., "Claude Desktop")
            protocol: Connection protocol (mcp, cli, rest, python)
            metadata: Additional agent info (version, capabilities, etc.)

        Returns:
            Agent record dict
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string")

        valid_protocols = ("mcp", "cli", "rest", "python")
        if protocol not in valid_protocols:
            raise ValueError(f"Invalid protocol: {protocol}. Must be one of {valid_protocols}")

        now = datetime.now().isoformat()
        meta_json = json.dumps(metadata or {})

        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _upsert(conn):
                conn.execute('''
                    INSERT INTO agent_registry (agent_id, agent_name, protocol, first_seen, last_seen, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(agent_id) DO UPDATE SET
                        last_seen = excluded.last_seen,
                        metadata = excluded.metadata,
                        agent_name = COALESCE(excluded.agent_name, agent_registry.agent_name)
                ''', (agent_id, agent_name, protocol, now, now, meta_json))
                conn.commit()

            mgr.execute_write(_upsert)
        except ImportError:
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute('''
                    INSERT INTO agent_registry (agent_id, agent_name, protocol, first_seen, last_seen, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(agent_id) DO UPDATE SET
                        last_seen = excluded.last_seen,
                        metadata = excluded.metadata,
                        agent_name = COALESCE(excluded.agent_name, agent_registry.agent_name)
                ''', (agent_id, agent_name, protocol, now, now, meta_json))
                conn.commit()
            finally:
                conn.close()

        # Emit agent.connected event
        try:
            from event_bus import EventBus
            bus = EventBus.get_instance(self.db_path)
            bus.emit("agent.connected", payload={
                "agent_id": agent_id,
                "agent_name": agent_name,
                "protocol": protocol,
            })
        except Exception:
            pass

        logger.info("Agent registered: id=%s, protocol=%s", agent_id, protocol)
        return self.get_agent(agent_id) or {"agent_id": agent_id}

    def record_write(self, agent_id: str):
        """Increment memories_written counter and update last_seen."""
        self._increment_counter(agent_id, "memories_written")

    def record_recall(self, agent_id: str):
        """Increment memories_recalled counter and update last_seen."""
        self._increment_counter(agent_id, "memories_recalled")

    def _increment_counter(self, agent_id: str, column: str):
        """Increment a counter column for an agent."""
        if column not in ("memories_written", "memories_recalled"):
            return

        now = datetime.now().isoformat()

        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _inc(conn):
                conn.execute(
                    f"UPDATE agent_registry SET {column} = {column} + 1, last_seen = ? WHERE agent_id = ?",
                    (now, agent_id)
                )
                conn.commit()

            mgr.execute_write(_inc)
        except Exception as e:
            logger.error("Failed to increment %s for %s: %s", column, agent_id, e)

    # =========================================================================
    # Query Agents
    # =========================================================================

    def get_agent(self, agent_id: str) -> Optional[dict]:
        """Get a specific agent by ID."""
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT agent_id, agent_name, protocol, first_seen, last_seen,
                           memories_written, memories_recalled, trust_score, metadata
                    FROM agent_registry WHERE agent_id = ?
                """, (agent_id,))
                row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_dict(row)
        except Exception as e:
            logger.error("Failed to get agent %s: %s", agent_id, e)
            return None

    def list_agents(
        self,
        protocol: Optional[str] = None,
        limit: int = 50,
        active_since_hours: Optional[int] = None,
    ) -> List[dict]:
        """
        List registered agents with optional filtering.

        Args:
            protocol: Filter by protocol (mcp, cli, rest, python)
            limit: Max agents to return
            active_since_hours: Only agents seen within N hours

        Returns:
            List of agent dicts, ordered by last_seen descending
        """
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()

                query = """
                    SELECT agent_id, agent_name, protocol, first_seen, last_seen,
                           memories_written, memories_recalled, trust_score, metadata
                    FROM agent_registry WHERE 1=1
                """
                params = []

                if protocol:
                    query += " AND protocol = ?"
                    params.append(protocol)

                if active_since_hours:
                    query += " AND last_seen >= datetime('now', '-' || ? || ' hours')"
                    params.append(active_since_hours)

                query += " ORDER BY last_seen DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

            return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to list agents: %s", e)
            return []

    def list_active_agents(self, timeout_minutes: int = 5) -> List[dict]:
        """
        List only active agents (seen within timeout_minutes).

        Used by dashboard to filter out ghost/disconnected agents.
        Default: agents seen within last 5 minutes are considered active.

        Args:
            timeout_minutes: Consider agents active if seen within this many minutes

        Returns:
            List of active agent dicts
        """
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT agent_id, agent_name, protocol, first_seen, last_seen,
                           memories_written, memories_recalled, trust_score, metadata
                    FROM agent_registry
                    WHERE last_seen >= datetime('now', '-' || ? || ' minutes')
                    ORDER BY last_seen DESC
                """, (timeout_minutes,))
                rows = cursor.fetchall()

            return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to list active agents: %s", e)
            return []

    def get_stats(self) -> dict:
        """Get agent registry statistics."""
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM agent_registry")
                total = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT protocol, COUNT(*) FROM agent_registry
                    GROUP BY protocol ORDER BY COUNT(*) DESC
                """)
                by_protocol = dict(cursor.fetchall())

                cursor.execute("""
                    SELECT SUM(memories_written), SUM(memories_recalled)
                    FROM agent_registry
                """)
                sums = cursor.fetchone()

                cursor.execute("""
                    SELECT COUNT(*) FROM agent_registry
                    WHERE last_seen >= datetime('now', '-24 hours')
                """)
                active_24h = cursor.fetchone()[0]

            return {
                "total_agents": total,
                "active_last_24h": active_24h,
                "by_protocol": by_protocol,
                "total_writes": sums[0] or 0,
                "total_recalls": sums[1] or 0,
            }
        except Exception as e:
            logger.error("Failed to get agent stats: %s", e)
            return {"total_agents": 0, "error": str(e)}

    def _row_to_dict(self, row: tuple) -> dict:
        """Convert a database row to an agent dict."""
        metadata = {}
        try:
            metadata = json.loads(row[8]) if row[8] else {}
        except (json.JSONDecodeError, TypeError):
            pass

        return {
            "agent_id": row[0],
            "agent_name": row[1],
            "protocol": row[2],
            "first_seen": row[3],
            "last_seen": row[4],
            "memories_written": row[5],
            "memories_recalled": row[6],
            "trust_score": row[7],
            "metadata": metadata,
        }
