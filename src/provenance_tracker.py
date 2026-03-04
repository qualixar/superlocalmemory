#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
ProvenanceTracker — Tracks the origin and lineage of every memory.

Adds provenance columns to the memories table:
    created_by       — Agent ID that created this memory (e.g., "mcp:claude-desktop")
    source_protocol  — Protocol used (mcp, cli, rest, python)
    trust_score      — Trust score at time of creation (default 1.0)
    provenance_chain — JSON array of derivation history

This enables:
    - "Who wrote this?" queries for the dashboard
    - Trust-weighted recall (v2.6 — higher trust = higher ranking)
    - Audit trail for enterprise compliance (v3.0)
    - Memory lineage tracking (if agent B derives from agent A's memory)

Column migration is safe: uses ALTER TABLE ADD COLUMN with try/except.
Old databases without provenance columns work fine — values default.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("superlocalmemory.provenance")


class ProvenanceTracker:
    """
    Tracks provenance (origin) metadata for memories.

    Singleton per database path. Thread-safe.
    """

    _instances: Dict[str, "ProvenanceTracker"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "ProvenanceTracker":
        """Get or create the singleton ProvenanceTracker."""
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
        self._init_schema()
        logger.info("ProvenanceTracker initialized: db=%s", self.db_path)

    def _init_schema(self):
        """
        Add provenance columns to memories table (safe migration).

        Uses ALTER TABLE ADD COLUMN wrapped in try/except — safe for:
        - Fresh databases (columns don't exist yet)
        - Existing databases (columns might already exist)
        - Concurrent migrations (OperationalError caught)
        """
        provenance_columns = {
            'created_by': "TEXT DEFAULT 'user'",
            'source_protocol': "TEXT DEFAULT 'cli'",
            'trust_score': "REAL DEFAULT 1.0",
            'provenance_chain': "TEXT DEFAULT '[]'",
        }

        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _migrate(conn):
                cursor = conn.cursor()

                # Check existing columns
                cursor.execute("PRAGMA table_info(memories)")
                existing = {row[1] for row in cursor.fetchall()}

                for col_name, col_type in provenance_columns.items():
                    if col_name not in existing:
                        try:
                            cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
                        except sqlite3.OperationalError:
                            pass  # Column already exists (concurrent migration)

                # Index for provenance queries
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_by ON memories(created_by)")
                except sqlite3.OperationalError:
                    pass

                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_protocol ON memories(source_protocol)")
                except sqlite3.OperationalError:
                    pass

                conn.commit()

            mgr.execute_write(_migrate)

        except ImportError:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("PRAGMA table_info(memories)")
            existing = {row[1] for row in cursor.fetchall()}

            for col_name, col_type in provenance_columns.items():
                if col_name not in existing:
                    try:
                        cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError:
                        pass

            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_by ON memories(created_by)")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_protocol ON memories(source_protocol)")
            except sqlite3.OperationalError:
                pass

            conn.commit()
            conn.close()

    # =========================================================================
    # Record Provenance
    # =========================================================================

    def record_provenance(
        self,
        memory_id: int,
        created_by: str = "user",
        source_protocol: str = "cli",
        trust_score: float = 1.0,
        derived_from: Optional[int] = None,
    ) -> None:
        """
        Record provenance metadata for a memory.

        Called after a memory is created. Updates the provenance columns
        on the memories table row.

        Args:
            memory_id: ID of the memory to annotate
            created_by: Agent ID that created this memory
            source_protocol: Protocol used (mcp, cli, rest, python)
            trust_score: Trust score at time of creation
            derived_from: If this memory was derived from another, its ID
        """
        trust_score = max(0.0, min(1.0, trust_score))
        chain = json.dumps([derived_from] if derived_from else [])

        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            def _update(conn):
                conn.execute('''
                    UPDATE memories
                    SET created_by = ?, source_protocol = ?,
                        trust_score = ?, provenance_chain = ?
                    WHERE id = ?
                ''', (created_by, source_protocol, trust_score, chain, memory_id))
                conn.commit()

            mgr.execute_write(_update)

        except Exception as e:
            # Provenance failure must never break core operations
            logger.error("Failed to record provenance for memory %d: %s", memory_id, e)

    # =========================================================================
    # Query Provenance
    # =========================================================================

    def get_provenance(self, memory_id: int) -> Optional[dict]:
        """
        Get provenance metadata for a specific memory.

        Args:
            memory_id: Memory ID to query

        Returns:
            Dict with created_by, source_protocol, trust_score, provenance_chain
            or None if memory not found
        """
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, created_by, source_protocol, trust_score, provenance_chain
                    FROM memories WHERE id = ?
                """, (memory_id,))
                row = cursor.fetchone()

            if not row:
                return None

            chain = []
            try:
                chain = json.loads(row[4]) if row[4] else []
            except (json.JSONDecodeError, TypeError):
                pass

            return {
                "memory_id": row[0],
                "created_by": row[1] or "user",
                "source_protocol": row[2] or "cli",
                "trust_score": row[3] if row[3] is not None else 1.0,
                "provenance_chain": chain,
            }

        except Exception as e:
            logger.error("Failed to get provenance for memory %d: %s", memory_id, e)
            return None

    def get_memories_by_agent(self, agent_id: str, limit: int = 50) -> list:
        """
        Get all memories created by a specific agent.

        Args:
            agent_id: Agent ID to query
            limit: Max results

        Returns:
            List of (memory_id, created_at, trust_score) tuples
        """
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, created_at, trust_score
                    FROM memories
                    WHERE created_by = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (agent_id, limit))
                return [
                    {"memory_id": r[0], "created_at": r[1], "trust_score": r[2]}
                    for r in cursor.fetchall()
                ]

        except Exception as e:
            logger.error("Failed to get memories by agent %s: %s", agent_id, e)
            return []

    def get_provenance_stats(self) -> dict:
        """Get provenance statistics across all memories."""
        try:
            from db_connection_manager import DbConnectionManager
            mgr = DbConnectionManager.get_instance(self.db_path)

            with mgr.read_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT created_by, COUNT(*) as count
                    FROM memories
                    WHERE created_by IS NOT NULL
                    GROUP BY created_by
                    ORDER BY count DESC
                """)
                by_agent = dict(cursor.fetchall())

                cursor.execute("""
                    SELECT source_protocol, COUNT(*) as count
                    FROM memories
                    WHERE source_protocol IS NOT NULL
                    GROUP BY source_protocol
                    ORDER BY count DESC
                """)
                by_protocol = dict(cursor.fetchall())

                cursor.execute("SELECT AVG(trust_score) FROM memories WHERE trust_score IS NOT NULL")
                avg_trust = cursor.fetchone()[0]

            return {
                "by_agent": by_agent,
                "by_protocol": by_protocol,
                "avg_trust_score": round(avg_trust, 3) if avg_trust else 1.0,
            }

        except Exception as e:
            logger.error("Failed to get provenance stats: %s", e)
            return {"by_agent": {}, "by_protocol": {}, "error": str(e)}
