# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tamper-proof hash-chain audit log for compliance.

Every operation is logged with a SHA-256 hash that includes the previous
entry's hash, creating a chain. Tampering with any entry breaks the chain
and is detectable via verify_integrity().

The audit chain uses its OWN sqlite3 connection (not shared DB manager)
for independence — audit must survive even if the main DB is corrupted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

_GENESIS_HASH = "genesis"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_chain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    agent_id TEXT DEFAULT '',
    profile_id TEXT DEFAULT '',
    content_hash TEXT DEFAULT '',
    prev_hash TEXT DEFAULT '',
    event_hash TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
"""


def _compute_hash(
    prev_hash: str,
    operation: str,
    agent_id: str,
    profile_id: str,
    content_hash: str,
    timestamp: str,
) -> str:
    """Compute SHA-256 hash for an audit entry.

    The hash incorporates: prev_hash + operation + agent_id +
    profile_id + content_hash + timestamp.
    """
    payload = (
        f"{prev_hash}{operation}{agent_id}"
        f"{profile_id}{content_hash}{timestamp}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditChain:
    """Tamper-proof hash-chain audit log for compliance.

    Every operation is logged with a hash that includes the previous hash,
    creating a chain. Tampering with any entry breaks the chain.

    Uses its own SQLite connection for independence from main DB.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self._db_path = str(db_path) if db_path else ":memory:"
        self._is_memory = self._db_path == ":memory:"
        self._lock = threading.Lock()
        # For in-memory DBs, keep a persistent connection (each connect()
        # to ":memory:" creates a separate empty database).
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if self._is_memory:
            self._persistent_conn = self._make_conn()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_conn_from_path(path: str) -> sqlite3.Connection:
        """Create a configured SQLite connection."""
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        # C4: audit chain holds a tamper-evident record — keep it owner-only.
        try:
            from superlocalmemory.core.security_primitives import harden_db_perms
            harden_db_perms(path)
        except Exception:
            pass
        return conn

    def _make_conn(self) -> sqlite3.Connection:
        """Create a new configured connection to the audit database."""
        return self._make_conn_from_path(self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        """Get a connection to the audit database.

        For in-memory databases, returns the persistent connection.
        For file databases, creates a new connection each time.
        """
        if self._is_memory and self._persistent_conn is not None:
            return self._persistent_conn
        return self._make_conn()

    def _release_conn(self, conn: sqlite3.Connection) -> None:
        """Release a connection. Only closes file-based connections."""
        if not self._is_memory:
            conn.close()

    def _init_db(self) -> None:
        """Initialize the audit_chain table."""
        conn = self._get_conn()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            self._release_conn(conn)

    def _get_last_hash(self, conn: sqlite3.Connection) -> str:
        """Get the hash of the most recent entry, or genesis."""
        row = conn.execute(
            "SELECT event_hash FROM audit_chain "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["event_hash"] if row else _GENESIS_HASH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        operation: str,
        agent_id: str = "",
        profile_id: str = "",
        content_hash: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log an audit event. Returns the event hash.

        Args:
            operation: Type of operation (store, recall, delete, etc.).
            agent_id: ID of the agent performing the operation.
            profile_id: ID of the profile being accessed.
            content_hash: Hash of the content involved (optional).
            metadata: Additional metadata dict (optional).

        Returns:
            The SHA-256 event hash for this entry.
        """
        ts = datetime.now(timezone.utc).isoformat()
        metadata_str = json.dumps(metadata or {}, sort_keys=True)

        with self._lock:
            conn = self._get_conn()
            try:
                prev_hash = self._get_last_hash(conn)
                event_hash = _compute_hash(
                    prev_hash, operation, agent_id,
                    profile_id, content_hash, ts,
                )
                conn.execute(
                    "INSERT INTO audit_chain "
                    "(timestamp, operation, agent_id, profile_id, "
                    " content_hash, prev_hash, event_hash, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ts, operation, agent_id, profile_id,
                        content_hash, prev_hash, event_hash, metadata_str,
                    ),
                )
                conn.commit()
                return event_hash
            finally:
                self._release_conn(conn)

    def query(
        self,
        profile_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        operation: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit events with filters.

        Args:
            profile_id: Filter by profile ID.
            agent_id: Filter by agent ID.
            operation: Filter by operation type.
            start_date: ISO date string for range start.
            end_date: ISO date string for range end.
            limit: Maximum number of events to return.

        Returns:
            List of event dicts ordered by id descending.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if profile_id is not None:
            clauses.append("profile_id = ?")
            params.append(profile_id)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if operation is not None:
            clauses.append("operation = ?")
            params.append(operation)
        if start_date is not None:
            clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("timestamp <= ?")
            params.append(end_date)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT id, timestamp, operation, agent_id, profile_id, "
            f"content_hash, prev_hash, event_hash, metadata "
            f"FROM audit_chain {where} "
            f"ORDER BY id DESC LIMIT ?"
        )
        params.append(limit)

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                self._release_conn(conn)

    def verify_integrity(self) -> bool:
        """Verify the entire hash chain. Returns True if chain is intact.

        Walks every entry in order, recomputes each hash, and verifies
        it matches the stored hash. Any mismatch means tampering.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT id, timestamp, operation, agent_id, profile_id, "
                "content_hash, prev_hash, event_hash "
                "FROM audit_chain ORDER BY id"
            ).fetchall()
        finally:
            self._release_conn(conn)

        if not rows:
            return True

        expected_prev = _GENESIS_HASH
        for row in rows:
            row_dict = dict(row)

            # Check the prev_hash link
            if row_dict["prev_hash"] != expected_prev:
                logger.warning(
                    "Audit chain prev_hash mismatch at entry %d",
                    row_dict["id"],
                )
                return False

            # Recompute the entry hash
            computed = _compute_hash(
                row_dict["prev_hash"],
                row_dict["operation"],
                row_dict["agent_id"],
                row_dict["profile_id"],
                row_dict["content_hash"],
                row_dict["timestamp"],
            )
            if computed != row_dict["event_hash"]:
                logger.warning(
                    "Audit chain event_hash mismatch at entry %d",
                    row_dict["id"],
                )
                return False

            expected_prev = row_dict["event_hash"]

        return True

    def get_stats(self) -> dict[str, int]:
        """Get audit statistics (event counts by operation type).

        Returns:
            Dict mapping operation names to their counts, plus
            a 'total' key with the chain length.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT operation, COUNT(*) AS cnt "
                "FROM audit_chain GROUP BY operation"
            ).fetchall()
            stats: dict[str, int] = {}
            total = 0
            for row in rows:
                count = row["cnt"]
                stats[row["operation"]] = count
                total += count
            stats["total"] = total
            return stats
        finally:
            self._release_conn(conn)
