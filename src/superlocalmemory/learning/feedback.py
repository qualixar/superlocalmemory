#!/usr/bin/env python3
# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
"""
FeedbackCollector -- Multi-signal feedback collection for V3 learning.

Collects implicit and explicit relevance signals:
    - Implicit: recall_hit (fact returned), recall_miss (fact not in results)
    - Explicit: user_positive, user_negative, user_correction
    - Derived: access_pattern (frequent recall = positive signal)

Privacy:
    - Full query text is NEVER stored.
    - Queries are hashed to SHA-256[:16] for grouping.

Storage:
    Uses direct sqlite3 with a self-contained ``learning_feedback`` table.
    NOT coupled to V3 DatabaseManager -- this is a standalone data collector.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("superlocalmemory.learning.feedback")

# Signal type -> numeric value for downstream consumers
SIGNAL_VALUES: Dict[str, float] = {
    "recall_hit": 0.7,
    "recall_miss": 0.0,
    "user_positive": 1.0,
    "user_negative": 0.0,
    "user_correction": 0.2,
    "access_pattern": 0.6,
}

# Dashboard UI vocabulary -> (signal_type, signal_value). The dashboard speaks
# thumbs_up/thumbs_down/pin (explicit) and dwell_positive/dwell_negative
# (derived from modal dwell time). Unknown types fall back to a neutral
# user_correction signal rather than being dropped.
_DASHBOARD_SIGNAL_MAP: Dict[str, tuple[str, float]] = {
    "thumbs_up": ("user_positive", 1.0),
    "thumbs_down": ("user_negative", 0.0),
    "pin": ("user_pin", 1.0),
    "dwell_positive": ("dwell_positive", 0.6),
    "dwell_negative": ("dwell_negative", 0.2),
}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS learning_feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id   TEXT    NOT NULL,
    fact_id      TEXT    NOT NULL,
    signal_type  TEXT    NOT NULL,
    signal_value REAL    NOT NULL,
    query_hash   TEXT,
    created_at   TEXT    NOT NULL,
    metadata     TEXT
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_feedback_profile
    ON learning_feedback (profile_id, created_at DESC)
"""


def _utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _hash_query(query: str) -> str:
    """Privacy-preserving SHA-256[:16] query hash."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


class FeedbackCollector:
    """
    Collects multi-signal relevance feedback for the V3 learning system.

    Each instance owns a sqlite3 database at *db_path*.  All writes are
    serialised through a threading lock for safety.

    Args:
        db_path: Path to the sqlite3 database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create tables/indexes if they do not exist."""
        conn = self._connect()
        try:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL mode and busy timeout."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API: record implicit feedback
    # ------------------------------------------------------------------

    def record_implicit(
        self,
        profile_id: str,
        query: str,
        fact_ids_returned: List[str],
        fact_ids_available: List[str],
    ) -> int:
        """
        Record implicit feedback from a recall operation.

        Facts in *fact_ids_returned* get a ``recall_hit`` signal.
        Facts in *fact_ids_available* but NOT in *fact_ids_returned* get
        a ``recall_miss`` signal.

        Args:
            profile_id:        Profile that performed the recall.
            query:             The recall query (hashed, never stored raw).
            fact_ids_returned: Fact IDs that appeared in results.
            fact_ids_available: All candidate fact IDs for this query.

        Returns:
            Number of feedback records created.
        """
        if not profile_id or not query:
            return 0

        qhash = _hash_query(query)
        returned_set = set(fact_ids_returned)
        now = _utcnow_iso()
        records: list[tuple] = []

        for fid in returned_set:
            records.append((
                profile_id, fid, "recall_hit",
                SIGNAL_VALUES["recall_hit"], qhash, now, None,
            ))

        for fid in fact_ids_available:
            if fid not in returned_set:
                records.append((
                    profile_id, fid, "recall_miss",
                    SIGNAL_VALUES["recall_miss"], qhash, now, None,
                ))

        if not records:
            return 0

        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    "INSERT INTO learning_feedback "
                    "(profile_id, fact_id, signal_type, signal_value, "
                    "query_hash, created_at, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    records,
                )
                conn.commit()
                return len(records)
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API: record explicit feedback
    # ------------------------------------------------------------------

    def record_explicit(
        self,
        profile_id: str,
        fact_id: str,
        signal_type: str,
        value: float,
    ) -> Optional[int]:
        """
        Record explicit user feedback on a specific fact.

        Args:
            profile_id:  Profile providing feedback.
            fact_id:     The fact being rated.
            signal_type: One of ``user_positive``, ``user_negative``,
                         ``user_correction``, or any custom type.
            value:       Numeric signal value (0.0 to 1.0).

        Returns:
            Row ID of the inserted record, or None on error.
        """
        if not profile_id or not fact_id:
            return None

        clamped = max(0.0, min(1.0, float(value)))
        now = _utcnow_iso()

        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "INSERT INTO learning_feedback "
                    "(profile_id, fact_id, signal_type, signal_value, "
                    "query_hash, created_at, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (profile_id, fact_id, signal_type, clamped, None, now, None),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API: record dashboard feedback
    # ------------------------------------------------------------------

    def record_dashboard_feedback(
        self,
        memory_id: str,
        query: str = "",
        feedback_type: str = "",
        profile_id: str = "default",
    ) -> Optional[int]:
        """Record an explicit feedback signal raised from the dashboard UI.

        Maps the dashboard's vocabulary (``thumbs_up``/``thumbs_down``/``pin``
        and the dwell-derived ``dwell_positive``/``dwell_negative``) onto a
        stored ``(signal_type, signal_value)`` pair. ``memory_id`` is the fact
        id; the raw ``query`` is hashed and never stored. Returns the inserted
        row id, or ``None`` on missing ``memory_id``.

        This method restores the dashboard feedback path: the HTTP routes in
        ``server/routes/learning.py`` called it before it existed, so every
        thumbs/pin/dwell write raised ``AttributeError`` (issues #53/#59).
        """
        if not memory_id:
            return None
        signal_type, value = _DASHBOARD_SIGNAL_MAP.get(
            feedback_type, ("user_correction", 0.5),
        )
        qhash = _hash_query(query) if query else None
        now = _utcnow_iso()

        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "INSERT INTO learning_feedback "
                    "(profile_id, fact_id, signal_type, signal_value, "
                    "query_hash, created_at, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (profile_id or "default", str(memory_id), signal_type,
                     value, qhash, now, None),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API: read feedback
    # ------------------------------------------------------------------

    def get_feedback(
        self,
        profile_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent feedback records for a profile.

        Args:
            profile_id: Profile to query.
            limit:      Maximum records to return.

        Returns:
            List of dicts with keys: id, fact_id, signal_type,
            signal_value, query_hash, created_at.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, fact_id, signal_type, signal_value, "
                "query_hash, created_at "
                "FROM learning_feedback "
                "WHERE profile_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (profile_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_feedback_count(self, profile_id: str) -> int:
        """
        Return the total number of feedback records for a profile.

        Args:
            profile_id: Profile to query.

        Returns:
            Integer count of feedback records.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM learning_feedback WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API: summary
    # ------------------------------------------------------------------

    def get_summary(self, profile_id: str) -> Dict[str, Any]:
        """
        Return summary statistics for a profile's feedback.

        Returns:
            Dict with total, by_type counts, and latest timestamp.
        """
        conn = self._connect()
        try:
            total_row = conn.execute(
                "SELECT COUNT(*) FROM learning_feedback WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            total = total_row[0] if total_row else 0

            type_rows = conn.execute(
                "SELECT signal_type, COUNT(*) AS cnt "
                "FROM learning_feedback WHERE profile_id = ? "
                "GROUP BY signal_type",
                (profile_id,),
            ).fetchall()
            by_type = {r["signal_type"]: r["cnt"] for r in type_rows}

            latest_row = conn.execute(
                "SELECT MAX(created_at) FROM learning_feedback "
                "WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            latest = latest_row[0] if latest_row else None

            return {
                "total": total,
                "by_type": by_type,
                "latest": latest,
            }
        finally:
            conn.close()

    # Alias used by dashboard routes
    get_feedback_summary = get_summary
