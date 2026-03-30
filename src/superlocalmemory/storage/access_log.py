# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""Access log for fact retrieval events.

Tracks when facts are accessed (recall, auto_invoke, search).
Used by Phase 2 auto-invoke for recency scoring (H1 fix).
All SQL parameterized (Rule 11). Silent errors (Rule 19).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: MIT
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


def _new_log_id() -> str:
    """Generate a short unique log ID."""
    return uuid.uuid4().hex[:16]


class AccessLog:
    """CRUD operations for fact_access_log table."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def store_access(
        self,
        fact_id: str,
        profile_id: str,
        access_type: str = "recall",
        session_id: str = "",
    ) -> str:
        """Record a fact access event. Returns log_id.

        Validates access_type against allowed values.
        Returns empty string on failure (Rule 19).
        """
        allowed = ("recall", "auto_invoke", "search", "consolidation")
        if access_type not in allowed:
            access_type = "recall"

        log_id = _new_log_id()
        try:
            self._db.execute(
                "INSERT INTO fact_access_log "
                "(log_id, fact_id, profile_id, access_type, session_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (log_id, fact_id, profile_id, access_type, session_id),
            )
            return log_id
        except Exception as exc:
            logger.debug("store_access failed for fact_id=%s: %s", fact_id, exc)
            return ""

    def store_access_batch(
        self,
        fact_ids: list[str],
        profile_id: str,
        access_type: str = "recall",
        session_id: str = "",
    ) -> int:
        """Record access events for multiple facts. Returns count stored.

        Catches exceptions per-fact (Rule 19: silent errors).
        """
        count = 0
        for fact_id in fact_ids:
            result = self.store_access(
                fact_id, profile_id, access_type, session_id,
            )
            if result:
                count += 1
        return count

    def get_latest_access_time(
        self, fact_id: str, profile_id: str,
    ) -> str | None:
        """Most recent access timestamp for a fact.

        Returns ISO datetime string or None if never accessed.
        """
        try:
            rows = self._db.execute(
                "SELECT MAX(accessed_at) AS latest "
                "FROM fact_access_log "
                "WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
            if rows and rows[0]["latest"] is not None:
                return str(rows[0]["latest"])
            return None
        except Exception as exc:
            logger.debug("get_latest_access_time failed: %s", exc)
            return None

    def get_access_count(
        self, fact_id: str, profile_id: str,
    ) -> int:
        """Total access count for a fact."""
        try:
            rows = self._db.execute(
                "SELECT COUNT(*) AS c "
                "FROM fact_access_log "
                "WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
            if rows:
                return int(rows[0]["c"])
            return 0
        except Exception as exc:
            logger.debug("get_access_count failed: %s", exc)
            return 0

    def get_all_access_times(
        self, profile_id: str, limit: int = 1000,
    ) -> dict[str, str]:
        """Latest access time for all facts in a profile.

        Returns {fact_id: latest_accessed_at} dict.
        """
        try:
            rows = self._db.execute(
                "SELECT fact_id, MAX(accessed_at) AS latest "
                "FROM fact_access_log "
                "WHERE profile_id = ? "
                "GROUP BY fact_id "
                "ORDER BY latest DESC "
                "LIMIT ?",
                (profile_id, limit),
            )
            return {str(r["fact_id"]): str(r["latest"]) for r in rows}
        except Exception as exc:
            logger.debug("get_all_access_times failed: %s", exc)
            return {}

    def get_frequently_accessed(
        self, profile_id: str, min_count: int = 3, limit: int = 100,
    ) -> list[tuple[str, int]]:
        """Facts accessed at least min_count times.

        Returns [(fact_id, access_count)] sorted by count desc.
        """
        try:
            rows = self._db.execute(
                "SELECT fact_id, COUNT(*) AS cnt "
                "FROM fact_access_log "
                "WHERE profile_id = ? "
                "GROUP BY fact_id "
                "HAVING cnt >= ? "
                "ORDER BY cnt DESC "
                "LIMIT ?",
                (profile_id, min_count, limit),
            )
            return [(str(r["fact_id"]), int(r["cnt"])) for r in rows]
        except Exception as exc:
            logger.debug("get_frequently_accessed failed: %s", exc)
            return []
