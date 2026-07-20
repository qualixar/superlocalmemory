#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
"""
SourceQualityScorer -- Beta-binomial source quality scoring for V3 learning.

Each memory source (agent, URL, manual, etc.) gets a quality score based on
how often its memories are confirmed vs contradicted or ignored.

Scoring (Beta-Binomial with Laplace smoothing):
    quality = (alpha + positives) / (alpha + beta + total)

    With alpha=1, beta=1 (uniform prior):
        - New source, 0 evidence  -> 1/2 = 0.50
        - 8 positive out of 10   -> 9/12 = 0.75
        - 1 positive out of 10   -> 2/12 = 0.17

Storage:
    Uses direct sqlite3 with a self-contained ``source_quality`` table.
    NOT coupled to V3 DatabaseManager.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("superlocalmemory.learning.source_quality")

# Beta-Binomial prior (Laplace / uniform)
_ALPHA = 1.0
_BETA = 1.0

# Default quality for unknown sources = alpha / (alpha + beta)
DEFAULT_QUALITY = _ALPHA / (_ALPHA + _BETA)  # 0.5

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS source_quality (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id   TEXT    NOT NULL,
    source_id    TEXT    NOT NULL,
    alpha        REAL    NOT NULL DEFAULT 1.0,
    beta         REAL    NOT NULL DEFAULT 1.0,
    updated_at   TEXT    NOT NULL
)
"""

_CREATE_UNIQUE = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_sq_profile_source
    ON source_quality (profile_id, source_id)
"""


def _utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SourceQualityScorer:
    """
    Beta-binomial source quality scoring.

    Maintains per-(profile, source) alpha/beta parameters.  Positive
    outcomes increment alpha; negative outcomes increment beta.
    Quality = alpha / (alpha + beta).

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
        conn = self._connect()
        try:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_UNIQUE)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API: record outcome
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        profile_id: str,
        source_id: str,
        outcome: str,
    ) -> None:
        """
        Record an observation for a source.

        Args:
            profile_id: Profile context.
            source_id:  Identifier of the source (agent name, URL, etc.).
            outcome:    ``"positive"`` or ``"negative"``.

        Raises:
            ValueError: If outcome is not ``"positive"`` or ``"negative"``.
        """
        if outcome not in ("positive", "negative"):
            raise ValueError(
                f"outcome must be 'positive' or 'negative', got {outcome!r}"
            )
        if not profile_id or not source_id:
            return

        now = _utcnow_iso()

        with self._lock:
            conn = self._connect()
            try:
                # Ensure row exists (INSERT OR IGNORE with defaults)
                conn.execute(
                    "INSERT OR IGNORE INTO source_quality "
                    "(profile_id, source_id, alpha, beta, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (profile_id, source_id, _ALPHA, _BETA, now),
                )

                # Update the appropriate parameter
                if outcome == "positive":
                    conn.execute(
                        "UPDATE source_quality "
                        "SET alpha = alpha + 1.0, updated_at = ? "
                        "WHERE profile_id = ? AND source_id = ?",
                        (now, profile_id, source_id),
                    )
                else:
                    conn.execute(
                        "UPDATE source_quality "
                        "SET beta = beta + 1.0, updated_at = ? "
                        "WHERE profile_id = ? AND source_id = ?",
                        (now, profile_id, source_id),
                    )

                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API: read quality
    # ------------------------------------------------------------------

    def get_quality(self, profile_id: str, source_id: str) -> float:
        """
        Get the quality score for a specific source.

        Returns the Beta-binomial posterior mean:
            quality = alpha / (alpha + beta)

        If the source has never been observed, returns the prior
        mean (0.5).

        Args:
            profile_id: Profile context.
            source_id:  Source identifier.

        Returns:
            Quality score in [0.0, 1.0].
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT alpha, beta FROM source_quality "
                "WHERE profile_id = ? AND source_id = ?",
                (profile_id, source_id),
            ).fetchone()

            if row is None:
                return DEFAULT_QUALITY

            alpha = float(row["alpha"])
            beta = float(row["beta"])
            denom = alpha + beta
            if denom <= 0:
                return DEFAULT_QUALITY
            return alpha / denom
        finally:
            conn.close()

    def get_all_qualities(self, profile_id: str) -> Dict[str, float]:
        """
        Get quality scores for all sources observed under a profile.

        Args:
            profile_id: Profile context.

        Returns:
            Dict mapping source_id -> quality score (0.0 to 1.0).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_id, alpha, beta FROM source_quality "
                "WHERE profile_id = ?",
                (profile_id,),
            ).fetchall()

            result: Dict[str, float] = {}
            for r in rows:
                alpha = float(r["alpha"])
                beta = float(r["beta"])
                denom = alpha + beta
                score = alpha / denom if denom > 0 else DEFAULT_QUALITY
                result[r["source_id"]] = score
            return result
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API: diagnostics
    # ------------------------------------------------------------------

    def get_detailed(
        self, profile_id: str, source_id: str,
    ) -> Dict[str, Any]:
        """
        Get detailed quality information for a single source.

        Returns:
            Dict with alpha, beta, quality, updated_at.
            Returns defaults if the source has not been observed.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT alpha, beta, updated_at FROM source_quality "
                "WHERE profile_id = ? AND source_id = ?",
                (profile_id, source_id),
            ).fetchone()

            if row is None:
                return {
                    "alpha": _ALPHA,
                    "beta": _BETA,
                    "quality": DEFAULT_QUALITY,
                    "updated_at": None,
                }

            alpha = float(row["alpha"])
            beta = float(row["beta"])
            denom = alpha + beta
            return {
                "alpha": alpha,
                "beta": beta,
                "quality": alpha / denom if denom > 0 else DEFAULT_QUALITY,
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()

    def get_all_detailed(self, profile_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed quality data for all sources under a profile.

        Returns:
            Dict mapping source_id -> detail dict.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_id, alpha, beta, updated_at "
                "FROM source_quality WHERE profile_id = ? "
                "ORDER BY (alpha / (alpha + beta)) DESC",
                (profile_id,),
            ).fetchall()

            result: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                alpha = float(r["alpha"])
                beta = float(r["beta"])
                denom = alpha + beta
                result[r["source_id"]] = {
                    "alpha": alpha,
                    "beta": beta,
                    "quality": alpha / denom if denom > 0 else DEFAULT_QUALITY,
                    "updated_at": r["updated_at"],
                }
            return result
        finally:
            conn.close()
