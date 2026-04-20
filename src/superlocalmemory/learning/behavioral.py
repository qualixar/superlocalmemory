# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Behavioral Pattern Store.

Stores, retrieves, and transfers behavioral patterns per profile.
Ported from V2's _store_patterns.py + cross_project_transfer.py
into a single unified module with direct sqlite3 access.

Key features:
- Record detected patterns (refinement, interest, archival, etc.)
- Query patterns by profile and type
- Summarize pattern counts by type
- Transfer patterns across profiles (cross-project learning)
- Confidence scoring: min(evidence/10, 1.0) * abs(rate - 0.5) * 2

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum observations before emitting a pattern
MIN_EVIDENCE = 3

# S9-DASH-01: filter orphan entity_ids that leak into patterns. The
# canonical_entities primary key is 16 hex chars; some historic rows
# also carry 17 (off-by-one). Any pattern value matching this shape
# with no accompanying ``metadata.value`` is skipped at read time so
# the dashboard no longer shows raw ids as "preferences".
_HEX_ID_RE = re.compile(r"^[0-9a-f]{15,20}$")

# Transfer eligibility thresholds
TRANSFER_MIN_CONFIDENCE = 0.3
TRANSFER_MIN_EVIDENCE = 2

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS _store_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id TEXT NOT NULL,
        pattern_type TEXT NOT NULL,
        pattern_key TEXT DEFAULT '',
        success_rate REAL DEFAULT 0.0,
        evidence_count INTEGER DEFAULT 1,
        confidence REAL DEFAULT 0.0,
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
"""

_CREATE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_sp_profile_type
    ON _store_patterns(profile_id, pattern_type)
"""


class BehavioralPatternStore:
    """Store and query behavioral patterns per profile.

    Uses direct sqlite3 for storage. Thread-safe via a lock.
    Creates the _store_patterns table on first use.

    Ported from V2's BehavioralPatternExtractor + CrossProjectTransfer,
    unified into a single class with profile-scoped operations.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._ensure_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_pattern(
        self,
        profile_id: str,
        pattern_type: str,
        data: Optional[Dict[str, Any]] = None,
        success_rate: float = 0.0,
        confidence: float = 0.0,
    ) -> int:
        """Store a detected behavioral pattern.

        Args:
            profile_id: Profile scope for the pattern.
            pattern_type: Category (e.g. "refinement", "interest", "archival").
            data: Arbitrary metadata dict (stored as JSON).
            success_rate: Success rate if applicable (0.0-1.0).
            confidence: Confidence score (0.0-1.0).

        Returns:
            The row ID of the inserted pattern.
        """
        now = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(data or {})
        pattern_key = (data or {}).get("topic", (data or {}).get("pattern_key", ""))

        with self._lock:
            conn = self._connect()
            try:
                # Check for existing pattern of same type+key for this profile
                existing = conn.execute(
                    "SELECT id, evidence_count FROM _store_patterns "
                    "WHERE profile_id = ? AND pattern_type = ? AND pattern_key = ?",
                    (profile_id, pattern_type, pattern_key),
                ).fetchone()

                if existing:
                    new_count = existing[1] + 1
                    new_confidence = self._compute_confidence(
                        new_count, success_rate
                    ) if success_rate > 0 else min(1.0, new_count / 100.0)
                    conn.execute(
                        "UPDATE _store_patterns "
                        "SET evidence_count = ?, confidence = ?, "
                        "    success_rate = ?, metadata = ?, updated_at = ? "
                        "WHERE id = ?",
                        (new_count, new_confidence, success_rate,
                         metadata_json, now, existing[0]),
                    )
                    conn.commit()
                    return existing[0]

                initial_confidence = confidence or 0.01
                cur = conn.execute(
                    "INSERT INTO _store_patterns "
                    "(profile_id, pattern_type, pattern_key, success_rate, "
                    " evidence_count, confidence, metadata, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)",
                    (profile_id, pattern_type, pattern_key, success_rate,
                     initial_confidence, metadata_json, now, now),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_patterns(
        self,
        profile_id: str,
        pattern_type: Optional[str] = None,
        limit: int = 50,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get stored patterns for a profile.

        Args:
            profile_id: Profile to query.
            pattern_type: If given, filter by type.
            limit: Max rows to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of pattern dicts with deserialized metadata.
        """
        with self._lock:
            conn = self._connect()
            try:
                query = (
                    "SELECT * FROM _store_patterns "
                    "WHERE profile_id = ? AND confidence >= ?"
                )
                params: List[Any] = [profile_id, min_confidence]

                if pattern_type is not None:
                    query += " AND pattern_type = ?"
                    params.append(pattern_type)

                query += " ORDER BY confidence DESC, updated_at DESC LIMIT ?"
                params.append(limit)

                rows = conn.execute(query, params).fetchall()
                out: List[Dict[str, Any]] = []
                for r in rows:
                    d = self._row_to_dict(r)
                    # S9-DASH-01: historic rows wrote raw hex entity_ids
                    # (16-20 hex chars, no row in canonical_entities) into
                    # ``pattern_key``/``metadata.value``. They surface as
                    # "entity_preferences: ea701bf01f1ff4df8" on the
                    # dashboard, which is noise. Skip them at read time so
                    # existing installs don't require a destructive DB
                    # cleanup migration.
                    key = d.get("pattern_key") or ""
                    meta_val = (d.get("metadata") or {}).get("value") or ""
                    if _HEX_ID_RE.match(str(key).split(":", 1)[-1]):
                        continue
                    if _HEX_ID_RE.match(str(meta_val)):
                        continue
                    out.append(d)
                return out
            finally:
                conn.close()

    def get_summary(self, profile_id: str) -> Dict[str, int]:
        """Get pattern counts by type for a profile.

        Returns:
            Dict mapping pattern_type -> count.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT pattern_type, COUNT(*) as cnt "
                    "FROM _store_patterns "
                    "WHERE profile_id = ? "
                    "GROUP BY pattern_type",
                    (profile_id,),
                ).fetchall()
                return {row[0]: row[1] for row in rows}
            finally:
                conn.close()

    def transfer_patterns(
        self,
        source_profile: str,
        target_profile: str,
        min_confidence: float = 0.0,
    ) -> int:
        """Copy eligible patterns from source to target profile.

        Only metadata (type, key, success_rate, confidence) is transferred.
        Memory content is never transferred. Creates new rows in the target
        profile scope.

        Args:
            source_profile: Profile to copy patterns from.
            target_profile: Profile to copy patterns to.
            min_confidence: Only transfer patterns above this threshold.

        Returns:
            Number of patterns transferred.
        """
        if source_profile == target_profile:
            return 0

        source_patterns = self.get_patterns(
            source_profile, min_confidence=min_confidence
        )

        transferred = 0
        for pattern in source_patterns:
            # Skip if target already has this pattern
            existing = self.get_patterns(
                target_profile,
                pattern_type=pattern["pattern_type"],
            )
            already_exists = any(
                p["pattern_key"] == pattern.get("pattern_key", "")
                for p in existing
            )
            if already_exists:
                continue

            self.record_pattern(
                profile_id=target_profile,
                pattern_type=pattern["pattern_type"],
                data={
                    "topic": pattern.get("pattern_key", ""),
                    "transferred_from": source_profile,
                    "original_confidence": pattern.get("confidence", 0.0),
                },
                success_rate=pattern.get("success_rate", 0.0),
                confidence=pattern.get("confidence", 0.0) * 0.8,
            )
            transferred += 1

        return transferred

    def delete_patterns(
        self,
        profile_id: str,
        pattern_type: Optional[str] = None,
    ) -> int:
        """Delete patterns for a profile. Optionally filter by type.

        Returns:
            Number of patterns deleted.
        """
        with self._lock:
            conn = self._connect()
            try:
                if pattern_type:
                    cur = conn.execute(
                        "DELETE FROM _store_patterns "
                        "WHERE profile_id = ? AND pattern_type = ?",
                        (profile_id, pattern_type),
                    )
                else:
                    cur = conn.execute(
                        "DELETE FROM _store_patterns WHERE profile_id = ?",
                        (profile_id,),
                    )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def delete_pattern_by_key(
        self,
        profile_id: str,
        pattern_type: str,
        pattern_key: str,
    ) -> int:
        """S9-DASH-04: delete a single pattern row by its user-visible
        identity ``(profile_id, pattern_type, pattern_key)``.

        Used by the dashboard "Delete pattern" button so an operator
        can kill a wrong auto-detected pattern without wiping the full
        table. Returns 1 if a row was deleted, 0 if no match.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM _store_patterns "
                    "WHERE profile_id = ? AND pattern_type = ? "
                    "AND pattern_key = ?",
                    (profile_id, pattern_type, pattern_key),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(evidence_count: int, success_rate: float) -> float:
        """Confidence = min(evidence/10, 1.0) * abs(rate - 0.5) * 2.

        High confidence requires both sufficient evidence AND a success
        rate that deviates significantly from the 50% baseline.
        """
        evidence_factor = min(evidence_count / 10.0, 1.0)
        deviation_factor = abs(success_rate - 0.5) * 2.0
        return round(evidence_factor * deviation_factor, 4)

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with row factory enabled."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        """Create the _store_patterns table if it does not exist."""
        conn = self._connect()
        try:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row into a plain dict with parsed JSON."""
        d = dict(row)
        meta = d.get("metadata", "{}")
        d["metadata"] = json.loads(meta) if isinstance(meta, str) else meta
        return d


# ---------------------------------------------------------------------------
# V3 API — BehavioralTracker (uses DatabaseManager + V3 schema)
# ---------------------------------------------------------------------------

from superlocalmemory.storage.models import BehavioralPattern  # noqa: E402


class BehavioralTracker:
    """V3 behavioral pattern tracker using DatabaseManager.

    Records query patterns (time of day, query type, entity preferences)
    and provides analytics (active hours, type distribution, preferences).

    Uses the ``behavioral_patterns`` table from the V3 schema.
    """

    def __init__(self, db) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_query(
        self,
        query: str,
        query_type: str,
        entities: list[str],
        profile_id: str,
    ) -> None:
        """Record a query and extract behavioral patterns.

        Creates up to 3 pattern types per call:
        - ``time_of_day``: hour_N for current hour
        - ``query_type``: keyed by the query_type string
        - ``entity_pref``: one per entity (max 5, lowercased)
        """
        # 1. Time of day
        hour = datetime.now(timezone.utc).hour
        self._upsert_pattern(profile_id, "time_of_day", f"hour_{hour}")

        # 2. Query type
        if query_type:
            self._upsert_pattern(profile_id, "query_type", query_type)

        # 3. Entity preferences (max 5, lowercased)
        for entity in entities[:5]:
            self._upsert_pattern(profile_id, "entity_pref", entity.lower())

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_patterns(
        self,
        pattern_type: str,
        profile_id: str,
        min_confidence: float = 0.0,
    ) -> list[BehavioralPattern]:
        """Get patterns filtered by type, profile, and min confidence."""
        rows = self._db.execute(
            "SELECT * FROM behavioral_patterns "
            "WHERE profile_id = ? AND pattern_type = ? AND confidence >= ? "
            "ORDER BY confidence DESC",
            (profile_id, pattern_type, min_confidence),
        )
        return [self._row_to_pattern(r) for r in rows]

    def get_entity_preferences(
        self, profile_id: str, top_k: int = 10
    ) -> list[str]:
        """Top-K preferred entities by confidence, highest first."""
        rows = self._db.execute(
            "SELECT pattern_key FROM behavioral_patterns "
            "WHERE profile_id = ? AND pattern_type = 'entity_pref' "
            "ORDER BY confidence DESC, observation_count DESC LIMIT ?",
            (profile_id, top_k),
        )
        return [dict(r)["pattern_key"] for r in rows]

    def get_active_hours(self, profile_id: str) -> list[int]:
        """Top 5 active hours by observation count."""
        rows = self._db.execute(
            "SELECT pattern_key FROM behavioral_patterns "
            "WHERE profile_id = ? AND pattern_type = 'time_of_day' "
            "ORDER BY observation_count DESC LIMIT 5",
            (profile_id,),
        )
        result: list[int] = []
        for r in rows:
            key = dict(r)["pattern_key"]
            if key.startswith("hour_"):
                try:
                    result.append(int(key[5:]))
                except ValueError:
                    pass
        return result

    def get_query_type_distribution(self, profile_id: str) -> dict[str, float]:
        """Proportional distribution of query types."""
        rows = self._db.execute(
            "SELECT pattern_key, observation_count FROM behavioral_patterns "
            "WHERE profile_id = ? AND pattern_type = 'query_type'",
            (profile_id,),
        )
        counts: dict[str, int] = {}
        for r in rows:
            d = dict(r)
            counts[d["pattern_key"]] = d["observation_count"]

        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: round(v / total, 4) for k, v in counts.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _upsert_pattern(
        self, profile_id: str, pattern_type: str, pattern_key: str
    ) -> None:
        """Insert or increment a pattern. Confidence = min(count/100, 1.0)."""
        from superlocalmemory.storage.models import _new_id, _now

        rows = self._db.execute(
            "SELECT pattern_id, observation_count FROM behavioral_patterns "
            "WHERE profile_id = ? AND pattern_type = ? AND pattern_key = ?",
            (profile_id, pattern_type, pattern_key),
        )
        if rows:
            d = dict(rows[0])
            new_count = d["observation_count"] + 1
            new_conf = min(new_count / 100.0, 1.0)
            self._db.execute(
                "UPDATE behavioral_patterns "
                "SET observation_count = ?, confidence = ?, last_updated = ? "
                "WHERE pattern_id = ?",
                (new_count, new_conf, _now(), d["pattern_id"]),
            )
        else:
            self._db.execute(
                "INSERT INTO behavioral_patterns "
                "(pattern_id, profile_id, pattern_type, pattern_key, "
                " pattern_value, confidence, observation_count, last_updated) "
                "VALUES (?, ?, ?, ?, '', ?, 1, ?)",
                (_new_id(), profile_id, pattern_type, pattern_key, 0.01, _now()),
            )

    @staticmethod
    def _row_to_pattern(row) -> BehavioralPattern:
        """Convert a DB row to BehavioralPattern."""
        d = dict(row)
        return BehavioralPattern(
            pattern_id=d["pattern_id"],
            profile_id=d["profile_id"],
            pattern_type=d.get("pattern_type", ""),
            pattern_key=d.get("pattern_key", ""),
            pattern_value=d.get("pattern_value", ""),
            confidence=d.get("confidence", 0.0),
            observation_count=d.get("observation_count", 0),
            last_updated=d.get("last_updated", ""),
        )
