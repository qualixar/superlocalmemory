#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
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
    - Queries are keyed-hashed for local grouping; the key never enters SQLite.

Storage:
    Every explicit-feedback event is written to the CANONICAL learning store
    -- a ``learning_signals`` row paired 1:1 with a ``learning_features`` row
    -- in the same transaction as the historic ``learning_feedback`` row.

    ``learning_signals`` is canonical because every live consumer already
    reads it: the dashboard's Living Brain panel and ranker-phase card
    (``server/routes/brain.py``, ``server/routes/learning.py``), the LightGBM
    retrainer, and -- since issue #106 -- the recall phase gate.
    ``learning_feedback`` is the pre-v3.4.22 table: ``legacy_migration``
    copies it forward into ``learning_signals``, the dashboard reports it as
    ``legacy_feedback_rows`` with a "pending migration" card, and the phase
    gate's own docstring calls it legacy. Writing feedback only there (the
    v3.8.11 attempt at issue #102) put the durable write in a table no phase
    counter consumes, which is why reported feedback still changed nothing.

    It is kept written for one more release (LLD-07 D5) so ``pattern_miner``
    channel mining and GDPR erasure keep working; the shared identity from
    ``legacy_migration.legacy_query_id`` stops the two writers double-counting.

    NOT coupled to V3 DatabaseManager -- this is a standalone data collector.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass
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

# ``channel`` records WHICH retrieval channel surfaced the fact (semantic,
# bm25, entity_graph, temporal, ...). ``pattern_miner._mine_channel_and_
# coretrieval`` groups on it to mine ``channel_performance`` patterns. It was
# read by the miner but never defined here, so every fresh database raised
# "no such column: channel" — swallowed at debug level, which silently killed
# BOTH channel mining and the co-retrieval mining that followed it in the same
# try block. Defined here for new databases; M033 back-fills existing ones.
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS learning_feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id   TEXT    NOT NULL,
    fact_id      TEXT    NOT NULL,
    signal_type  TEXT    NOT NULL,
    signal_value REAL    NOT NULL,
    query_hash   TEXT,
    created_at   TEXT    NOT NULL,
    metadata     TEXT,
    channel      TEXT    DEFAULT 'unknown'
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_feedback_profile
    ON learning_feedback (profile_id, created_at DESC)
"""

_CREATE_CHANNEL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_feedback_channel
    ON learning_feedback (profile_id, channel)
"""


# Signal type stamped on the canonical ``learning_signals`` row for an
# explicit-feedback event. Identical to what ``legacy_migration`` writes when
# it carries a ``learning_feedback`` row forward, so a row recorded eagerly
# and a row migrated in batch are indistinguishable to every consumer.
CANONICAL_SIGNAL_TYPE = "legacy_feedback"

# ``learning_features.features_json`` for a feedback event. Feedback arrives
# out of band -- there is no ranked candidate list to extract a real feature
# vector from -- so the row is empty and flagged ``is_synthetic=1``. The
# LightGBM retrainer selects ``WHERE is_synthetic=0``, so these rows move the
# phase counters and the bandit without ever polluting model training.
_SYNTHETIC_FEATURES_JSON = "{}"


@dataclass(frozen=True)
class FeedbackWrite:
    """Outcome of one explicit-feedback write.

    ``canonical`` is the only field callers should gate user-facing success
    on: it is True when the ``learning_signals`` row that every phase counter
    reads actually landed. ``feedback_row_id`` alone means the legacy row was
    written, which on its own influences nothing.
    """

    feedback_row_id: Optional[int]
    signal_row_id: Optional[int]
    canonical: bool


def _utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _canonical_schema_ready(conn: sqlite3.Connection) -> bool:
    """Return True when learning.db can accept a canonical feedback event.

    Both tables must exist AND carry the LLD-02 columns the event needs
    (``learning_signals.query_id`` for the shared identity that keeps the
    batch migration from double-counting, ``learning_features.is_synthetic``
    for the flag that keeps these rows out of LightGBM training). A table
    that exists without them cannot hold the event correctly, so treating
    mere existence as readiness would write a row that silently breaks both
    invariants.
    """
    try:
        signal_cols = {
            row[1] for row in conn.execute(
                "PRAGMA table_info(learning_signals)",
            )
        }
        feature_cols = {
            row[1] for row in conn.execute(
                "PRAGMA table_info(learning_features)",
            )
        }
    except sqlite3.Error:
        return False
    return (
        "query_id" in signal_cols
        and "query_text_hash" in signal_cols
        and "signal_id" in feature_cols
        and "is_synthetic" in feature_cols
    )


def _load_or_create_hash_key(db_path: Path) -> bytes:
    """Return an owner-only per-install key for feedback query HMACs."""
    key_path = db_path.parent / ".feedback-hash-key"
    try:
        key = key_path.read_bytes()
        if len(key) >= 32:
            os.chmod(key_path, 0o600)
            return key
    except OSError:
        pass

    key = secrets.token_bytes(32)
    try:
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = key_path.read_bytes()
        if len(existing) < 32:
            raise RuntimeError("feedback hash key is truncated")
        os.chmod(key_path, 0o600)
        return existing
    except OSError as exc:
        # Do not fall back to a guessable digest on a read-only data root.
        # A process-local key preserves privacy; only cross-restart grouping
        # is lost, and the operator gets an explicit warning.
        logger.warning(
            "cannot persist feedback hash key beside %s: %s; using a "
            "process-local key", db_path, exc,
        )
        return key
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(key_path, 0o600)
    return key


def _hash_query(query: str, key: bytes) -> str:
    """Return a keyed, truncated SHA-256 digest for local query grouping."""
    return hmac.digest(key, query.encode("utf-8"), "sha256").hex()[:16]


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
        self._query_hash_key = _load_or_create_hash_key(self._db_path)
        self._lock = threading.Lock()
        # Latched once the canonical LLD-02 tables are confirmed present, so
        # the sqlite_master probe runs at most once per collector instead of
        # on every feedback write.
        self._canonical_ready = False
        self._ensure_schema()
        self._bootstrap_canonical_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create tables/indexes if they do not exist."""
        conn = self._connect()
        try:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
            # Pre-3.8.11 databases created ``learning_feedback`` without the
            # ``channel`` column. M033 covers migrated installs; this ADD keeps
            # a collector pointed at a legacy file self-healing rather than
            # failing every channel query for the life of the process.
            existing = {
                row[1] for row in
                conn.execute("PRAGMA table_info(learning_feedback)")
            }
            if "channel" not in existing:
                conn.execute(
                    "ALTER TABLE learning_feedback "
                    "ADD COLUMN channel TEXT DEFAULT 'unknown'"
                )
            conn.execute(_CREATE_CHANNEL_INDEX)
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

        qhash = _hash_query(query, self._query_hash_key)
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
        query: str = "",
        channel: str = "unknown",
    ) -> Optional[int]:
        """Record explicit user feedback on a specific fact.

        Back-compatible wrapper: returns the ``learning_feedback`` row id.
        Callers that must tell a user whether the feedback actually reached
        the store the phase counters read should use
        :meth:`record_explicit_event` and check ``FeedbackWrite.canonical`` —
        a legacy row id on its own influences no consumer.
        """
        return self.record_explicit_event(
            profile_id=profile_id, fact_id=fact_id, signal_type=signal_type,
            value=value, query=query, channel=channel,
        ).feedback_row_id

    def record_explicit_event(
        self,
        profile_id: str,
        fact_id: str,
        signal_type: str,
        value: float,
        query: str = "",
        channel: str = "unknown",
    ) -> FeedbackWrite:
        """
        Record explicit user feedback as ONE atomic learning event.

        Writes three rows in a single transaction: the historic
        ``learning_feedback`` row (kept one more release for ``pattern_miner``
        channel mining and GDPR erasure) plus the canonical
        ``learning_signals`` + ``learning_features`` pair that the dashboard,
        the recall phase gate, and the retrainer all read. Either the whole
        event is durable or none of it is — a partial write would leave the
        legacy table and the phase counters permanently disagreeing, which is
        the shape of issue #106.

        Recall itself is deliberately read-only (it must never open a writer —
        see
        ``test_readonly_bandit_uses_uri_read_connection_and_never_records_play``),
        so explicit feedback is the only path that grows these tables outside
        the signal worker.

        Args:
            profile_id:  Profile providing feedback.
            fact_id:     The fact being rated.
            signal_type: One of ``user_positive``, ``user_negative``,
                         ``user_correction``, or any custom type.
            value:       Numeric signal value (0.0 to 1.0).
            query:       Originating query. Stored only as a keyed truncated
                         digest — full text and the HMAC key are never stored
                         in SQLite.
            channel:     Retrieval channel that surfaced the fact.

        Returns:
            A :class:`FeedbackWrite` describing exactly which rows landed.
        """
        if not profile_id or not fact_id:
            return FeedbackWrite(None, None, False)

        clamped = max(0.0, min(1.0, float(value)))
        now = _utcnow_iso()
        query_hash = (
            _hash_query(query, self._query_hash_key) if query else None
        )

        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "INSERT INTO learning_feedback "
                    "(profile_id, fact_id, signal_type, signal_value, "
                    "query_hash, created_at, metadata, channel) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (profile_id, fact_id, signal_type, clamped, query_hash,
                     now, None, channel or "unknown"),
                )
                feedback_row_id = cursor.lastrowid
                signal_row_id = self._insert_canonical_pair(
                    conn,
                    profile_id=profile_id,
                    fact_id=fact_id,
                    value=clamped,
                    query_hash=query_hash,
                    created_at=now,
                    feedback_row_id=feedback_row_id,
                )
                conn.commit()
                return FeedbackWrite(
                    feedback_row_id, signal_row_id, signal_row_id is not None,
                )
            except sqlite3.Error:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Canonical store
    # ------------------------------------------------------------------

    def _insert_canonical_pair(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        fact_id: str,
        value: float,
        query_hash: Optional[str],
        created_at: str,
        feedback_row_id: Optional[int],
    ) -> Optional[int]:
        """Insert the ``learning_signals`` + ``learning_features`` pair.

        Runs inside the caller's open transaction so the canonical rows commit
        with the legacy row or not at all. Returns the new signal row id, or
        None when the canonical tables are absent — they are owned by the
        migration runner (LLD-06 H15 forbids DDL here), so on a database that
        predates them the caller is told the truth rather than handed a
        fabricated success.
        """
        if feedback_row_id is None:
            return None
        if not self._canonical_tables_present(conn):
            return None

        from superlocalmemory.learning.legacy_migration import legacy_query_id

        query_id = legacy_query_id(feedback_row_id)
        # Pad to 32 hex chars so an eagerly-written row has the same shape as
        # both a migrated row and a fresh signal-worker row.
        padded_hash = ((query_hash or "") + ("0" * 32))[:32]

        cursor = conn.execute(
            "INSERT INTO learning_signals "
            "(profile_id, query, fact_id, signal_type, value, created_at, "
            " query_id, query_text_hash, position, channel_scores, "
            " cross_encoder) "
            "VALUES (?, '', ?, ?, ?, ?, ?, ?, 0, '{}', NULL)",
            (profile_id, fact_id, CANONICAL_SIGNAL_TYPE, value, created_at,
             query_id, padded_hash),
        )
        signal_row_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO learning_features "
            "(profile_id, query_id, fact_id, features_json, label, "
            " created_at, signal_id, is_synthetic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            (profile_id, query_id, fact_id, _SYNTHETIC_FEATURES_JSON,
             value, created_at, signal_row_id),
        )
        return signal_row_id

    def _bootstrap_canonical_schema(self) -> None:
        """Make sure the canonical store exists before any feedback arrives.

        Explicit feedback can be the very first learning write on a machine —
        a user can rate a recall before the daemon has ever run the migration
        runner. Without this the write would honestly, but uselessly, report
        that it never reached the store that gates ranking.

        No DDL is authored here (LLD-06 H15). The base tables come from
        ``LearningDatabase``, which the migration runner itself calls as its
        first-boot bootstrap, and the LLD-02 columns come from M001's own DDL.
        Applying M001's DDL without a ``migration_log`` row is safe: when the
        real runner reaches M001 its ALTERs fail, ``M001.verify`` passes, and
        the runner records it as "already applied (verified via schema
        inspection)".

        Never fatal — a read-only or unwritable learning.db must not stop a
        collector from being constructed.
        """
        try:
            from superlocalmemory.learning.database import LearningDatabase
            from superlocalmemory.storage.migrations import (
                M001_add_signal_features_columns as _m001,
            )

            LearningDatabase(self._db_path)
            conn = self._connect()
            try:
                if not _m001.verify(conn):
                    conn.executescript(_m001.DDL)
                    conn.commit()
                self._canonical_ready = _canonical_schema_ready(conn)
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 — construction must not fail
            logger.warning(
                "canonical learning schema bootstrap failed for %s: %s",
                self._db_path, exc,
            )

    def _canonical_tables_present(self, conn: sqlite3.Connection) -> bool:
        """Return True when both canonical LLD-02 tables exist.

        Re-probes while unready so a collector constructed before the
        migration runner ran starts writing canonically as soon as the tables
        appear, instead of degrading for the life of the process.
        """
        if self._canonical_ready:
            return True
        self._canonical_ready = _canonical_schema_ready(conn)
        if not self._canonical_ready:
            logger.warning(
                "learning.db at %s has no usable learning_signals/"
                "learning_features schema; explicit feedback cannot reach the "
                "store that gates adaptive ranking.",
                self._db_path,
            )
        return self._canonical_ready

    def get_signal_count(self, profile_id: str) -> int:
        """Return the canonical signal count that gates the ranking phase.

        This is the single number the recall phase gate, the dashboard's
        Living Brain panel, and the ranker-phase card all resolve their phase
        from. Reporting anything else to a user — as ``report_feedback`` did
        with ``feedback_records`` before issue #106 — shows progress toward a
        threshold nothing is actually measuring.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM learning_signals WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            return row[0] if row else 0
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

        Routed through :meth:`record_explicit_event` so a thumbs-up from the
        dashboard lands in exactly the same canonical store as a thumbs-up
        from MCP. Before issue #106 this path wrote only ``learning_feedback``,
        so the dashboard's own Living Brain counter — which reads
        ``learning_signals`` — never moved in response to its own buttons.
        """
        if not memory_id:
            return None
        signal_type, value = _DASHBOARD_SIGNAL_MAP.get(
            feedback_type, ("user_correction", 0.5),
        )
        return self.record_explicit_event(
            profile_id=profile_id or "default",
            fact_id=str(memory_id),
            signal_type=signal_type,
            value=value,
            query=query,
            channel="dashboard",
        ).feedback_row_id

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
