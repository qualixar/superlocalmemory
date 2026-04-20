# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22

"""Legacy ``learning_feedback`` row migration — data movement only.

LLD reference: ``.backup/active-brain/lld/LLD-07-schema-migrations-and-security-primitives.md``
Section 5 (Legacy learning_feedback Migration).

Hard rule H15 (LLD-06 §10): this module MUST NOT contain any schema
DDL. All schema definitions live in
``src/superlocalmemory/storage/migrations/`` modules. This file is
data-copy only — the tables it reads from and writes into are created
by the LLD-07 runner before this runs.

S8-ARC-01 fix: the earlier stub always returned ``copied=0`` while the
``/api/v3/brain`` endpoint surfaced the pre-existing ``learning_feedback``
row count as ``legacy_migrated_count`` — a silent integrity lie. This
implementation does the real row copy and is idempotent (guarded by a
``migration_log`` sentinel row under the name ``LEG001_feedback_to_signals``).
D5 (keep ``learning_feedback`` for one release) still holds — this
migration does NOT delete from ``learning_feedback``; it only inserts
into ``learning_signals`` + ``learning_features``.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Marker used in migration_log to signal the data-copy has run.
MIGRATION_NAME = "LEG001_feedback_to_signals"

# Batch size for the row copy. 500 keeps the single transaction short
# enough that any concurrent SQLite reader only sees brief locks, while
# still amortising the per-row overhead.
_COPY_BATCH_SIZE = 500


def migrate_legacy_feedback(
    learning_db: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Copy ``learning_feedback`` rows forward into LLD-02 tables.

    Policy (LLD-07 §5 + D5 + D9):
      * Idempotent: if ``migration_log`` already has ``LEG001_feedback_to_signals``
        with status 'complete', returns a stats dict with ``already_done=True``.
      * Preserves the original ``learning_feedback`` table (D5 — one release).
      * Flags synthetic rows: ``learning_features.is_synthetic=1`` so the
        LightGBM trainer in ``consolidation_worker._retrain_ranker`` can
        exclude them from training (``WHERE is_synthetic=0``).
      * Never raises; any sqlite3 error collapses into ``failed=N`` and a
        WARN log so the daemon lifespan can continue.

    Returns:
        ``{"copied": int, "skipped": int, "failed": int,
          "already_done": bool, "details": {...}}``
    """
    stats = {
        "copied": 0, "skipped": 0, "failed": 0,
        "already_done": False,
        "details": {"name": MIGRATION_NAME, "dry_run": dry_run},
    }
    if not learning_db.exists():
        stats["details"]["reason"] = "db_missing"
        return stats

    try:
        conn = sqlite3.connect(str(learning_db), isolation_level=None, timeout=10.0)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        logger.warning("legacy migration: open failed: %s", exc)
        stats["failed"] = 1
        stats["details"]["reason"] = f"open_failed:{type(exc).__name__}"
        return stats

    try:
        # Every required table must exist. If any is missing we defer to
        # the migration runner (LLD-07 M001/M003) and record a skip.
        if not _tables_exist(conn, ("migration_log", "learning_feedback",
                                     "learning_signals", "learning_features")):
            stats["details"]["reason"] = "required_tables_missing"
            return stats

        if _already_complete(conn, MIGRATION_NAME):
            stats["already_done"] = True
            stats["details"]["reason"] = "migration_log_complete"
            return stats

        # Row count of candidates up front so the stats dict can report
        # progress even under a dry_run.
        total = _count_feedback(conn)
        stats["details"]["source_rows"] = total
        if total == 0:
            _record_migration(conn, MIGRATION_NAME,
                              status="complete", rows_affected=0,
                              dry_run=dry_run)
            return stats

        if dry_run:
            stats["details"]["reason"] = "dry_run_noop"
            stats["skipped"] = total
            return stats

        copied, failed = _copy_rows(conn)
        stats["copied"] = copied
        stats["failed"] = failed
        _record_migration(conn, MIGRATION_NAME,
                          status="complete" if failed == 0 else "partial",
                          rows_affected=copied, dry_run=False)
        return stats
    finally:
        try:
            conn.close()
        except sqlite3.Error:  # pragma: no cover — defensive close
            pass


# ---------------------------------------------------------------------------
# Helpers — parameterised SQL only; no DDL here per H15.
# ---------------------------------------------------------------------------


def _tables_exist(conn: sqlite3.Connection, names: tuple[str, ...]) -> bool:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
        "(" + ",".join("?" * len(names)) + ")",
        names,
    ).fetchall()
    return len(rows) == len(names)


def _already_complete(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT status FROM migration_log WHERE name = ?",
        (name,),
    ).fetchone()
    return row is not None and str(row[0]).lower() == "complete"


def _count_feedback(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM learning_feedback",
    ).fetchone()
    return int(row[0]) if row else 0


def _copy_rows(conn: sqlite3.Connection) -> tuple[int, int]:
    """Copy all ``learning_feedback`` rows forward.

    Returns ``(copied, failed)``. Does not raise. Commits per batch so
    a later failure still leaves the earlier batches durable.
    """
    copied = 0
    failed = 0
    offset = 0
    while True:
        try:
            batch = conn.execute(
                "SELECT id, profile_id, query_hash, fact_id, "
                "       signal_type, signal_value, created_at "
                "FROM learning_feedback "
                "ORDER BY id LIMIT ? OFFSET ?",
                (_COPY_BATCH_SIZE, offset),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("legacy migration: batch read failed: %s", exc)
            failed += 1
            break
        if not batch:
            break

        try:
            conn.execute("BEGIN IMMEDIATE")
            for row in batch:
                try:
                    _copy_single_row(conn, row)
                    copied += 1
                except sqlite3.Error as exc:
                    logger.debug("legacy migration: row %s failed: %s",
                                 row[0], exc)
                    failed += 1
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            logger.warning("legacy migration: batch commit failed: %s", exc)
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:  # pragma: no cover
                pass
            failed += len(batch)
        offset += len(batch)

    return copied, failed


def _copy_single_row(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    """Insert one legacy row into learning_signals + learning_features.

    Synthetic features are minimal (position from the legacy row if present,
    zeros elsewhere). The LightGBM trainer filters these out via
    ``is_synthetic=0``; the bandit/heuristic layer can still learn from
    signal counts.

    Note: ``learning_feedback.query_hash`` is already a privacy-hashed
    digest (LLD-02 §4.1 S2), so we copy it forward as-is. Padding to
    32 chars keeps a stable shape when upstream hashes are shorter.
    """
    stored_hash = str(row["query_hash"] or "")
    # Keep the hash at 32 hex chars for shape-stability with fresh signals.
    query_hash = (stored_hash + ("0" * 32))[:32]
    created_at = str(row["created_at"] or
                     datetime.now(timezone.utc).isoformat(timespec="seconds"))
    profile_id = str(row["profile_id"] or "default")
    fact_id = str(row["fact_id"] or "")
    legacy_query_id = f"legacy:{row['id']}"

    # Insert the signal row. ``signal_type='legacy_feedback'`` marks it
    # clearly so consumers (dashboard, labeler) can treat it correctly.
    cur = conn.execute(
        "INSERT INTO learning_signals "
        "(profile_id, query, fact_id, signal_type, value, created_at, "
        " query_id, query_text_hash, position, channel_scores, cross_encoder) "
        "VALUES (?, '', ?, 'legacy_feedback', ?, ?, ?, ?, 0, '{}', NULL)",
        (profile_id, fact_id,
         float(row["signal_value"] or 1.0),
         created_at, legacy_query_id, query_hash),
    )
    sid = cur.lastrowid

    # Synthetic 20-dim feature vector (zeros). The real FEATURE_NAMES
    # come from ``learning.features`` at training time; we don't import
    # it here to keep this module boot-order-independent.
    conn.execute(
        "INSERT INTO learning_features "
        "(profile_id, query_id, fact_id, features_json, label, created_at, "
        " signal_id, is_synthetic) "
        "VALUES (?, ?, ?, '{}', 0.0, ?, ?, 1)",
        (profile_id, legacy_query_id, fact_id, created_at, sid),
    )


def _record_migration(
    conn: sqlite3.Connection,
    name: str,
    *,
    status: str,
    rows_affected: int,
    dry_run: bool,
) -> None:
    """Record migration status in ``migration_log``. No DDL here (H15).

    The ``migration_log`` schema uses ``applied_at`` + ``ddl_sha256``
    (see storage/migrations/M003_migration_log.py). This is a data-only
    migration, so ``ddl_sha256`` is the empty string — the sentinel row
    is for idempotency, not schema-drift detection.
    """
    applied_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if dry_run:
        return
    try:
        conn.execute(
            "INSERT OR REPLACE INTO migration_log "
            "(name, applied_at, ddl_sha256, rows_affected, status) "
            "VALUES (?, ?, '', ?, ?)",
            (name, applied_at, rows_affected, status),
        )
    except sqlite3.Error as exc:
        logger.warning("legacy migration: log record failed: %s", exc)


__all__ = ("migrate_legacy_feedback", "MIGRATION_NAME")
