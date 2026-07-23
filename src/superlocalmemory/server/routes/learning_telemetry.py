# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Read-only persisted telemetry used by the Living Brain routes."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("superlocalmemory.routes.learning")

SOURCE_SCORE_LIMIT = 50


class ReadOnlyRankerStore:
    """Minimal model-cache adapter that never creates or migrates tables."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    def count_signals(self, profile_id: str) -> int:
        connection = _readonly_connection(self._db_path)
        try:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM learning_signals "
                "WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            return int(row["count"] if row else 0)
        finally:
            connection.close()

    def load_active_model(self, profile_id: str) -> dict | None:
        connection = _readonly_connection(self._db_path)
        try:
            row = connection.execute(
                "SELECT state_bytes, bytes_sha256, feature_names, trained_at, "
                "model_version FROM learning_model_state "
                "WHERE profile_id = ? AND is_active = 1 LIMIT 1",
                (profile_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "state_bytes": bytes(row["state_bytes"]),
            "bytes_sha256": row["bytes_sha256"],
            "feature_names": row["feature_names"],
            "trained_at": row["trained_at"],
            "model_version": row["model_version"],
        }


def _readonly_connection(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, timeout=1.0,
    )
    connection.execute("PRAGMA busy_timeout=1000")
    connection.row_factory = sqlite3.Row
    return connection


def sqlite_status(exc: sqlite3.Error) -> str:
    """Return a stable, non-sensitive dashboard status for SQLite failures."""
    message = str(exc).lower()
    return (
        "database_busy"
        if "locked" in message or "busy" in message
        else "query_error"
    )


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def load_source_quality_state(profile_id: str, db_path: Path) -> dict:
    """Read bounded source scores plus the profile's full source count."""
    if not db_path.exists():
        return {
            "scores": {}, "tracked_sources": 0,
            "status": "missing_database",
        }
    try:
        connection = _readonly_connection(db_path)
        try:
            if not _table_exists(connection, "source_quality"):
                return {
                    "scores": {}, "tracked_sources": 0,
                    "status": "missing_table",
                }
            count_row = connection.execute(
                "SELECT COUNT(DISTINCT source_id) AS count "
                "FROM source_quality WHERE profile_id = ? "
                "AND source_id IS NOT NULL AND source_id != ''",
                (profile_id,),
            ).fetchone()
            rows = connection.execute(
                "SELECT source_id, alpha, beta FROM source_quality "
                "WHERE profile_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (profile_id, SOURCE_SCORE_LIMIT),
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        status = sqlite_status(exc)
        logger.warning("source quality telemetry %s: %s", status, exc)
        return {"scores": {}, "tracked_sources": 0, "status": status}

    scores: dict[str, float] = {}
    for row in rows:
        alpha, beta = float(row["alpha"] or 0), float(row["beta"] or 0)
        source_id = str(row["source_id"] or "")
        if alpha + beta > 0 and source_id and source_id not in scores:
            scores[source_id] = round(alpha / (alpha + beta), 4)
    return {
        "scores": scores,
        "tracked_sources": int(count_row["count"] if count_row else 0),
        "status": "available",
    }


def load_model_state(profile_id: str, db_path: Path) -> dict:
    """Count persisted model artifacts without inferring them from signals."""
    if not db_path.exists():
        return {"models_trained": 0, "status": "missing_database"}
    try:
        connection = _readonly_connection(db_path)
        try:
            if not _table_exists(connection, "learning_model_state"):
                return {"models_trained": 0, "status": "missing_table"}
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM learning_model_state "
                "WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        status = sqlite_status(exc)
        logger.warning("model-state telemetry %s: %s", status, exc)
        return {"models_trained": 0, "status": status}
    return {
        "models_trained": int(row["count"] if row else 0),
        "status": "available",
    }
