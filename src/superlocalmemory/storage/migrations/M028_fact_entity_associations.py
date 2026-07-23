# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""M028 — normalized fact/entity associations for O(1) ingest effects.

The table is both an association index and an idempotency ledger.  Its
composite primary key means a fact can contribute to an entity's ``fact_count``
once, regardless of ingestion retries or how many operations converge on the
same consolidated fact.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

NAME = "M028_fact_entity_associations"
DB_TARGET = "memory"

DDL = """
CREATE TABLE IF NOT EXISTS fact_entity_associations (
    profile_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    first_operation_id TEXT NOT NULL DEFAULT '',
    count_applied INTEGER NOT NULL DEFAULT 0
        CHECK (count_applied IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    PRIMARY KEY (profile_id, fact_id, entity_id),
    FOREIGN KEY (fact_id) REFERENCES atomic_facts(fact_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id)
        REFERENCES canonical_entities(entity_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fact_entity_associations_entity
    ON fact_entity_associations(profile_id, entity_id, fact_id);
CREATE TABLE IF NOT EXISTS fact_entity_association_repair_state (
    repair_key TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'running', 'retrying', 'complete')),
    target_fact_rowid INTEGER NOT NULL DEFAULT -1,
    last_fact_rowid INTEGER NOT NULL DEFAULT 0,
    scanned INTEGER NOT NULL DEFAULT 0,
    inserted INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
"""


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def apply(conn: sqlite3.Connection) -> None:
    """Install schema and capture a constant-time historical rowid boundary."""
    conn.executescript(DDL)
    association_columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(fact_entity_associations)"
        ).fetchall()
    }
    if "count_applied" not in association_columns:
        conn.execute(
            "ALTER TABLE fact_entity_associations "
            "ADD COLUMN count_applied INTEGER NOT NULL DEFAULT 0 "
            "CHECK (count_applied IN (0, 1))"
        )
    repair_columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(fact_entity_association_repair_state)"
        ).fetchall()
    }
    if "target_fact_rowid" not in repair_columns:
        conn.execute(
            "ALTER TABLE fact_entity_association_repair_state "
            "ADD COLUMN target_fact_rowid INTEGER NOT NULL DEFAULT -1"
        )
    target = int(conn.execute(
        "SELECT COALESCE(MAX(rowid), 0) FROM atomic_facts"
    ).fetchone()[0])
    conn.executescript(
        "INSERT OR IGNORE INTO fact_entity_association_repair_state "
        "(repair_key,state,target_fact_rowid,updated_at) "
        f"VALUES ('historical-backfill','pending',{target},'');"
        "UPDATE fact_entity_association_repair_state "
        f"SET target_fact_rowid={target} "
        "WHERE repair_key='historical-backfill' AND target_fact_rowid < 0;"
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_repair_status(db_path: Path) -> dict[str, int | str]:
    """Read durable backfill progress without inferring it from schema."""
    conn = _connect(Path(db_path))
    try:
        row = conn.execute(
            "SELECT state,target_fact_rowid,last_fact_rowid,scanned,inserted,"
            "last_error,updated_at "
            "FROM fact_entity_association_repair_state "
            "WHERE repair_key='historical-backfill'"
        ).fetchone()
        if row is None:
            return {
                "state": "pending", "target_fact_rowid": 0,
                "last_fact_rowid": 0,
                "scanned": 0, "inserted": 0,
                "last_error": "", "updated_at": "",
            }
        return dict(row)
    finally:
        conn.close()


def _entity_ids(raw: object) -> tuple[str, ...]:
    try:
        values = json.loads(str(raw or "[]"))
    except (TypeError, ValueError):
        return ()
    if not isinstance(values, list):
        return ()
    return tuple(dict.fromkeys(str(value) for value in values if value))


def _repair_batch(conn: sqlite3.Connection, batch_size: int) -> dict[str, int | bool]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        status = conn.execute(
            "SELECT last_fact_rowid,target_fact_rowid "
            "FROM fact_entity_association_repair_state "
            "WHERE repair_key='historical-backfill'"
        ).fetchone()
        cursor = int(status["last_fact_rowid"] or 0)
        target = int(status["target_fact_rowid"])
        rows = conn.execute(
            "SELECT rowid,fact_id,profile_id,canonical_entities_json "
            "FROM atomic_facts WHERE rowid>? AND rowid<=? "
            "ORDER BY rowid LIMIT ?",
            (cursor, target, batch_size),
        ).fetchall()
        if not rows:
            conn.execute(
                "UPDATE fact_entity_association_repair_state "
                "SET state='complete',last_error='',updated_at=? "
                "WHERE repair_key='historical-backfill'",
                (_now(),),
            )
            conn.commit()
            return {"scanned": 0, "inserted": 0, "complete": True}
        inserted = 0
        for row in rows:
            for entity_id in _entity_ids(row["canonical_entities_json"]):
                result = conn.execute(
                    "INSERT OR IGNORE INTO fact_entity_associations "
                    "(profile_id,fact_id,entity_id,first_operation_id,"
                    "count_applied) "
                    "SELECT ?,?,?,?,? FROM canonical_entities "
                    "WHERE profile_id=? AND entity_id=?",
                    (
                        row["profile_id"], row["fact_id"], entity_id,
                        "migration-backfill", 0,
                        row["profile_id"], entity_id,
                    ),
                )
                inserted += max(0, result.rowcount)
        conn.execute(
            "UPDATE fact_entity_association_repair_state SET "
            "state='running',last_fact_rowid=?,scanned=scanned+?,"
            "inserted=inserted+?,last_error='',updated_at=? "
            "WHERE repair_key='historical-backfill'",
            (int(rows[-1]["rowid"]), len(rows), inserted, _now()),
        )
        conn.commit()
        return {"scanned": len(rows), "inserted": inserted, "complete": False}
    except Exception:
        conn.rollback()
        raise


def repair_fact_entity_associations(
    db_path: Path,
    *,
    batch_size: int = 250,
    max_batches: int = 1,
) -> dict[str, int | bool]:
    """Run bounded, restartable short-transaction backfill batches."""
    if batch_size < 1 or max_batches < 1:
        raise ValueError("batch_size and max_batches must be positive")
    totals = {"scanned": 0, "inserted": 0, "complete": False}
    conn = _connect(Path(db_path))
    try:
        for _ in range(max_batches):
            result = _repair_batch(conn, batch_size)
            totals["scanned"] += int(result["scanned"])
            totals["inserted"] += int(result["inserted"])
            totals["complete"] = bool(result["complete"])
            if totals["complete"]:
                break
        return totals
    except sqlite3.Error as exc:
        try:
            conn.execute(
                "UPDATE fact_entity_association_repair_state SET "
                "state='retrying',last_error=?,updated_at=? "
                "WHERE repair_key='historical-backfill'",
                (type(exc).__name__, _now()),
            )
            conn.commit()
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


def verify(conn: sqlite3.Connection) -> bool:
    """Return true when the normalized association contract is indexed."""
    if not (
        _table_exists(conn, "fact_entity_associations")
        and _table_exists(conn, "fact_entity_association_repair_state")
    ):
        return False
    association_columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(fact_entity_associations)"
        ).fetchall()
    }
    repair_columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(fact_entity_association_repair_state)"
        ).fetchall()
    }
    if (
        "count_applied" not in association_columns
        or "target_fact_rowid" not in repair_columns
    ):
        return False
    indexes = {
        row[1]
        for row in conn.execute(
            "PRAGMA index_list(fact_entity_associations)"
        ).fetchall()
    }
    return "idx_fact_entity_associations_entity" in indexes


def repair(conn: sqlite3.Connection) -> None:
    """Restore additive M028 schema without recapturing its high-water mark."""
    apply(conn)
