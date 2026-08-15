"""M040 — profile-scoped Agent Experience receipts in ``learning.db``.

The receipt plane deliberately has no foreign keys into ``memory.db``.  It is
therefore not part of recall's lock domain: profile lifecycle coordinates its
cross-store erasure as a retryable saga instead of using ``ATTACH``.
"""

from __future__ import annotations

import sqlite3

NAME = "M040_agent_experience_receipts"
DB_TARGET = "learning"

DDL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS agent_experiences (
    profile_id TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    task_class TEXT NOT NULL,
    project_scope TEXT NOT NULL,
    route_json TEXT NOT NULL,
    verification_authority TEXT NOT NULL,
    verification_digest TEXT NOT NULL,
    verification_reference TEXT,
    producer_claim TEXT NOT NULL,
    terminal_status TEXT NOT NULL,
    failure_class TEXT,
    human_intervention INTEGER CHECK (human_intervention IN (0, 1)),
    lessons TEXT,
    receipt_digest TEXT,
    artifact_digests_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, experience_id)
);
CREATE TABLE IF NOT EXISTS cognitive_turn_receipts (
    profile_id TEXT NOT NULL,
    receipt_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    project_scope TEXT NOT NULL,
    query_digest TEXT NOT NULL,
    fact_decisions_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('open', 'finalized', 'abandoned', 'reconciled')),
    outcome_json TEXT,
    payload_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, receipt_id)
);
CREATE TABLE IF NOT EXISTS agent_receipt_profile_closures (
    profile_id TEXT PRIMARY KEY,
    closed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_experiences_profile_occurred
    ON agent_experiences (profile_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_experiences_profile_project_occurred
    ON agent_experiences (profile_id, project_scope, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_cognitive_turns_profile_task_updated
    ON cognitive_turn_receipts (profile_id, task_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_cognitive_turns_profile_state_updated
    ON cognitive_turn_receipts (profile_id, state, updated_at DESC);
COMMIT;
"""

_TABLE_COLUMNS = {
    "agent_experiences": (
        "profile_id",
        "experience_id",
        "occurred_at",
        "task_class",
        "project_scope",
        "route_json",
        "verification_authority",
        "verification_digest",
        "verification_reference",
        "producer_claim",
        "terminal_status",
        "failure_class",
        "human_intervention",
        "lessons",
        "receipt_digest",
        "artifact_digests_json",
        "payload_sha256",
        "created_at",
    ),
    "cognitive_turn_receipts": (
        "profile_id",
        "receipt_id",
        "task_id",
        "project_scope",
        "query_digest",
        "fact_decisions_json",
        "state",
        "outcome_json",
        "payload_sha256",
        "created_at",
        "updated_at",
    ),
    "agent_receipt_profile_closures": ("profile_id", "closed_at"),
}
_PRIMARY_KEYS = {
    "agent_experiences": ("profile_id", "experience_id"),
    "cognitive_turn_receipts": ("profile_id", "receipt_id"),
    "agent_receipt_profile_closures": ("profile_id",),
}
_COLUMN_TYPES = {
    "agent_experiences": (
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "INTEGER",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
        "TEXT",
    ),
    "cognitive_turn_receipts": ("TEXT",) * len(_TABLE_COLUMNS["cognitive_turn_receipts"]),
    "agent_receipt_profile_closures": ("TEXT", "TEXT"),
}
_REQUIRED_NOT_NULL = {
    "agent_experiences": frozenset(_TABLE_COLUMNS["agent_experiences"])
    - {
        "verification_reference",
        "failure_class",
        "human_intervention",
        "lessons",
        "receipt_digest",
    },
    "cognitive_turn_receipts": frozenset(_TABLE_COLUMNS["cognitive_turn_receipts"])
    - {"outcome_json"},
    "agent_receipt_profile_closures": frozenset(_TABLE_COLUMNS["agent_receipt_profile_closures"]),
}
_INDEXES = {
    "idx_agent_experiences_profile_occurred": ("agent_experiences", ("profile_id", "occurred_at")),
    "idx_agent_experiences_profile_project_occurred": (
        "agent_experiences",
        ("profile_id", "project_scope", "occurred_at"),
    ),
    "idx_cognitive_turns_profile_task_updated": (
        "cognitive_turn_receipts",
        ("profile_id", "task_id", "updated_at"),
    ),
    "idx_cognitive_turns_profile_state_updated": (
        "cognitive_turn_receipts",
        ("profile_id", "state", "updated_at"),
    ),
}


def apply(conn: sqlite3.Connection) -> None:
    """Atomically install M040 or leave the learning DB unchanged."""
    if verify(conn):
        return
    if _tables_are_malformed(conn):
        raise sqlite3.OperationalError("M040 receipt tables are malformed; refusing rebuild")
    if all(name in _tables(conn) for name in _TABLE_COLUMNS):
        repair(conn)
        return
    conn.executescript(DDL)
    if not verify(conn):
        # A pre-existing same-named index can make CREATE INDEX IF NOT EXISTS
        # a no-op even though that index belongs to another table. Rebuild the
        # derived index set transactionally after all required tables exist.
        repair(conn)
    if not verify(conn):
        raise sqlite3.OperationalError("M040 schema did not reach its required end-state")


def repair(conn: sqlite3.Connection) -> None:
    """Restore missing/wrong indexes only; never rebuild user evidence."""
    if _tables_are_malformed(conn):
        raise sqlite3.OperationalError("M040 receipt tables are malformed; refusing rebuild")
    if not all(name in _tables(conn) for name in _TABLE_COLUMNS):
        apply(conn)
        return
    drops = "\n".join(f"DROP INDEX IF EXISTS {name};" for name in _INDEXES)
    creates = "\n".join(
        f"CREATE INDEX {name} ON {table} ({', '.join(columns)});"
        for name, (table, columns) in _INDEXES.items()
    )
    conn.executescript(f"BEGIN IMMEDIATE;\n{drops}\n{creates}\nCOMMIT;")
    if not verify(conn):
        raise sqlite3.OperationalError("M040 index repair did not restore required end-state")


def verify(conn: sqlite3.Connection) -> bool:
    return not _tables_are_malformed(conn) and all(
        _index_columns(conn, name) == columns for name, (_, columns) in _INDEXES.items()
    )


def _tables_are_malformed(conn: sqlite3.Connection) -> bool:
    tables = _tables(conn)
    for table, columns in _TABLE_COLUMNS.items():
        if table not in tables:
            continue
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        actual = tuple(row[1] for row in info)
        types = tuple(str(row[2]).upper() for row in info)
        primary_key = tuple(row[1] for row in sorted(info, key=lambda row: row[5]) if row[5])
        not_null = {row[1] for row in info if row[3] or row[5]}
        if (
            actual != columns
            or types != _COLUMN_TYPES[table]
            or primary_key != _PRIMARY_KEYS[table]
            or not_null != _REQUIRED_NOT_NULL[table]
            or not _required_checks_present(conn, table)
        ):
            return True
        if conn.execute(f"PRAGMA foreign_key_list({table})").fetchone() is not None:
            return True
    return False


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _index_columns(conn: sqlite3.Connection, name: str) -> tuple[str, ...] | None:
    row = conn.execute(
        "SELECT tbl_name FROM sqlite_master WHERE type='index' AND name=?", (name,)
    ).fetchone()
    expected = _INDEXES.get(name)
    if row is None or expected is None or row[0] != expected[0]:
        return None
    return tuple(
        row[2]
        for row in conn.execute(f"PRAGMA index_xinfo({name})")
        if row[5] and row[2] is not None
    )


def _required_checks_present(conn: sqlite3.Connection, table: str) -> bool:
    sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    sql = "" if sql_row is None or sql_row[0] is None else "".join(str(sql_row[0]).lower().split())
    if table == "agent_experiences":
        return "check(human_interventionin(0,1))" in sql
    if table == "cognitive_turn_receipts":
        return "check(statein('open','finalized','abandoned','reconciled'))" in sql
    return True
