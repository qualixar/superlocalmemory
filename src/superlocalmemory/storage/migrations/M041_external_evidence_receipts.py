"""M041 — typed, observation-only external evidence in ``learning.db``."""

from __future__ import annotations

import sqlite3

NAME = "M041_external_evidence_receipts"
DB_TARGET = "learning"

DDL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS external_evidence_receipts (
    profile_id TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    run_ref TEXT NOT NULL,
    run_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    run_state TEXT NOT NULL,
    demonstration INTEGER NOT NULL CHECK (demonstration IN (0, 1)),
    eligible_for_learning INTEGER NOT NULL CHECK (eligible_for_learning IN (0, 1)),
    terminal_at TEXT NOT NULL,
    graph_digest TEXT NOT NULL,
    plan_digest TEXT NOT NULL,
    policy_digest TEXT NOT NULL,
    receipt_sequence INTEGER NOT NULL,
    receipt_head_digest TEXT NOT NULL,
    receipt_trust TEXT NOT NULL,
    nodes_json TEXT NOT NULL,
    artifact_digests_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, contract_id, workspace_id, run_ref)
);
CREATE INDEX IF NOT EXISTS idx_external_evidence_profile_terminal
    ON external_evidence_receipts (profile_id, terminal_at DESC);
CREATE INDEX IF NOT EXISTS idx_external_evidence_profile_workspace
    ON external_evidence_receipts (profile_id, workspace_id, terminal_at DESC);
COMMIT;
"""

_TABLE = "external_evidence_receipts"
_COLUMNS = (
    "profile_id",
    "contract_id",
    "workspace_id",
    "run_ref",
    "run_id",
    "outcome",
    "run_state",
    "demonstration",
    "eligible_for_learning",
    "terminal_at",
    "graph_digest",
    "plan_digest",
    "policy_digest",
    "receipt_sequence",
    "receipt_head_digest",
    "receipt_trust",
    "nodes_json",
    "artifact_digests_json",
    "payload_sha256",
    "observed_at",
)
_TYPES = (
    "TEXT",
    "TEXT",
    "TEXT",
    "TEXT",
    "TEXT",
    "TEXT",
    "TEXT",
    "INTEGER",
    "INTEGER",
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
    "TEXT",
)
_PRIMARY_KEY = ("profile_id", "contract_id", "workspace_id", "run_ref")
_INDEXES = {
    "idx_external_evidence_profile_terminal": (
        "external_evidence_receipts", ("profile_id", "terminal_at"),
    ),
    "idx_external_evidence_profile_workspace": (
        "external_evidence_receipts", ("profile_id", "workspace_id", "terminal_at"),
    ),
}


def apply(conn: sqlite3.Connection) -> None:
    """Install the additive receipt table atomically and idempotently."""
    if _table_exists(conn) and not _table_is_valid(conn):
        raise sqlite3.OperationalError(
            "M041 external evidence table is malformed; refusing rebuild"
        )
    conn.executescript(DDL)
    if not verify(conn):
        repair(conn)
    if not verify(conn):
        raise sqlite3.OperationalError("M041 external evidence schema did not reach its end-state")


def repair(conn: sqlite3.Connection) -> None:
    """Restore only M041's derived indexes without touching stored evidence."""
    if _table_exists(conn) and not _table_is_valid(conn):
        raise sqlite3.OperationalError(
            "M041 external evidence table is malformed; refusing rebuild"
        )
    if not _table_exists(conn):
        apply(conn)
        return
    drops = "\n".join(f"DROP INDEX IF EXISTS {name};" for name in _INDEXES)
    creates = "\n".join(
        f"CREATE INDEX {name} ON {table} ({', '.join(columns)});"
        for name, (table, columns) in _INDEXES.items()
    )
    conn.executescript(f"BEGIN IMMEDIATE;\n{drops}\n{creates}\nCOMMIT;")
    if not verify(conn):
        raise sqlite3.OperationalError("M041 index repair did not restore required end-state")


def verify(conn: sqlite3.Connection) -> bool:
    if not _table_is_valid(conn):
        return False
    for name, (table, columns) in _INDEXES.items():
        row = conn.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type='index' AND name=?", (name,)
        ).fetchone()
        if row is None or row[0] != table:
            return False
        actual = tuple(
            item[2]
            for item in conn.execute(f"PRAGMA index_xinfo({name})")
            if item[5] and item[2] is not None
        )
        if actual != columns:
            return False
    return True


def _table_exists(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (_TABLE,)
        ).fetchone()
        is not None
    )


def _table_is_valid(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn):
        return False
    info = conn.execute(f"PRAGMA table_info({_TABLE})").fetchall()
    columns = tuple(row[1] for row in info)
    types = tuple(str(row[2]).upper() for row in info)
    primary_key = tuple(row[1] for row in sorted(info, key=lambda row: row[5]) if row[5])
    not_null = {row[1] for row in info if row[3] or row[5]}
    return (
        columns == _COLUMNS
        and types == _TYPES
        and primary_key == _PRIMARY_KEY
        and not_null == set(_COLUMNS)
        and conn.execute(f"PRAGMA foreign_key_list({_TABLE})").fetchone() is None
        and _required_checks_present(conn)
    )


def _required_checks_present(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (_TABLE,)
    ).fetchone()
    sql = "" if row is None or row[0] is None else "".join(str(row[0]).lower().split())
    return (
        "check(demonstrationin(0,1))" in sql
        and "check(eligible_for_learningin(0,1))" in sql
    )
