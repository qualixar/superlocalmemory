"""M050 — immutable bridge-v2 receipts and rebuildable execution learning."""

from __future__ import annotations

import sqlite3

NAME = "M050_execution_learning_v2"
DB_TARGET = "learning"

DDL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS execution_learning_receipts (
    profile_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    run_ref TEXT NOT NULL,
    run_id TEXT NOT NULL,
    receipt_head_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    producer_identity TEXT NOT NULL,
    capability_digest TEXT NOT NULL,
    terminal_listing_digest TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, workspace_id, run_ref)
);
CREATE TABLE IF NOT EXISTS execution_learning_events (
    profile_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    run_ref TEXT NOT NULL,
    receipt_head_digest TEXT NOT NULL,
    route_key TEXT NOT NULL,
    signal INTEGER NOT NULL CHECK (signal IN (-1, 1)),
    created_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, workspace_id, run_ref, receipt_head_digest)
);
CREATE INDEX IF NOT EXISTS idx_execution_learning_events_profile_route
    ON execution_learning_events (profile_id, route_key);
COMMIT;
"""


def apply(conn: sqlite3.Connection) -> None:
    """Install only additive, independently-owned v2 tables."""
    conn.executescript(DDL)
    if not verify(conn):
        raise sqlite3.OperationalError("M050 execution-learning schema did not reach its end-state")


def verify(conn: sqlite3.Connection) -> bool:
    required = {"execution_learning_receipts", "execution_learning_events"}
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if not required <= tables:
        return False
    receipt_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(execution_learning_receipts)")
    }
    return {
        "producer_identity", "capability_digest", "terminal_listing_digest",
    } <= receipt_columns


def repair(conn: sqlite3.Connection) -> None:
    """Restore missing derived indexes without mutating immutable receipts."""
    if not verify(conn):
        apply(conn)
        return
    conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_learning_events_profile_route "
                 "ON execution_learning_events (profile_id, route_key)")
