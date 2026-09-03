"""M042 — review-gated correction cases in ``memory.db``.

The ledger contains identifiers and lifecycle metadata only.  It never stores
fact text.  The saved predecessor temporal tuple lets a reviewed rollback
restore the exact lifecycle state without deleting fact history.
"""

from __future__ import annotations

import sqlite3

NAME = "M042_correction_case_ledger"
DB_TARGET = "memory"

DDL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS correction_cases (
    case_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('personal', 'project', 'shared', 'global')),
    predecessor_fact_id TEXT NOT NULL,
    successor_fact_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('proposed', 'applied', 'rejected', 'rolled_back')),
    version INTEGER NOT NULL CHECK (version >= 0),
    idempotency_key TEXT NOT NULL,
    proposed_by_actor_id TEXT NOT NULL,
    proposed_by_actor_kind TEXT NOT NULL,
    proposed_by_trust_tier TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reviewed_by_actor_id TEXT,
    reviewed_at TEXT,
    applied_at TEXT,
    system_effective_at TEXT,
    event_valid_from TEXT,
    event_valid_until TEXT,
    predecessor_temporal_existed INTEGER,
    predecessor_valid_from TEXT,
    predecessor_valid_until TEXT,
    predecessor_system_created_at TEXT,
    predecessor_system_expired_at TEXT,
    predecessor_invalidated_by TEXT,
    predecessor_invalidation_reason TEXT,
    UNIQUE (profile_id, idempotency_key),
    FOREIGN KEY (predecessor_fact_id) REFERENCES atomic_facts(fact_id) ON DELETE RESTRICT,
    FOREIGN KEY (successor_fact_id) REFERENCES atomic_facts(fact_id) ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS correction_events (
    event_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('personal', 'project', 'shared', 'global')),
    event_type TEXT NOT NULL
        CHECK (event_type IN ('proposed', 'applied', 'rejected', 'rolled_back')),
    operation_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    actor_kind TEXT NOT NULL,
    actor_trust_tier TEXT NOT NULL,
    expected_version INTEGER,
    resulting_version INTEGER NOT NULL CHECK (resulting_version >= 0),
    system_occurred_at TEXT NOT NULL,
    event_valid_from TEXT,
    event_valid_until TEXT,
    UNIQUE (case_id, operation_id),
    FOREIGN KEY (case_id) REFERENCES correction_cases(case_id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_correction_cases_profile_status
    ON correction_cases (profile_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_correction_events_case_sequence
    ON correction_events (case_id, system_occurred_at ASC);
CREATE INDEX IF NOT EXISTS idx_correction_cases_successor_admission
    ON correction_cases (profile_id, successor_fact_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_correction_cases_active_predecessor
    ON correction_cases (profile_id, predecessor_fact_id)
    WHERE status IN ('proposed', 'applied');
COMMIT;
"""

_TABLES = frozenset({"correction_cases", "correction_events"})
_FORBIDDEN_RAW_COLUMNS = frozenset({"content", "fact_text", "raw_text", "query"})
_REQUIRED_CASE_COLUMNS = frozenset(
    {
        "case_id",
        "profile_id",
        "scope",
        "predecessor_fact_id",
        "successor_fact_id",
        "reason_code",
        "status",
        "version",
        "idempotency_key",
        "proposed_by_actor_id",
        "proposed_by_actor_kind",
        "proposed_by_trust_tier",
        "created_at",
        "updated_at",
        "reviewed_by_actor_id",
        "reviewed_at",
        "applied_at",
        "system_effective_at",
        "event_valid_from",
        "event_valid_until",
        "predecessor_temporal_existed",
        "predecessor_valid_from",
        "predecessor_valid_until",
        "predecessor_system_created_at",
        "predecessor_system_expired_at",
        "predecessor_invalidated_by",
        "predecessor_invalidation_reason",
    }
)
_REQUIRED_EVENT_COLUMNS = frozenset(
    {
        "event_id",
        "case_id",
        "profile_id",
        "scope",
        "event_type",
        "operation_id",
        "actor_id",
        "actor_kind",
        "actor_trust_tier",
        "expected_version",
        "resulting_version",
        "system_occurred_at",
        "event_valid_from",
        "event_valid_until",
    }
)
_INDEX_SPECS = {
    "idx_correction_cases_profile_status": (
        "correction_cases",
        ("profile_id", "status", "updated_at"),
        False,
        None,
    ),
    "idx_correction_events_case_sequence": (
        "correction_events",
        ("case_id", "system_occurred_at"),
        False,
        None,
    ),
    "idx_correction_cases_successor_admission": (
        "correction_cases",
        ("profile_id", "successor_fact_id", "status"),
        False,
        None,
    ),
    "uq_correction_cases_active_predecessor": (
        "correction_cases",
        ("profile_id", "predecessor_fact_id"),
        True,
        "wherestatusin('proposed','applied')",
    ),
}


def apply(conn: sqlite3.Connection) -> None:
    """Install the additive review ledger atomically and idempotently."""
    if any(_table_exists(conn, table) for table in _TABLES) and not verify(conn):
        raise sqlite3.OperationalError("M042 correction ledger is malformed; refusing rebuild")
    conn.executescript(DDL)
    if not verify(conn):
        raise sqlite3.OperationalError("M042 correction ledger did not reach its end-state")


def verify(conn: sqlite3.Connection) -> bool:
    if not all(_table_exists(conn, table) for table in _TABLES):
        return False
    case_columns = _columns(conn, "correction_cases")
    event_columns = _columns(conn, "correction_events")
    if (
        set(case_columns) != _REQUIRED_CASE_COLUMNS
        or set(event_columns) != _REQUIRED_EVENT_COLUMNS
        or _FORBIDDEN_RAW_COLUMNS & (set(case_columns) | set(event_columns))
    ):
        return False
    if not _required_checks_present(conn):
        return False
    return all(_index_matches(conn, name, *spec) for name, spec in _INDEX_SPECS.items())


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))


def _required_checks_present(conn: sqlite3.Connection) -> bool:
    statements: dict[str, str] = {}
    for table in ("correction_cases", "correction_events"):
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if row is None or row[0] is None:
            return False
        statements[table] = "".join(str(row[0]).lower().split())
    case_sql = statements["correction_cases"]
    event_sql = statements["correction_events"]
    scope_check = "check(scopein('personal','project','shared','global'))"
    return (
        scope_check in case_sql
        and scope_check in event_sql
        and "check(statusin('proposed','applied','rejected','rolled_back'))" in case_sql
        and "check(event_typein('proposed','applied','rejected','rolled_back'))" in event_sql
        and "unique(profile_id,idempotency_key)" in case_sql
        and "unique(case_id,operation_id)" in event_sql
    )


def _index_matches(
    conn: sqlite3.Connection,
    name: str,
    table: str,
    columns: tuple[str, ...],
    unique: bool,
    where_clause: str | None,
) -> bool:
    """Verify an index's table, ordered columns, uniqueness, and partial predicate."""
    row = conn.execute(
        "SELECT tbl_name, sql FROM sqlite_master WHERE type='index' AND name=?", (name,)
    ).fetchone()
    if row is None or row[0] != table:
        return False
    index_rows = conn.execute(f"PRAGMA index_xinfo({name})").fetchall()
    indexed_columns = tuple(
        entry[2] for entry in index_rows if entry[5] == 1 and entry[2] is not None
    )
    if indexed_columns != columns:
        return False
    index_list = conn.execute(f"PRAGMA index_list({table})").fetchall()
    indexed = next((entry for entry in index_list if entry[1] == name), None)
    if indexed is None or bool(indexed[2]) is not unique:
        return False
    if where_clause is None:
        return True
    return row[1] is not None and where_clause in "".join(str(row[1]).lower().split())


def repair(conn: sqlite3.Connection) -> None:
    """Re-run the idempotent apply as end-state repair (4.1.14 #133).

    A malformed ledger raises instead of rebuilding: the framework
    reports it honestly rather than claiming a repaired end-state.
    """
    apply(conn)
