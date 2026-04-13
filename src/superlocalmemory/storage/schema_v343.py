# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3.4.3 "The Unified Brain" — Schema Extensions.

All new tables for v3.4.3 features:
  - Phase C: Mesh broker tables (mesh_peers, mesh_messages, mesh_state, mesh_locks, mesh_events)
  - Phase D: Entity compilation (ALTER entity_profiles + new index)
  - Phase E: Ingestion log (ingestion_log)

Design rules (inherited from schema_v32.py):
  - CREATE IF NOT EXISTS for idempotency
  - profile_id where applicable
  - Never ALTER existing column types (add new columns only)

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table names (for reference and rollback)
# ---------------------------------------------------------------------------

V343_TABLES: Final[tuple[str, ...]] = (
    "mesh_peers",
    "mesh_messages",
    "mesh_state",
    "mesh_locks",
    "mesh_events",
    "ingestion_log",
)

# ---------------------------------------------------------------------------
# DDL — Phase C: Mesh Broker Tables
# ---------------------------------------------------------------------------

_MESH_DDL = """
-- Mesh Peers
CREATE TABLE IF NOT EXISTS mesh_peers (
    peer_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    summary TEXT DEFAULT '',
    status TEXT DEFAULT 'active',
    host TEXT DEFAULT '127.0.0.1',
    port INTEGER DEFAULT 0,
    registered_at TEXT NOT NULL,
    last_heartbeat TEXT NOT NULL
);

-- Mesh Messages
CREATE TABLE IF NOT EXISTS mesh_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_peer TEXT NOT NULL,
    to_peer TEXT NOT NULL,
    msg_type TEXT DEFAULT 'text',
    content TEXT NOT NULL,
    read INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

-- Mesh State (shared key-value store)
CREATE TABLE IF NOT EXISTS mesh_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    set_by TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Mesh Locks (file-level locks for coordination)
CREATE TABLE IF NOT EXISTS mesh_locks (
    file_path TEXT PRIMARY KEY,
    locked_by TEXT NOT NULL,
    locked_at TEXT NOT NULL,
    expires_at TEXT NOT NULL DEFAULT '9999-12-31T23:59:59Z'
);

-- Mesh Events (audit log)
CREATE TABLE IF NOT EXISTS mesh_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload TEXT DEFAULT '{}',
    emitted_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mesh_messages_to
    ON mesh_messages(to_peer, read);
CREATE INDEX IF NOT EXISTS idx_mesh_events_type
    ON mesh_events(event_type, created_at);
"""

# ---------------------------------------------------------------------------
# DDL — Phase D: Entity Compilation (ALTER entity_profiles)
# ---------------------------------------------------------------------------

_ENTITY_COMPILATION_ALTERS = [
    "ALTER TABLE entity_profiles ADD COLUMN project_name TEXT DEFAULT ''",
    "ALTER TABLE entity_profiles ADD COLUMN compiled_truth TEXT DEFAULT ''",
    "ALTER TABLE entity_profiles ADD COLUMN timeline TEXT DEFAULT '[]'",
    "ALTER TABLE entity_profiles ADD COLUMN compilation_confidence REAL DEFAULT 0.5",
    "ALTER TABLE entity_profiles ADD COLUMN last_compiled_at TEXT DEFAULT NULL",
]

_ENTITY_COMPILATION_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_profiles_compilation
    ON entity_profiles(entity_id, profile_id, project_name);
CREATE INDEX IF NOT EXISTS idx_entity_profiles_project
    ON entity_profiles(profile_id, project_name);
"""

# ---------------------------------------------------------------------------
# DDL — Phase E: Ingestion Log
# ---------------------------------------------------------------------------

_INGESTION_DDL = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    dedup_key TEXT NOT NULL,
    fact_ids TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    status TEXT DEFAULT 'ingested',
    ingested_at TEXT NOT NULL,
    UNIQUE(source_type, dedup_key)
);
CREATE INDEX IF NOT EXISTS idx_ingestion_dedup
    ON ingestion_log(source_type, dedup_key);
"""

# ---------------------------------------------------------------------------
# Version marker
# ---------------------------------------------------------------------------

_VERSION_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DDL — v3.4.6: Mesh Connected Brain (broadcast, project routing, offline queue)
# ---------------------------------------------------------------------------

_MESH_V346_ALTERS = [
    "ALTER TABLE mesh_peers ADD COLUMN project_path TEXT DEFAULT ''",
    "ALTER TABLE mesh_peers ADD COLUMN agent_type TEXT DEFAULT 'unknown'",
    "ALTER TABLE mesh_messages ADD COLUMN expires_at TEXT DEFAULT NULL",
    "ALTER TABLE mesh_messages ADD COLUMN target_type TEXT DEFAULT 'peer'",
    "ALTER TABLE mesh_messages ADD COLUMN project_path TEXT DEFAULT ''",
]

_MESH_V346_DDL = """
-- Read tracking for broadcast/project messages (each peer tracks own reads)
CREATE TABLE IF NOT EXISTS mesh_reads (
    message_id INTEGER NOT NULL,
    peer_id TEXT NOT NULL,
    read_at TEXT NOT NULL,
    PRIMARY KEY (message_id, peer_id)
);

CREATE INDEX IF NOT EXISTS idx_mesh_messages_target
    ON mesh_messages(target_type, project_path);
CREATE INDEX IF NOT EXISTS idx_mesh_messages_expires
    ON mesh_messages(expires_at);
CREATE INDEX IF NOT EXISTS idx_mesh_peers_project
    ON mesh_peers(project_path);
"""


def apply_v346_schema(db_path: str | sqlite3.Connection) -> dict:
    """Apply v3.4.6 mesh enhancements. Idempotent."""
    result = {"applied": [], "errors": []}

    if isinstance(db_path, sqlite3.Connection):
        conn = db_path
        own_connection = False
    else:
        conn = sqlite3.connect(str(db_path))
        own_connection = True

    try:
        for alter_sql in _MESH_V346_ALTERS:
            try:
                conn.execute(alter_sql)
                col_name = alter_sql.split("ADD COLUMN")[1].strip().split()[0]
                result["applied"].append(col_name)
            except sqlite3.OperationalError:
                pass  # Column already exists

        try:
            conn.executescript(_MESH_V346_DDL)
            result["applied"].append("mesh_reads table + indexes")
        except sqlite3.OperationalError as e:
            result["errors"].append(f"mesh_v346: {e}")

        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
            ("3.4.6", __import__("datetime").datetime.now().isoformat()),
        )
        conn.commit()

        if result["applied"]:
            logger.info("Schema v3.4.6 applied: %s", ", ".join(result["applied"]))

    except Exception as e:
        result["errors"].append(f"fatal: {e}")
    finally:
        if own_connection:
            conn.close()

    return result


def apply_v343_schema(db_path: str | sqlite3.Connection) -> dict:
    """Apply all v3.4.3 schema changes. Idempotent — safe to call multiple times.

    Returns dict with migration status and any errors.
    """
    result = {"applied": [], "skipped": [], "errors": []}

    if isinstance(db_path, sqlite3.Connection):
        conn = db_path
        own_connection = False
    else:
        conn = sqlite3.connect(str(db_path))
        own_connection = True

    try:
        # Check if already applied
        try:
            row = conn.execute(
                "SELECT version FROM schema_version WHERE version = '3.4.3'"
            ).fetchone()
            if row:
                result["skipped"].append("v3.4.3 already applied")
                return result
        except sqlite3.OperationalError:
            pass  # schema_version table doesn't exist yet

        # Apply version table
        conn.executescript(_VERSION_DDL)

        # Phase C: Mesh tables
        try:
            conn.executescript(_MESH_DDL)
            result["applied"].append("mesh tables (5 tables, 2 indexes)")
        except sqlite3.OperationalError as e:
            result["errors"].append(f"mesh: {e}")

        # Phase D: Entity compilation ALTER TABLE
        for alter_sql in _ENTITY_COMPILATION_ALTERS:
            try:
                conn.execute(alter_sql)
                result["applied"].append(alter_sql.split("ADD COLUMN")[1].strip().split()[0])
            except sqlite3.OperationalError:
                # Column already exists — fine (idempotent)
                pass

        try:
            conn.executescript(_ENTITY_COMPILATION_INDEX)
            result["applied"].append("entity_profiles indexes")
        except sqlite3.OperationalError as e:
            result["errors"].append(f"entity indexes: {e}")

        # Phase E: Ingestion log
        try:
            conn.executescript(_INGESTION_DDL)
            result["applied"].append("ingestion_log table")
        except sqlite3.OperationalError as e:
            result["errors"].append(f"ingestion: {e}")

        # Mark version
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
            ("3.4.3", __import__("datetime").datetime.now().isoformat()),
        )
        conn.commit()

        if result["applied"]:
            logger.info("Schema v3.4.3 applied: %s", ", ".join(result["applied"]))

    except Exception as e:
        result["errors"].append(f"fatal: {e}")
        logger.error("Schema v3.4.3 migration failed: %s", e)
    finally:
        if own_connection:
            conn.close()

    return result
