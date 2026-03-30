# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Database Schema.

SQL DDL for the entire memory system. Zero dead tables: every table
has both a writer and a reader. Profile-scoped by design.

Design principles:
  - 17 tables, each justified by a write path AND a read path
  - profile_id on every row — instant profile switching
  - FTS5 virtual table on atomic_facts for full-text search
  - Foreign keys with CASCADE deletes — no orphans
  - WAL journal mode for concurrent reads
  - Indexes on all hot query paths (entity lookup, temporal range,
    fact_type filter, profile scoping)

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import sqlite3
from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: Final[int] = 1

# All table names in creation order (respects FK dependencies).
# FTS5 virtual tables listed separately — they cannot have FKs.
_TABLES: Final[tuple[str, ...]] = (
    "schema_version",
    "profiles",
    "memories",
    "atomic_facts",
    "canonical_entities",
    "entity_aliases",
    "entity_profiles",
    "memory_scenes",
    "temporal_events",
    "graph_edges",
    "consolidation_log",
    "trust_scores",
    "provenance",
    "feedback_records",
    "behavioral_patterns",
    "action_outcomes",
    "compliance_audit",
    "bm25_tokens",
    "config",
)

_FTS_TABLES: Final[tuple[str, ...]] = (
    "atomic_facts_fts",
)


# ---------------------------------------------------------------------------
# Pragmas
# ---------------------------------------------------------------------------

def _set_pragmas(conn: sqlite3.Connection) -> None:
    """Set SQLite pragmas for performance and integrity."""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")


# ---------------------------------------------------------------------------
# Schema version (must be first — no FK dependencies)
# ---------------------------------------------------------------------------

_SQL_SCHEMA_VERSION: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    description TEXT    NOT NULL DEFAULT ''
);
"""


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

_SQL_PROFILES: Final[str] = """
CREATE TABLE IF NOT EXISTS profiles (
    profile_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    personality  TEXT NOT NULL DEFAULT '',
    mode         TEXT NOT NULL DEFAULT 'a'
                      CHECK (mode IN ('a', 'b', 'c')),
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    last_used    TEXT,
    config_json  TEXT NOT NULL DEFAULT '{}'
);
"""


# ---------------------------------------------------------------------------
# Memories (raw conversation turns)
# ---------------------------------------------------------------------------

_SQL_MEMORIES: Final[str] = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id    TEXT PRIMARY KEY,
    profile_id   TEXT NOT NULL DEFAULT 'default',
    content      TEXT NOT NULL,
    session_id   TEXT NOT NULL DEFAULT '',
    speaker      TEXT NOT NULL DEFAULT '',
    role         TEXT NOT NULL DEFAULT '',
    session_date TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    metadata_json TEXT NOT NULL DEFAULT '{}',

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memories_profile
    ON memories (profile_id);
CREATE INDEX IF NOT EXISTS idx_memories_session
    ON memories (profile_id, session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created
    ON memories (created_at);
"""


# ---------------------------------------------------------------------------
# Atomic facts — THE primary retrieval unit
# ---------------------------------------------------------------------------

_SQL_ATOMIC_FACTS: Final[str] = """
CREATE TABLE IF NOT EXISTS atomic_facts (
    fact_id            TEXT PRIMARY KEY,
    memory_id          TEXT NOT NULL,
    profile_id         TEXT NOT NULL DEFAULT 'default',
    content            TEXT NOT NULL,
    fact_type          TEXT NOT NULL DEFAULT 'semantic'
                            CHECK (fact_type IN (
                                'episodic', 'semantic', 'opinion', 'temporal'
                            )),

    -- Entities (JSON arrays)
    entities_json           TEXT NOT NULL DEFAULT '[]',
    canonical_entities_json TEXT NOT NULL DEFAULT '[]',

    -- Temporal (3-date model)
    observation_date   TEXT,
    referenced_date    TEXT,
    interval_start     TEXT,
    interval_end       TEXT,

    -- Quality scores
    confidence         REAL NOT NULL DEFAULT 1.0,
    importance         REAL NOT NULL DEFAULT 0.5,
    evidence_count     INTEGER NOT NULL DEFAULT 1,
    access_count       INTEGER NOT NULL DEFAULT 0,

    -- Source tracing
    source_turn_ids_json TEXT NOT NULL DEFAULT '[]',
    session_id           TEXT NOT NULL DEFAULT '',

    -- Embeddings (JSON arrays — TEXT for simplicity and portability)
    embedding          TEXT,
    fisher_mean        TEXT,
    fisher_variance    TEXT,

    -- Lifecycle
    lifecycle          TEXT NOT NULL DEFAULT 'active'
                            CHECK (lifecycle IN (
                                'active', 'warm', 'cold', 'archived'
                            )),
    langevin_position  TEXT,

    -- Emotional
    emotional_valence  REAL NOT NULL DEFAULT 0.0,
    emotional_arousal  REAL NOT NULL DEFAULT 0.0,

    -- Signal type (V2 compatible)
    signal_type        TEXT NOT NULL DEFAULT 'factual',

    created_at         TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (memory_id) REFERENCES memories (memory_id)
        ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_facts_profile
    ON atomic_facts (profile_id);
CREATE INDEX IF NOT EXISTS idx_facts_memory
    ON atomic_facts (memory_id);
CREATE INDEX IF NOT EXISTS idx_facts_type
    ON atomic_facts (profile_id, fact_type);
CREATE INDEX IF NOT EXISTS idx_facts_lifecycle
    ON atomic_facts (profile_id, lifecycle);
CREATE INDEX IF NOT EXISTS idx_facts_session
    ON atomic_facts (profile_id, session_id);
CREATE INDEX IF NOT EXISTS idx_facts_referenced_date
    ON atomic_facts (profile_id, referenced_date)
    WHERE referenced_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_facts_interval
    ON atomic_facts (profile_id, interval_start, interval_end)
    WHERE interval_start IS NOT NULL;
"""


# ---------------------------------------------------------------------------
# FTS5 virtual table on atomic_facts for full-text search
# ---------------------------------------------------------------------------

_SQL_V2_MIGRATION_CLEANUP: Final[str] = """
-- Clean up stale V2 triggers that fire on active tables but reference
-- renamed backup FTS tables. The V2→V3 migration renames tables via
-- ALTER TABLE RENAME, which auto-updates trigger bodies to reference
-- _v2_bak_* tables but leaves FTS5 delete-command column names stale.
-- This causes: "table _v2_bak_*_fts has no column named *_fts"

-- Drop V2-era triggers on memories table (memories_ai/ad/au)
DROP TRIGGER IF EXISTS memories_ai;
DROP TRIGGER IF EXISTS memories_ad;
DROP TRIGGER IF EXISTS memories_au;

-- Drop stale V3 triggers (may have been corrupted by V2 rename)
DROP TRIGGER IF EXISTS atomic_facts_fts_insert;
DROP TRIGGER IF EXISTS atomic_facts_fts_delete;
DROP TRIGGER IF EXISTS atomic_facts_fts_update;

-- Drop renamed V2 backup FTS virtual tables (and their shadow tables)
DROP TABLE IF EXISTS "_v2_bak_atomic_facts_fts";
DROP TABLE IF EXISTS "_v2_bak_memories_fts";
"""

_SQL_ATOMIC_FACTS_FTS: Final[str] = """
CREATE VIRTUAL TABLE IF NOT EXISTS atomic_facts_fts
    USING fts5(
        fact_id UNINDEXED,
        content,
        content='atomic_facts',
        content_rowid='rowid'
    );

-- Triggers to keep FTS in sync with atomic_facts.
-- Always DROP+CREATE (not IF NOT EXISTS) to replace any stale triggers
-- left by V2 migration.

-- INSERT trigger
CREATE TRIGGER atomic_facts_fts_insert
    AFTER INSERT ON atomic_facts
BEGIN
    INSERT INTO atomic_facts_fts (rowid, fact_id, content)
        VALUES (NEW.rowid, NEW.fact_id, NEW.content);
END;

-- DELETE trigger
CREATE TRIGGER atomic_facts_fts_delete
    AFTER DELETE ON atomic_facts
BEGIN
    INSERT INTO atomic_facts_fts (atomic_facts_fts, rowid, fact_id, content)
        VALUES ('delete', OLD.rowid, OLD.fact_id, OLD.content);
END;

-- UPDATE trigger
CREATE TRIGGER atomic_facts_fts_update
    AFTER UPDATE OF content ON atomic_facts
BEGIN
    INSERT INTO atomic_facts_fts (atomic_facts_fts, rowid, fact_id, content)
        VALUES ('delete', OLD.rowid, OLD.fact_id, OLD.content);
    INSERT INTO atomic_facts_fts (rowid, fact_id, content)
        VALUES (NEW.rowid, NEW.fact_id, NEW.content);
END;
"""


# ---------------------------------------------------------------------------
# Canonical entities
# ---------------------------------------------------------------------------

_SQL_CANONICAL_ENTITIES: Final[str] = """
CREATE TABLE IF NOT EXISTS canonical_entities (
    entity_id       TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL DEFAULT 'default',
    canonical_name  TEXT NOT NULL,
    entity_type     TEXT NOT NULL DEFAULT '',
    first_seen      TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen       TEXT NOT NULL DEFAULT (datetime('now')),
    fact_count      INTEGER NOT NULL DEFAULT 0,

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entities_profile
    ON canonical_entities (profile_id);
CREATE INDEX IF NOT EXISTS idx_entities_name_lower
    ON canonical_entities (profile_id, canonical_name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_entities_type
    ON canonical_entities (profile_id, entity_type);
"""


# ---------------------------------------------------------------------------
# Entity aliases
# ---------------------------------------------------------------------------

_SQL_ENTITY_ALIASES: Final[str] = """
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_id    TEXT PRIMARY KEY,
    entity_id   TEXT NOT NULL,
    alias       TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    source      TEXT NOT NULL DEFAULT '',

    FOREIGN KEY (entity_id) REFERENCES canonical_entities (entity_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_aliases_entity
    ON entity_aliases (entity_id);
CREATE INDEX IF NOT EXISTS idx_aliases_lookup
    ON entity_aliases (alias COLLATE NOCASE);
"""


# ---------------------------------------------------------------------------
# Entity profiles (accumulated knowledge per entity)
# ---------------------------------------------------------------------------

_SQL_ENTITY_PROFILES: Final[str] = """
CREATE TABLE IF NOT EXISTS entity_profiles (
    profile_entry_id   TEXT PRIMARY KEY,
    entity_id          TEXT NOT NULL,
    profile_id         TEXT NOT NULL DEFAULT 'default',
    knowledge_summary  TEXT NOT NULL DEFAULT '',
    fact_ids_json      TEXT NOT NULL DEFAULT '[]',
    last_updated       TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (entity_id) REFERENCES canonical_entities (entity_id)
        ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eprofiles_entity
    ON entity_profiles (entity_id);
CREATE INDEX IF NOT EXISTS idx_eprofiles_profile
    ON entity_profiles (profile_id);
"""


# ---------------------------------------------------------------------------
# Memory scenes (clustered groups of related facts)
# ---------------------------------------------------------------------------

_SQL_MEMORY_SCENES: Final[str] = """
CREATE TABLE IF NOT EXISTS memory_scenes (
    scene_id        TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL DEFAULT 'default',
    theme           TEXT NOT NULL DEFAULT '',
    fact_ids_json   TEXT NOT NULL DEFAULT '[]',
    entity_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated    TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scenes_profile
    ON memory_scenes (profile_id);
"""


# ---------------------------------------------------------------------------
# Temporal events (per-entity timeline entries)
# ---------------------------------------------------------------------------

_SQL_TEMPORAL_EVENTS: Final[str] = """
CREATE TABLE IF NOT EXISTS temporal_events (
    event_id         TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL DEFAULT 'default',
    entity_id        TEXT NOT NULL,
    fact_id          TEXT NOT NULL,
    observation_date TEXT,
    referenced_date  TEXT,
    interval_start   TEXT,
    interval_end     TEXT,
    description      TEXT NOT NULL DEFAULT '',

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES canonical_entities (entity_id)
        ON DELETE CASCADE,
    FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tevents_profile
    ON temporal_events (profile_id);
CREATE INDEX IF NOT EXISTS idx_tevents_entity
    ON temporal_events (profile_id, entity_id);
CREATE INDEX IF NOT EXISTS idx_tevents_date_range
    ON temporal_events (profile_id, referenced_date)
    WHERE referenced_date IS NOT NULL;
"""


# ---------------------------------------------------------------------------
# Graph edges (knowledge graph between facts/entities)
# ---------------------------------------------------------------------------

_SQL_GRAPH_EDGES: Final[str] = """
CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id     TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL DEFAULT 'default',
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL DEFAULT 'entity'
                     CHECK (edge_type IN (
                         'entity', 'temporal', 'semantic',
                         'causal', 'contradiction', 'supersedes'
                     )),
    weight      REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_profile
    ON graph_edges (profile_id);
CREATE INDEX IF NOT EXISTS idx_edges_source
    ON graph_edges (profile_id, source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target
    ON graph_edges (profile_id, target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type
    ON graph_edges (profile_id, edge_type);
"""


# ---------------------------------------------------------------------------
# Consolidation log (ADD / UPDATE / SUPERSEDE / NOOP decisions)
# ---------------------------------------------------------------------------

_SQL_CONSOLIDATION_LOG: Final[str] = """
CREATE TABLE IF NOT EXISTS consolidation_log (
    action_id        TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL DEFAULT 'default',
    action_type      TEXT NOT NULL DEFAULT 'add'
                          CHECK (action_type IN (
                              'add', 'update', 'supersede', 'noop'
                          )),
    new_fact_id      TEXT NOT NULL DEFAULT '',
    existing_fact_id TEXT NOT NULL DEFAULT '',
    reason           TEXT NOT NULL DEFAULT '',
    timestamp        TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conslog_profile
    ON consolidation_log (profile_id);
CREATE INDEX IF NOT EXISTS idx_conslog_new_fact
    ON consolidation_log (new_fact_id);
"""


# ---------------------------------------------------------------------------
# Trust scores (Bayesian per source / entity / fact)
# ---------------------------------------------------------------------------

_SQL_TRUST_SCORES: Final[str] = """
CREATE TABLE IF NOT EXISTS trust_scores (
    trust_id       TEXT PRIMARY KEY,
    profile_id     TEXT NOT NULL DEFAULT 'default',
    target_type    TEXT NOT NULL DEFAULT '',
    target_id      TEXT NOT NULL DEFAULT '',
    trust_score    REAL NOT NULL DEFAULT 0.5,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    last_updated   TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trust_profile
    ON trust_scores (profile_id);
CREATE INDEX IF NOT EXISTS idx_trust_target
    ON trust_scores (profile_id, target_type, target_id);
"""


# ---------------------------------------------------------------------------
# Provenance (who/what created this memory and how)
# ---------------------------------------------------------------------------

_SQL_PROVENANCE: Final[str] = """
CREATE TABLE IF NOT EXISTS provenance (
    provenance_id TEXT PRIMARY KEY,
    profile_id    TEXT NOT NULL DEFAULT 'default',
    fact_id       TEXT NOT NULL,
    source_type   TEXT NOT NULL DEFAULT '',
    source_id     TEXT NOT NULL DEFAULT '',
    created_by    TEXT NOT NULL DEFAULT '',
    timestamp     TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE,
    FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_prov_profile
    ON provenance (profile_id);
CREATE INDEX IF NOT EXISTS idx_prov_fact
    ON provenance (fact_id);
"""


# ---------------------------------------------------------------------------
# Feedback records (user feedback on retrieval results)
# ---------------------------------------------------------------------------

_SQL_FEEDBACK_RECORDS: Final[str] = """
CREATE TABLE IF NOT EXISTS feedback_records (
    feedback_id   TEXT PRIMARY KEY,
    profile_id    TEXT NOT NULL DEFAULT 'default',
    query         TEXT NOT NULL DEFAULT '',
    fact_id       TEXT NOT NULL,
    feedback_type TEXT NOT NULL DEFAULT '',
    dwell_time_ms INTEGER NOT NULL DEFAULT 0,
    timestamp     TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE,
    FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_profile
    ON feedback_records (profile_id);
CREATE INDEX IF NOT EXISTS idx_feedback_fact
    ON feedback_records (fact_id);
"""


# ---------------------------------------------------------------------------
# Behavioral patterns (learned query habits / topic prefs)
# ---------------------------------------------------------------------------

_SQL_BEHAVIORAL_PATTERNS: Final[str] = """
CREATE TABLE IF NOT EXISTS behavioral_patterns (
    pattern_id        TEXT PRIMARY KEY,
    profile_id        TEXT NOT NULL DEFAULT 'default',
    pattern_type      TEXT NOT NULL DEFAULT '',
    pattern_key       TEXT NOT NULL DEFAULT '',
    pattern_value     TEXT NOT NULL DEFAULT '',
    confidence        REAL NOT NULL DEFAULT 0.0,
    observation_count INTEGER NOT NULL DEFAULT 0,
    last_updated      TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_bpatterns_profile
    ON behavioral_patterns (profile_id);
CREATE INDEX IF NOT EXISTS idx_bpatterns_type
    ON behavioral_patterns (profile_id, pattern_type);
"""


# ---------------------------------------------------------------------------
# Action outcomes (did the retrieved facts help?)
# ---------------------------------------------------------------------------

_SQL_ACTION_OUTCOMES: Final[str] = """
CREATE TABLE IF NOT EXISTS action_outcomes (
    outcome_id    TEXT PRIMARY KEY,
    profile_id    TEXT NOT NULL DEFAULT 'default',
    query         TEXT NOT NULL DEFAULT '',
    fact_ids_json TEXT NOT NULL DEFAULT '[]',
    outcome       TEXT NOT NULL DEFAULT '',
    context_json  TEXT NOT NULL DEFAULT '{}',
    timestamp     TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_outcomes_profile
    ON action_outcomes (profile_id);
"""


# ---------------------------------------------------------------------------
# Compliance audit (GDPR, EU AI Act trail)
# ---------------------------------------------------------------------------

_SQL_COMPLIANCE_AUDIT: Final[str] = """
CREATE TABLE IF NOT EXISTS compliance_audit (
    audit_id    TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL DEFAULT 'default',
    action      TEXT NOT NULL DEFAULT '',
    target_type TEXT NOT NULL DEFAULT '',
    target_id   TEXT NOT NULL DEFAULT '',
    details     TEXT NOT NULL DEFAULT '',
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_audit_profile
    ON compliance_audit (profile_id);
CREATE INDEX IF NOT EXISTS idx_audit_action
    ON compliance_audit (profile_id, action);
CREATE INDEX IF NOT EXISTS idx_audit_target
    ON compliance_audit (profile_id, target_type, target_id);
"""


# ---------------------------------------------------------------------------
# BM25 Token Persistence
# ---------------------------------------------------------------------------

_SQL_BM25_TOKENS: Final[str] = """
CREATE TABLE IF NOT EXISTS bm25_tokens (
    fact_id    TEXT NOT NULL,
    profile_id TEXT NOT NULL DEFAULT 'default',
    tokens     TEXT NOT NULL DEFAULT '[]',

    PRIMARY KEY (fact_id, profile_id),
    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
        ON DELETE CASCADE
);
"""

# ---------------------------------------------------------------------------
# Key-Value Config Store
# ---------------------------------------------------------------------------

_SQL_CONFIG: Final[str] = """
CREATE TABLE IF NOT EXISTS config (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# ---------------------------------------------------------------------------
# Ordered DDL list (tables before FTS, respects FK order)
# ---------------------------------------------------------------------------

_DDL_ORDERED: Final[tuple[str, ...]] = (
    _SQL_SCHEMA_VERSION,
    _SQL_PROFILES,
    _SQL_MEMORIES,
    _SQL_ATOMIC_FACTS,
    _SQL_CANONICAL_ENTITIES,
    _SQL_ENTITY_ALIASES,
    _SQL_ENTITY_PROFILES,
    _SQL_MEMORY_SCENES,
    _SQL_TEMPORAL_EVENTS,
    _SQL_GRAPH_EDGES,
    _SQL_CONSOLIDATION_LOG,
    _SQL_TRUST_SCORES,
    _SQL_PROVENANCE,
    _SQL_FEEDBACK_RECORDS,
    _SQL_BEHAVIORAL_PATTERNS,
    _SQL_ACTION_OUTCOMES,
    _SQL_COMPLIANCE_AUDIT,
    _SQL_BM25_TOKENS,
    _SQL_CONFIG,
    # V2 migration cleanup — drop stale triggers/FTS before recreating
    _SQL_V2_MIGRATION_CLEANUP,
    # FTS5 must come after atomic_facts (content table) AND after cleanup
    _SQL_ATOMIC_FACTS_FTS,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_all_tables(conn: sqlite3.Connection) -> None:
    """Create every table, index, trigger, and FTS virtual table.

    Safe to call repeatedly — all statements use IF NOT EXISTS.
    Inserts the initial schema_version row on first run.

    Args:
        conn: An open SQLite connection. Caller manages commit.
    """
    _set_pragmas(conn)

    for ddl in _DDL_ORDERED:
        conn.executescript(ddl)

    # --- V3.2 schema extension (additive only) ---
    from superlocalmemory.storage.schema_v32 import V32_DDL
    for ddl in V32_DDL:
        conn.executescript(ddl)

    # Seed schema version on first run.
    existing = conn.execute(
        "SELECT COUNT(*) AS n FROM schema_version"
    ).fetchone()
    if existing[0] == 0:
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES (?, ?)",
            (SCHEMA_VERSION, "Initial schema — zero dead tables"),
        )

    # Ensure the 'default' profile exists.
    conn.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        ("default", "Default Profile"),
    )


def drop_all_tables(conn: sqlite3.Connection) -> None:
    """Drop every table and FTS virtual table. For testing only.

    Args:
        conn: An open SQLite connection. Caller manages commit.
    """
    # V32 tables first (they may FK to base tables)
    from superlocalmemory.storage.schema_v32 import V32_ROLLBACK
    for sql in V32_ROLLBACK:
        conn.execute(sql)

    # FTS + triggers first (depend on atomic_facts).
    for fts in _FTS_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {fts}")
    for trigger in (
        "atomic_facts_fts_insert",
        "atomic_facts_fts_delete",
        "atomic_facts_fts_update",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")

    # Drop regular tables in reverse FK order.
    for table in reversed(_TABLES):
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def get_table_names() -> tuple[str, ...]:
    """Return all regular table names (excludes FTS virtual tables)."""
    return _TABLES


def get_fts_table_names() -> tuple[str, ...]:
    """Return all FTS virtual table names."""
    return _FTS_TABLES
