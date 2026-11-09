# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
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

import logging
import sqlite3
from typing import Final

from superlocalmemory.storage import scene_member_triggers as _scene_member_triggers

logger = logging.getLogger(__name__)

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
    "fact_entity_associations",
    "fact_entity_association_repair_state",
    "entity_aliases",
    "entity_profiles",
    "memory_scenes",
    "scene_fact_members",
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
    "entity_communities",
    "community_summaries",
    "consolidated_summaries",
    "persona_summary",
    "projection_outbox",
)

_FTS_TABLES: Final[tuple[str, ...]] = (
    "atomic_facts_fts",
    "fact_expansion_fts",
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
    scope        TEXT NOT NULL DEFAULT 'personal',
    shared_with  TEXT,
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
    ON memories (created_at);"""


# ---------------------------------------------------------------------------
# Atomic facts — THE primary retrieval unit
# ---------------------------------------------------------------------------

_SQL_ATOMIC_FACTS: Final[str] = """
CREATE TABLE IF NOT EXISTS atomic_facts (
    fact_id            TEXT PRIMARY KEY,
    memory_id          TEXT NOT NULL,
    profile_id         TEXT NOT NULL DEFAULT 'default',
    scope              TEXT NOT NULL DEFAULT 'personal',
    shared_with        TEXT,
    content            TEXT NOT NULL,
    fact_type          TEXT NOT NULL DEFAULT 'semantic'
                            CHECK (fact_type IN (
                                'episodic', 'semantic', 'opinion', 'prospective'
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
    -- Core-memory injection priority (M015 on upgraded databases)
    pinned             INTEGER NOT NULL DEFAULT 0,

    -- Source tracing
    source_turn_ids_json TEXT NOT NULL DEFAULT '[]',
    session_id           TEXT NOT NULL DEFAULT '',

    -- Embeddings (JSON arrays — TEXT for simplicity and portability)
    embedding          TEXT,
    fisher_mean        TEXT,
    fisher_variance    TEXT,
    -- Tracks how many accesses had Fisher applied; delta-based update in maintenance
    fisher_last_applied_access INTEGER NOT NULL DEFAULT 0,

    -- Lifecycle
    lifecycle          TEXT NOT NULL DEFAULT 'active'
                            CHECK (lifecycle IN (
                                'active', 'warm', 'cold', 'archived'
                            )),
    -- Withheld from retrieval without being destroyed. Set by repair, never
    -- by a normal write; enforced in exactly one place,
    -- DatabaseManager.get_facts_by_ids, which every channel's candidates are
    -- re-authorised through and which the engine hydrates from.
    quarantined        INTEGER NOT NULL DEFAULT 0,
    -- Stored but never an answer (operational stubs such as "session ended");
    -- listed only on request. Same enforcement point as quarantined (M060).
    non_answer         INTEGER DEFAULT 0,
    -- The user's own decision (0 = "this is an answer"); NULL = none. It wins
    -- over the automatic stub rule forever (WP-19 review M1).
    non_answer_user    INTEGER,
    -- Memory kind (M052 on upgraded stores). NULL = untyped; no CHECK by design.
    memory_kind            TEXT,
    memory_kind_source     TEXT,
    memory_kind_confidence REAL,
    memory_kind_recipe     TEXT,
    memory_kind_at         TEXT,
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
CREATE INDEX IF NOT EXISTS idx_facts_pinned
    ON atomic_facts (profile_id, pinned);
CREATE INDEX IF NOT EXISTS idx_facts_session
    ON atomic_facts (profile_id, session_id);
CREATE INDEX IF NOT EXISTS idx_facts_referenced_date
    ON atomic_facts (profile_id, referenced_date)
    WHERE referenced_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_facts_interval
    ON atomic_facts (profile_id, interval_start, interval_end)
    WHERE interval_start IS NOT NULL;"""


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
CREATE TRIGGER IF NOT EXISTS atomic_facts_fts_insert
    AFTER INSERT ON atomic_facts
BEGIN
    INSERT INTO atomic_facts_fts (rowid, fact_id, content)
        VALUES (NEW.rowid, NEW.fact_id, NEW.content);
END;

-- DELETE trigger
CREATE TRIGGER IF NOT EXISTS atomic_facts_fts_delete
    AFTER DELETE ON atomic_facts
BEGIN
    INSERT INTO atomic_facts_fts (atomic_facts_fts, rowid, fact_id, content)
        VALUES ('delete', OLD.rowid, OLD.fact_id, OLD.content);
END;

-- UPDATE trigger
CREATE TRIGGER IF NOT EXISTS atomic_facts_fts_update
    AFTER UPDATE OF content ON atomic_facts
BEGIN
    INSERT INTO atomic_facts_fts (atomic_facts_fts, rowid, fact_id, content)
        VALUES ('delete', OLD.rowid, OLD.fact_id, OLD.content);
    INSERT INTO atomic_facts_fts (rowid, fact_id, content)
        VALUES (NEW.rowid, NEW.fact_id, NEW.content);
END;
"""


# ---------------------------------------------------------------------------
# Fact expansion FTS (Phase 4, T3b — fact-augmented key expansion)
# ---------------------------------------------------------------------------
# Standalone (NOT external-content) FTS5 holding per-fact alternate keys
# (entity aliases + paraphrases). Kept separate from atomic_facts_fts so the
# proven content index and its triggers are never touched. Populated by
# core.key_expander on store (and backfilled), queried as an additive UNION in
# the BM25 channel. fact_id is stored so results can be scope-JOINed to
# atomic_facts.
_SQL_FACT_EXPANSION_FTS: Final[str] = """
CREATE VIRTUAL TABLE IF NOT EXISTS fact_expansion_fts
    USING fts5(
        fact_id UNINDEXED,
        alt_keys
    );
"""


# ---------------------------------------------------------------------------
# Canonical entities
# ---------------------------------------------------------------------------

_SQL_CANONICAL_ENTITIES: Final[str] = """
CREATE TABLE IF NOT EXISTS canonical_entities (
    entity_id       TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL DEFAULT 'default',
    scope           TEXT NOT NULL DEFAULT 'personal',
    shared_with     TEXT,
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
CREATE INDEX IF NOT EXISTS idx_entities_profile_fact_count
    ON canonical_entities (profile_id, fact_count DESC);
CREATE INDEX IF NOT EXISTS idx_entities_profile_type_fact_count
    ON canonical_entities (
        profile_id, entity_type COLLATE NOCASE, fact_count DESC
    );"""


# ---------------------------------------------------------------------------
# Normalized fact/entity associations (also the ingestion effect ledger)
# ---------------------------------------------------------------------------

_SQL_FACT_ENTITY_ASSOCIATIONS: Final[str] = """
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
INSERT OR IGNORE INTO fact_entity_association_repair_state
    (repair_key, state, target_fact_rowid, updated_at)
VALUES ('historical-backfill', 'pending', -1, '');
"""


# ---------------------------------------------------------------------------
# Entity aliases
# ---------------------------------------------------------------------------

_SQL_ENTITY_ALIASES: Final[str] = """
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_id    TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL DEFAULT 'default',
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
-- NOTE: the (profile_id, entity_id) index is created by migration M022, NOT
-- here. On an upgrading DB this DDL runs at engine init BEFORE the deferred
-- M022 adds the profile_id column, so referencing it here would fail engine
-- init with "no such column: profile_id".
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
CREATE UNIQUE INDEX IF NOT EXISTS uq_scenes_profile_scene
    ON memory_scenes (profile_id, scene_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_facts_profile_fact
    ON atomic_facts (profile_id, fact_id);
"""


# ---------------------------------------------------------------------------
# Normalized scene/fact membership projection (bounded scene assignment)
# ---------------------------------------------------------------------------

_SQL_SCENE_FACT_MEMBERS: Final[str] = """
CREATE TABLE IF NOT EXISTS scene_fact_members (
    profile_id TEXT NOT NULL,
    scene_id   TEXT NOT NULL,
    fact_id    TEXT NOT NULL,
    position   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scene_id, fact_id),
    FOREIGN KEY (profile_id, scene_id)
        REFERENCES memory_scenes(profile_id, scene_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id, fact_id)
        REFERENCES atomic_facts(profile_id, fact_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scene_fact_members_lookup
    ON scene_fact_members (profile_id, fact_id, scene_id);
CREATE INDEX IF NOT EXISTS idx_scene_fact_members_order
    ON scene_fact_members (scene_id, position);
""" + _scene_member_triggers.DDL


# ---------------------------------------------------------------------------
# Temporal events (per-entity timeline entries)
# ---------------------------------------------------------------------------

_SQL_TEMPORAL_EVENTS: Final[str] = """
CREATE TABLE IF NOT EXISTS temporal_events (
    event_id         TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL DEFAULT 'default',
    scope            TEXT NOT NULL DEFAULT 'personal',
    shared_with      TEXT,
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
    WHERE referenced_date IS NOT NULL;"""


# ---------------------------------------------------------------------------
# Graph edges (knowledge graph between facts/entities)
# ---------------------------------------------------------------------------

_SQL_GRAPH_EDGES: Final[str] = """
CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id     TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL DEFAULT 'default',
    scope       TEXT NOT NULL DEFAULT 'personal',
    shared_with TEXT,
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
CREATE INDEX IF NOT EXISTS idx_edges_exists_check
    ON graph_edges (profile_id, source_id, target_id, edge_type);"""


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

# Wave Q: entity-community backbone (additive; safe on existing DBs).
# One row per (profile, entity) mapping to the community it belongs to,
# computed by Louvain over the entity co-occurrence graph in the background.
# Shared spine for Q2 community summaries and Q3 progressive abstraction.
_SQL_ENTITY_COMMUNITIES: Final[str] = """
CREATE TABLE IF NOT EXISTS entity_communities (
    profile_id  TEXT    NOT NULL,
    entity_id   TEXT    NOT NULL,
    community_id INTEGER NOT NULL,
    computed_at TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (profile_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_comm_profile
    ON entity_communities(profile_id);
CREATE INDEX IF NOT EXISTS idx_entity_comm_cid
    ON entity_communities(profile_id, community_id);
"""

# Wave Q2: one synthesized report per entity community (additive; safe on
# existing DBs). Generated in the background after entity_communities; the
# summary excludes superseded facts. member_fact_ids enables drill-down.
_SQL_COMMUNITY_SUMMARIES: Final[str] = """
CREATE TABLE IF NOT EXISTS community_summaries (
    profile_id      TEXT    NOT NULL,
    community_id    INTEGER NOT NULL,
    summary         TEXT    NOT NULL DEFAULT '',
    keywords        TEXT    NOT NULL DEFAULT '',
    entity_ids_json TEXT    NOT NULL DEFAULT '[]',
    fact_ids_json   TEXT    NOT NULL DEFAULT '[]',
    fact_count      INTEGER NOT NULL DEFAULT 0,
    computed_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (profile_id, community_id)
);
CREATE INDEX IF NOT EXISTS idx_comm_summ_profile
    ON community_summaries(profile_id);
"""

# Display-only consolidated summaries.
#
# A summary of a cluster of facts is a VIEW of memory, not a memory. Between
# v3.6.4 and 4.0.9 the fact consolidator wrote its summaries straight into
# atomic_facts with a raw INSERT, which put model-authored prose into the
# retrieval corpus alongside the user's own words — where it out-ranked them,
# because those rows carried every entity in their cluster and so had more
# entity links than any real fact.
#
# This table restores the boundary. community_summaries is the precedent to
# read it by: written by one owner, read only after retrieval has finished, and
# named by no channel. tests/test_retrieval/test_summaries_stay_out_of_recall.py
# fails if a retrieval module so much as mentions it.
#
# source_earliest / source_latest are the honest dates for a derived row. A
# summary has no observation_date of its own — it was never observed — but the
# span of what it summarises is real, and it is what lets the dashboard say
# which stretch of work a summary covers.
CONSOLIDATED_SUMMARIES_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS consolidated_summaries (
    summary_id       TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL,
    entity_id        TEXT NOT NULL DEFAULT '',
    entity_name      TEXT NOT NULL DEFAULT '',
    content          TEXT NOT NULL,
    source_fact_ids  TEXT NOT NULL DEFAULT '[]',
    source_count     INTEGER NOT NULL DEFAULT 0,
    char_count       INTEGER NOT NULL DEFAULT 0,
    generated_by     TEXT NOT NULL DEFAULT 'extractive'
                        CHECK (generated_by IN (
                            'extractive', 'ollama', 'cloud', 'migrated'
                        )),
    scope            TEXT NOT NULL DEFAULT 'personal',
    shared_with      TEXT,
    source_earliest  TEXT,
    source_latest    TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (profile_id, entity_id, content)
);
CREATE INDEX IF NOT EXISTS idx_consolidated_summaries_profile
    ON consolidated_summaries(profile_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_consolidated_summaries_entity
    ON consolidated_summaries(profile_id, entity_id);
"""

# Wave Q3: progressive-abstraction top tier — one persona roll-up per profile
# consuming the top community summaries (additive; safe on existing DBs).
# Recall-gated (never auto-injected into hot recall) and size-bounded to avoid
# the V3.4.40 summary-pollution regression. Drill-down: community_ids_json →
# communities → their member facts.
_SQL_PERSONA_SUMMARY: Final[str] = """
CREATE TABLE IF NOT EXISTS persona_summary (
    profile_id         TEXT PRIMARY KEY,
    summary            TEXT NOT NULL DEFAULT '',
    keywords           TEXT NOT NULL DEFAULT '',
    community_ids_json TEXT NOT NULL DEFAULT '[]',
    computed_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# ---------------------------------------------------------------------------
# Ordered DDL list (tables before FTS, respects FK order)
# ---------------------------------------------------------------------------

#: Imported rather than restated so the table has exactly one definition; the
#: module that owns the queue owns its shape.
from superlocalmemory.storage.projection_outbox import DDL as _PROJECTION_OUTBOX_DDL  # noqa: E402

_DDL_ORDERED: Final[tuple[str, ...]] = (
    _SQL_SCHEMA_VERSION,
    _SQL_PROFILES,
    _SQL_MEMORIES,
    _SQL_ATOMIC_FACTS,
    _SQL_CANONICAL_ENTITIES,
    _SQL_FACT_ENTITY_ASSOCIATIONS,
    _SQL_ENTITY_ALIASES,
    _SQL_ENTITY_PROFILES,
    _SQL_MEMORY_SCENES,
    _SQL_SCENE_FACT_MEMBERS,
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
    # T3b: standalone expansion FTS (additive; safe on existing DBs)
    _SQL_FACT_EXPANSION_FTS,
    # Wave Q: entity-community backbone (additive; safe on existing DBs)
    _SQL_ENTITY_COMMUNITIES,
    # Wave Q2: community summaries (additive; safe on existing DBs)
    _SQL_COMMUNITY_SUMMARIES,
    CONSOLIDATED_SUMMARIES_DDL,
    # Wave Q3: persona roll-up tier (additive; safe on existing DBs)
    _SQL_PERSONA_SUMMARY,
    # The queue that makes a graph/vector projection write survive a crash.
    # Created here, at every engine init, rather than only by its migration:
    # if this table is missing, facts stop being projected and the only symptom
    # is a memory that cannot be recalled. That invariant must not be
    # contingent on a migration pass having succeeded.
    _PROJECTION_OUTBOX_DDL,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: (table, column, column definition). Applied with ALTER TABLE ADD COLUMN,
#: which SQLite offers no IF NOT EXISTS form of, so presence is checked first.
_ADDITIVE_COLUMNS: Final[tuple[tuple[str, str, str], ...]] = (
    ("atomic_facts", "quarantined", "INTEGER NOT NULL DEFAULT 0"),
    # ``pinned`` arrives with M015, which is a DEFERRED migration -- it runs
    # after the engine is up. But the DDL below indexes it
    # (``idx_facts_pinned``), and that DDL runs during engine start. On a store
    # old enough to predate M015 the index therefore raised "no such column:
    # pinned" before anything could add it, and every start failed the same way:
    # the deferred pass could not run because the engine could not start, and
    # the engine could not start because the deferred pass had not run.
    # ``slm db migrate`` did not break the loop either -- it reports Failed=0
    # and skips deferred migrations by definition.
    # The column has to exist before its own index either way, so it belongs
    # here, where a store gets it at start regardless of migration state.
    ("atomic_facts", "pinned", "INTEGER NOT NULL DEFAULT 0"),
    # WP-19: keeping operational stubs out of answers must not wait for the
    # deferred M060 pass, and the visibility indexes cover it from the start.
    ("atomic_facts", "non_answer", "INTEGER DEFAULT 0"),
    ("atomic_facts", "non_answer_user", "INTEGER"),
)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Add columns that upgraded databases predate. Idempotent.

    A missing table is not an error: this runs inside create_all_tables, so the
    table is created moments earlier in the same call, and a database old enough
    to lack it entirely has nothing to alter.
    """
    for table, column, definition in _ADDITIVE_COLUMNS:
        try:
            present = any(
                row[1] == column
                for row in conn.execute(f"PRAGMA table_info({table})")
            )
            if not present:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                )
        except sqlite3.Error as exc:
            # Never fatal. A store that cannot take the column keeps working;
            # get_facts_by_ids checks for the column before filtering on it.
            logger.warning(
                "additive column %s.%s not applied: %s", table, column, exc,
            )


#: M052's index, created only when the column exists: an index on a missing
#: column would fail engine start on a store M052 has not reached (the ``pinned`` trap).
def _create_memory_kind_index(conn: sqlite3.Connection) -> None:
    try:
        if any(r[1] == "memory_kind" for r in conn.execute("PRAGMA table_info(atomic_facts)")):
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_memory_kind ON atomic_facts "
                         "(profile_id, memory_kind) WHERE memory_kind IS NOT NULL")
    except sqlite3.Error as exc:
        logger.warning("memory-kind index not created: %s", exc)


def create_all_tables(conn: sqlite3.Connection) -> None:
    """Create every table, index, trigger, and FTS virtual table.

    Safe to call repeatedly — all statements use IF NOT EXISTS.
    Inserts the initial schema_version row on first run.

    Args:
        conn: An open SQLite connection. Caller manages commit.
    """
    _set_pragmas(conn)

    # Before the DDL, not only after it. The DDL below indexes columns that an
    # upgraded store may not have yet, and an index on a column that does not
    # exist is a hard error, not a skipped statement -- so the whole of
    # create_all_tables would raise and the engine would never start.
    #
    # On a fresh database this pass does nothing: the tables do not exist yet,
    # which the helper treats as "nothing to alter". On an upgraded one the
    # tables are already there and this is exactly where the columns are owed.
    # It runs again at the end for tables created during this call.
    _add_missing_columns(conn)

    for ddl in _DDL_ORDERED:
        conn.executescript(ddl)

    # --- V3.2 schema extension (additive only) ---
    from superlocalmemory.storage.schema_v32 import V32_DDL
    for ddl in V32_DDL:
        conn.executescript(ddl)

    # The graph's change counter, part of the activation-cache key. After the
    # V32 DDL: its triggers sit on association_edges and empty activation_cache.
    from superlocalmemory.storage import fact_search_changes, graph_generation
    conn.executescript(graph_generation.DDL + fact_search_changes.ddl(conn))
    from superlocalmemory.core.overtaken_cases import ensure_table
    ensure_table(conn)

    # Additive columns on tables that predate them.
    #
    # CREATE TABLE IF NOT EXISTS cannot add a column to a table that already
    # exists, so an upgraded database gets the column here rather than only from
    # a migration. Doing it at every engine init makes the invariant "if
    # atomic_facts exists then quarantined exists" hold even when the migration
    # pass failed or was never reached — which matters because withholding a
    # poisoned row from retrieval must not be contingent on a migration having
    # succeeded.
    _add_missing_columns(conn)
    _create_memory_kind_index(conn)
    from superlocalmemory.storage import recency_index, visibility_index
    from superlocalmemory.storage.fts_residue import enable_quietly
    visibility_index.ensure(conn)
    recency_index.ensure(conn)
    _scene_member_triggers.ensure(conn)  # 4.2.4 stores: a scene save scanned every fact
    enable_quietly(conn)  # a deleted memory's words leave the keyword index at once

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
    # The graph counter's triggers reference V32 tables: drop them first.
    from superlocalmemory.storage import fact_search_changes, graph_generation
    for trigger in (*graph_generation.trigger_names(), *fact_search_changes.trigger_names()):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    for table in (graph_generation.TABLE, fact_search_changes.TABLE, "correction_cases_overtaken"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")

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
        "trg_scene_fact_members_insert",
        "trg_scene_fact_members_update",
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
