# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""SuperLocalMemory V3.2 -- Schema Extensions (Associative Memory).

Phase 0.5 creates this file with EMPTY V32_DDL.
Each implementing phase OWNS its DDL and appends here:
  - Phase 1 adds: fact_access_log, fact_embeddings (vec0), embedding_metadata
  - Phase 2 adds: fact_context
  - Phase 3 adds: association_edges, activation_cache, fact_importance
  - Phase 4 adds: fact_temporal_validity
  - Phase 5 adds: core_memory_blocks

Design rules:
  - profile_id + FK CASCADE on every table (Rule 01)
  - CREATE IF NOT EXISTS for idempotency (Rule 02)
  - Rollback SQL in V32_ROLLBACK (Rule 20)
  - Never ALTER existing tables (Rule 02)

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Table names (for drop_all parity -- Rule 20)
# All 9 names listed upfront. DDL added by each phase.
# ---------------------------------------------------------------------------

V32_TABLES: Final[tuple[str, ...]] = (
    "fact_retention",
    "polar_embeddings",
    "embedding_quantization_metadata",
    "fact_access_log",
    "fact_embeddings",
    "embedding_metadata",
    "fact_context",
    "association_edges",
    "activation_cache",
    "fact_importance",
    "fact_temporal_validity",
    "core_memory_blocks",
    "ccq_consolidated_blocks",
    "ccq_audit_log",
    "soft_prompt_templates",
)

# ---------------------------------------------------------------------------
# DDL Statements -- EMPTY at Phase 0.5. Each phase appends its DDL.
# Phase LLDs are AUTHORITATIVE for column names and constraints.
# ---------------------------------------------------------------------------

V32_DDL: list[str] = [
    # --- Phase A: Forgetting Brain (fact_retention) ---
    """
    CREATE TABLE IF NOT EXISTS fact_retention (
        fact_id          TEXT PRIMARY KEY,
        profile_id       TEXT NOT NULL,
        retention_score  REAL NOT NULL DEFAULT 1.0,
        memory_strength  REAL NOT NULL DEFAULT 1.0,
        access_count     INTEGER NOT NULL DEFAULT 0,
        last_accessed_at TEXT,
        last_computed_at TEXT NOT NULL DEFAULT (datetime('now')),
        lifecycle_zone   TEXT NOT NULL DEFAULT 'active'
                         CHECK (lifecycle_zone IN ('active', 'warm', 'cold', 'archive', 'forgotten')),

        FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id) ON DELETE CASCADE,
        FOREIGN KEY (profile_id) REFERENCES profiles (profile_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_retention_profile
        ON fact_retention (profile_id, lifecycle_zone);
    CREATE INDEX IF NOT EXISTS idx_retention_score
        ON fact_retention (profile_id, retention_score DESC);
    """,
    # --- Phase B: PolarQuant Embedding Quantization ---
    """
    CREATE TABLE IF NOT EXISTS polar_embeddings (
        fact_id     TEXT PRIMARY KEY,
        profile_id  TEXT NOT NULL,
        radius      REAL NOT NULL,
        angle_indices BLOB NOT NULL,
        qjl_bits    BLOB,
        bit_width   INTEGER NOT NULL DEFAULT 4,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),

        FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id) ON DELETE CASCADE,
        FOREIGN KEY (profile_id) REFERENCES profiles (profile_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_polar_profile
        ON polar_embeddings (profile_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_quantization_metadata (
        fact_id               TEXT PRIMARY KEY,
        profile_id            TEXT NOT NULL,
        quantization_level    TEXT NOT NULL DEFAULT 'float32'
                              CHECK (quantization_level IN ('float32', 'int8', 'polar4', 'polar2', 'deleted')),
        bit_width             INTEGER NOT NULL DEFAULT 32,
        compressed_size_bytes INTEGER,
        created_at            TEXT NOT NULL DEFAULT (datetime('now')),

        FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id) ON DELETE CASCADE,
        FOREIGN KEY (profile_id) REFERENCES profiles (profile_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_eqm_profile_level
        ON embedding_quantization_metadata (profile_id, quantization_level);
    """,
    # --- Phase 1: Vector Foundation ---
    """
    CREATE TABLE IF NOT EXISTS fact_access_log (
        log_id      TEXT PRIMARY KEY,
        fact_id     TEXT NOT NULL,
        profile_id  TEXT NOT NULL DEFAULT 'default',
        accessed_at TEXT NOT NULL DEFAULT (datetime('now')),
        access_type TEXT NOT NULL DEFAULT 'recall'
                         CHECK (access_type IN ('recall', 'auto_invoke', 'search', 'consolidation')),
        session_id  TEXT NOT NULL DEFAULT '',

        FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id)
            ON DELETE CASCADE,
        FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
            ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_access_log_fact
        ON fact_access_log (fact_id, accessed_at DESC);
    CREATE INDEX IF NOT EXISTS idx_access_log_profile
        ON fact_access_log (profile_id, accessed_at DESC);
    CREATE INDEX IF NOT EXISTS idx_access_log_profile_fact
        ON fact_access_log (profile_id, fact_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_metadata (
        vec_rowid   INTEGER PRIMARY KEY,
        fact_id     TEXT NOT NULL UNIQUE,
        profile_id  TEXT NOT NULL DEFAULT 'default',
        model_name  TEXT NOT NULL DEFAULT '',
        dimension   INTEGER NOT NULL DEFAULT 768,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),

        FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id)
            ON DELETE CASCADE,
        FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
            ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_embmeta_fact
        ON embedding_metadata (fact_id);
    CREATE INDEX IF NOT EXISTS idx_embmeta_profile
        ON embedding_metadata (profile_id);
    """,
    # --- Phase 2: Auto-Invoke Engine ---
    """
    CREATE TABLE IF NOT EXISTS fact_context (
        fact_id     TEXT PRIMARY KEY,
        profile_id  TEXT NOT NULL,
        contextual_description TEXT NOT NULL,
        keywords    TEXT,
        generated_by TEXT NOT NULL DEFAULT 'rules',
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),

        FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id)
            ON DELETE CASCADE,
        FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
            ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_fact_context_profile
        ON fact_context (profile_id);
    """,
    # Phase 4 will add: fact_temporal_validity
    # Phase 5 will add: core_memory_blocks

    # --- Phase 3: Association Graph ---
    """
    CREATE TABLE IF NOT EXISTS association_edges (
        edge_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        source_fact_id TEXT NOT NULL,
        target_fact_id TEXT NOT NULL,
        association_type TEXT NOT NULL CHECK(association_type IN (
            'auto_link', 'hebbian', 'consolidation', 'user_defined'
        )),
        weight REAL NOT NULL DEFAULT 0.5,
        co_access_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        last_strengthened TEXT,
        FOREIGN KEY (source_fact_id) REFERENCES atomic_facts(fact_id) ON DELETE CASCADE,
        FOREIGN KEY (target_fact_id) REFERENCES atomic_facts(fact_id) ON DELETE CASCADE,
        FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_assoc_source
        ON association_edges(source_fact_id, profile_id);
    CREATE INDEX IF NOT EXISTS idx_assoc_target
        ON association_edges(target_fact_id, profile_id);
    CREATE INDEX IF NOT EXISTS idx_assoc_profile
        ON association_edges(profile_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_assoc_unique_pair
        ON association_edges(profile_id, source_fact_id, target_fact_id, association_type);
    """,
    """
    CREATE TABLE IF NOT EXISTS activation_cache (
        cache_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        query_hash TEXT NOT NULL,
        node_id TEXT NOT NULL,
        activation_value REAL NOT NULL,
        iteration INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        expires_at TEXT NOT NULL DEFAULT (datetime('now', '+1 hour')),
        FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_actcache_profile_query
        ON activation_cache(profile_id, query_hash);
    CREATE INDEX IF NOT EXISTS idx_actcache_expires
        ON activation_cache(expires_at);
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_importance (
        fact_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        pagerank_score REAL NOT NULL DEFAULT 0.0,
        community_id INTEGER,
        degree_centrality REAL DEFAULT 0.0,
        computed_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (fact_id) REFERENCES atomic_facts(fact_id) ON DELETE CASCADE,
        FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_fact_importance_profile
        ON fact_importance(profile_id);
    CREATE INDEX IF NOT EXISTS idx_fact_importance_pagerank
        ON fact_importance(profile_id, pagerank_score DESC);
    CREATE INDEX IF NOT EXISTS idx_fact_importance_community
        ON fact_importance(profile_id, community_id);
    """,
    # --- Phase 5: Core Memory Blocks (Sleep-Time Consolidation) ---
    """
    CREATE TABLE IF NOT EXISTS core_memory_blocks (
        block_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        block_type TEXT NOT NULL CHECK(block_type IN (
            'user_profile', 'project_context', 'behavioral_patterns',
            'active_decisions', 'learned_preferences', 'custom'
        )),
        content TEXT NOT NULL,
        source_fact_ids TEXT NOT NULL DEFAULT '[]',
        char_count INTEGER NOT NULL DEFAULT 0,
        version INTEGER NOT NULL DEFAULT 1,
        compiled_by TEXT NOT NULL DEFAULT 'rules',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_core_blocks_profile
        ON core_memory_blocks(profile_id);
    CREATE INDEX IF NOT EXISTS idx_core_blocks_type
        ON core_memory_blocks(profile_id, block_type);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_core_blocks_unique
        ON core_memory_blocks(profile_id, block_type);
    """,
    # --- Phase E: CCQ Consolidated Blocks (dedicated table, many-per-profile) ---
    """
    CREATE TABLE IF NOT EXISTS ccq_consolidated_blocks (
        block_id        TEXT PRIMARY KEY,
        profile_id      TEXT NOT NULL,
        content         TEXT NOT NULL,
        source_fact_ids TEXT NOT NULL DEFAULT '[]',
        gist_embedding_rowid INTEGER,
        char_count      INTEGER NOT NULL DEFAULT 0,
        compiled_by     TEXT NOT NULL DEFAULT 'ccq',
        cluster_id      TEXT NOT NULL,
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),

        FOREIGN KEY (profile_id) REFERENCES profiles (profile_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_ccq_blocks_profile
        ON ccq_consolidated_blocks (profile_id);
    CREATE INDEX IF NOT EXISTS idx_ccq_blocks_cluster
        ON ccq_consolidated_blocks (cluster_id);
    """,
    # --- Phase E: CCQ Audit Log ---
    """
    CREATE TABLE IF NOT EXISTS ccq_audit_log (
        audit_id        TEXT PRIMARY KEY,
        profile_id      TEXT NOT NULL,
        cluster_id      TEXT NOT NULL,
        block_id        TEXT NOT NULL,
        fact_ids        TEXT NOT NULL DEFAULT '[]',
        fact_count      INTEGER NOT NULL DEFAULT 0,
        gist_text       TEXT NOT NULL,
        extraction_mode TEXT NOT NULL DEFAULT 'rules'
                        CHECK (extraction_mode IN ('rules', 'llm')),
        bytes_before    INTEGER NOT NULL DEFAULT 0,
        bytes_after     INTEGER NOT NULL DEFAULT 0,
        compression_ratio REAL NOT NULL DEFAULT 0.0,
        shared_entities TEXT NOT NULL DEFAULT '[]',
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),

        FOREIGN KEY (profile_id) REFERENCES profiles (profile_id) ON DELETE CASCADE,
        FOREIGN KEY (block_id) REFERENCES ccq_consolidated_blocks (block_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_ccq_audit_profile
        ON ccq_audit_log (profile_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_ccq_audit_block
        ON ccq_audit_log (block_id);
    """,
    # --- Phase 4: Temporal Intelligence ---
    """
    CREATE TABLE IF NOT EXISTS fact_temporal_validity (
        fact_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,

        -- Event Time (when was this fact true in the real world?)
        valid_from TEXT,
        valid_until TEXT,

        -- Transaction Time (when did the system learn/invalidate?)
        system_created_at TEXT NOT NULL DEFAULT (datetime('now')),
        system_expired_at TEXT,

        -- Invalidation metadata
        invalidated_by TEXT,
        invalidation_reason TEXT,

        FOREIGN KEY (fact_id) REFERENCES atomic_facts(fact_id) ON DELETE CASCADE,
        FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_temporal_valid
        ON fact_temporal_validity(profile_id, valid_until);
    CREATE INDEX IF NOT EXISTS idx_temporal_system_expired
        ON fact_temporal_validity(profile_id, system_expired_at);
    CREATE INDEX IF NOT EXISTS idx_temporal_invalidated_by
        ON fact_temporal_validity(invalidated_by);
    """,
    # --- Phase F: The Learning Brain (Memory Parameterization) ---
    """
    CREATE TABLE IF NOT EXISTS soft_prompt_templates (
        prompt_id       TEXT PRIMARY KEY,
        profile_id      TEXT NOT NULL,
        category        TEXT NOT NULL CHECK (category IN (
            'identity', 'tech_preference', 'communication_style',
            'workflow_pattern', 'project_context', 'decision_history',
            'avoidance', 'custom'
        )),
        content         TEXT NOT NULL,
        source_pattern_ids TEXT NOT NULL DEFAULT '[]',
        confidence      REAL NOT NULL DEFAULT 0.0,
        effectiveness   REAL NOT NULL DEFAULT 0.5,
        token_count     INTEGER NOT NULL DEFAULT 0,
        retention_score REAL NOT NULL DEFAULT 1.0,
        active          INTEGER NOT NULL DEFAULT 1,
        version         INTEGER NOT NULL DEFAULT 1,
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at      TEXT NOT NULL DEFAULT (datetime('now')),

        FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_soft_prompt_profile
        ON soft_prompt_templates(profile_id, active);
    CREATE INDEX IF NOT EXISTS idx_soft_prompt_category
        ON soft_prompt_templates(profile_id, category);
    CREATE INDEX IF NOT EXISTS idx_soft_prompts_profile_active_category
        ON soft_prompt_templates(profile_id, active, category, prompt_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_soft_prompt_unique_cat
        ON soft_prompt_templates(profile_id, category)
        WHERE active = 1;
    """,
]

# ---------------------------------------------------------------------------
# Rollback DDL (reverse FK order -- Rule 20)
# ---------------------------------------------------------------------------

V32_ROLLBACK: Final[tuple[str, ...]] = tuple(
    f"DROP TABLE IF EXISTS {table}" for table in reversed(V32_TABLES)
)


def rollback_v32(conn) -> None:
    """Drop all V32 tables in reverse FK order. For testing/rollback only."""
    for sql in V32_ROLLBACK:
        conn.execute(sql)
