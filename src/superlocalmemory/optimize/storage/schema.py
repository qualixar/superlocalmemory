"""DDL for llmcache.db — the Optimize module's dedicated SQLite database.

ISOLATION GUARANTEE: This schema MUST NEVER reference memory.db tables.
No FKs to profiles, atomic_facts, memories, or any SLM core table.
"""

from __future__ import annotations

import sqlite3
from typing import Final

CACHE_SCHEMA_VERSION: Final[int] = 1

LLMCACHE_DB_FILENAME: Final[str] = "llmcache.db"


_DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS llmcache_schema_version (
        version     INTEGER NOT NULL,
        applied_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        description TEXT    NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llmcache_entries (
        entry_id        TEXT    PRIMARY KEY,
        cache_key       TEXT    NOT NULL,
        tenant_id       TEXT    NOT NULL DEFAULT 'default',
        model           TEXT    NOT NULL DEFAULT '',
        provider        TEXT    NOT NULL DEFAULT '',
        value_blob      BLOB    NOT NULL,
        compressed      INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        last_hit_at     TEXT,
        ttl_expires     REAL,
        hit_count       INTEGER NOT NULL DEFAULT 0,
        byte_size       INTEGER NOT NULL DEFAULT 0,
        tag_json        TEXT    NOT NULL DEFAULT '[]',
        cache_tier      TEXT    NOT NULL DEFAULT 'exact'
                                CHECK (cache_tier IN ('exact', 'semantic'))
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_llmcache_key_tenant ON llmcache_entries (cache_key, tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_llmcache_ttl ON llmcache_entries (ttl_expires) WHERE ttl_expires IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_llmcache_tenant ON llmcache_entries (tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_llmcache_provider ON llmcache_entries (tenant_id, provider)",
    "CREATE INDEX IF NOT EXISTS idx_llmcache_lru ON llmcache_entries (last_hit_at)",
    """
    CREATE TABLE IF NOT EXISTS llmcache_semantic_vectors (
        entry_id        TEXT    PRIMARY KEY,
        tenant_id       TEXT    NOT NULL DEFAULT 'default',
        vector_blob     BLOB    NOT NULL,
        vector_dim      INTEGER NOT NULL DEFAULT 768,
        model_name      TEXT    NOT NULL DEFAULT 'nomic-ai/nomic-embed-text-v1.5',
        context_fp      TEXT    NOT NULL DEFAULT '',
        created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_llmcache_vec_tenant ON llmcache_semantic_vectors (tenant_id)",
    """
    CREATE TABLE IF NOT EXISTS llmcache_ccr_originals (
        ccr_id          TEXT    PRIMARY KEY,
        tenant_id       TEXT    NOT NULL DEFAULT 'default',
        original_blob   BLOB    NOT NULL,
        compressed_hash TEXT    NOT NULL,
        byte_size_orig  INTEGER NOT NULL DEFAULT 0,
        byte_size_comp  INTEGER NOT NULL DEFAULT 0,
        model           TEXT    NOT NULL DEFAULT '',
        created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        ttl_expires     REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_llmcache_ccr_hash ON llmcache_ccr_originals (compressed_hash)",
    "CREATE INDEX IF NOT EXISTS idx_llmcache_ccr_tenant ON llmcache_ccr_originals (tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_llmcache_ccr_ttl ON llmcache_ccr_originals (ttl_expires) WHERE ttl_expires IS NOT NULL",
    """
    CREATE TABLE IF NOT EXISTS llmcache_tags (
        tag         TEXT NOT NULL,
        cache_key   TEXT NOT NULL,
        tenant_id   TEXT NOT NULL DEFAULT 'default',
        PRIMARY KEY (tag, cache_key, tenant_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_llmcache_tags_tag ON llmcache_tags (tag)",
    """
    CREATE TABLE IF NOT EXISTS llmcache_boundaries (
        entry_id        TEXT PRIMARY KEY,
        logistic_t      REAL NOT NULL DEFAULT 0.95,
        logistic_gamma  REAL NOT NULL DEFAULT 10.0,
        sample_count    INTEGER NOT NULL DEFAULT 0,
        updated_at      REAL NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llmcache_centroids (
        tenant_id       TEXT PRIMARY KEY,
        centroid_blob   BLOB NOT NULL,
        n               INTEGER NOT NULL DEFAULT 0,
        updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llmcache_metrics (
        id                          INTEGER PRIMARY KEY CHECK (id = 1),
        hits                        INTEGER NOT NULL DEFAULT 0,
        misses                      INTEGER NOT NULL DEFAULT 0,
        calls_skipped               INTEGER NOT NULL DEFAULT 0,
        tokens_saved_input          INTEGER NOT NULL DEFAULT 0,
        tokens_saved_output         INTEGER NOT NULL DEFAULT 0,
        tokens_saved_compress       INTEGER NOT NULL DEFAULT 0,
        evictions                   INTEGER NOT NULL DEFAULT 0,
        latency_overhead_ms_sum     REAL    NOT NULL DEFAULT 0,
        latency_samples             INTEGER NOT NULL DEFAULT 0,
        compress_runs               INTEGER NOT NULL DEFAULT 0,
        compress_bytes_original     INTEGER NOT NULL DEFAULT 0,
        compress_bytes_after        INTEGER NOT NULL DEFAULT 0,
        cache_size_bytes            INTEGER NOT NULL DEFAULT 0,
        cache_entry_count           INTEGER NOT NULL DEFAULT 0,
        updated_at                  REAL    NOT NULL DEFAULT 0
    )
    """,
)


def create_all_tables(conn: sqlite3.Connection) -> None:
    """Create all llmcache_* tables, indexes, and seed schema_version.

    Safe to call repeatedly — all DDL uses IF NOT EXISTS.
    Also seeds the single llmcache_metrics row (id=1) via INSERT OR IGNORE.
    C-10: adds context_fp column to existing DBs via ALTER TABLE migration.
    """
    for stmt in _DDL_STATEMENTS:
        conn.execute(stmt)
    conn.execute("INSERT OR IGNORE INTO llmcache_metrics(id) VALUES (1)")
    # C-10 migration: add context_fp column if missing (existing installs pre-v3.6.10)
    existing_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(llmcache_semantic_vectors)")
    }
    if "context_fp" not in existing_cols:
        conn.execute(
            "ALTER TABLE llmcache_semantic_vectors ADD COLUMN context_fp TEXT NOT NULL DEFAULT ''"
        )
    row = conn.execute(
        "SELECT 1 FROM llmcache_schema_version WHERE version = ?",
        (CACHE_SCHEMA_VERSION,),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO llmcache_schema_version (version, description) VALUES (?, ?)",
            (CACHE_SCHEMA_VERSION, ""),
        )


def drop_all_tables(conn: sqlite3.Connection) -> None:
    """Drop all llmcache_* tables. Testing only."""
    for name in get_table_names():
        conn.execute(f"DROP TABLE IF EXISTS {name}")


def get_table_names() -> tuple[str, ...]:
    """Return all table names in creation order."""
    return (
        "llmcache_schema_version",
        "llmcache_entries",
        "llmcache_semantic_vectors",
        "llmcache_ccr_originals",
        "llmcache_tags",
        "llmcache_boundaries",
        "llmcache_centroids",
        "llmcache_metrics",
    )


def assert_no_memory_db_tables(conn: sqlite3.Connection) -> None:
    """Assert that no memory.db tables are present. Raises RuntimeError on violation.

    Use this after opening a connection to confirm we are NOT talking to memory.db.
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {r[0] for r in rows}
    forbidden = {
        "memories", "atomic_facts", "profiles", "canonical_entities",
        "entity_aliases", "consolidation_log", "trust_scores", "bm25_tokens",
        "fact_retention", "core_memory_blocks", "ccq_consolidated_blocks",
    }
    found = forbidden & table_names
    if found:
        raise RuntimeError(
            f"ISOLATION VIOLATION: llmcache.db contains memory.db tables: {sorted(found)}. "
            "This means the wrong database file was opened. Aborting."
        )
