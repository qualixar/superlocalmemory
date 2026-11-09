# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4 — CodeGraph Module

"""DDL for the code_graph.db database.

Single source of truth for all CodeGraph tables.
No other module should contain CREATE TABLE statements.

Tables:
  1. graph_nodes       — Code entities (functions, classes, files, modules)
  2. graph_edges       — Relationships (calls, imports, inherits, contains, tested_by)
  3. graph_files       — File tracking for incremental updates
  4. graph_metadata    — Key-value store for graph-level config
  5. code_memory_links — Bridge table linking code nodes to SLM memory facts
  6. code_node_embeddings — vec0 virtual table for semantic search (optional)
  7. graph_nodes_fts   — FTS5 virtual table for text search
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL Statements (executed in order)
# ---------------------------------------------------------------------------

_DDL_STATEMENTS: tuple[str, ...] = (
    # ── Table 1: graph_nodes ──────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS graph_nodes (
        node_id         TEXT PRIMARY KEY,
        kind            TEXT NOT NULL CHECK (kind IN ('file', 'class', 'function', 'method', 'module')),
        name            TEXT NOT NULL,
        qualified_name  TEXT NOT NULL UNIQUE,
        file_path       TEXT NOT NULL,
        line_start      INTEGER NOT NULL DEFAULT 0,
        line_end        INTEGER NOT NULL DEFAULT 0,
        language        TEXT NOT NULL DEFAULT '',
        parent_name     TEXT,
        signature       TEXT,
        docstring       TEXT,
        is_test         INTEGER NOT NULL DEFAULT 0,
        content_hash    TEXT,
        community_id    INTEGER,
        extra_json      TEXT NOT NULL DEFAULT '{}',
        created_at      REAL NOT NULL,
        updated_at      REAL NOT NULL
    )
    """,

    # ── Table 2: graph_edges ──────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS graph_edges (
        edge_id         TEXT PRIMARY KEY,
        kind            TEXT NOT NULL CHECK (kind IN ('calls', 'imports', 'inherits', 'contains', 'tested_by', 'depends_on')),
        source_node_id  TEXT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
        target_node_id  TEXT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
        file_path       TEXT NOT NULL,
        line            INTEGER NOT NULL DEFAULT 0,
        confidence      REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
        extra_json      TEXT NOT NULL DEFAULT '{}',
        created_at      REAL NOT NULL,
        updated_at      REAL NOT NULL
    )
    """,  # noqa: E501

    # ── Table 3: graph_files ──────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS graph_files (
        file_path       TEXT PRIMARY KEY,
        content_hash    TEXT NOT NULL,
        mtime           REAL NOT NULL,
        language        TEXT NOT NULL,
        node_count      INTEGER NOT NULL DEFAULT 0,
        edge_count      INTEGER NOT NULL DEFAULT 0,
        last_indexed    REAL NOT NULL
    )
    """,

    # ── Table 4: graph_metadata ───────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS graph_metadata (
        key             TEXT PRIMARY KEY,
        value           TEXT NOT NULL,
        updated_at      REAL NOT NULL
    )
    """,

    # ── Table 5: code_memory_links ────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS code_memory_links (
        link_id         TEXT PRIMARY KEY,
        code_node_id    TEXT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
        slm_fact_id     TEXT NOT NULL,
        slm_entity_id   TEXT,
        link_type       TEXT NOT NULL CHECK (link_type IN (
            'mentions', 'decision_about', 'bug_fix', 'refactor', 'design_rationale'
        )),
        confidence      REAL NOT NULL DEFAULT 0.8 CHECK (confidence >= 0.0 AND confidence <= 1.0),
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        last_verified   TEXT,
        is_stale        INTEGER NOT NULL DEFAULT 0,
        enriched_description TEXT
    )
    """,
)

#: Columns added to existing tables after their first release.
#:
#: This file's DDL is all ``CREATE TABLE IF NOT EXISTS`` and code_graph.db has
#: no migration framework, so appending a column to a _DDL_STATEMENTS block
#: reaches NEW databases only — every database created by an earlier version
#: skips the statement entirely and never gains the column. That silent
#: divergence is what this list exists to close.
#:
#: ADDITIVE ONLY: ``ALTER TABLE ... ADD COLUMN`` with no NOT NULL and no
#: default, so it cannot fail on a populated table and cannot rewrite a row.
#: Never put a DROP, a RENAME, or a type change here.
_ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # (table, column, type) — enrichment text for a code↔memory link. Lives
    # here rather than in memory.db so the user's own fact wording is never
    # overwritten, and so recall (which never opens code_graph.db) is unaffected.
    ("code_memory_links", "enriched_description", "TEXT"),
)


def _apply_additive_columns(cursor: sqlite3.Cursor) -> None:
    """Add any missing column from _ADDITIVE_COLUMNS. Idempotent."""
    for table, column, coltype in _ADDITIVE_COLUMNS:
        try:
            existing = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
        except sqlite3.Error as exc:  # table absent on a partial database
            logger.debug("additive column probe skipped for %s: %s", table, exc)
            continue
        if not existing or column in existing:
            continue
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            logger.info("code_graph schema: added %s.%s", table, column)
        except sqlite3.OperationalError as exc:
            # Concurrent initialiser won the race, or the column appeared
            # between the probe and the ALTER. Both are benign.
            logger.debug("additive column %s.%s not applied: %s", table, column, exc)

# Indexes (separate from tables for clarity)
_INDEX_STATEMENTS: tuple[str, ...] = (
    # graph_nodes indexes
    "CREATE INDEX IF NOT EXISTS idx_gn_file_path ON graph_nodes(file_path)",
    "CREATE INDEX IF NOT EXISTS idx_gn_kind ON graph_nodes(kind)",
    "CREATE INDEX IF NOT EXISTS idx_gn_name ON graph_nodes(name)",
    "CREATE INDEX IF NOT EXISTS idx_gn_qualified ON graph_nodes(qualified_name)",
    "CREATE INDEX IF NOT EXISTS idx_gn_parent ON graph_nodes(parent_name)",
    "CREATE INDEX IF NOT EXISTS idx_gn_language ON graph_nodes(language)",
    "CREATE INDEX IF NOT EXISTS idx_gn_community ON graph_nodes(community_id)",
    # graph_edges indexes
    "CREATE INDEX IF NOT EXISTS idx_ge_source ON graph_edges(source_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_ge_target ON graph_edges(target_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_ge_kind ON graph_edges(kind)",
    "CREATE INDEX IF NOT EXISTS idx_ge_file ON graph_edges(file_path)",
    "CREATE INDEX IF NOT EXISTS idx_ge_source_kind ON graph_edges(source_node_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_ge_target_kind ON graph_edges(target_node_id, kind)",
    # code_memory_links indexes
    "CREATE INDEX IF NOT EXISTS idx_cml_node ON code_memory_links(code_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_cml_fact ON code_memory_links(slm_fact_id)",
    "CREATE INDEX IF NOT EXISTS idx_cml_entity ON code_memory_links(slm_entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_cml_type ON code_memory_links(link_type)",
    "CREATE INDEX IF NOT EXISTS idx_cml_stale ON code_memory_links(is_stale)",
)

# FTS5 virtual table + sync triggers
_FTS5_STATEMENTS: tuple[str, ...] = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS graph_nodes_fts USING fts5(
        name,
        qualified_name,
        file_path,
        signature,
        content='graph_nodes',
        content_rowid='rowid',
        tokenize='porter unicode61'
    )
    """,
    # Auto-sync trigger: INSERT
    """
    CREATE TRIGGER IF NOT EXISTS trg_gn_fts_insert AFTER INSERT ON graph_nodes
    BEGIN
        INSERT INTO graph_nodes_fts(rowid, name, qualified_name, file_path, signature)
        VALUES (NEW.rowid, NEW.name, NEW.qualified_name, NEW.file_path, NEW.signature);
    END
    """,
    # Auto-sync trigger: DELETE
    """
    CREATE TRIGGER IF NOT EXISTS trg_gn_fts_delete AFTER DELETE ON graph_nodes
    BEGIN
        INSERT INTO graph_nodes_fts(graph_nodes_fts, rowid, name, qualified_name, file_path, signature)
        VALUES ('delete', OLD.rowid, OLD.name, OLD.qualified_name, OLD.file_path, OLD.signature);
    END
    """,
    # Auto-sync trigger: UPDATE
    """
    CREATE TRIGGER IF NOT EXISTS trg_gn_fts_update AFTER UPDATE ON graph_nodes
    BEGIN
        INSERT INTO graph_nodes_fts(graph_nodes_fts, rowid, name, qualified_name, file_path, signature)
        VALUES ('delete', OLD.rowid, OLD.name, OLD.qualified_name, OLD.file_path, OLD.signature);
        INSERT INTO graph_nodes_fts(rowid, name, qualified_name, file_path, signature)
        VALUES (NEW.rowid, NEW.name, NEW.qualified_name, NEW.file_path, NEW.signature);
    END
    """,
)


# ---------------------------------------------------------------------------
# Public API (matches SLM's schema.py pattern)
# ---------------------------------------------------------------------------

def create_all_tables(conn: sqlite3.Connection) -> None:
    """Create all CodeGraph tables, indexes, and triggers.

    Idempotent — safe to call multiple times (all DDL uses IF NOT EXISTS).
    """
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")

    # Core tables
    for ddl in _DDL_STATEMENTS:
        cursor.execute(ddl)

    # Columns added after a table's first release. Must run AFTER the CREATEs
    # (so a fresh database already has them and this is a no-op) and BEFORE the
    # indexes (in case one is ever declared on an added column).
    _apply_additive_columns(cursor)

    # Indexes
    for idx in _INDEX_STATEMENTS:
        cursor.execute(idx)

    # FTS5 + triggers (may fail if SQLite lacks FTS5 — non-fatal)
    for stmt in _FTS5_STATEMENTS:
        try:
            cursor.execute(stmt)
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 setup failed (non-fatal): %s", exc)

    # vec0 virtual table for embeddings (may fail if sqlite-vec not loaded)
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS code_node_embeddings USING vec0(
                node_id TEXT PRIMARY KEY,
                embedding float[768] distance_metric=cosine
            )
        """)
    except sqlite3.OperationalError as exc:
        logger.warning("vec0 setup failed (non-fatal, embeddings disabled): %s", exc)

    conn.commit()
    logger.info("CodeGraph schema initialized (%d tables, %d indexes)",
                len(_DDL_STATEMENTS), len(_INDEX_STATEMENTS))


def drop_all_tables(conn: sqlite3.Connection) -> None:
    """Drop all CodeGraph tables. Used in tests only."""
    cursor = conn.cursor()
    for table in (
        "graph_nodes_fts", "code_node_embeddings",
        "code_memory_links", "graph_metadata",
        "graph_files", "graph_edges", "graph_nodes",
    ):
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        except sqlite3.OperationalError:
            pass
    # Drop triggers
    for trigger in ("trg_gn_fts_insert", "trg_gn_fts_delete", "trg_gn_fts_update"):
        cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.commit()
