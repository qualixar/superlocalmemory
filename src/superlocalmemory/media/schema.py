# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The images-and-documents database layout (version 1).

Every table later features need exists from the first version, so adding
documents or folder sources never needs a migration. Vectors live in one
``media_vec_<space_id>`` table per embedding space (created by the store),
the same shape memory.db uses; ``media_vector_rows`` maps a vector row back
to its item.

Originals on disk are content-addressed and shared by all profiles:
``<data_root>/media/<sha256[:2]>/<sha256>.<ext>``; ``original_relpath``
is relative to ``<data_root>/media/`` and never contains a profile id.
"""

from __future__ import annotations

import sqlite3

MEDIA_SCHEMA_VERSION = 1

#: Tables that carry a profile_id column and are moved/erased with a profile.
PROFILE_TABLES = ("media_items", "documents", "jobs", "sources", "media_vector_rows")

_DDL = (
    "CREATE TABLE IF NOT EXISTS media_schema (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS media_spaces (
      space_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, model_revision TEXT NOT NULL,
      dim INTEGER NOT NULL CHECK (dim > 0),
      state TEXT NOT NULL CHECK (state IN ('active','building','previous')),
      created_at TEXT NOT NULL)""",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_media_spaces_active ON media_spaces(state) WHERE state = 'active'",
    """CREATE TABLE IF NOT EXISTS media_items (
      media_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
      kind TEXT NOT NULL CHECK (kind IN ('image','page')),
      sha256 TEXT NOT NULL, phash TEXT, mime TEXT NOT NULL, bytes INTEGER NOT NULL,
      width INTEGER, height INTEGER, original_relpath TEXT,
      exif_json TEXT NOT NULL DEFAULT '{}',
      captured_at TEXT, anchor_memory_id TEXT,
      document_id TEXT, page_no INTEGER, source_id TEXT,
      origin TEXT NOT NULL CHECK (origin IN ('tool','dashboard','download','folder','document')),
      state TEXT NOT NULL DEFAULT 'active'
        CHECK (state IN ('active','tombstoned','quarantined','media_missing')),
      thumb_webp BLOB, created_at TEXT NOT NULL, tombstoned_at TEXT)""",
    "CREATE INDEX IF NOT EXISTS ix_media_profile_sha ON media_items(profile_id, sha256)",
    "CREATE INDEX IF NOT EXISTS ix_media_anchor ON media_items(anchor_memory_id)",
    "CREATE INDEX IF NOT EXISTS ix_media_doc ON media_items(document_id, page_no)",
    """CREATE TABLE IF NOT EXISTS media_vector_rows (
      space_id TEXT NOT NULL, vec_rowid INTEGER NOT NULL, media_id TEXT NOT NULL,
      profile_id TEXT NOT NULL, PRIMARY KEY (space_id, vec_rowid))""",
    "CREATE INDEX IF NOT EXISTS ix_mvr_media ON media_vector_rows(media_id)",
    """CREATE TABLE IF NOT EXISTS documents (
      document_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, sha256 TEXT NOT NULL,
      title TEXT NOT NULL, mime TEXT NOT NULL, page_count INTEGER NOT NULL DEFAULT 0,
      pages_text_layer INTEGER NOT NULL DEFAULT 0, pages_ocr INTEGER NOT NULL DEFAULT 0,
      pages_empty INTEGER NOT NULL DEFAULT 0, source_id TEXT, source_relpath TEXT,
      state TEXT NOT NULL CHECK (state IN ('processing','ready','failed','tombstoned')),
      created_at TEXT NOT NULL, updated_at TEXT NOT NULL, tombstoned_at TEXT)""",
    "CREATE INDEX IF NOT EXISTS ix_documents_profile ON documents(profile_id, state)",
    """CREATE TABLE IF NOT EXISTS doc_pages (
      document_id TEXT NOT NULL, page_no INTEGER NOT NULL, media_id TEXT,
      memory_ids_json TEXT NOT NULL DEFAULT '[]',
      text_origin TEXT NOT NULL CHECK (text_origin IN ('text_layer','ocr','none')),
      char_count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (document_id, page_no))""",
    """CREATE TABLE IF NOT EXISTS jobs (
      job_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
      kind TEXT NOT NULL CHECK (kind IN ('document','source_scan','media_reembed','gc','env_install')),
      state TEXT NOT NULL CHECK (state IN ('queued','running','done','failed','cancelled')),
      done INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0, error TEXT,
      lease_owner TEXT, lease_until TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_jobs_state ON jobs(state, created_at)",
    """CREATE TABLE IF NOT EXISTS sources (
      source_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
      kind TEXT NOT NULL CHECK (kind IN ('folder','obsidian')),
      root_path TEXT NOT NULL, display_name TEXT NOT NULL, include_types_json TEXT NOT NULL,
      state TEXT NOT NULL CHECK (state IN ('preview','active','paused','offline','removed')),
      remote_visible INTEGER NOT NULL DEFAULT 0, watch INTEGER NOT NULL DEFAULT 1,
      created_at TEXT NOT NULL, last_scan_at TEXT, last_scan_stats_json TEXT NOT NULL DEFAULT '{}')""",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_sources_root ON sources(profile_id, root_path) WHERE state != 'removed'",
    """CREATE TABLE IF NOT EXISTS source_files (
      source_id TEXT NOT NULL, relpath TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
      file_id TEXT, sha256 TEXT,
      state TEXT NOT NULL CHECK (state IN ('pending','indexed','skipped','quarantined',
        'cloud_placeholder','tombstoned','error')),
      reason TEXT, memory_ids_json TEXT NOT NULL DEFAULT '[]', document_id TEXT, media_id TEXT,
      tombstoned_at TEXT, updated_at TEXT NOT NULL, PRIMARY KEY (source_id, relpath))""",
    "CREATE INDEX IF NOT EXISTS ix_source_files_sha ON source_files(source_id, sha256)",
    "CREATE INDEX IF NOT EXISTS ix_source_files_state ON source_files(source_id, state)",
    """CREATE TABLE IF NOT EXISTS source_links (
      source_id TEXT NOT NULL, from_relpath TEXT NOT NULL, target TEXT NOT NULL,
      link_kind TEXT NOT NULL CHECK (link_kind IN ('wikilink','embed','heading','block','md_link','canvas_edge')),
      label TEXT, PRIMARY KEY (source_id, from_relpath, target, link_kind))""",
)


def apply_schema(conn: sqlite3.Connection, *, created_by: str) -> None:
    """Create every table (idempotent) and stamp the version once."""
    for statement in _DDL:
        conn.execute(statement)
    conn.execute("INSERT OR IGNORE INTO media_schema(key, value) VALUES ('version', ?)",
                 (str(MEDIA_SCHEMA_VERSION),))
    conn.execute("INSERT OR IGNORE INTO media_schema(key, value) VALUES ('created_by', ?)",
                 (created_by,))


def stored_version(conn: sqlite3.Connection) -> int:
    """Schema version on disk; 0 when the stamp is missing or unreadable."""
    try:
        row = conn.execute("SELECT value FROM media_schema WHERE key = 'version'").fetchone()
        return int(row[0]) if row else 0
    except (sqlite3.Error, ValueError):
        return 0
