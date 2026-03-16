# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""V2 to V3 database migration.

Detects V2 installations, backs up data, extends schema with V3 tables,
and creates backward-compatible symlinks.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime, UTC
from pathlib import Path

logger = logging.getLogger(__name__)

V2_BASE = Path.home() / ".claude-memory"
V3_BASE = Path.home() / ".superlocalmemory"
V2_DB_NAME = "memory.db"
BACKUP_NAME = "memory-v2-backup.db"

# V3 tables to add during migration
V3_TABLES_SQL = [
    """CREATE TABLE IF NOT EXISTS semantic_facts (
        fact_id TEXT PRIMARY KEY,
        memory_id TEXT,
        profile_id TEXT NOT NULL DEFAULT 'default',
        content TEXT NOT NULL,
        fact_type TEXT DEFAULT 'world',
        confidence REAL DEFAULT 0.7,
        speaker TEXT DEFAULT '',
        embedding BLOB,
        fisher_mean BLOB,
        fisher_variance BLOB,
        access_count INTEGER DEFAULT 0,
        observation_date TEXT,
        referenced_date TEXT,
        interval_start TEXT,
        interval_end TEXT,
        canonical_entities TEXT DEFAULT '[]',
        created_at TEXT,
        updated_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS kg_nodes (
        node_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL DEFAULT 'default',
        entity_name TEXT NOT NULL,
        entity_type TEXT DEFAULT 'unknown',
        aliases TEXT DEFAULT '[]',
        fact_count INTEGER DEFAULT 0,
        created_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS memory_edges (
        edge_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL DEFAULT 'default',
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        edge_type TEXT DEFAULT 'semantic',
        weight REAL DEFAULT 1.0,
        created_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS memory_scenes (
        scene_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL DEFAULT 'default',
        label TEXT DEFAULT '',
        fact_ids TEXT DEFAULT '[]',
        created_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS bm25_tokens (
        fact_id TEXT NOT NULL,
        profile_id TEXT NOT NULL DEFAULT 'default',
        tokens TEXT NOT NULL,
        doc_length INTEGER DEFAULT 0,
        PRIMARY KEY (fact_id, profile_id)
    )""",
    """CREATE TABLE IF NOT EXISTS temporal_events (
        event_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL DEFAULT 'default',
        entity_id TEXT,
        fact_id TEXT,
        observation_date TEXT,
        referenced_date TEXT,
        interval_start TEXT,
        interval_end TEXT,
        description TEXT DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS memory_observations (
        obs_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL DEFAULT 'default',
        entity_id TEXT NOT NULL,
        observation TEXT NOT NULL,
        source_fact_id TEXT,
        created_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS contradictions (
        contradiction_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL DEFAULT 'default',
        fact_id_a TEXT NOT NULL,
        fact_id_b TEXT NOT NULL,
        severity REAL DEFAULT 0.5,
        resolved INTEGER DEFAULT 0,
        created_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS langevin_state (
        fact_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL DEFAULT 'default',
        position REAL DEFAULT 0.5,
        velocity REAL DEFAULT 0.0,
        updated_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS sheaf_sections (
        section_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL DEFAULT 'default',
        fact_id TEXT NOT NULL,
        section_data BLOB,
        created_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS v3_config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT
    )""",
]

# Indexes for V3 tables
V3_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_facts_profile ON semantic_facts(profile_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_memory ON semantic_facts(memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_profile ON kg_nodes(profile_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id)",
    "CREATE INDEX IF NOT EXISTS idx_scenes_profile ON memory_scenes(profile_id)",
    "CREATE INDEX IF NOT EXISTS idx_temporal_entity ON temporal_events(entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_observations_entity ON memory_observations(entity_id)",
]


class V2Migrator:
    """Migrate V2 database to V3 schema."""

    def __init__(self, home: Path | None = None):
        self._home = home or Path.home()
        self._v2_base = self._home / ".claude-memory"
        self._v3_base = self._home / ".superlocalmemory"
        self._v2_db = self._v2_base / V2_DB_NAME
        self._v3_db = self._v3_base / V2_DB_NAME
        self._backup_db = self._v3_base / BACKUP_NAME

    def detect_v2(self) -> bool:
        """Check if a V2 installation exists."""
        return self._v2_db.exists() and self._v2_db.is_file()

    def is_already_migrated(self) -> bool:
        """Check if migration has already been performed."""
        if not self._v3_db.exists():
            return False
        try:
            conn = sqlite3.connect(str(self._v3_db))
            try:
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                return "semantic_facts" in tables and "v3_config" in tables
            finally:
                conn.close()
        except Exception:
            return False

    def get_v2_stats(self) -> dict:
        """Get statistics about the V2 database."""
        if not self.detect_v2():
            return {"exists": False}
        conn = None
        try:
            conn = sqlite3.connect(str(self._v2_db))
            memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            # Check for profiles
            profile_count = 1
            try:
                profiles = conn.execute(
                    "SELECT DISTINCT profile FROM memories WHERE profile IS NOT NULL"
                ).fetchall()
                profile_count = max(len(profiles), 1)
            except Exception:
                pass
            return {
                "exists": True,
                "memory_count": memory_count,
                "profile_count": profile_count,
                "table_count": len(tables),
                "db_path": str(self._v2_db),
                "db_size_mb": round(self._v2_db.stat().st_size / 1024 / 1024, 2),
            }
        except Exception as exc:
            return {"exists": True, "error": str(exc)}
        finally:
            if conn is not None:
                conn.close()

    def migrate(self) -> dict:
        """Run the full V2 to V3 migration.

        Steps:
        1. Create V3 directory
        2. Backup V2 database
        3. Copy database to V3 location
        4. Extend schema with V3 tables
        5. Create symlink for backward compat
        6. Mark migration complete

        Returns dict with migration stats.
        """
        if not self.detect_v2():
            return {"success": False, "error": "No V2 installation found"}

        if self.is_already_migrated():
            return {"success": True, "message": "Already migrated"}

        stats = {"steps": []}

        try:
            # Step 1: Create V3 directory
            self._v3_base.mkdir(parents=True, exist_ok=True)
            (self._v3_base / "embeddings").mkdir(exist_ok=True)
            (self._v3_base / "models").mkdir(exist_ok=True)
            stats["steps"].append("Created V3 directory")

            # Step 2: Backup
            shutil.copy2(str(self._v2_db), str(self._backup_db))
            stats["steps"].append(f"Backed up to {self._backup_db}")

            # Step 3: Copy to V3 location
            shutil.copy2(str(self._v2_db), str(self._v3_db))
            stats["steps"].append("Copied database to V3 location")

            # Step 4: Extend schema + alter V2 tables for V3 compatibility
            conn = sqlite3.connect(str(self._v3_db))

            # Add missing V3 columns to V2 memories table
            existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
            v3_columns = [
                ("profile_id", 'TEXT DEFAULT "default"'),
                ("memory_id", "TEXT"),
                ("session_id", 'TEXT DEFAULT ""'),
                ("speaker", 'TEXT DEFAULT ""'),
                ("role", 'TEXT DEFAULT "user"'),
                ("session_date", "TEXT"),
                ("metadata_json", 'TEXT DEFAULT "{}"'),
            ]
            for col, coltype in v3_columns:
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {coltype}")
            # Backfill V3 columns from V2 data
            conn.execute('UPDATE memories SET profile_id = COALESCE(profile, "default") WHERE profile_id IS NULL')
            conn.execute("UPDATE memories SET memory_id = 'v2_' || CAST(id AS TEXT) WHERE memory_id IS NULL")
            # Create unique index on memory_id so V3 FKs work
            try:
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_memory_id ON memories (memory_id)")
            except Exception:
                pass
            # Disable FK enforcement for migrated DBs (V2 schema is incompatible)
            conn.execute("PRAGMA foreign_keys=OFF")

            # Rename ALL tables with incompatible schemas (V2 + old alpha)
            # User data is in 'memories' table (already upgraded above)
            # Everything else is computed/derived and will be recreated by V3
            all_existing = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '_v2_bak_%'"
            ).fetchall()}
            # Keep only: memories (upgraded), profiles, schema_version, v3_config, sqlite_sequence
            keep_tables = {"memories", "profiles", "schema_version", "v3_config", "sqlite_sequence"}
            v2_conflicting = [t for t in all_existing if t not in keep_tables and not t.startswith("_")]
            for table in v2_conflicting:
                try:
                    conn.execute(f'ALTER TABLE "{table}" RENAME TO "_v2_bak_{table}"')
                except Exception:
                    pass  # Table may not exist

            conn.commit()

            # Use the FULL V3 schema (not the partial V3_TABLES_SQL)
            from superlocalmemory.storage import schema
            schema.create_all_tables(conn)
            conn.commit()
            # Mark migration in config table
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
                    ("migration_date", datetime.now(UTC).isoformat(), datetime.now(UTC).isoformat()),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
                    ("migration_version", "3.0.0", datetime.now(UTC).isoformat()),
                )
                conn.commit()
            except Exception:
                pass  # Schema handles this on engine init
            # Step 4b: Convert V2 memories → V3 atomic_facts
            try:
                now = datetime.now(UTC).isoformat()
                rows = conn.execute("SELECT memory_id, profile_id, content, created_at FROM memories").fetchall()
                converted = 0
                for row in rows:
                    mid, pid, content, created = row[0], row[1], row[2], row[3]
                    if not content or not content.strip():
                        continue
                    fid = f"v2_fact_{mid}"
                    conn.execute(
                        "INSERT OR IGNORE INTO atomic_facts "
                        "(fact_id, memory_id, profile_id, content, fact_type, "
                        " entities_json, canonical_entities_json, confidence, importance, "
                        " evidence_count, access_count, source_turn_ids_json, session_id, "
                        " lifecycle, emotional_valence, emotional_arousal, signal_type, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (fid, mid, pid or "default", content, "factual",
                         "[]", "[]", 0.8, 0.5, 1, 0, "[]", "",
                         "active", 0.0, 0.0, "factual", created or now),
                    )
                    converted += 1
                conn.commit()
                stats["steps"].append(f"Converted {converted} V2 memories to V3 facts")
            except Exception as exc:
                stats["steps"].append(f"V2 conversion partial: {exc}")

            conn.close()
            stats["steps"].append("Created V3 schema")

            # Step 5: Symlink (only if .claude-memory is not already a symlink)
            if not self._v2_base.is_symlink():
                # Rename original to .claude-memory-v2-original
                original_backup = self._home / ".claude-memory-v2-original"
                if not original_backup.exists():
                    self._v2_base.rename(original_backup)
                    self._v2_base.symlink_to(self._v3_base)
                    stats["steps"].append("Created symlink: .claude-memory -> .superlocalmemory")
                else:
                    stats["steps"].append("Symlink skipped (backup dir already exists)")
            else:
                stats["steps"].append("Symlink already exists")

            stats["success"] = True
            stats["v3_db"] = str(self._v3_db)
            stats["backup_db"] = str(self._backup_db)

        except Exception as exc:
            stats["success"] = False
            stats["error"] = str(exc)
            logger.error("Migration failed: %s", exc)

        return stats

    def rollback(self) -> dict:
        """Rollback migration -- restore V2 state.

        Returns dict with rollback stats.
        """
        stats = {"steps": []}

        try:
            # Remove symlink
            if self._v2_base.is_symlink():
                self._v2_base.unlink()
                stats["steps"].append("Removed symlink")

            # Restore original V2 directory
            original_backup = self._home / ".claude-memory-v2-original"
            if original_backup.exists():
                if not self._v2_base.exists():
                    original_backup.rename(self._v2_base)
                    stats["steps"].append("Restored original .claude-memory")
            elif self._backup_db.exists():
                # Restore from backup
                self._v2_base.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(self._backup_db), str(self._v2_db))
                stats["steps"].append("Restored database from backup")

            stats["success"] = True

        except Exception as exc:
            stats["success"] = False
            stats["error"] = str(exc)

        return stats
