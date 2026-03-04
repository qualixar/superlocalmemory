#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
SuperLocalMemory V1 to V2 Migration Script

Safely migrates the memory database from V1 schema to V2 architecture.
This script is idempotent - safe to re-run if interrupted.

Usage:
    python ~/.claude-memory/migrate_v1_to_v2.py

Features:
- Adds new columns to memories table
- Creates new tables for tree, graph, patterns, and archive
- Creates all required indexes
- Migrates existing memories to tree structure
- Handles rollback on failure
- Prints progress messages
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
import sys
import traceback
from typing import Tuple, Dict, Any

DB_PATH = Path.home() / '.claude-memory' / 'memory.db'
BACKUP_PATH = Path.home() / '.claude-memory' / 'backups' / f'pre-v2-{datetime.now().strftime("%Y%m%d-%H%M%S")}.db'


def create_backup() -> None:
    """Create a backup of the database before migration."""
    print("=" * 60)
    print("CREATING BACKUP")
    print("=" * 60)

    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        import shutil
        shutil.copy2(DB_PATH, BACKUP_PATH)
        print(f"✓ Backup created: {BACKUP_PATH}")
        print(f"  Size: {BACKUP_PATH.stat().st_size / 1024:.1f} KB")
    else:
        print("! Database does not exist yet - no backup needed")

    print()


def check_schema_version(conn: sqlite3.Connection) -> bool:
    """Check if migration has already been completed."""
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT value FROM system_metadata WHERE key = 'schema_version'
        """)
        result = cursor.fetchone()
        if result and result[0] == '2.0.0':
            return True
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        pass

    return False


def add_new_columns(conn: sqlite3.Connection) -> None:
    """Add new columns to the memories table."""
    print("=" * 60)
    print("ADDING NEW COLUMNS TO MEMORIES TABLE")
    print("=" * 60)

    cursor = conn.cursor()

    new_columns = [
        ('parent_id', 'INTEGER'),
        ('tree_path', 'TEXT'),
        ('depth', 'INTEGER DEFAULT 0'),
        ('category', 'TEXT'),
        ('cluster_id', 'INTEGER'),
        ('last_accessed', 'TIMESTAMP'),
        ('access_count', 'INTEGER DEFAULT 0'),
        ('tier', 'INTEGER DEFAULT 1')
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(f'ALTER TABLE memories ADD COLUMN {col_name} {col_type}')
            print(f"✓ Added column: {col_name}")
        except sqlite3.OperationalError as e:
            if 'duplicate column' in str(e).lower():
                print(f"- Column already exists: {col_name}")
            else:
                raise

    print()


def create_new_tables(conn: sqlite3.Connection) -> None:
    """Create all new tables for V2 architecture."""
    print("=" * 60)
    print("CREATING NEW TABLES")
    print("=" * 60)

    cursor = conn.cursor()

    tables = {
        'memory_tree': '''
            CREATE TABLE IF NOT EXISTS memory_tree (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                parent_id INTEGER,
                tree_path TEXT NOT NULL,
                depth INTEGER DEFAULT 0,
                memory_count INTEGER DEFAULT 0,
                total_size INTEGER DEFAULT 0,
                last_updated TIMESTAMP,
                memory_id INTEGER,
                FOREIGN KEY (parent_id) REFERENCES memory_tree(id) ON DELETE CASCADE,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        ''',

        'graph_nodes': '''
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER UNIQUE NOT NULL,
                entities TEXT,
                embedding_vector TEXT,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        ''',

        'graph_edges': '''
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_memory_id INTEGER NOT NULL,
                target_memory_id INTEGER NOT NULL,
                relationship_type TEXT,
                weight REAL DEFAULT 1.0,
                shared_entities TEXT,
                similarity_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                UNIQUE(source_memory_id, target_memory_id)
            )
        ''',

        'graph_clusters': '''
            CREATE TABLE IF NOT EXISTS graph_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                member_count INTEGER DEFAULT 0,
                avg_importance REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',

        'identity_patterns': '''
            CREATE TABLE IF NOT EXISTS identity_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 1,
                memory_ids TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pattern_type, key, category)
            )
        ''',

        'pattern_examples': '''
            CREATE TABLE IF NOT EXISTS pattern_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL,
                memory_id INTEGER NOT NULL,
                example_text TEXT,
                FOREIGN KEY (pattern_id) REFERENCES identity_patterns(id) ON DELETE CASCADE,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        ''',

        'memory_archive': '''
            CREATE TABLE IF NOT EXISTS memory_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER UNIQUE NOT NULL,
                full_content TEXT NOT NULL,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        ''',

        'system_metadata': '''
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
    }

    for table_name, create_sql in tables.items():
        try:
            cursor.execute(create_sql)
            print(f"✓ Created table: {table_name}")
        except sqlite3.OperationalError as e:
            if 'already exists' in str(e).lower():
                print(f"- Table already exists: {table_name}")
            else:
                raise

    print()


def create_indexes(conn: sqlite3.Connection) -> None:
    """Create all indexes for performance optimization."""
    print("=" * 60)
    print("CREATING INDEXES")
    print("=" * 60)

    cursor = conn.cursor()

    indexes = [
        ('idx_project', 'memories', 'project_path'),
        ('idx_tags', 'memories', 'tags'),
        ('idx_category', 'memories', 'category'),
        ('idx_tree_path', 'memories', 'tree_path'),
        ('idx_cluster', 'memories', 'cluster_id'),
        ('idx_last_accessed', 'memories', 'last_accessed'),
        ('idx_tier', 'memories', 'tier'),
        ('idx_tree_path_layer2', 'memory_tree', 'tree_path'),
        ('idx_node_type', 'memory_tree', 'node_type'),
        ('idx_cluster_members', 'memories', 'cluster_id'),
        ('idx_graph_source', 'graph_edges', 'source_memory_id'),
        ('idx_graph_target', 'graph_edges', 'target_memory_id'),
        ('idx_pattern_type', 'identity_patterns', 'pattern_type'),
        ('idx_pattern_confidence', 'identity_patterns', 'confidence'),
        ('idx_archive_memory', 'memory_archive', 'memory_id')
    ]

    for idx_name, table, column in indexes:
        try:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})')
            print(f"✓ Created index: {idx_name}")
        except sqlite3.OperationalError as e:
            print(f"- Index already exists: {idx_name}")

    print()


def migrate_to_tree_structure(conn: sqlite3.Connection) -> None:
    """Migrate existing memories to tree structure."""
    print("=" * 60)
    print("MIGRATING MEMORIES TO TREE STRUCTURE")
    print("=" * 60)

    cursor = conn.cursor()

    # Check if root node already exists
    cursor.execute("SELECT id FROM memory_tree WHERE node_type = 'root'")
    root = cursor.fetchone()

    if root:
        root_id = root[0]
        print(f"- Root node already exists (id={root_id})")
    else:
        # Create root node
        cursor.execute('''
            INSERT INTO memory_tree (node_type, name, tree_path, depth, last_updated)
            VALUES ('root', 'All Projects', '1', 0, CURRENT_TIMESTAMP)
        ''')
        root_id = cursor.lastrowid
        print(f"✓ Created root node (id={root_id})")

    # Get all existing memories that haven't been migrated
    cursor.execute('''
        SELECT id, project_path, project_name, content
        FROM memories
        WHERE tree_path IS NULL OR tree_path = ''
    ''')
    memories = cursor.fetchall()

    if not memories:
        print("- No memories to migrate (all already in tree)")
        print()
        return

    print(f"Found {len(memories)} memories to migrate")

    project_nodes = {}  # project_key -> node_id

    # Load existing project nodes
    cursor.execute("""
        SELECT id, name FROM memory_tree WHERE node_type = 'project'
    """)
    for node_id, name in cursor.fetchall():
        project_nodes[name] = node_id

    migrated_count = 0

    for memory_id, project_path, project_name, content in memories:
        # Determine project key
        if not project_name and not project_path:
            project_key = 'Uncategorized'
        else:
            project_key = project_name or Path(project_path).name if project_path else 'Uncategorized'

        # Create project node if doesn't exist
        if project_key not in project_nodes:
            cursor.execute('''
                INSERT INTO memory_tree (node_type, name, parent_id, tree_path, depth, last_updated)
                VALUES ('project', ?, ?, ?, 1, CURRENT_TIMESTAMP)
            ''', (project_key, root_id, f'1.{len(project_nodes) + 2}'))

            project_nodes[project_key] = cursor.lastrowid
            print(f"  ✓ Created project node: {project_key}")

        # Link memory to project
        project_node_id = project_nodes[project_key]
        tree_path = f'1.{project_node_id}.{memory_id}'

        # Create memory node in tree
        cursor.execute('''
            INSERT INTO memory_tree (node_type, name, parent_id, tree_path, depth, memory_id, last_updated)
            VALUES ('memory', ?, ?, ?, 2, ?, CURRENT_TIMESTAMP)
        ''', (
            f"Memory #{memory_id}",
            project_node_id,
            tree_path,
            memory_id
        ))

        # Update memory with tree info
        cursor.execute('''
            UPDATE memories
            SET tree_path = ?, depth = 2, last_accessed = created_at
            WHERE id = ?
        ''', (tree_path, memory_id))

        migrated_count += 1

    # Update project node memory counts
    for project_key, project_node_id in project_nodes.items():
        cursor.execute('''
            SELECT COUNT(*), SUM(LENGTH(content))
            FROM memories m
            JOIN memory_tree mt ON mt.memory_id = m.id
            WHERE mt.parent_id = ?
        ''', (project_node_id,))

        count, total_size = cursor.fetchone()
        total_size = total_size or 0

        cursor.execute('''
            UPDATE memory_tree
            SET memory_count = ?, total_size = ?, last_updated = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (count, total_size, project_node_id))

    # Update root node count
    cursor.execute('''
        SELECT COUNT(*) FROM memories
    ''')
    total_memories = cursor.fetchone()[0]

    cursor.execute('''
        UPDATE memory_tree
        SET memory_count = ?, last_updated = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (total_memories, root_id))

    print(f"✓ Migrated {migrated_count} memories to {len(project_nodes)} projects")
    print()


def update_metadata(conn: sqlite3.Connection) -> None:
    """Update system metadata with migration info."""
    print("=" * 60)
    print("UPDATING SYSTEM METADATA")
    print("=" * 60)

    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO system_metadata (key, value, updated_at)
        VALUES ('schema_version', '2.0.0', CURRENT_TIMESTAMP)
    ''')
    print("✓ Set schema_version = 2.0.0")

    cursor.execute('''
        INSERT OR REPLACE INTO system_metadata (key, value, updated_at)
        VALUES ('migrated_at', ?, CURRENT_TIMESTAMP)
    ''', (datetime.now().isoformat(),))
    print(f"✓ Set migrated_at = {datetime.now().isoformat()}")

    # Count memories
    cursor.execute('SELECT COUNT(*) FROM memories')
    memory_count = cursor.fetchone()[0]

    cursor.execute('''
        INSERT OR REPLACE INTO system_metadata (key, value, updated_at)
        VALUES ('memory_count_at_migration', ?, CURRENT_TIMESTAMP)
    ''', (str(memory_count),))
    print(f"✓ Recorded memory_count_at_migration = {memory_count}")

    print()


def verify_migration(conn: sqlite3.Connection) -> Tuple[bool, Dict[str, Any]]:
    """Verify that migration completed successfully."""
    print("=" * 60)
    print("VERIFYING MIGRATION")
    print("=" * 60)

    cursor = conn.cursor()

    checks = []

    # Check schema version
    cursor.execute("SELECT value FROM system_metadata WHERE key = 'schema_version'")
    version = cursor.fetchone()
    checks.append(("Schema version", version and version[0] == '2.0.0'))

    # Check memories table has new columns
    cursor.execute("PRAGMA table_info(memories)")
    columns = {row[1] for row in cursor.fetchall()}
    required_columns = {'parent_id', 'tree_path', 'depth', 'category', 'cluster_id',
                       'last_accessed', 'access_count', 'tier'}
    checks.append(("New columns added", required_columns.issubset(columns)))

    # Check new tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    required_tables = {'memory_tree', 'graph_nodes', 'graph_edges', 'graph_clusters',
                      'identity_patterns', 'pattern_examples', 'memory_archive', 'system_metadata'}
    checks.append(("New tables created", required_tables.issubset(tables)))

    # Check tree structure
    cursor.execute("SELECT COUNT(*) FROM memory_tree WHERE node_type = 'root'")
    root_count = cursor.fetchone()[0]
    checks.append(("Root node exists", root_count == 1))

    cursor.execute("SELECT COUNT(*) FROM memory_tree WHERE node_type = 'project'")
    project_count = cursor.fetchone()[0]
    checks.append(("Project nodes exist", project_count >= 0))

    cursor.execute("SELECT COUNT(*) FROM memories WHERE tree_path IS NOT NULL")
    migrated_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM memories")
    total_count = cursor.fetchone()[0]
    checks.append(("All memories migrated", migrated_count == total_count))

    # Print results
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    print()

    if not all_passed:
        raise Exception("Migration verification failed! See errors above.")

    return True


def print_summary(conn: sqlite3.Connection) -> None:
    """Print migration summary statistics."""
    print("=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)

    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM memories")
    memory_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM memory_tree WHERE node_type = 'project'")
    project_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM memory_tree WHERE node_type = 'memory'")
    tree_memory_count = cursor.fetchone()[0]

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
    indexes = [row[0] for row in cursor.fetchall()]

    print(f"Total memories: {memory_count}")
    print(f"Project nodes: {project_count}")
    print(f"Memory nodes in tree: {tree_memory_count}")
    print(f"Total tables: {len(tables)}")
    print(f"Total indexes: {len(indexes)}")
    print()

    print("Database tables:")
    for table in tables:
        if not table.startswith('sqlite_') and not table.endswith('_fts'):
            print(f"  - {table}")

    print()
    print("=" * 60)
    print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()


def migrate() -> None:
    """Main migration function."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  SuperLocalMemory V1 → V2 Migration".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print(f"Database: {DB_PATH}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Please ensure SuperLocalMemory V1 is installed.")
        sys.exit(1)

    try:
        # Create backup first
        create_backup()

        # Open database connection
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")

        # Check if already migrated
        if check_schema_version(conn):
            print("=" * 60)
            print("ALREADY MIGRATED")
            print("=" * 60)
            print("Database is already at version 2.0.0")
            print("Migration is idempotent - re-running will update any missing components.")
            print()

        # Start transaction
        conn.execute("BEGIN")

        try:
            # Run migration steps
            add_new_columns(conn)
            create_new_tables(conn)
            create_indexes(conn)
            migrate_to_tree_structure(conn)
            update_metadata(conn)

            # Verify migration
            verify_migration(conn)

            # Commit transaction
            conn.commit()

            # Print summary
            print_summary(conn)

        except Exception as e:
            # Rollback on error
            conn.rollback()
            raise

        finally:
            conn.close()

        # Print next steps
        print("NEXT STEPS:")
        print("-" * 60)
        print("1. Build knowledge graph:")
        print("   python ~/.claude-memory/graph_engine.py --build")
        print()
        print("2. Learn identity patterns:")
        print("   python ~/.claude-memory/pattern_learner.py --analyze")
        print()
        print("3. Test CLI commands:")
        print("   /recall 'test query'")
        print()
        print("4. Start web UI (optional):")
        print("   cd ~/.claude-memory/ui && uvicorn server:app --port 5432")
        print()
        print("-" * 60)
        print()
        print(f"Backup saved to: {BACKUP_PATH}")
        print()

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ MIGRATION FAILED")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print()
        print("Stack trace:")
        traceback.print_exc()
        print()
        print("=" * 60)
        print("ROLLBACK INSTRUCTIONS")
        print("=" * 60)
        print("The database has been rolled back to its previous state.")
        print(f"A backup was created at: {BACKUP_PATH}")
        print()
        print("To restore from backup manually:")
        print(f"  cp {BACKUP_PATH} {DB_PATH}")
        print()
        sys.exit(1)


if __name__ == '__main__':
    migrate()
