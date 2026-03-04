#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
SuperLocalMemory V2 - Reset & Reinitialize Utility

Provides safe ways to reset the memory system:
1. Soft reset - Clear memories but keep schema
2. Hard reset - Delete everything and reinitialize
3. Selective reset - Clear specific layers only
4. Backup & reset - Always create backup first
"""

import sqlite3
import shutil
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"
BACKUP_DIR = MEMORY_DIR / "backups"
VENV_PATH = MEMORY_DIR / "venv"


class MemoryReset:
    def __init__(self, auto_backup=True):
        self.db_path = DB_PATH
        self.backup_dir = BACKUP_DIR
        self.auto_backup = auto_backup

    def create_backup(self) -> str:
        """Create timestamped backup before any reset operation."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_file = self.backup_dir / f"pre-reset-{timestamp}.db"

        self.backup_dir.mkdir(exist_ok=True)

        if self.db_path.exists():
            shutil.copy2(self.db_path, backup_file)
            print(f"✓ Backup created: {backup_file}")
            return str(backup_file)
        else:
            print("⚠ No database found to backup")
            return None

    def soft_reset(self) -> None:
        """Clear all memories but keep V2 schema structure."""
        print("\n" + "="*60)
        print("SOFT RESET - Clear Memories, Keep Schema")
        print("="*60)

        if self.auto_backup:
            self.create_backup()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete all data from tables (keeps schema)
        tables = [
            'pattern_examples',
            'identity_patterns',
            'memory_archive',
            'graph_edges',
            'graph_clusters',
            'graph_nodes',
            'memory_tree',
            'memories',
            'sessions'
        ]

        VALID_TABLES = frozenset(tables)  # Whitelist from hardcoded list above

        for table in tables:
            try:
                if table not in VALID_TABLES:
                    raise ValueError(f"Invalid table name: {table}")
                cursor.execute(f'DELETE FROM {table}')  # Safe: validated against whitelist
                count = cursor.rowcount
                print(f"  ✓ Cleared {table}: {count} rows deleted")
            except sqlite3.OperationalError as e:
                print(f"  - Skipped {table}: {e}")

        # Reset sequences
        cursor.execute('DELETE FROM sqlite_sequence')
        print(f"  ✓ Reset auto-increment sequences")

        conn.commit()
        conn.close()

        print("\n✅ Soft reset complete!")
        print("   Schema preserved, all memories cleared")
        print("   You can now add new memories to clean system")

    def hard_reset(self) -> None:
        """Delete database completely and reinitialize with V2 schema."""
        print("\n" + "="*60)
        print("HARD RESET - Delete Everything & Reinitialize")
        print("="*60)

        if self.auto_backup:
            self.create_backup()

        # Delete database file
        if self.db_path.exists():
            self.db_path.unlink()
            print(f"  ✓ Deleted database: {self.db_path}")

        # Reinitialize with V2 schema
        print("\n  Reinitializing V2 schema...")
        self._initialize_v2_schema()

        print("\n✅ Hard reset complete!")
        print("   Fresh V2 database created")
        print("   Ready for new memories")

    def layer_reset(self, layers: list) -> None:
        """Reset specific layers only."""
        print("\n" + "="*60)
        print(f"LAYER RESET - Clearing Layers: {', '.join(layers)}")
        print("="*60)

        if self.auto_backup:
            self.create_backup()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        layer_tables = {
            'graph': ['graph_edges', 'graph_clusters', 'graph_nodes'],
            'patterns': ['identity_patterns', 'pattern_examples'],
            'tree': ['memory_tree'],
            'archive': ['memory_archive']
        }

        VALID_LAYER_TABLES = frozenset(
            t for tables_list in layer_tables.values() for t in tables_list
        )  # Whitelist from hardcoded dict above

        for layer in layers:
            if layer in layer_tables:
                print(f"\n  Clearing Layer: {layer.upper()}")
                for table in layer_tables[layer]:
                    try:
                        if table not in VALID_LAYER_TABLES:
                            raise ValueError(f"Invalid table name: {table}")
                        cursor.execute(f'DELETE FROM {table}')  # Safe: validated against whitelist
                        count = cursor.rowcount
                        print(f"    ✓ Cleared {table}: {count} rows")
                    except sqlite3.OperationalError as e:
                        print(f"    - Skipped {table}: {e}")

                # Clear related columns in memories table
                if layer == 'graph':
                    cursor.execute('UPDATE memories SET cluster_id = NULL')
                    print(f"    ✓ Cleared cluster_id from memories")
                elif layer == 'tree':
                    cursor.execute('UPDATE memories SET parent_id = NULL, tree_path = NULL, depth = 0')
                    print(f"    ✓ Cleared tree fields from memories")
            else:
                print(f"  ⚠ Unknown layer: {layer}")

        conn.commit()
        conn.close()

        print("\n✅ Layer reset complete!")

    def _initialize_v2_schema(self):
        """Initialize fresh V2 database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Layer 1: Memories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    summary TEXT,
                    project_path TEXT,
                    project_name TEXT,
                    tags TEXT,
                    category TEXT,
                    memory_type TEXT DEFAULT 'session',
                    importance INTEGER DEFAULT 5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    content_hash TEXT UNIQUE,
                    parent_id INTEGER,
                    tree_path TEXT,
                    depth INTEGER DEFAULT 0,
                    cluster_id INTEGER,
                    tier INTEGER DEFAULT 1,
                    FOREIGN KEY (parent_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            ''')
            print("    ✓ Created memories table")

            # FTS5 index
            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, summary, tags, content='memories', content_rowid='id')
            ''')
            print("    ✓ Created FTS index")

            # Layer 2: Tree
            cursor.execute('''
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
            ''')
            print("    ✓ Created memory_tree table")

            # Layer 3: Graph
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER UNIQUE NOT NULL,
                    entities TEXT,
                    embedding_vector TEXT,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            ''')

            cursor.execute('''
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
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS graph_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    member_count INTEGER DEFAULT 0,
                    avg_importance REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            print("    ✓ Created graph tables")

            # Layer 4: Patterns
            cursor.execute('''
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
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id INTEGER NOT NULL,
                    memory_id INTEGER NOT NULL,
                    example_text TEXT,
                    FOREIGN KEY (pattern_id) REFERENCES identity_patterns(id) ON DELETE CASCADE,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            ''')
            print("    ✓ Created pattern tables")

            # Archive table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER UNIQUE NOT NULL,
                    full_content TEXT NOT NULL,
                    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            ''')
            print("    ✓ Created archive table")

            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    project_path TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    summary TEXT
                )
            ''')
            print("    ✓ Created sessions table")

            # System metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                INSERT OR REPLACE INTO system_metadata (key, value)
                VALUES ('schema_version', '2.0.0')
            ''')
            print("    ✓ Created metadata table")

            # Create indexes
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_project ON memories(project_path)',
                'CREATE INDEX IF NOT EXISTS idx_category ON memories(category)',
                'CREATE INDEX IF NOT EXISTS idx_cluster ON memories(cluster_id)',
                'CREATE INDEX IF NOT EXISTS idx_tree_path ON memories(tree_path)',
                'CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)',
                'CREATE INDEX IF NOT EXISTS idx_graph_source ON graph_edges(source_memory_id)',
                'CREATE INDEX IF NOT EXISTS idx_graph_target ON graph_edges(target_memory_id)',
            ]

            for idx_sql in indexes:
                cursor.execute(idx_sql)
            print("    ✓ Created indexes")

            conn.commit()
        finally:
            conn.close()

    def show_stats(self):
        """Show current database statistics."""
        print("\n" + "="*60)
        print("CURRENT MEMORY SYSTEM STATUS")
        print("="*60)

        if not self.db_path.exists():
            print("\n⚠ No database found!")
            print("   Run 'hard_reset' to initialize fresh V2 system")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get counts for all tables
        tables = {
            'Memories': 'memories',
            'Tree Nodes': 'memory_tree',
            'Graph Nodes': 'graph_nodes',
            'Graph Edges': 'graph_edges',
            'Graph Clusters': 'graph_clusters',
            'Identity Patterns': 'identity_patterns',
            'Archived Memories': 'memory_archive'
        }

        VALID_STAT_TABLES = frozenset(tables.values())  # Whitelist from hardcoded dict above

        print("\nTable Statistics:")
        for name, table in tables.items():
            try:
                if table not in VALID_STAT_TABLES:
                    raise ValueError(f"Invalid table name: {table}")
                cursor.execute(f'SELECT COUNT(*) FROM {table}')  # Safe: validated against whitelist
                count = cursor.fetchone()[0]
                print(f"  {name:20s}: {count:>5} rows")
            except sqlite3.OperationalError:
                print(f"  {name:20s}: Table not found")

        # Database size
        db_size = self.db_path.stat().st_size / (1024 * 1024)
        print(f"\nDatabase Size: {db_size:.2f} MB")

        # Backup count
        if self.backup_dir.exists():
            backup_count = len(list(self.backup_dir.glob("*.db")))
            print(f"Backups Available: {backup_count}")

        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='SuperLocalMemory V2 - Reset & Reinitialize Utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Show current status
  python memory-reset.py status

  # Soft reset (clear memories, keep schema)
  python memory-reset.py soft

  # Hard reset (delete everything, reinitialize)
  python memory-reset.py hard --confirm

  # Reset specific layers only
  python memory-reset.py layer --layers graph patterns

  # Reset without automatic backup
  python memory-reset.py soft --no-backup
        '''
    )

    parser.add_argument('command', choices=['status', 'soft', 'hard', 'layer'],
                       help='Reset command to execute')
    parser.add_argument('--layers', nargs='+',
                       choices=['graph', 'patterns', 'tree', 'archive'],
                       help='Layers to reset (for layer command)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip automatic backup')
    parser.add_argument('--confirm', action='store_true',
                       help='Confirm destructive operations (required for hard reset)')

    args = parser.parse_args()

    resetter = MemoryReset(auto_backup=not args.no_backup)

    if args.command == 'status':
        resetter.show_stats()

    elif args.command == 'soft':
        print("\n⚠️  WARNING: This will delete ALL memories but keep the V2 schema.")
        print("   Backup will be created automatically unless --no-backup specified.")
        response = input("\nProceed with soft reset? (yes/no): ")

        if response.lower() == 'yes':
            resetter.soft_reset()
        else:
            print("Cancelled.")

    elif args.command == 'hard':
        if not args.confirm:
            print("\n❌ ERROR: Hard reset requires --confirm flag")
            print("   This operation deletes EVERYTHING and reinitializes.")
            print("   Use: python memory-reset.py hard --confirm")
            sys.exit(1)

        print("\n⚠️  WARNING: This will DELETE the entire database!")
        print("   All memories, patterns, graphs will be permanently removed.")
        print("   A backup will be created automatically unless --no-backup specified.")
        response = input("\nType 'DELETE EVERYTHING' to confirm: ")

        if response == 'DELETE EVERYTHING':
            resetter.hard_reset()
        else:
            print("Cancelled.")

    elif args.command == 'layer':
        if not args.layers:
            print("❌ ERROR: --layers required for layer reset")
            print("   Example: python memory-reset.py layer --layers graph patterns")
            sys.exit(1)

        print(f"\n⚠️  WARNING: This will clear layers: {', '.join(args.layers)}")
        response = input("\nProceed? (yes/no): ")

        if response.lower() == 'yes':
            resetter.layer_reset(args.layers)
        else:
            print("Cancelled.")


if __name__ == '__main__':
    main()
