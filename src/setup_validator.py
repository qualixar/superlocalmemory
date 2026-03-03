#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V2 - Setup Validator

First-run setup validator that ensures all dependencies are installed
and the database is properly initialized.
"""
import sys
import os
import sqlite3
from pathlib import Path
from typing import Tuple, List, Optional

# Configuration
MEMORY_DIR = Path(os.environ.get('SUPERLOCALEMORY_DIR', str(Path.home() / ".claude-memory")))
DB_PATH = MEMORY_DIR / "memory.db"

# Required Python version
MIN_PYTHON_VERSION = (3, 8)

# Required tables for V2
REQUIRED_TABLES = [
    'memories',
    'graph_edges',
    'graph_nodes',
    'graph_clusters',
    'identity_patterns',
    'pattern_examples',
    'memory_tree',
    'memory_archive'
]


def print_banner():
    """Print product banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  SuperLocalMemory - Setup Validator                          ║
║  https://superlocalmemory.com                                ║
╚══════════════════════════════════════════════════════════════╝
""")


def check_python_version() -> Tuple[bool, str]:
    """Check Python version meets requirements."""
    current = sys.version_info[:2]
    if current >= MIN_PYTHON_VERSION:
        return True, f"Python {current[0]}.{current[1]}"
    else:
        return False, f"Python {current[0]}.{current[1]} (need {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+)"


def check_core_dependencies() -> List[Tuple[str, bool, str]]:
    """Check core required dependencies."""
    results = []

    # SQLite3 (built-in)
    try:
        import sqlite3
        results.append(("sqlite3", True, "Built-in"))
    except ImportError:
        results.append(("sqlite3", False, "pip install sqlite3"))

    # JSON (built-in)
    try:
        import json
        results.append(("json", True, "Built-in"))
    except ImportError:
        results.append(("json", False, "Built-in - should be available"))

    # hashlib (built-in)
    try:
        import hashlib
        results.append(("hashlib", True, "Built-in"))
    except ImportError:
        results.append(("hashlib", False, "Built-in - should be available"))

    return results


def check_optional_dependencies() -> List[Tuple[str, bool, str, str]]:
    """Check optional dependencies for advanced features."""
    results = []

    # scikit-learn (for knowledge graph)
    try:
        import sklearn
        results.append(("scikit-learn", True, sklearn.__version__, "Knowledge Graph"))
    except ImportError:
        results.append(("scikit-learn", False, "pip install scikit-learn", "Knowledge Graph"))

    # numpy (for vector operations)
    try:
        import numpy
        results.append(("numpy", True, numpy.__version__, "Vector Operations"))
    except ImportError:
        results.append(("numpy", False, "pip install numpy", "Vector Operations"))

    # igraph (for clustering)
    try:
        import igraph
        results.append(("python-igraph", True, igraph.__version__, "Graph Clustering"))
    except ImportError:
        results.append(("python-igraph", False, "pip install python-igraph", "Graph Clustering"))

    # leidenalg (for Leiden algorithm)
    try:
        import leidenalg
        results.append(("leidenalg", True, leidenalg.__version__, "Leiden Clustering"))
    except ImportError:
        results.append(("leidenalg", False, "pip install leidenalg", "Leiden Clustering"))

    # FastAPI (for UI server)
    try:
        import fastapi
        results.append(("fastapi", True, fastapi.__version__, "UI Server"))
    except ImportError:
        results.append(("fastapi", False, "pip install fastapi uvicorn", "UI Server"))

    return results


def check_database() -> Tuple[bool, str, List[str]]:
    """Check database status and tables."""
    if not DB_PATH.exists():
        return False, "Not created yet (will be created on first use)", []

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get existing tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        # Check for required tables
        missing_tables = []
        for table in REQUIRED_TABLES:
            if table not in existing_tables:
                missing_tables.append(table)

        # Get memory count
        try:
            cursor.execute("SELECT COUNT(*) FROM memories")
            memory_count = cursor.fetchone()[0]
        except:
            memory_count = 0

        conn.close()

        if missing_tables:
            return False, f"Missing tables: {', '.join(missing_tables)}", missing_tables
        else:
            return True, f"OK ({memory_count} memories)", []

    except Exception as e:
        return False, f"Error: {str(e)}", []


def check_directory_structure() -> List[Tuple[str, bool, str]]:
    """Check required directories exist."""
    results = []

    # Memory directory
    if MEMORY_DIR.exists():
        results.append(("Memory Directory", True, str(MEMORY_DIR)))
    else:
        results.append(("Memory Directory", False, f"Will be created at {MEMORY_DIR}"))

    # Profiles directory
    profiles_dir = MEMORY_DIR / "profiles"
    if profiles_dir.exists():
        results.append(("Profiles Directory", True, str(profiles_dir)))
    else:
        results.append(("Profiles Directory", False, "Will be created on first profile"))

    # Backups directory
    backups_dir = MEMORY_DIR / "backups"
    if backups_dir.exists():
        results.append(("Backups Directory", True, str(backups_dir)))
    else:
        results.append(("Backups Directory", False, "Will be created on first backup"))

    return results


def initialize_database() -> Tuple[bool, str]:
    """Initialize database with required schema if needed."""
    try:
        # Create memory directory if needed
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create memories table (core)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                summary TEXT,
                project_path TEXT,
                project_name TEXT,
                tags TEXT DEFAULT '[]',
                category TEXT,
                parent_id INTEGER,
                tree_path TEXT DEFAULT '/',
                depth INTEGER DEFAULT 0,
                memory_type TEXT DEFAULT 'session',
                importance INTEGER DEFAULT 5,
                content_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                compressed_at TIMESTAMP,
                tier INTEGER DEFAULT 1,
                cluster_id INTEGER,
                FOREIGN KEY (parent_id) REFERENCES memories(id)
            )
        ''')

        # Create graph tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER UNIQUE NOT NULL,
                entities TEXT DEFAULT '[]',
                embedding_vector BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_memory_id INTEGER NOT NULL,
                target_memory_id INTEGER NOT NULL,
                similarity REAL NOT NULL,
                relationship_type TEXT,
                shared_entities TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_memory_id) REFERENCES memories(id),
                FOREIGN KEY (target_memory_id) REFERENCES memories(id),
                UNIQUE(source_memory_id, target_memory_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_name TEXT,
                name TEXT,
                description TEXT,
                summary TEXT,
                memory_count INTEGER DEFAULT 0,
                member_count INTEGER DEFAULT 0,
                avg_importance REAL DEFAULT 5.0,
                top_entities TEXT DEFAULT '[]',
                parent_cluster_id INTEGER,
                depth INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_cluster_id) REFERENCES graph_clusters(id) ON DELETE SET NULL
            )
        ''')

        # Create pattern learning tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS identity_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_key TEXT NOT NULL,
                pattern_value TEXT,
                confidence REAL DEFAULT 0.0,
                frequency INTEGER DEFAULT 1,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pattern_type, pattern_key)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL,
                memory_id INTEGER NOT NULL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pattern_id) REFERENCES identity_patterns(id),
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
        ''')

        # Create tree table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_tree (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_type TEXT NOT NULL,
                name TEXT NOT NULL,
                parent_id INTEGER,
                tree_path TEXT DEFAULT '/',
                depth INTEGER DEFAULT 0,
                memory_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES memory_tree(id)
            )
        ''')

        # Create archive table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_memory_id INTEGER,
                compressed_content TEXT NOT NULL,
                compression_type TEXT DEFAULT 'tier2',
                original_size INTEGER,
                compressed_size INTEGER,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create system metadata table for watermarking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')

        # Add system watermark
        cursor.execute('''
            INSERT OR REPLACE INTO system_metadata (key, value) VALUES
            ('product', 'SuperLocalMemory'),
            ('website', 'https://superlocalmemory.com'),
            ('repository', 'https://github.com/varun369/SuperLocalMemoryV2'),
            ('license', 'MIT'),
            ('schema_version', '2.0.0')
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_cluster ON memories(cluster_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_memory_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_memory_id)')

        conn.commit()
        conn.close()

        return True, "Database initialized successfully"

    except Exception as e:
        return False, f"Error initializing database: {str(e)}"


def validate_setup(auto_fix: bool = False) -> bool:
    """
    Run complete setup validation.

    Args:
        auto_fix: If True, automatically initialize missing components

    Returns:
        True if all required checks pass
    """
    print_banner()

    all_passed = True
    optional_issues = []

    # Check Python version
    print("Checking Python version...")
    py_ok, py_msg = check_python_version()
    print(f"  {'✓' if py_ok else '✗'} {py_msg}")
    if not py_ok:
        all_passed = False

    print()

    # Check core dependencies
    print("Checking core dependencies...")
    for name, ok, msg in check_core_dependencies():
        print(f"  {'✓' if ok else '✗'} {name}: {msg}")
        if not ok:
            all_passed = False

    print()

    # Check optional dependencies
    print("Checking optional dependencies...")
    for name, ok, msg, feature in check_optional_dependencies():
        status = '✓' if ok else '○'
        print(f"  {status} {name}: {msg if ok else 'Not installed'} ({feature})")
        if not ok:
            optional_issues.append((name, msg, feature))

    print()

    # Check directory structure
    print("Checking directory structure...")
    for name, ok, msg in check_directory_structure():
        print(f"  {'✓' if ok else '○'} {name}: {msg}")

    print()

    # Check database
    print("Checking database...")
    db_ok, db_msg, missing_tables = check_database()
    print(f"  {'✓' if db_ok else '○'} Database: {db_msg}")

    if not db_ok and auto_fix:
        print("\n  Initializing database...")
        init_ok, init_msg = initialize_database()
        print(f"  {'✓' if init_ok else '✗'} {init_msg}")
        if init_ok:
            db_ok = True

    print()
    print("=" * 60)

    if all_passed and db_ok:
        print("\n✓ All required checks passed!")
        print("\nQuick Start Commands:")
        print("  1. Add a memory:")
        print("     superlocalmemoryv2-remember 'Your content here'")
        print("\n  2. Search memories:")
        print("     superlocalmemoryv2-recall 'search query'")
        print("\n  3. Build knowledge graph (after adding 2+ memories):")
        print("     python ~/.claude-memory/graph_engine.py build")
        print("\n  4. Start UI server:")
        print("     python ~/.claude-memory/api_server.py")
        print("\nDocumentation: https://github.com/varun369/SuperLocalMemoryV2")
        return True
    else:
        print("\n⚠ Some checks need attention:")
        if not all_passed:
            print("  - Fix required dependency issues above")
        if not db_ok:
            print("  - Database needs initialization")
            print("    Run: python setup_validator.py --init")

        if optional_issues:
            print("\nOptional (for full features):")
            print("  pip install scikit-learn numpy python-igraph leidenalg fastapi uvicorn")

        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SuperLocalMemory V2 Setup Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_validator.py          # Check setup
  python setup_validator.py --init   # Check and initialize database

Author: Varun Pratap Bhardwaj
Repository: https://github.com/varun369/SuperLocalMemoryV2
        """
    )
    parser.add_argument(
        '--init', '-i',
        action='store_true',
        help='Initialize database if needed'
    )

    args = parser.parse_args()
    success = validate_setup(auto_fix=args.init)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
