# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""Tree Schema — Database initialization and root node management.

Provides the TreeSchemaMixin with DB init and root-node bootstrap logic
for the materialized-path tree structure.
"""
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime

MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"


class TreeSchemaMixin:
    """Database schema and root-node management for the memory tree."""

    def _init_db(self):
        """Initialize memory_tree table."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

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

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tree_path_layer2 ON memory_tree(tree_path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_type ON memory_tree(node_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_parent_id_tree ON memory_tree(parent_id)')

            conn.commit()
        finally:
            conn.close()

    def _ensure_root(self) -> int:
        """Ensure root node exists and return its ID."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute('SELECT id FROM memory_tree WHERE node_type = ? AND parent_id IS NULL', ('root',))
            result = cursor.fetchone()

            if result:
                root_id = result[0]
            else:
                cursor.execute('''
                    INSERT INTO memory_tree (node_type, name, tree_path, depth, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                ''', ('root', 'Root', '1', 0, datetime.now().isoformat()))
                root_id = cursor.lastrowid

                # Update tree_path with actual ID
                cursor.execute('UPDATE memory_tree SET tree_path = ? WHERE id = ?', (str(root_id), root_id))
                conn.commit()

        finally:
            conn.close()
        return root_id
