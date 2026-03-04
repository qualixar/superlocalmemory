# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""Tree Nodes — CRUD operations and count aggregation.

Provides TreeNodesMixin with add_node, delete_node, update_counts,
and internal helper methods for materialized-path management.
"""
import sqlite3
from typing import Optional, Dict, Any
from datetime import datetime


class TreeNodesMixin:
    """Node CRUD and aggregation logic for the memory tree."""

    def add_node(
        self,
        node_type: str,
        name: str,
        parent_id: int,
        description: Optional[str] = None,
        memory_id: Optional[int] = None
    ) -> int:
        """
        Add a new node to the tree.

        Args:
            node_type: Type of node ('root', 'project', 'category', 'memory')
            name: Display name
            parent_id: Parent node ID
            description: Optional description
            memory_id: Link to memories table (for leaf nodes)

        Returns:
            New node ID
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get parent path and depth
            cursor.execute('SELECT tree_path, depth FROM memory_tree WHERE id = ?', (parent_id,))
            result = cursor.fetchone()

            if not result:
                raise ValueError(f"Parent node {parent_id} not found")

            parent_path, parent_depth = result

            # Calculate new node position
            depth = parent_depth + 1

            cursor.execute('''
                INSERT INTO memory_tree (
                    node_type, name, description,
                    parent_id, tree_path, depth,
                    memory_id, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_type,
                name,
                description,
                parent_id,
                '',  # Placeholder, updated below
                depth,
                memory_id,
                datetime.now().isoformat()
            ))

            node_id = cursor.lastrowid

            # Update tree_path with actual node_id
            tree_path = f"{parent_path}.{node_id}"
            cursor.execute('UPDATE memory_tree SET tree_path = ? WHERE id = ?', (tree_path, node_id))

            conn.commit()
        finally:
            conn.close()

        return node_id

    def delete_node(self, node_id: int) -> bool:
        """
        Delete a node and all its descendants.

        Args:
            node_id: Node ID to delete

        Returns:
            True if deleted, False if not found
        """
        if node_id == self.root_id:
            raise ValueError("Cannot delete root node")

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get tree_path
            cursor.execute('SELECT tree_path, parent_id FROM memory_tree WHERE id = ?', (node_id,))
            result = cursor.fetchone()

            if not result:
                return False

            tree_path, parent_id = result

            # Delete node and all descendants (CASCADE handles children)
            cursor.execute('DELETE FROM memory_tree WHERE id = ? OR tree_path LIKE ?',
                          (node_id, f"{tree_path}.%"))

            deleted = cursor.rowcount > 0
            conn.commit()
        finally:
            conn.close()

        # Update parent counts
        if deleted and parent_id:
            self.update_counts(parent_id)

        return deleted

    def update_counts(self, node_id: int):
        """
        Update aggregated counts for a node (memory_count, total_size).
        Recursively updates all ancestors.

        Args:
            node_id: Node ID to update
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get all descendant memory nodes
            cursor.execute('SELECT tree_path FROM memory_tree WHERE id = ?', (node_id,))
            result = cursor.fetchone()

            if not result:
                return

            tree_path = result[0]

            # Count memories in subtree
            cursor.execute('''
                SELECT COUNT(*), COALESCE(SUM(LENGTH(m.content)), 0)
                FROM memory_tree t
                LEFT JOIN memories m ON t.memory_id = m.id
                WHERE t.tree_path LIKE ? AND t.memory_id IS NOT NULL
            ''', (f"{tree_path}%",))

            memory_count, total_size = cursor.fetchone()

            # Update node
            cursor.execute('''
                UPDATE memory_tree
                SET memory_count = ?, total_size = ?, last_updated = ?
                WHERE id = ?
            ''', (memory_count, total_size, datetime.now().isoformat(), node_id))

            # Update all ancestors
            path_ids = [int(x) for x in tree_path.split('.')]
            for ancestor_id in path_ids[:-1]:  # Exclude current node
                self.update_counts(ancestor_id)

            conn.commit()
        finally:
            conn.close()

    def _update_all_counts(self):
        """Update counts for all nodes (used after build_tree)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get all nodes in reverse depth order (leaves first)
            cursor.execute('''
                SELECT id FROM memory_tree
                ORDER BY depth DESC
            ''')

            node_ids = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

        # Update each node (will cascade to parents)
        processed = set()
        for node_id in node_ids:
            if node_id not in processed:
                self.update_counts(node_id)
                processed.add(node_id)

    def _generate_tree_path(self, parent_path: str, node_id: int) -> str:
        """Generate tree_path for a new node."""
        if parent_path:
            return f"{parent_path}.{node_id}"
        return str(node_id)

    def _calculate_depth(self, tree_path: str) -> int:
        """Calculate depth from tree_path (count dots)."""
        return tree_path.count('.')
