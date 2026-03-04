# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""Tree Queries — Read-only tree traversal and statistics.

Provides TreeQueriesMixin with get_tree, get_subtree,
get_path_to_root, and get_stats.
"""
import sqlite3
from typing import Optional, List, Dict, Any


class TreeQueriesMixin:
    """Read-only query methods for the memory tree."""

    def get_tree(
        self,
        project_name: Optional[str] = None,
        max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get tree structure as nested dictionary.

        Args:
            project_name: Filter by specific project
            max_depth: Maximum depth to retrieve

        Returns:
            Nested dictionary representing tree structure
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Build query
            if project_name:
                # Find project node
                cursor.execute('''
                    SELECT id, tree_path FROM memory_tree
                    WHERE node_type = 'project' AND name = ?
                ''', (project_name,))
                result = cursor.fetchone()

                if not result:
                    return {'error': f"Project '{project_name}' not found"}

                project_id, project_path = result

                # Get subtree
                if max_depth is not None:
                    cursor.execute('''
                        SELECT id, node_type, name, description, parent_id, tree_path,
                               depth, memory_count, total_size, last_updated, memory_id
                        FROM memory_tree
                        WHERE (id = ? OR tree_path LIKE ?) AND depth <= ?
                        ORDER BY tree_path
                    ''', (project_id, f"{project_path}.%", max_depth))
                else:
                    cursor.execute('''
                        SELECT id, node_type, name, description, parent_id, tree_path,
                               depth, memory_count, total_size, last_updated, memory_id
                        FROM memory_tree
                        WHERE id = ? OR tree_path LIKE ?
                        ORDER BY tree_path
                    ''', (project_id, f"{project_path}.%"))
            else:
                # Get entire tree
                if max_depth is not None:
                    cursor.execute('''
                        SELECT id, node_type, name, description, parent_id, tree_path,
                               depth, memory_count, total_size, last_updated, memory_id
                        FROM memory_tree
                        WHERE depth <= ?
                        ORDER BY tree_path
                    ''', (max_depth,))
                else:
                    cursor.execute('''
                        SELECT id, node_type, name, description, parent_id, tree_path,
                               depth, memory_count, total_size, last_updated, memory_id
                        FROM memory_tree
                        ORDER BY tree_path
                    ''')

            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return {'error': 'No tree nodes found'}

        # Build nested structure
        nodes = {}
        root = None

        for row in rows:
            node = {
                'id': row[0],
                'type': row[1],
                'name': row[2],
                'description': row[3],
                'parent_id': row[4],
                'tree_path': row[5],
                'depth': row[6],
                'memory_count': row[7],
                'total_size': row[8],
                'last_updated': row[9],
                'memory_id': row[10],
                'children': []
            }
            nodes[node['id']] = node

            if node['parent_id'] is None or (project_name and node['type'] == 'project'):
                root = node
            elif node['parent_id'] in nodes:
                nodes[node['parent_id']]['children'].append(node)

        return root or {'error': 'Root node not found'}

    def get_subtree(self, node_id: int) -> List[Dict[str, Any]]:
        """
        Get all descendants of a specific node (flat list).

        Args:
            node_id: Node ID to get subtree for

        Returns:
            List of descendant nodes
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get node's tree_path
            cursor.execute('SELECT tree_path FROM memory_tree WHERE id = ?', (node_id,))
            result = cursor.fetchone()

            if not result:
                return []

            tree_path = result[0]

            # Get all descendants
            cursor.execute('''
                SELECT id, node_type, name, description, parent_id, tree_path,
                       depth, memory_count, total_size, last_updated, memory_id
                FROM memory_tree
                WHERE tree_path LIKE ?
                ORDER BY tree_path
            ''', (f"{tree_path}.%",))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'type': row[1],
                    'name': row[2],
                    'description': row[3],
                    'parent_id': row[4],
                    'tree_path': row[5],
                    'depth': row[6],
                    'memory_count': row[7],
                    'total_size': row[8],
                    'last_updated': row[9],
                    'memory_id': row[10]
                })
        finally:
            conn.close()

        return results

    def get_path_to_root(self, node_id: int) -> List[Dict[str, Any]]:
        """
        Get path from node to root (breadcrumb trail).

        Args:
            node_id: Starting node ID

        Returns:
            List of nodes from root to target node
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get node's tree_path
            cursor.execute('SELECT tree_path FROM memory_tree WHERE id = ?', (node_id,))
            result = cursor.fetchone()

            if not result:
                return []

            tree_path = result[0]

            # Parse path to get all ancestor IDs
            path_ids = [int(x) for x in tree_path.split('.')]

            # Get all ancestor nodes
            placeholders = ','.join('?' * len(path_ids))
            cursor.execute(f'''
                SELECT id, node_type, name, description, parent_id, tree_path,
                       depth, memory_count, total_size, last_updated, memory_id
                FROM memory_tree
                WHERE id IN ({placeholders})
                ORDER BY depth
            ''', path_ids)

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'type': row[1],
                    'name': row[2],
                    'description': row[3],
                    'parent_id': row[4],
                    'tree_path': row[5],
                    'depth': row[6],
                    'memory_count': row[7],
                    'total_size': row[8],
                    'last_updated': row[9],
                    'memory_id': row[10]
                })
        finally:
            conn.close()

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get tree statistics."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM memory_tree')
            total_nodes = cursor.fetchone()[0]

            cursor.execute('SELECT node_type, COUNT(*) FROM memory_tree GROUP BY node_type')
            by_type = dict(cursor.fetchall())

            cursor.execute('SELECT MAX(depth) FROM memory_tree')
            max_depth = cursor.fetchone()[0] or 0

            cursor.execute('''
                SELECT SUM(memory_count), SUM(total_size)
                FROM memory_tree
                WHERE node_type = 'root'
            ''')
            total_memories, total_size = cursor.fetchone()

        finally:
            conn.close()

        return {
            'total_nodes': total_nodes,
            'by_type': by_type,
            'max_depth': max_depth,
            'total_memories': total_memories or 0,
            'total_size_bytes': total_size or 0
        }
