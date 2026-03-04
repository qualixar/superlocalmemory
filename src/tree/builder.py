# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""Tree Builder — Constructs the full tree from the memories table.

Provides TreeBuilderMixin with build_tree, plus the CLI entry point.
"""
import sqlite3


class TreeBuilderMixin:
    """Builds the hierarchical tree from flat memory records."""

    def build_tree(self):
        """
        Build complete tree structure from memories table.

        Process:
        1. Clear existing tree (except root)
        2. Group memories by project
        3. Group by category within projects
        4. Link individual memories as leaf nodes
        5. Update aggregated counts
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Clear existing tree (keep root)
            cursor.execute('DELETE FROM memory_tree WHERE node_type != ?', ('root',))

            # Step 1: Create project nodes
            cursor.execute('''
                SELECT DISTINCT project_path, project_name
                FROM memories
                WHERE project_path IS NOT NULL
                ORDER BY project_path
            ''')
            projects = cursor.fetchall()

            project_map = {}  # project_path -> node_id

            for project_path, project_name in projects:
                name = project_name or project_path.split('/')[-1]
                node_id = self.add_node('project', name, self.root_id, description=project_path)
                project_map[project_path] = node_id

            # Step 2: Create category nodes within projects
            cursor.execute('''
                SELECT DISTINCT project_path, category
                FROM memories
                WHERE project_path IS NOT NULL AND category IS NOT NULL
                ORDER BY project_path, category
            ''')
            categories = cursor.fetchall()

            category_map = {}  # (project_path, category) -> node_id

            for project_path, category in categories:
                parent_id = project_map.get(project_path)
                if parent_id:
                    node_id = self.add_node('category', category, parent_id)
                    category_map[(project_path, category)] = node_id

            # Step 3: Link memories as leaf nodes
            cursor.execute('''
                SELECT id, content, summary, project_path, category, importance, created_at
                FROM memories
                ORDER BY created_at DESC
            ''')
            memories = cursor.fetchall()

            for mem_id, content, summary, project_path, category, importance, created_at in memories:
                # Determine parent node
                if project_path and category and (project_path, category) in category_map:
                    parent_id = category_map[(project_path, category)]
                elif project_path and project_path in project_map:
                    parent_id = project_map[project_path]
                else:
                    parent_id = self.root_id

                # Create memory node
                name = summary or content[:60].replace('\n', ' ')
                self.add_node('memory', name, parent_id, memory_id=mem_id, description=content[:200])

            # Step 4: Update aggregated counts
            self._update_all_counts()

            conn.commit()
        finally:
            conn.close()


def run_cli():
    """CLI entry point for tree_manager."""
    import sys
    import json
    from src.tree import TreeManager

    tree_mgr = TreeManager()

    if len(sys.argv) < 2:
        print("TreeManager CLI")
        print("\nCommands:")
        print("  python tree_manager.py build                  # Build tree from memories")
        print("  python tree_manager.py show [project] [depth] # Show tree structure")
        print("  python tree_manager.py subtree <node_id>      # Get subtree")
        print("  python tree_manager.py path <node_id>         # Get path to root")
        print("  python tree_manager.py stats                  # Show statistics")
        print("  python tree_manager.py add <type> <name> <parent_id>  # Add node")
        print("  python tree_manager.py delete <node_id>       # Delete node")
        sys.exit(0)

    command = sys.argv[1]

    if command == "build":
        print("Building tree from memories...")
        tree_mgr.build_tree()
        stats = tree_mgr.get_stats()
        print(f"Tree built: {stats['total_nodes']} nodes, {stats['total_memories']} memories")

    elif command == "show":
        project = sys.argv[2] if len(sys.argv) > 2 else None
        max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else None

        tree = tree_mgr.get_tree(project, max_depth)

        def print_tree(node, indent=0):
            if 'error' in node:
                print(node['error'])
                return

            prefix = "  " * indent
            icon = {"root": "\U0001f333", "project": "\U0001f4c1", "category": "\U0001f4c2", "memory": "\U0001f4c4"}.get(node['type'], "\u2022")

            print(f"{prefix}{icon} {node['name']} (id={node['id']}, memories={node['memory_count']})")

            for child in node.get('children', []):
                print_tree(child, indent + 1)

        print_tree(tree)

    elif command == "subtree" and len(sys.argv) >= 3:
        node_id = int(sys.argv[2])
        nodes = tree_mgr.get_subtree(node_id)

        if not nodes:
            print(f"No subtree found for node {node_id}")
        else:
            print(f"Subtree of node {node_id}:")
            for node in nodes:
                indent = "  " * (node['depth'] - nodes[0]['depth'] + 1)
                print(f"{indent}- {node['name']} (id={node['id']})")

    elif command == "path" and len(sys.argv) >= 3:
        node_id = int(sys.argv[2])
        path = tree_mgr.get_path_to_root(node_id)

        if not path:
            print(f"Node {node_id} not found")
        else:
            print("Path to root:")
            print(" > ".join([f"{n['name']} (id={n['id']})" for n in path]))

    elif command == "stats":
        stats = tree_mgr.get_stats()
        print(json.dumps(stats, indent=2))

    elif command == "add" and len(sys.argv) >= 5:
        node_type = sys.argv[2]
        name = sys.argv[3]
        parent_id = int(sys.argv[4])

        node_id = tree_mgr.add_node(node_type, name, parent_id)
        print(f"Node created with ID: {node_id}")

    elif command == "delete" and len(sys.argv) >= 3:
        node_id = int(sys.argv[2])
        if tree_mgr.delete_node(node_id):
            print(f"Node {node_id} deleted")
        else:
            print(f"Node {node_id} not found")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
