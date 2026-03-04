#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Command-line interface for MemoryStoreV2.

This module contains the CLI implementation extracted from memory_store_v2.py
to reduce file size and improve maintainability.
"""

import sys
import json
from .helpers import format_content


def run_cli():
    """Main CLI entry point for MemoryStoreV2."""
    # Import here to avoid circular dependency
    from memory_store_v2 import MemoryStoreV2

    store = MemoryStoreV2()

    if len(sys.argv) < 2:
        print("MemoryStore V2 CLI")
        print("\nV1 Compatible Commands:")
        print("  python memory_store_v2.py add <content> [--project <path>] [--tags tag1,tag2]")
        print("  python memory_store_v2.py search <query> [--full]")
        print("  python memory_store_v2.py list [limit] [--full]")
        print("  python memory_store_v2.py get <id>")
        print("  python memory_store_v2.py recent [limit] [--full]")
        print("  python memory_store_v2.py stats")
        print("  python memory_store_v2.py context <query>")
        print("  python memory_store_v2.py delete <id>")
        print("\nV2 Extensions:")
        print("  python memory_store_v2.py tree [parent_id]")
        print("  python memory_store_v2.py cluster <cluster_id> [--full]")
        print("\nOptions:")
        print("  --full    Show complete content (default: smart truncation at 5000 chars)")
        sys.exit(0)

    command = sys.argv[1]

    if command == "tree":
        parent_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
        results = store.get_tree(parent_id)

        if not results:
            print("No memories in tree.")
        else:
            for r in results:
                indent = "  " * r['depth']
                print(f"{indent}[{r['id']}] {r['content'][:50]}...")
                if r.get('category'):
                    print(f"{indent}    Category: {r['category']}")

    elif command == "cluster" and len(sys.argv) >= 3:
        cluster_id = int(sys.argv[2])
        show_full = '--full' in sys.argv
        results = store.get_by_cluster(cluster_id)

        if not results:
            print(f"No memories in cluster {cluster_id}.")
        else:
            print(f"Cluster {cluster_id} - {len(results)} memories:")
            for r in results:
                print(f"\n[{r['id']}] Importance: {r['importance']}")
                print(f"  {format_content(r['content'], full=show_full)}")

    elif command == "stats":
        stats = store.get_stats()
        print(json.dumps(stats, indent=2))

    elif command == "add":
        # Parse content and options
        if len(sys.argv) < 3:
            print("Error: Content required")
            print("Usage: python memory_store_v2.py add <content> [--project <path>] [--tags tag1,tag2]")
            sys.exit(1)

        content = sys.argv[2]
        project_path = None
        tags = []

        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == '--project' and i + 1 < len(sys.argv):
                project_path = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--tags' and i + 1 < len(sys.argv):
                tags = [t.strip() for t in sys.argv[i + 1].split(',')]
                i += 2
            else:
                i += 1

        mem_id = store.add_memory(content, project_path=project_path, tags=tags)
        print(f"Memory added with ID: {mem_id}")

    elif command == "search":
        if len(sys.argv) < 3:
            print("Error: Search query required")
            print("Usage: python memory_store_v2.py search <query> [--full]")
            sys.exit(1)

        query = sys.argv[2]
        show_full = '--full' in sys.argv
        results = store.search(query, limit=5)

        if not results:
            print("No results found.")
        else:
            for r in results:
                print(f"\n[{r['id']}] Score: {r['score']:.2f}")
                if r.get('project_name'):
                    print(f"Project: {r['project_name']}")
                if r.get('tags'):
                    print(f"Tags: {', '.join(r['tags'])}")
                print(f"Content: {format_content(r['content'], full=show_full)}")
                print(f"Created: {r['created_at']}")

    elif command == "recent":
        show_full = '--full' in sys.argv
        # Parse limit (skip --full flag)
        limit = 10
        for i, arg in enumerate(sys.argv[2:], start=2):
            if arg != '--full' and arg.isdigit():
                limit = int(arg)
                break

        results = store.get_recent(limit)

        if not results:
            print("No memories found.")
        else:
            for r in results:
                print(f"\n[{r['id']}] {r['created_at']}")
                if r.get('project_name'):
                    print(f"Project: {r['project_name']}")
                if r.get('tags'):
                    print(f"Tags: {', '.join(r['tags'])}")
                print(f"Content: {format_content(r['content'], full=show_full)}")

    elif command == "list":
        show_full = '--full' in sys.argv
        # Parse limit (skip --full flag)
        limit = 10
        for i, arg in enumerate(sys.argv[2:], start=2):
            if arg != '--full' and arg.isdigit():
                limit = int(arg)
                break

        results = store.get_recent(limit)

        if not results:
            print("No memories found.")
        else:
            for r in results:
                print(f"[{r['id']}] {format_content(r['content'], full=show_full)}")

    elif command == "get":
        if len(sys.argv) < 3:
            print("Error: Memory ID required")
            print("Usage: python memory_store_v2.py get <id>")
            sys.exit(1)

        mem_id = int(sys.argv[2])
        memory = store.get_by_id(mem_id)

        if not memory:
            print(f"Memory {mem_id} not found.")
        else:
            print(f"\nID: {memory['id']}")
            print(f"Content: {memory['content']}")
            if memory.get('summary'):
                print(f"Summary: {memory['summary']}")
            if memory.get('project_name'):
                print(f"Project: {memory['project_name']}")
            if memory.get('tags'):
                print(f"Tags: {', '.join(memory['tags'])}")
            print(f"Created: {memory['created_at']}")
            print(f"Importance: {memory['importance']}")
            print(f"Access Count: {memory['access_count']}")

    elif command == "context":
        if len(sys.argv) < 3:
            print("Error: Query required")
            print("Usage: python memory_store_v2.py context <query>")
            sys.exit(1)

        query = sys.argv[2]
        context = store.export_for_context(query)
        print(context)

    elif command == "delete":
        if len(sys.argv) < 3:
            print("Error: Memory ID required")
            print("Usage: python memory_store_v2.py delete <id>")
            sys.exit(1)

        mem_id = int(sys.argv[2])
        store.delete_memory(mem_id)
        print(f"Memory {mem_id} deleted.")

    else:
        print(f"Unknown command: {command}")
        print("Run without arguments to see available commands.")
