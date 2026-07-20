#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V2 - LlamaIndex Integration Example

Demonstrates how to use SuperLocalMemory V2 as a LlamaIndex chat store
backend. All data stays local in a SQLite database -- no API keys,
no cloud calls, no telemetry.

Usage:
    python examples/llamaindex_example.py

Requirements:
    pip install llama-index-core
    SuperLocalMemory V2 installed (https://github.com/qualixar/superlocalmemory)
"""
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Path setup -- allow importing from the integration source tree
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

# Add the LlamaIndex integration package to the import path
sys.path.insert(0, os.path.join(_REPO_ROOT, "integrations", "llamaindex"))

# Add the SLM core source so MemoryStoreV2 can be imported
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from llama_index.storage.chat_store.superlocalmemory import SuperLocalMemoryChatStore
from llama_index.core.llms import ChatMessage, MessageRole


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_messages(messages: list, label: str = "Messages") -> None:
    """Print a list of LlamaIndex ChatMessages in a readable format."""
    print(f"\n  {label} ({len(messages)} total):")
    if not messages:
        print("    (empty)")
        return
    for i, msg in enumerate(messages):
        role = msg.role.value
        # Truncate long content for display
        content = msg.content
        if len(content) > 80:
            content = content[:77] + "..."
        print(f"    [{i}] {role}: {content}")


def main() -> None:
    # Use a temporary directory so this example never touches the user's
    # real SuperLocalMemory database at ~/.claude-memory/memory.db
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "example_memory.db")
        print(f"Using temporary database: {db_path}")

        # ==================================================================
        # 1. Create a SuperLocalMemoryChatStore
        # ==================================================================
        print_header("1. Create Chat Store")

        store = SuperLocalMemoryChatStore(db_path=db_path)

        print(f"  Store class : {store.class_name()}")
        print(f"  DB Path     : {db_path}")
        print(f"  Active keys : {store.get_keys()}")

        # ==================================================================
        # 2. Add messages with add_message()
        # ==================================================================
        print_header("2. Add Messages with add_message()")

        store.add_message(
            "user-alice",
            ChatMessage(role=MessageRole.SYSTEM, content="You are a Python tutor."),
        )
        print("  Added: SYSTEM  -> 'You are a Python tutor.'")

        store.add_message(
            "user-alice",
            ChatMessage(role=MessageRole.USER, content="What is a list comprehension?"),
        )
        print("  Added: USER    -> 'What is a list comprehension?'")

        store.add_message(
            "user-alice",
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="A list comprehension is a concise way to create lists:\n"
                "  squares = [x**2 for x in range(10)]",
            ),
        )
        print("  Added: ASSISTANT -> 'A list comprehension is a concise way...'")

        # ==================================================================
        # 3. Get messages with get_messages()
        # ==================================================================
        print_header("3. Retrieve Messages with get_messages()")

        alice_messages = store.get_messages("user-alice")
        print_messages(alice_messages, "user-alice")

        # Verify role types
        print(f"\n  Role checks:")
        print(f"    messages[0].role == SYSTEM    : {alice_messages[0].role == MessageRole.SYSTEM}")
        print(f"    messages[1].role == USER      : {alice_messages[1].role == MessageRole.USER}")
        print(f"    messages[2].role == ASSISTANT  : {alice_messages[2].role == MessageRole.ASSISTANT}")

        # ==================================================================
        # 4. Use set_messages() to replace an entire session
        # ==================================================================
        print_header("4. Replace Messages with set_messages()")

        print(f"  Before set_messages: {len(store.get_messages('user-alice'))} messages")

        # Replace all messages with a fresh conversation
        store.set_messages("user-alice", [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a database expert."),
            ChatMessage(role=MessageRole.USER, content="What is SQLite?"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="SQLite is a self-contained, serverless, zero-configuration "
                "SQL database engine. It is the most widely deployed database "
                "in the world.",
            ),
            ChatMessage(role=MessageRole.USER, content="Is it good for local apps?"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Yes! SQLite excels at local, single-user applications. "
                "SuperLocalMemory V2 uses SQLite for exactly this reason.",
            ),
        ])

        print(f"  After set_messages : {len(store.get_messages('user-alice'))} messages")
        print_messages(store.get_messages("user-alice"), "Replaced Conversation")

        # ==================================================================
        # 5. Get keys with get_keys()
        # ==================================================================
        print_header("5. Get All Session Keys with get_keys()")

        # Add messages for a second user
        store.add_message(
            "user-bob",
            ChatMessage(role=MessageRole.USER, content="Hello from Bob!"),
        )
        store.add_message(
            "user-bob",
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi Bob, how can I help?"),
        )

        # Add messages for a third user
        store.add_message(
            "user-carol",
            ChatMessage(role=MessageRole.USER, content="Carol here."),
        )

        keys = store.get_keys()
        print(f"  Active session keys: {keys}")
        print(f"  Total sessions     : {len(keys)}")

        for key in keys:
            count = len(store.get_messages(key))
            print(f"    {key}: {count} messages")

        # ==================================================================
        # 6. Delete messages
        # ==================================================================
        print_header("6. Delete Messages")

        # 6a. Delete a specific message by index
        print("  --- Delete by index ---")
        bob_before = store.get_messages("user-bob")
        print(f"  user-bob before: {len(bob_before)} messages")
        print_messages(bob_before, "user-bob Before")

        deleted_msg = store.delete_message("user-bob", 0)
        print(f"\n  Deleted index 0: {deleted_msg.role.value} -> '{deleted_msg.content}'")

        bob_after = store.get_messages("user-bob")
        print(f"  user-bob after : {len(bob_after)} messages")
        print_messages(bob_after, "user-bob After")

        # 6b. Delete the last message
        print("\n  --- Delete last message ---")
        # Re-add a message so we have something to delete
        store.add_message(
            "user-bob",
            ChatMessage(role=MessageRole.USER, content="Another question from Bob."),
        )
        bob_current = store.get_messages("user-bob")
        print(f"  user-bob now has: {len(bob_current)} messages")

        last_deleted = store.delete_last_message("user-bob")
        print(f"  Deleted last   : {last_deleted.role.value} -> '{last_deleted.content}'")
        print(f"  user-bob now   : {len(store.get_messages('user-bob'))} messages")

        # 6c. Delete all messages for a key
        print("\n  --- Delete all messages for a key ---")
        print(f"  user-carol before: {len(store.get_messages('user-carol'))} messages")

        deleted_all = store.delete_messages("user-carol")
        print(f"  Deleted {len(deleted_all)} messages from user-carol")
        print(f"  user-carol after : {len(store.get_messages('user-carol'))} messages")

        print(f"\n  Remaining keys: {store.get_keys()}")

        # ==================================================================
        # 7. Session isolation
        # ==================================================================
        print_header("7. Session Isolation")

        # Reset for a clean demo
        store.delete_messages("user-alice")
        store.delete_messages("user-bob")

        store.set_messages("project-frontend", [
            ChatMessage(role=MessageRole.USER, content="How do I center a div?"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Use flexbox: display: flex; justify-content: center; align-items: center;"),
        ])

        store.set_messages("project-backend", [
            ChatMessage(role=MessageRole.USER, content="How do I connect to Postgres?"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Use psycopg2 or asyncpg for Python."),
            ChatMessage(role=MessageRole.USER, content="Which is faster?"),
        ])

        frontend_msgs = store.get_messages("project-frontend")
        backend_msgs = store.get_messages("project-backend")

        print(f"  project-frontend : {len(frontend_msgs)} messages")
        print(f"  project-backend  : {len(backend_msgs)} messages")
        print_messages(frontend_msgs, "project-frontend")
        print_messages(backend_msgs, "project-backend")

        # Verify isolation
        frontend_contents = {m.content for m in frontend_msgs}
        backend_contents = {m.content for m in backend_msgs}
        overlap = frontend_contents & backend_contents
        print(f"\n  Overlapping messages: {len(overlap)} (expected: 0)")
        assert len(overlap) == 0, "Sessions should be completely isolated!"
        print("  Sessions are fully isolated.")

        # ==================================================================
        # 8. Persistence across store instances
        # ==================================================================
        print_header("8. Persistence Across Instances")

        print("  Creating a NEW SuperLocalMemoryChatStore instance")
        print("  pointing at the SAME database file...")

        store_reloaded = SuperLocalMemoryChatStore(db_path=db_path)

        reloaded_keys = store_reloaded.get_keys()
        print(f"\n  Original store keys : {store.get_keys()}")
        print(f"  Reloaded store keys : {reloaded_keys}")
        print(f"  Keys match          : {store.get_keys() == reloaded_keys}")

        for key in reloaded_keys:
            original_count = len(store.get_messages(key))
            reloaded_count = len(store_reloaded.get_messages(key))
            status = "OK" if original_count == reloaded_count else "MISMATCH"
            print(f"    {key}: {reloaded_count} messages [{status}]")

        # ==================================================================
        # Summary
        # ==================================================================
        print_header("Summary")
        print("  SuperLocalMemory V2 + LlamaIndex integration provides:")
        print("    - Persistent chat store backed by local SQLite")
        print("    - Full ChatMessage support (USER, ASSISTANT, SYSTEM)")
        print("    - Session key isolation with get_keys() discovery")
        print("    - set_messages() for bulk replace, add_message() for append")
        print("    - delete_message(), delete_last_message(), delete_messages()")
        print("    - Zero cloud calls, zero API keys required")
        print("    - Data accessible from CLI, MCP, Skills, and REST API")
        print()


if __name__ == "__main__":
    main()
