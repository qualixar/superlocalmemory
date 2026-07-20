#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V2 - LangChain Integration Example

Demonstrates how to use SuperLocalMemory V2 as a LangChain chat message
history backend. All data stays local in a SQLite database -- no API keys,
no cloud calls, no telemetry.

Usage:
    python examples/langchain_example.py

Requirements:
    pip install langchain-core
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

# Add the LangChain integration package to the import path
sys.path.insert(0, os.path.join(_REPO_ROOT, "integrations", "langchain"))

# Add the SLM core source so MemoryStoreV2 can be imported
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from langchain_superlocalmemory import SuperLocalMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_messages(messages: list, label: str = "Messages") -> None:
    """Print a list of LangChain messages in a readable format."""
    print(f"\n  {label} ({len(messages)} total):")
    if not messages:
        print("    (empty)")
        return
    for i, msg in enumerate(messages):
        role = type(msg).__name__.replace("Message", "")
        print(f"    [{i}] {role}: {msg.content}")


def main() -> None:
    # Use a temporary directory so this example never touches the user's
    # real SuperLocalMemory database at ~/.claude-memory/memory.db
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "example_memory.db")
        print(f"Using temporary database: {db_path}")

        # ==================================================================
        # 1. Create a chat message history for a session
        # ==================================================================
        print_header("1. Create Chat Message History")

        history = SuperLocalMemoryChatMessageHistory(
            session_id="demo-session-001",
            db_path=db_path,
        )

        print(f"  Session ID : demo-session-001")
        print(f"  DB Path    : {db_path}")
        print(f"  Messages   : {len(history.messages)}")

        # ==================================================================
        # 2. Add messages (HumanMessage, AIMessage, SystemMessage)
        # ==================================================================
        print_header("2. Add Messages")

        # Add a system message first
        history.add_messages([
            SystemMessage(content="You are a helpful coding assistant."),
        ])
        print("  Added: SystemMessage('You are a helpful coding assistant.')")

        # Add a conversation exchange
        history.add_messages([
            HumanMessage(content="How do I read a file in Python?"),
            AIMessage(content="Use the built-in open() function with a context manager:\n"
                      "  with open('file.txt', 'r') as f:\n"
                      "      content = f.read()"),
        ])
        print("  Added: HumanMessage('How do I read a file in Python?')")
        print("  Added: AIMessage('Use the built-in open() function...')")

        # Add another turn
        history.add_messages([
            HumanMessage(content="What about reading line by line?"),
            AIMessage(content="Use readlines() or iterate over the file object:\n"
                      "  for line in f:\n"
                      "      print(line.strip())"),
        ])
        print("  Added: HumanMessage('What about reading line by line?')")
        print("  Added: AIMessage('Use readlines() or iterate...')")

        # ==================================================================
        # 3. Retrieve messages
        # ==================================================================
        print_header("3. Retrieve Messages")

        messages = history.messages
        print_messages(messages)

        # Verify types
        print(f"\n  Type checks:")
        print(f"    messages[0] is SystemMessage : {isinstance(messages[0], SystemMessage)}")
        print(f"    messages[1] is HumanMessage  : {isinstance(messages[1], HumanMessage)}")
        print(f"    messages[2] is AIMessage     : {isinstance(messages[2], AIMessage)}")

        # ==================================================================
        # 4. Demonstrate persistence (new instance, same session_id)
        # ==================================================================
        print_header("4. Persistence Across Instances")

        print("  Creating a NEW SuperLocalMemoryChatMessageHistory instance")
        print("  with the SAME session_id and db_path...")

        history_reloaded = SuperLocalMemoryChatMessageHistory(
            session_id="demo-session-001",
            db_path=db_path,
        )

        reloaded_messages = history_reloaded.messages
        print(f"\n  Original instance had  : {len(messages)} messages")
        print(f"  Reloaded instance has  : {len(reloaded_messages)} messages")
        print(f"  Data persisted         : {len(messages) == len(reloaded_messages)}")
        print_messages(reloaded_messages, "Reloaded Messages")

        # ==================================================================
        # 5. Session isolation (different session_id = different messages)
        # ==================================================================
        print_header("5. Session Isolation")

        # Create a second session using the same database
        history_other = SuperLocalMemoryChatMessageHistory(
            session_id="other-session-002",
            db_path=db_path,
        )

        history_other.add_messages([
            HumanMessage(content="This message belongs to session 002 only."),
            AIMessage(content="Confirmed -- session 002 is isolated."),
        ])

        session_001_msgs = history.messages
        session_002_msgs = history_other.messages

        print(f"  Session 'demo-session-001' : {len(session_001_msgs)} messages")
        print(f"  Session 'other-session-002': {len(session_002_msgs)} messages")
        print_messages(session_001_msgs, "Session 001")
        print_messages(session_002_msgs, "Session 002")

        # Verify no cross-contamination
        session_001_contents = {m.content for m in session_001_msgs}
        session_002_contents = {m.content for m in session_002_msgs}
        overlap = session_001_contents & session_002_contents
        print(f"\n  Overlapping messages: {len(overlap)} (expected: 0)")
        assert len(overlap) == 0, "Sessions should be completely isolated!"
        print("  Sessions are fully isolated.")

        # ==================================================================
        # 6. Clear a session
        # ==================================================================
        print_header("6. Clear a Session")

        print(f"  Before clear -- session 001: {len(history.messages)} messages")
        print(f"  Before clear -- session 002: {len(history_other.messages)} messages")

        history.clear()

        print(f"\n  Cleared session 'demo-session-001'.")
        print(f"\n  After clear  -- session 001: {len(history.messages)} messages")
        print(f"  After clear  -- session 002: {len(history_other.messages)} messages")

        # Verify the other session is untouched
        assert len(history.messages) == 0, "Cleared session should be empty"
        assert len(history_other.messages) == 2, "Other session should be untouched"
        print("\n  Clear only affected the target session. Other session intact.")

        # ==================================================================
        # Summary
        # ==================================================================
        print_header("Summary")
        print("  SuperLocalMemory V2 + LangChain integration provides:")
        print("    - Persistent chat history backed by local SQLite")
        print("    - Full message type support (Human, AI, System)")
        print("    - Session isolation via tagged memory entries")
        print("    - Zero cloud calls, zero API keys required")
        print("    - Data accessible from CLI, MCP, Skills, and REST API")
        print()


if __name__ == "__main__":
    main()
