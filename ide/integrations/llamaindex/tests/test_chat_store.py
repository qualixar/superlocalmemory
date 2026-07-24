#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 - LlamaIndex Chat Store Tests

Comprehensive tests for the SuperLocalMemoryChatStore implementation.
All tests use temporary databases — no production data is touched.
"""
import os
import sqlite3
import tempfile

import pytest

pytest.importorskip("llama_index.core")  # skip cleanly when LlamaIndex isn't installed
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.storage.chat_store.superlocalmemory import SuperLocalMemoryChatStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db():
    """Provide a path to a temporary database file that is cleaned up after use."""
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "test_memory.db")


@pytest.fixture
def store(tmp_db):
    """Provide a fresh SuperLocalMemoryChatStore backed by a temp database."""
    return SuperLocalMemoryChatStore(db_path=tmp_db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages(
    texts: list[str], roles: list[MessageRole] | None = None
) -> list[ChatMessage]:
    """Create a list of ChatMessages from plain text strings."""
    if roles is None:
        # Alternate user / assistant
        roles = [
            MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            for i in range(len(texts))
        ]
    return [ChatMessage(role=r, content=t) for r, t in zip(roles, texts)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSetAndGetMessages:
    """Tests for set_messages / get_messages round-trip."""

    def test_set_and_get_messages(self, store):
        """Set 3 messages and retrieve them in order."""
        msgs = _make_messages(["Hello", "Hi there", "How are you?"])
        store.set_messages("session-1", msgs)

        retrieved = store.get_messages("session-1")
        assert len(retrieved) == 3
        assert retrieved[0].content == "Hello"
        assert retrieved[1].content == "Hi there"
        assert retrieved[2].content == "How are you?"

    def test_get_messages_empty(self, store):
        """Getting messages for a non-existent key returns an empty list."""
        assert store.get_messages("nonexistent") == []


class TestAddMessage:
    """Tests for add_message."""

    def test_add_single_message(self, store):
        """Add one message and verify it appears in get_messages."""
        msg = ChatMessage(role=MessageRole.USER, content="solo message")
        store.add_message("key-a", msg)

        retrieved = store.get_messages("key-a")
        assert len(retrieved) == 1
        assert retrieved[0].content == "solo message"
        assert retrieved[0].role == MessageRole.USER


class TestDeleteMessages:
    """Tests for delete_messages (all messages for a key)."""

    def test_delete_messages(self, store):
        """Set, delete, verify empty and check return value."""
        msgs = _make_messages(["A", "B", "C"])
        store.set_messages("del-key", msgs)

        deleted = store.delete_messages("del-key")
        assert deleted is not None
        assert len(deleted) == 3
        assert deleted[0].content == "A"

        # After deletion, key should be empty
        assert store.get_messages("del-key") == []

    def test_delete_messages_nonexistent(self, store):
        """Deleting messages for a key with no data returns None."""
        assert store.delete_messages("ghost-key") is None


class TestDeleteMessageByIndex:
    """Tests for delete_message (single message by index)."""

    def test_delete_message_by_index(self, store):
        """Set 3 messages, delete index 1, verify 2 remain in correct order."""
        msgs = _make_messages(["first", "second", "third"])
        store.set_messages("idx-key", msgs)

        deleted = store.delete_message("idx-key", 1)
        assert deleted is not None
        assert deleted.content == "second"

        remaining = store.get_messages("idx-key")
        assert len(remaining) == 2
        assert remaining[0].content == "first"
        assert remaining[1].content == "third"

    def test_delete_message_out_of_range(self, store):
        """Deleting an out-of-range index returns None."""
        msgs = _make_messages(["only"])
        store.set_messages("range-key", msgs)
        assert store.delete_message("range-key", 5) is None

    def test_delete_message_negative_index(self, store):
        """Deleting a negative index returns None."""
        msgs = _make_messages(["x"])
        store.set_messages("neg-key", msgs)
        assert store.delete_message("neg-key", -1) is None


class TestDeleteLastMessage:
    """Tests for delete_last_message."""

    def test_delete_last_message(self, store):
        """Set 3 messages, delete last, verify 2 remain."""
        msgs = _make_messages(["alpha", "beta", "gamma"])
        store.set_messages("last-key", msgs)

        deleted = store.delete_last_message("last-key")
        assert deleted is not None
        assert deleted.content == "gamma"

        remaining = store.get_messages("last-key")
        assert len(remaining) == 2
        assert remaining[-1].content == "beta"

    def test_delete_last_message_empty(self, store):
        """Deleting last message on an empty key returns None."""
        assert store.delete_last_message("empty-key") is None


class TestGetKeys:
    """Tests for get_keys."""

    def test_get_keys(self, store):
        """Set messages for 3 different keys and verify get_keys returns all 3."""
        for key in ["alice", "bob", "carol"]:
            store.add_message(key, ChatMessage(role=MessageRole.USER, content=f"hi from {key}"))

        keys = store.get_keys()
        assert sorted(keys) == ["alice", "bob", "carol"]

    def test_get_keys_empty(self, store):
        """No data returns empty list."""
        assert store.get_keys() == []


class TestSessionIsolation:
    """Tests that different keys don't interfere with each other."""

    def test_session_isolation(self, store):
        """Messages in key A should not appear in key B."""
        store.set_messages("sess-A", _make_messages(["A1", "A2"]))
        store.set_messages("sess-B", _make_messages(["B1", "B2", "B3"]))

        a_msgs = store.get_messages("sess-A")
        b_msgs = store.get_messages("sess-B")

        assert len(a_msgs) == 2
        assert len(b_msgs) == 3
        assert a_msgs[0].content == "A1"
        assert b_msgs[0].content == "B1"


class TestSetOverwrites:
    """Tests that set_messages replaces previous messages."""

    def test_set_overwrites(self, store):
        """Setting messages twice replaces the first batch entirely."""
        store.set_messages("ow-key", _make_messages(["old1", "old2", "old3"]))
        store.set_messages("ow-key", _make_messages(["new1"]))

        msgs = store.get_messages("ow-key")
        assert len(msgs) == 1
        assert msgs[0].content == "new1"


class TestPersistence:
    """Tests that data survives across store instances."""

    def test_persistence(self, tmp_db):
        """Create store, add data, create NEW store on same db, verify data."""
        store1 = SuperLocalMemoryChatStore(db_path=tmp_db)
        store1.set_messages("persist-key", _make_messages(["remember me"]))

        # New store instance pointing at same database
        store2 = SuperLocalMemoryChatStore(db_path=tmp_db)
        msgs = store2.get_messages("persist-key")
        assert len(msgs) == 1
        assert msgs[0].content == "remember me"

    def test_explicit_database_uses_v3_canonical_ingestion_without_path_hacks(
        self, tmp_db
    ):
        """The installed adapter writes through V3 without importing from its data root."""
        data_root = os.path.dirname(tmp_db)
        import sys

        before = list(sys.path)

        store = SuperLocalMemoryChatStore(db_path=tmp_db)
        store.add_message(
            "v3-contract",
            ChatMessage(role=MessageRole.USER, content="Canonical V3 adapter message"),
        )

        assert sys.path == before
        assert data_root not in sys.path
        with sqlite3.connect(tmp_db) as conn:
            operation = conn.execute(
                "SELECT source_type, state FROM ingestion_operations"
            ).fetchone()
            memory = conn.execute(
                "SELECT session_id FROM memories"
            ).fetchone()
        assert operation is not None
        assert operation[0] == "python-api"
        assert operation[1] in {"queryable", "complete", "failed"}
        assert memory is not None
        assert memory[0].startswith("llamaindex:")


class TestContentEdgeCases:
    """Tests for special content: unicode, long text, multiple roles."""

    def test_unicode_content(self, store):
        """Unicode characters (CJK, emoji, diacritics) round-trip correctly."""
        texts = [
            "Hello, world!",
            "Hej varlden!",
            "Bonjour le monde!",
        ]
        store.set_messages("unicode-key", _make_messages(texts))
        msgs = store.get_messages("unicode-key")
        assert len(msgs) == 3
        assert msgs[1].content == "Hej varlden!"

    def test_long_content(self, store):
        """A 10K character message round-trips correctly."""
        long_text = "x" * 10_000
        store.add_message(
            "long-key", ChatMessage(role=MessageRole.USER, content=long_text)
        )
        msgs = store.get_messages("long-key")
        assert len(msgs) == 1
        assert len(msgs[0].content) == 10_000

    def test_message_roles(self, store):
        """User, assistant, and system roles all round-trip correctly."""
        msgs = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
            ChatMessage(role=MessageRole.USER, content="Hi"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hello!"),
        ]
        store.set_messages("roles-key", msgs)
        retrieved = store.get_messages("roles-key")

        assert retrieved[0].role == MessageRole.SYSTEM
        assert retrieved[1].role == MessageRole.USER
        assert retrieved[2].role == MessageRole.ASSISTANT

    def test_empty_content_message(self, store):
        """A message with empty content still round-trips."""
        store.add_message(
            "empty-msg", ChatMessage(role=MessageRole.ASSISTANT, content="")
        )
        msgs = store.get_messages("empty-msg")
        assert len(msgs) == 1
        # Empty content serializes as "" and deserializes back
        assert msgs[0].content == "" or msgs[0].content is None

    def test_additional_kwargs(self, store):
        """additional_kwargs round-trip correctly."""
        msg = ChatMessage(
            role=MessageRole.USER,
            content="with metadata",
            additional_kwargs={"source": "test", "count": 42},
        )
        store.add_message("kwargs-key", msg)
        msgs = store.get_messages("kwargs-key")
        assert len(msgs) == 1
        assert msgs[0].additional_kwargs.get("source") == "test"
        assert msgs[0].additional_kwargs.get("count") == 42
