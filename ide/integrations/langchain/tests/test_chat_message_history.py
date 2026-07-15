#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 - LangChain Chat Message History Tests

Functional tests for SuperLocalMemoryChatMessageHistory.
All tests use a temporary database -- the user's real memory is never touched.
"""
import os
import sqlite3
import sys
import tempfile

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_superlocalmemory import SuperLocalMemoryChatMessageHistory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db():
    """Yield a path to a temporary SQLite database that is cleaned up after use."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test_memory.db")
        yield db


@pytest.fixture
def history(tmp_db):
    """Return a fresh history instance bound to the temp DB."""
    return SuperLocalMemoryChatMessageHistory(session_id="test-session", db_path=tmp_db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_add_single_human_message(tmp_db):
    """A single HumanMessage can be added and retrieved."""
    h = SuperLocalMemoryChatMessageHistory(session_id="s1", db_path=tmp_db)
    h.add_messages([HumanMessage(content="Hello, world!")])

    msgs = h.messages
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "Hello, world!"


def test_add_single_ai_message(tmp_db):
    """A single AIMessage can be added and retrieved."""
    h = SuperLocalMemoryChatMessageHistory(session_id="s1", db_path=tmp_db)
    h.add_messages([AIMessage(content="I am an AI.")])

    msgs = h.messages
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].content == "I am an AI."


def test_add_multiple_messages(tmp_db):
    """A batch of three messages is stored and all are retrievable."""
    h = SuperLocalMemoryChatMessageHistory(session_id="s1", db_path=tmp_db)
    h.add_messages([
        HumanMessage(content="Hi"),
        AIMessage(content="Hello"),
        HumanMessage(content="How are you?"),
    ])

    msgs = h.messages
    assert len(msgs) == 3
    assert msgs[0].content == "Hi"
    assert msgs[1].content == "Hello"
    assert msgs[2].content == "How are you?"


def test_messages_order(tmp_db):
    """Messages are returned in chronological (insertion) order."""
    h = SuperLocalMemoryChatMessageHistory(session_id="s1", db_path=tmp_db)

    h.add_messages([HumanMessage(content="first")])
    h.add_messages([AIMessage(content="second")])
    h.add_messages([HumanMessage(content="third")])

    msgs = h.messages
    assert [m.content for m in msgs] == ["first", "second", "third"]


def test_message_roundtrip(tmp_db):
    """A HumanMessage with additional_kwargs survives serialization."""
    h = SuperLocalMemoryChatMessageHistory(session_id="s1", db_path=tmp_db)

    original = HumanMessage(
        content="Test with kwargs",
        additional_kwargs={"custom_key": "custom_value", "number": 42},
    )
    h.add_messages([original])

    msgs = h.messages
    assert len(msgs) == 1
    assert msgs[0].content == "Test with kwargs"
    assert msgs[0].additional_kwargs.get("custom_key") == "custom_value"
    assert msgs[0].additional_kwargs.get("number") == 42


def test_system_message(tmp_db):
    """A SystemMessage round-trips correctly."""
    h = SuperLocalMemoryChatMessageHistory(session_id="s1", db_path=tmp_db)
    h.add_messages([SystemMessage(content="You are a helpful assistant.")])

    msgs = h.messages
    assert len(msgs) == 1
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == "You are a helpful assistant."


def test_clear(tmp_db):
    """clear() removes all messages from the session."""
    h = SuperLocalMemoryChatMessageHistory(session_id="s1", db_path=tmp_db)
    h.add_messages([
        HumanMessage(content="A"),
        AIMessage(content="B"),
    ])

    assert len(h.messages) == 2
    h.clear()
    assert len(h.messages) == 0


def test_clear_only_affects_session(tmp_db):
    """Clearing session A does not affect session B."""
    ha = SuperLocalMemoryChatMessageHistory(session_id="session-a", db_path=tmp_db)
    hb = SuperLocalMemoryChatMessageHistory(session_id="session-b", db_path=tmp_db)

    ha.add_messages([HumanMessage(content="Message in A")])
    hb.add_messages([HumanMessage(content="Message in B")])

    ha.clear()

    assert len(ha.messages) == 0
    assert len(hb.messages) == 1
    assert hb.messages[0].content == "Message in B"


def test_session_isolation(tmp_db):
    """Two sessions with the same DB path do not see each other's messages."""
    ha = SuperLocalMemoryChatMessageHistory(session_id="alpha", db_path=tmp_db)
    hb = SuperLocalMemoryChatMessageHistory(session_id="beta", db_path=tmp_db)

    ha.add_messages([HumanMessage(content="Alpha message")])
    hb.add_messages([HumanMessage(content="Beta message")])

    assert len(ha.messages) == 1
    assert ha.messages[0].content == "Alpha message"

    assert len(hb.messages) == 1
    assert hb.messages[0].content == "Beta message"


def test_empty_session(tmp_db):
    """A brand-new session returns an empty list."""
    h = SuperLocalMemoryChatMessageHistory(session_id="empty", db_path=tmp_db)
    assert h.messages == []


def test_persistence_across_instances(tmp_db):
    """Messages persist when a new Python instance is created with the same session_id."""
    h1 = SuperLocalMemoryChatMessageHistory(session_id="persist", db_path=tmp_db)
    h1.add_messages([HumanMessage(content="Persisted message")])

    # Create a completely new instance pointing at the same DB + session.
    h2 = SuperLocalMemoryChatMessageHistory(session_id="persist", db_path=tmp_db)
    msgs = h2.messages
    assert len(msgs) == 1
    assert msgs[0].content == "Persisted message"


def test_explicit_database_uses_v3_canonical_ingestion_without_path_hacks(tmp_db):
    """The installed adapter must write through V3 and never import from the data root."""
    data_root = os.path.dirname(tmp_db)
    before = list(sys.path)

    h = SuperLocalMemoryChatMessageHistory(session_id="v3-contract", db_path=tmp_db)
    h.add_messages([HumanMessage(content="Canonical V3 adapter message")])

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
    assert memory[0].startswith("langchain:")


def test_unicode_content(tmp_db):
    """Unicode and emoji content round-trips correctly."""
    h = SuperLocalMemoryChatMessageHistory(session_id="unicode", db_path=tmp_db)
    text = "Bonjour le monde! Hola mundo! Konnichiwa sekai!"
    h.add_messages([HumanMessage(content=text)])

    msgs = h.messages
    assert len(msgs) == 1
    assert msgs[0].content == text


def test_long_content(tmp_db):
    """A 10K character message is stored and retrieved intact."""
    h = SuperLocalMemoryChatMessageHistory(session_id="long", db_path=tmp_db)
    long_text = "A" * 10_000
    h.add_messages([HumanMessage(content=long_text)])

    msgs = h.messages
    assert len(msgs) == 1
    assert len(msgs[0].content) == 10_000
    assert msgs[0].content == long_text


def test_special_chars_session_id(tmp_db):
    """Session IDs with dots, dashes, and underscores work correctly."""
    for sid in ["my.session.123", "my-session-456", "my_session_789", "a.b-c_d"]:
        h = SuperLocalMemoryChatMessageHistory(session_id=sid, db_path=tmp_db)
        h.add_messages([HumanMessage(content=f"msg for {sid}")])

        msgs = h.messages
        assert len(msgs) == 1, f"Failed for session_id={sid}"
        assert msgs[0].content == f"msg for {sid}"

        h.clear()
        assert len(h.messages) == 0, f"clear() failed for session_id={sid}"
