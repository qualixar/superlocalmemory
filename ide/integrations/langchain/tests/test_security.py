#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 - LangChain Integration Security Tests

Security tests covering SQL injection, XSS payloads, oversized content,
and edge-case session identifiers.  These mirror OWASP agentic-AI concerns
relevant to a local-first memory store.
"""
import os
import tempfile

import pytest
from langchain_core.messages import HumanMessage
from langchain_superlocalmemory import SuperLocalMemoryChatMessageHistory


@pytest.fixture
def tmp_db():
    """Yield a path to a temporary SQLite database that is cleaned up after use."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test_memory.db")
        yield db


# ---------------------------------------------------------------------------
# SQL injection
# ---------------------------------------------------------------------------


def test_sql_injection_session_id(tmp_db):
    """SQL injection in session_id should not work."""
    history = SuperLocalMemoryChatMessageHistory(
        session_id="'; DROP TABLE memories; --", db_path=tmp_db
    )
    history.add_messages([HumanMessage(content="test")])
    assert len(history.messages) == 1  # Should work normally, not crash


def test_sql_injection_content(tmp_db):
    """SQL injection in message content is safely stored and retrieved."""
    history = SuperLocalMemoryChatMessageHistory(session_id="test", db_path=tmp_db)
    payload = "'; DROP TABLE memories; --"
    history.add_messages([HumanMessage(content=payload)])

    msgs = history.messages
    assert len(msgs) == 1
    assert msgs[0].content == payload


# ---------------------------------------------------------------------------
# XSS
# ---------------------------------------------------------------------------


def test_xss_content(tmp_db):
    """XSS payloads are stored verbatim (no sanitization needed for storage)."""
    history = SuperLocalMemoryChatMessageHistory(session_id="test", db_path=tmp_db)
    payload = "<script>alert('xss')</script>"
    history.add_messages([HumanMessage(content=payload)])

    msgs = history.messages
    assert len(msgs) == 1
    assert "<script>" in msgs[0].content


# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------


def test_large_content(tmp_db):
    """A large message (near the SLM 1 MB limit) is stored and retrieved intact.

    The V3 adapter enforces MAX_CONTENT_SIZE = 1_000_000 on the *stored*
    string, which is the JSON-serialized message (not the raw text).  The
    JSON envelope adds ~150 bytes of overhead, so we use 999_000 chars to
    stay safely under the limit.
    """
    history = SuperLocalMemoryChatMessageHistory(session_id="test", db_path=tmp_db)
    big_msg = "x" * 999_000
    history.add_messages([HumanMessage(content=big_msg)])

    msgs = history.messages
    assert len(msgs) == 1
    assert len(msgs[0].content) == 999_000


def test_oversized_content_raises(tmp_db):
    """Content that exceeds SLM's 1 MB limit after serialization raises ValueError."""
    history = SuperLocalMemoryChatMessageHistory(session_id="test", db_path=tmp_db)
    # 1_000_000 chars + JSON overhead will exceed the 1 MB limit.
    big_msg = "x" * 1_000_000
    with pytest.raises(ValueError, match="exceeds maximum size"):
        history.add_messages([HumanMessage(content=big_msg)])


# ---------------------------------------------------------------------------
# Edge-case session IDs
# ---------------------------------------------------------------------------


def test_empty_session_id(tmp_db):
    """An empty string session_id still functions correctly."""
    history = SuperLocalMemoryChatMessageHistory(session_id="", db_path=tmp_db)
    history.add_messages([HumanMessage(content="test")])
    assert len(history.messages) == 1


def test_null_bytes_session_id(tmp_db):
    """Null bytes in session_id do not cause crashes."""
    history = SuperLocalMemoryChatMessageHistory(
        session_id="test\x00evil", db_path=tmp_db
    )
    history.add_messages([HumanMessage(content="test")])
    assert len(history.messages) == 1
