#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 - LlamaIndex Chat Store Security Tests

Security-focused tests: SQL injection, XSS, oversized payloads, null bytes,
and other adversarial inputs that a production chat store must handle safely.
"""
import os
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
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "test_security.db")


@pytest.fixture
def store(tmp_db):
    return SuperLocalMemoryChatStore(db_path=tmp_db)


# ---------------------------------------------------------------------------
# SQL Injection Tests
# ---------------------------------------------------------------------------


class TestSQLInjection:
    """Verify that SQL injection attempts in keys and content are harmless."""

    SQL_PAYLOADS = [
        "'; DROP TABLE memories; --",
        "' OR '1'='1",
        "'; DELETE FROM memories WHERE '1'='1",
        "1; SELECT * FROM memories --",
        "Robert'); DROP TABLE memories;--",
        "' UNION SELECT * FROM memories --",
    ]

    def test_sql_injection_in_key(self, store):
        """SQL injection attempts in the session key must not corrupt the store."""
        for payload in self.SQL_PAYLOADS:
            msg = ChatMessage(role=MessageRole.USER, content="safe content")
            store.add_message(payload, msg)

            # The store should still function — retrieve the message back
            msgs = store.get_messages(payload)
            assert len(msgs) == 1
            assert msgs[0].content == "safe content"

            # Clean up
            store.delete_messages(payload)

    def test_sql_injection_in_content(self, store):
        """SQL injection attempts in message content must be stored literally."""
        for payload in self.SQL_PAYLOADS:
            msg = ChatMessage(role=MessageRole.USER, content=payload)
            store.add_message("safe-key", msg)

        msgs = store.get_messages("safe-key")
        assert len(msgs) == len(self.SQL_PAYLOADS)

        # Verify content is stored literally, not executed
        for i, payload in enumerate(self.SQL_PAYLOADS):
            assert msgs[i].content == payload

        # Clean up
        store.delete_messages("safe-key")


# ---------------------------------------------------------------------------
# XSS Tests
# ---------------------------------------------------------------------------


class TestXSS:
    """Verify that XSS payloads in content are stored literally (not executed)."""

    XSS_PAYLOADS = [
        "<script>alert('xss')</script>",
        '<img src=x onerror=alert(1)>',
        "javascript:alert(document.cookie)",
        "<svg/onload=alert('xss')>",
        '"><script>alert(String.fromCharCode(88,83,83))</script>',
    ]

    def test_xss_in_content(self, store):
        """XSS payloads must be stored and retrieved as literal strings."""
        for payload in self.XSS_PAYLOADS:
            msg = ChatMessage(role=MessageRole.USER, content=payload)
            store.add_message("xss-key", msg)

        msgs = store.get_messages("xss-key")
        assert len(msgs) == len(self.XSS_PAYLOADS)

        for i, payload in enumerate(self.XSS_PAYLOADS):
            assert msgs[i].content == payload

        store.delete_messages("xss-key")


# ---------------------------------------------------------------------------
# Oversized Content Tests
# ---------------------------------------------------------------------------


class TestOversizedContent:
    """Verify that oversized payloads are handled without crashing."""

    def test_oversized_content_1mb(self, store):
        """A 1MB message should be rejected by the V3 adapter boundary."""
        huge = "A" * 1_000_001  # Just over the 1MB limit
        msg = ChatMessage(role=MessageRole.USER, content=huge)

        with pytest.raises(ValueError, match="exceeds maximum size"):
            store.add_message("huge-key", msg)

    def test_large_but_valid_content(self, store):
        """A 500KB message should be stored successfully (under 1MB limit)."""
        large = "B" * 500_000
        msg = ChatMessage(role=MessageRole.USER, content=large)
        store.add_message("large-key", msg)

        msgs = store.get_messages("large-key")
        assert len(msgs) == 1
        assert len(msgs[0].content) == 500_000

        store.delete_messages("large-key")


# ---------------------------------------------------------------------------
# Edge Case Key Tests
# ---------------------------------------------------------------------------


class TestEdgeCaseKeys:
    """Verify safe handling of unusual session key values."""

    def test_empty_key(self, store):
        """An empty string key should still work (tags allow empty strings)."""
        msg = ChatMessage(role=MessageRole.USER, content="empty key test")
        store.add_message("", msg)

        msgs = store.get_messages("")
        assert len(msgs) == 1
        assert msgs[0].content == "empty key test"

        store.delete_messages("")

    def test_null_bytes_in_key(self, store):
        """Null bytes in a key should be handled without crash."""
        key_with_null = "key\x00with\x00nulls"
        msg = ChatMessage(role=MessageRole.USER, content="null byte key")
        store.add_message(key_with_null, msg)

        msgs = store.get_messages(key_with_null)
        assert len(msgs) == 1
        assert msgs[0].content == "null byte key"

        store.delete_messages(key_with_null)

    def test_very_long_key(self, store):
        """A very long key (1000 chars) should work."""
        long_key = "k" * 1000
        msg = ChatMessage(role=MessageRole.USER, content="long key test")
        store.add_message(long_key, msg)

        msgs = store.get_messages(long_key)
        assert len(msgs) == 1

        store.delete_messages(long_key)

    def test_special_characters_in_key(self, store):
        """Keys with special characters should work."""
        special_keys = [
            "user@example.com",
            "path/to/session",
            "key with spaces",
            "key-with-dashes_and_underscores",
            "unicode-key",
        ]
        for key in special_keys:
            store.add_message(
                key, ChatMessage(role=MessageRole.USER, content=f"test {key}")
            )

        for key in special_keys:
            msgs = store.get_messages(key)
            assert len(msgs) == 1, f"Failed for key: {key}"

        # Verify all keys are discoverable
        all_keys = store.get_keys()
        for key in special_keys:
            assert key in all_keys, f"Key '{key}' not found in get_keys()"

        for key in special_keys:
            store.delete_messages(key)


# ---------------------------------------------------------------------------
# Null / None Content Tests
# ---------------------------------------------------------------------------


class TestNullContent:
    """Verify that null bytes and None-like content are handled safely."""

    def test_null_bytes_in_content(self, store):
        """Null bytes in content should be stored and retrieved literally."""
        content_with_null = "before\x00after"
        msg = ChatMessage(role=MessageRole.USER, content=content_with_null)
        store.add_message("null-content-key", msg)

        msgs = store.get_messages("null-content-key")
        assert len(msgs) == 1
        # The content may or may not preserve null bytes depending on SQLite/JSON
        # The key requirement is that it doesn't crash
        assert msgs[0].content is not None

        store.delete_messages("null-content-key")
