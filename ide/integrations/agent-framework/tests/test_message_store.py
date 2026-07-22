"""Functional + security tests for the framework-free message store core.

Run without ``agent-framework-core`` installed, against the real SLM engine on
a temp database.
"""

import pytest


def test_append_and_order(message_store):
    message_store.append("s1", {"role": "user", "text": "hello"})
    message_store.append("s1", {"role": "assistant", "text": "hi there"})
    msgs = message_store.messages("s1")
    assert [m["text"] for m in msgs] == ["hello", "hi there"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_session_isolation(message_store):
    message_store.append("a", {"role": "user", "text": "in a"})
    message_store.append("b", {"role": "user", "text": "in b"})
    assert [m["text"] for m in message_store.messages("a")] == ["in a"]
    assert [m["text"] for m in message_store.messages("b")] == ["in b"]


def test_none_session_uses_default(message_store):
    message_store.append(None, {"role": "user", "text": "default"})
    assert len(message_store.messages(None)) == 1


def test_empty_session(message_store):
    assert message_store.messages("never") == []


def test_clear(message_store):
    message_store.append("s1", {"role": "user", "text": "one"})
    message_store.append("s2", {"role": "user", "text": "keep"})
    message_store.clear("s1")
    assert message_store.messages("s1") == []
    assert len(message_store.messages("s2")) == 1  # unaffected


def test_sql_injection_in_session_id(message_store):
    sid = "'; DROP TABLE memories; --"
    message_store.append(sid, {"role": "user", "text": "safe"})
    assert [m["text"] for m in message_store.messages(sid)] == ["safe"]
    message_store.append("plain", {"role": "user", "text": "still here"})
    assert [m["text"] for m in message_store.messages("plain")] == ["still here"]


def test_oversized_message_raises(message_store):
    with pytest.raises(ValueError, match="exceeds maximum size"):
        message_store.append("s", {"role": "user", "text": "x" * 1_000_001})


def test_unicode_roundtrip(message_store):
    text = "Ada — 日本語 — 🎯"
    message_store.append("u", {"role": "user", "text": text})
    assert message_store.messages("u")[0]["text"] == text
