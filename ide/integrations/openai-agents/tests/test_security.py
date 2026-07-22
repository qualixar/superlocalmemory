"""Security tests for the V3 session store core.

SQL injection via session_id / item content, oversized items, LIKE wildcard
collisions, and reserved-separator rejection.  Run without ``openai-agents``
installed (loaded via conftest by file path).
"""

import pytest


def _item(role: str = "user", content: str = "ok") -> dict:
    return {"role": role, "content": content}


def test_sql_injection_in_session_id(session_store):
    sid = "'; DROP TABLE memories; --"
    session_store.append_items(sid, [_item(content="injected")])
    items = session_store.get_items(sid)
    assert len(items) == 1
    assert items[0]["content"] == "injected"
    # Table still intact — injection didn't execute.
    session_store.append_items("safe", [_item(content="safe")])
    assert session_store.get_items("safe") == [{"role": "user", "content": "safe"}]


def test_sql_injection_in_item_content(session_store):
    payload = "'); DROP TABLE atomic_facts; --"
    session_store.append_items("sess", [_item(content=payload)])
    assert session_store.get_items("sess")[0]["content"] == payload


def test_like_wildcards_in_session_id_are_literal(session_store):
    session_store.append_items("a%b", [_item(content="wildcard")])
    session_store.append_items("axb", [_item(content="literal")])
    results = session_store.get_items("a%b")
    assert len(results) == 1
    assert results[0]["content"] == "wildcard"


def test_like_underscore_in_session_id_is_literal(session_store):
    session_store.append_items("u_1", [_item(content="u_1")])
    session_store.append_items("ua1", [_item(content="ua1")])
    results = session_store.get_items("u_1")
    assert len(results) == 1
    assert results[0]["content"] == "u_1"


def test_reserved_separator_in_session_id_rejected(session_store):
    with pytest.raises(ValueError, match="reserved separator"):
        session_store.append_items("s\x1fid", [_item()])


def test_clear_with_reserved_separator_rejected(session_store):
    with pytest.raises(ValueError, match="reserved separator"):
        session_store.clear_session("s\x1fid")


def test_pop_with_reserved_separator_rejected(session_store):
    with pytest.raises(ValueError, match="reserved separator"):
        session_store.pop_item("s\x1fid")


def test_get_with_reserved_separator_rejected(session_store):
    with pytest.raises(ValueError, match="reserved separator"):
        session_store.get_items("s\x1fid")


def test_oversized_item_raises(session_store):
    big = {"role": "user", "content": "x" * 1_000_001}
    with pytest.raises(ValueError, match="exceeds maximum size"):
        session_store.append_items("sess", [big])


def test_unicode_roundtrip(session_store):
    item = {"role": "assistant", "content": "Ada — 日本語 — 🎯"}
    session_store.append_items("sess", [item])
    assert session_store.get_items("sess")[0] == item


def test_session_id_prefix_no_bleed(session_store):
    """Session 'ab' must not match prefix of session 'abc'."""
    session_store.append_items("ab", [_item(content="in-ab")])
    session_store.append_items("abc", [_item(content="in-abc")])
    assert session_store.get_items("ab") == [{"role": "user", "content": "in-ab"}]
    assert session_store.get_items("abc") == [{"role": "user", "content": "in-abc"}]


def test_clear_session_prefix_no_bleed(session_store):
    session_store.append_items("ab", [_item(content="keep")])
    session_store.append_items("abc", [_item(content="also-keep")])
    session_store.clear_session("ab")
    assert session_store.get_items("ab") == []
    assert session_store.get_items("abc") == [{"role": "user", "content": "also-keep"}]
