"""Security tests for the V3 ADK store core.

SQL injection via app_name / user_id / session_id / event text, oversized
events, LIKE wildcard collisions, and reserved-separator rejection.  These
run without ``google-adk`` installed (loaded via conftest by file path).
"""

import pytest


def _evt(text: str) -> list[dict]:
    return [{"text": text, "author": "u", "timestamp_float": 0.0,
             "timestamp_iso": "2026-01-01T00:00:00+00:00"}]


def test_sql_injection_in_app_name(adk_store):
    app = "'; DROP TABLE memories; --"
    adk_store.add_events(app, "u", "s", _evt("safe"))
    stored = adk_store.list_events_for_namespace(app, "u")
    assert len(stored) == 1
    # Table still exists — injection didn't execute.
    adk_store.add_events("clean", "u", "s", _evt("x"))
    assert len(adk_store.list_events_for_namespace("clean", "u")) == 1


def test_sql_injection_in_user_id(adk_store):
    adk_store.add_events("app", "'; DELETE FROM memories; --", "s", _evt("ok"))
    stored = adk_store.list_events_for_namespace("app", "'; DELETE FROM memories; --")
    assert len(stored) == 1


def test_sql_injection_in_event_text(adk_store):
    payload = "'); DROP TABLE atomic_facts; --"
    adk_store.add_events("app", "u", "s", _evt(payload))
    stored = adk_store.list_events_for_session("app", "u", "s")
    assert stored[0]["event"]["text"] == payload


def test_like_wildcards_in_app_name_are_literal(adk_store):
    # 'a%b' must NOT match 'axb' via LIKE wildcard expansion.
    adk_store.add_events("a%b", "u", "s", _evt("wildcard"))
    adk_store.add_events("axb", "u", "s", _evt("literal"))
    wildcard_results = adk_store.list_events_for_namespace("a%b", "u")
    assert len(wildcard_results) == 1
    assert wildcard_results[0]["event"]["text"] == "wildcard"


def test_like_underscore_in_user_id_is_literal(adk_store):
    adk_store.add_events("app", "user_1", "s", _evt("u1"))
    adk_store.add_events("app", "user12", "s", _evt("u2"))
    # 'user_1' must not expand '_' to match 'user12'.
    results = adk_store.list_events_for_namespace("app", "user_1")
    assert len(results) == 1
    assert results[0]["event"]["text"] == "u1"


def test_reserved_separator_in_app_name_rejected(adk_store):
    with pytest.raises(ValueError, match="reserved separator"):
        adk_store.add_events("app\x1fname", "u", "s", _evt("bad"))


def test_reserved_separator_in_user_id_rejected(adk_store):
    with pytest.raises(ValueError, match="reserved separator"):
        adk_store.add_events("app", "u\x1fser", "s", _evt("bad"))


def test_reserved_separator_in_session_id_rejected(adk_store):
    with pytest.raises(ValueError, match="reserved separator"):
        adk_store.add_events("app", "u", "s\x1eid", _evt("bad"))


def test_oversized_event_raises(adk_store):
    big_text = "x" * 1_000_001
    with pytest.raises(ValueError, match="exceeds maximum size"):
        adk_store.add_events("app", "u", "s", _evt(big_text))


def test_unicode_roundtrip(adk_store):
    text = "Ada — 日本語 — emoji 🎯"
    adk_store.add_events("app", "u", "s", _evt(text))
    stored = adk_store.list_events_for_session("app", "u", "s")
    assert stored[0]["event"]["text"] == text


def test_multiple_sessions_no_prefix_bleed(adk_store):
    """Session 'ab' must not match prefix of session 'abc'."""
    adk_store.add_events("app", "u", "ab", _evt("session-ab"))
    adk_store.add_events("app", "u", "abc", _evt("session-abc"))
    ab_events = adk_store.list_events_for_session("app", "u", "ab")
    assert len(ab_events) == 1
    assert ab_events[0]["event"]["text"] == "session-ab"
    abc_events = adk_store.list_events_for_session("app", "u", "abc")
    assert len(abc_events) == 1
    assert abc_events[0]["event"]["text"] == "session-abc"
