"""Functional tests for the framework-free V3 ADK store core.

These run without ``google-adk`` installed (loaded via conftest by file path)
and exercise the real SuperLocalMemory engine against a temp database.
"""

import pytest


# ---------------------------------------------------------------------------
# Basic storage and retrieval
# ---------------------------------------------------------------------------

def _make_events(n: int = 2) -> list[dict]:
    return [
        {"text": f"Message {i}", "author": "user", "timestamp_float": float(i),
         "timestamp_iso": "2026-01-01T00:00:00+00:00"}
        for i in range(n)
    ]


def test_add_and_list_events(adk_store):
    events = _make_events(3)
    adk_store.add_events("app", "alice", "sess1", events)
    stored = adk_store.list_events_for_session("app", "alice", "sess1")
    assert len(stored) == 3
    texts = [e["event"]["text"] for e in stored]
    assert texts == ["Message 0", "Message 1", "Message 2"]


def test_add_events_preserves_author_and_timestamp(adk_store):
    events = [
        {"text": "Hello", "author": "agent", "timestamp_float": 1_700_000.0,
         "timestamp_iso": "2026-01-01T00:00:00+00:00"}
    ]
    adk_store.add_events("app", "bob", "s", events)
    stored = adk_store.list_events_for_session("app", "bob", "s")
    assert stored[0]["event"]["author"] == "agent"
    assert stored[0]["event"]["timestamp_float"] == 1_700_000.0


def test_get_missing_session_returns_empty(adk_store):
    assert adk_store.list_events_for_session("app", "nobody", "nope") == []


def test_add_events_idempotent_replace(adk_store):
    """Re-adding the same session replaces old events."""
    adk_store.add_events("app", "alice", "sess", _make_events(3))
    adk_store.add_events("app", "alice", "sess", _make_events(1))
    stored = adk_store.list_events_for_session("app", "alice", "sess")
    assert len(stored) == 1


def test_session_isolation(adk_store):
    adk_store.add_events("app", "alice", "s1", _make_events(2))
    adk_store.add_events("app", "alice", "s2", _make_events(1))
    s1 = adk_store.list_events_for_session("app", "alice", "s1")
    s2 = adk_store.list_events_for_session("app", "alice", "s2")
    assert len(s1) == 2 and len(s2) == 1


def test_user_isolation(adk_store):
    adk_store.add_events("app", "alice", "s", _make_events(2))
    adk_store.add_events("app", "bob", "s", _make_events(3))
    alice = adk_store.list_events_for_namespace("app", "alice")
    bob = adk_store.list_events_for_namespace("app", "bob")
    assert len(alice) == 2 and len(bob) == 3


def test_events_ordered_by_index(adk_store):
    events = [
        {"text": f"turn{i}", "author": "user", "timestamp_float": float(i),
         "timestamp_iso": "2026-01-01T00:00:00+00:00"}
        for i in range(5)
    ]
    adk_store.add_events("app", "carol", "s", events)
    stored = adk_store.list_events_for_session("app", "carol", "s")
    assert [e["event_index"] for e in stored] == list(range(5))


def test_empty_events_clears_session(adk_store):
    adk_store.add_events("app", "user1", "s", _make_events(2))
    adk_store.add_events("app", "user1", "s", [])
    assert adk_store.list_events_for_session("app", "user1", "s") == []


def test_envelope_fields_present(adk_store):
    """Envelope must include all required fields for SLM queryability."""
    adk_store.add_events("app", "u", "s", [{"text": "hi", "author": "x",
                                              "timestamp_float": 0.0,
                                              "timestamp_iso": "2026-01-01T00:00:00+00:00"}])
    stored = adk_store.list_events_for_session("app", "u", "s")
    env = stored[0]
    for field in ("adapter", "app_name", "user_id", "session_id", "event_index",
                  "event", "created_at"):
        assert field in env, f"missing field: {field}"
