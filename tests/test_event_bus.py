# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Tests for V3 Event Bus -- Task 2 of V3 build."""
import pytest
from pathlib import Path
from superlocalmemory.infra.event_bus import EventBus, VALID_EVENT_TYPES


@pytest.fixture
def bus(tmp_path):
    """Create a fresh EventBus for each test."""
    EventBus.reset_instance(tmp_path / "test.db")
    return EventBus.get_instance(tmp_path / "test.db")


def test_v3_event_types_present():
    assert "memory.stored" in VALID_EVENT_TYPES
    assert "trust.signal" in VALID_EVENT_TYPES
    assert "compliance.audit" in VALID_EVENT_TYPES
    assert "learning.feedback" in VALID_EVENT_TYPES


def test_publish_and_subscribe(bus):
    received = []
    bus.subscribe(lambda e: received.append(e))
    bus.publish("memory.stored", {"fact_id": "f1"})
    assert len(received) == 1
    assert received[0]["event_type"] == "memory.stored"


def test_emit_and_add_listener_aliases(bus):
    received = []
    bus.add_listener(lambda e: received.append(e))
    bus.emit("memory.recalled", {"query": "test"})
    assert len(received) == 1


def test_multiple_subscribers(bus):
    results = []
    bus.subscribe(lambda e: results.append("a"))
    bus.subscribe(lambda e: results.append("b"))
    bus.publish("memory.stored", {})
    assert results == ["a", "b"]


def test_unsubscribe(bus):
    received = []
    handler = lambda e: received.append(e)
    bus.subscribe(handler)
    bus.unsubscribe(handler)
    bus.publish("memory.stored", {})
    assert len(received) == 0


def test_subscriber_error_does_not_crash(bus):
    bus.subscribe(lambda e: 1/0)
    bus.publish("memory.stored", {})  # should not raise


def test_events_persisted(bus):
    bus.publish("memory.stored", {"fact_id": "f1"})
    bus.publish("memory.recalled", {"query": "test"})
    events = bus.get_recent_events(limit=10)
    assert len(events) >= 2


def test_event_stats(bus):
    bus.publish("memory.stored", {})
    bus.publish("memory.stored", {})
    bus.publish("memory.recalled", {})
    stats = bus.get_event_stats()
    assert stats.get("memory.stored", 0) >= 2
    assert stats.get("memory.recalled", 0) >= 1


def test_buffered_events(bus):
    bus.publish("memory.stored", {"n": 1})
    bus.publish("memory.stored", {"n": 2})
    buffered = bus.get_buffered_events(since_seq=0)
    assert len(buffered) >= 2


def test_singleton_pattern(tmp_path):
    db = tmp_path / "singleton.db"
    EventBus.reset_instance(db)
    bus1 = EventBus.get_instance(db)
    bus2 = EventBus.get_instance(db)
    assert bus1 is bus2


# ── Per-profile isolation (I-2) ──────────────────────────────────────────────

def test_events_are_isolated_by_profile(bus):
    """get_recent_events must only return the requested profile's events."""
    bus.emit("memory.stored", {"n": 1}, profile_id="work")
    bus.emit("memory.stored", {"n": 2}, profile_id="work")
    bus.emit("memory.stored", {"n": 3}, profile_id="home")

    work = bus.get_recent_events(profile_id="work")
    home = bus.get_recent_events(profile_id="home")
    empty = bus.get_recent_events(profile_id="unused")

    assert len(work) == 2
    assert len(home) == 1
    assert empty == []
    assert all(e["profile_id"] == "work" for e in work)


def test_event_stats_are_isolated_by_profile(bus):
    bus.emit("memory.stored", {}, profile_id="work")
    bus.emit("memory.recalled", {}, profile_id="work")
    bus.emit("memory.stored", {}, profile_id="home")

    assert bus.get_event_stats(profile_id="work")["total_events"] == 2
    assert bus.get_event_stats(profile_id="home")["total_events"] == 1
    assert bus.get_event_stats(profile_id="none")["total_events"] == 0


def test_wildcard_scope_returns_all_profiles(bus):
    """The internal '*' scope bypasses filtering (maintenance only)."""
    bus.emit("memory.stored", {}, profile_id="work")
    bus.emit("memory.stored", {}, profile_id="home")
    assert len(bus.get_recent_events(profile_id="*")) == 2


def test_persisted_event_carries_profile_id(bus):
    bus.emit("memory.stored", {"x": 1}, profile_id="alpha")
    rows = bus.get_recent_events(profile_id="alpha")
    assert rows and rows[0]["profile_id"] == "alpha"


def test_legacy_events_table_self_migrates(tmp_path):
    """A pre-isolation memory_events table gains profile_id + backfills 'default'."""
    import sqlite3
    db = tmp_path / "legacy_events.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE memory_events ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL,"
        " memory_id INTEGER, source_agent TEXT, source_protocol TEXT,"
        " payload TEXT, importance INTEGER, tier TEXT, created_at TIMESTAMP);"
        "INSERT INTO memory_events (event_type, payload, importance, tier, created_at) "
        "VALUES ('memory.stored', '{}', 5, 'hot', '2026-01-01T00:00:00Z');"
    )
    conn.commit()
    conn.close()

    EventBus.reset_instance(db)
    bus = EventBus.get_instance(db)
    cols = {r[1] for r in sqlite3.connect(str(db)).execute(
        "PRAGMA table_info(memory_events)").fetchall()}
    assert "profile_id" in cols
    # Legacy row backfilled to 'default'.
    assert len(bus.get_recent_events(profile_id="default")) == 1
    assert bus.get_recent_events(profile_id="other") == []
