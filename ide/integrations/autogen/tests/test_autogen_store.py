"""Functional + security tests for the framework-free AutoGen store core.

Run without ``autogen-agentchat`` installed (loaded via conftest by file path)
against the real SuperLocalMemory engine on a temp database.
"""

import pytest


# ── add and get_recent ────────────────────────────────────────────────────────

def test_add_stores_content(autogen_store):
    autogen_store.add("The sky is blue and vast.", mime_type="text/plain")
    recent = autogen_store.get_recent(limit=10)
    assert len(recent) >= 1
    assert any("sky" in r["content"] for r in recent)


def test_add_multiple_are_all_stored(autogen_store):
    autogen_store.add("First message about cats.")
    autogen_store.add("Second message about dogs.")
    recent = autogen_store.get_recent(limit=10)
    assert len(recent) == 2


def test_get_recent_respects_limit(autogen_store):
    for i in range(5):
        autogen_store.add(f"Memory item number {i} for testing recall limits.")
    recent = autogen_store.get_recent(limit=3)
    assert len(recent) <= 3


def test_get_recent_returns_in_order(autogen_store):
    autogen_store.add("Alpha content for ordering test.")
    autogen_store.add("Beta content for ordering test.")
    recent = autogen_store.get_recent(limit=10)
    assert recent[0]["content"].startswith("Alpha")
    assert recent[1]["content"].startswith("Beta")


def test_metadata_roundtrip(autogen_store):
    meta = {"source": "user_input", "turn": 3}
    autogen_store.add("Content with metadata attached.", metadata=meta)
    recent = autogen_store.get_recent(limit=1)
    stored_meta = recent[0].get("metadata", {})
    assert stored_meta.get("source") == "user_input"
    assert stored_meta.get("turn") == 3


# ── clear ─────────────────────────────────────────────────────────────────────

def test_clear_removes_all(autogen_store):
    autogen_store.add("To be cleared.")
    autogen_store.add("Also to be cleared.")
    autogen_store.clear()
    assert autogen_store.get_recent(limit=100) == []


def test_clear_is_scoped_to_autogen_entries(autogen_store):
    """clear() must not delete unrelated SLM memories."""
    # Store something via the raw engine (not autogen-mem: prefix).
    autogen_store._engine.store(
        "SLM core memory unrelated to autogen.",
        session_id="core-session-not-autogen",
        metadata={"integration": "core"},
    )
    autogen_store.add("AutoGen memory to clear.")
    autogen_store.clear()
    # Autogen memories gone.
    assert autogen_store.get_recent(limit=100) == []
    # Core memory still present (not touched by clear).
    rows = autogen_store._engine.db.execute(
        "SELECT COUNT(*) as cnt FROM memories WHERE session_id=? AND profile_id=?",
        ("core-session-not-autogen", autogen_store._engine.profile_id),
    )
    count = dict(list(rows)[0])["cnt"]
    assert count >= 1


# ── query ─────────────────────────────────────────────────────────────────────

def test_query_returns_list(autogen_store):
    autogen_store.add("Python programming language overview.")
    results = autogen_store.query_text("Python programming", limit=5)
    assert isinstance(results, list)


def test_query_result_structure(autogen_store):
    autogen_store.add("Machine learning fundamentals and neural networks.")
    results = autogen_store.query_text("machine learning", limit=5)
    # Each result must be a dict with 'content' and 'score'.
    for r in results:
        assert "content" in r
        assert "score" in r
        assert isinstance(r["score"], float)


def test_query_empty_store_returns_empty(autogen_store):
    results = autogen_store.query_text("anything at all", limit=5)
    assert results == []


# ── security ──────────────────────────────────────────────────────────────────

def test_sql_injection_in_content(autogen_store):
    content = "'; DROP TABLE memories; -- safe content follows"
    autogen_store.add(content)
    recent = autogen_store.get_recent(limit=10)
    assert len(recent) >= 1
    # Other adds are unaffected.
    autogen_store.add("Plain safe content remains accessible.")
    recent2 = autogen_store.get_recent(limit=10)
    assert any("Plain" in r["content"] for r in recent2)


def test_unicode_roundtrip(autogen_store):
    content = "Ada — 日本語 — 🎯"
    autogen_store.add(content)
    recent = autogen_store.get_recent(limit=1)
    assert recent[0]["content"] == content


def test_oversized_content_raises(autogen_store):
    with pytest.raises(ValueError, match="exceeds maximum size"):
        autogen_store.add("x" * 1_000_001)
