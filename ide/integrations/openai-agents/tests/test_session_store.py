"""Functional tests for the framework-free V3 session store core.

These run without ``openai-agents`` installed (loaded via conftest by file
path) and exercise the real SuperLocalMemory engine against a temp database.
"""

import time


def _item(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# Basic append / get
# ---------------------------------------------------------------------------

def test_append_and_get(session_store):
    session_store.append_items("sess1", [_item("user", "hello")])
    items = session_store.get_items("sess1")
    assert len(items) == 1
    assert items[0] == {"role": "user", "content": "hello"}


def test_get_empty_returns_empty(session_store):
    assert session_store.get_items("empty_sess") == []


def test_multiple_items_ordered_oldest_first(session_store):
    for i in range(4):
        session_store.append_items("sess", [_item("user", f"msg{i}")])
    items = session_store.get_items("sess")
    assert [it["content"] for it in items] == ["msg0", "msg1", "msg2", "msg3"]


def test_append_batch_preserves_order(session_store):
    batch = [_item("user", f"m{i}") for i in range(5)]
    session_store.append_items("sess", batch)
    items = session_store.get_items("sess")
    assert [it["content"] for it in items] == [f"m{i}" for i in range(5)]


def test_get_items_limit_returns_last_n(session_store):
    session_store.append_items("sess", [_item("user", f"m{i}") for i in range(6)])
    last_two = session_store.get_items("sess", limit=2)
    assert len(last_two) == 2
    assert last_two[0]["content"] == "m4"
    assert last_two[1]["content"] == "m5"


def test_get_items_limit_larger_than_count(session_store):
    session_store.append_items("sess", [_item("user", "only")])
    assert len(session_store.get_items("sess", limit=100)) == 1


# ---------------------------------------------------------------------------
# pop_item
# ---------------------------------------------------------------------------

def test_pop_item_removes_most_recent(session_store):
    for msg in ["first", "second", "third"]:
        session_store.append_items("sess", [_item("user", msg)])
    popped = session_store.pop_item("sess")
    assert popped is not None
    assert popped["content"] == "third"
    remaining = session_store.get_items("sess")
    assert len(remaining) == 2
    assert remaining[-1]["content"] == "second"


def test_pop_item_empty_returns_none(session_store):
    assert session_store.pop_item("empty") is None


def test_pop_item_until_empty(session_store):
    session_store.append_items("sess", [_item("user", "a"), _item("user", "b")])
    p1 = session_store.pop_item("sess")
    p2 = session_store.pop_item("sess")
    p3 = session_store.pop_item("sess")
    assert p1["content"] == "b"
    assert p2["content"] == "a"
    assert p3 is None


# ---------------------------------------------------------------------------
# clear_session
# ---------------------------------------------------------------------------

def test_clear_session_removes_all_items(session_store):
    session_store.append_items("sess", [_item("user", f"m{i}") for i in range(5)])
    session_store.clear_session("sess")
    assert session_store.get_items("sess") == []


def test_clear_session_does_not_affect_other_sessions(session_store):
    session_store.append_items("a", [_item("user", "msg-a")])
    session_store.append_items("b", [_item("user", "msg-b")])
    session_store.clear_session("a")
    assert session_store.get_items("a") == []
    assert session_store.get_items("b") == [{"role": "user", "content": "msg-b"}]


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------

def test_sessions_are_isolated(session_store):
    session_store.append_items("s1", [_item("user", "in s1")])
    session_store.append_items("s2", [_item("user", "in s2")])
    assert session_store.get_items("s1") == [{"role": "user", "content": "in s1"}]
    assert session_store.get_items("s2") == [{"role": "user", "content": "in s2"}]


# ---------------------------------------------------------------------------
# Envelope fields
# ---------------------------------------------------------------------------

def test_envelope_fields_present(session_store):
    """Stored envelope must carry seq and created_at for observability."""
    import importlib.util
    from pathlib import Path

    store_path = Path(__file__).resolve().parent.parent / "openai_agents_superlocalmemory" / "_v3_session_store.py"
    spec = importlib.util.spec_from_file_location("_slm_oa_session_core_local", store_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Access the raw DB to inspect the envelope.
    session_store.append_items("meta_sess", [_item("user", "check")])
    escaped = module._escape_like(module._item_prefix("meta_sess"))
    rows = session_store._engine.db.execute(
        "SELECT content FROM memories "
        "WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' LIMIT 1",
        (session_store._engine.profile_id, escaped + "%"),
    )
    import json
    for row in rows:
        data = json.loads(dict(row)["content"])
        assert "adapter" in data
        assert "seq" in data
        assert "item" in data
        assert "created_at" in data
        assert data["seq"] == 0
