"""Functional tests for the framework-free V3 key-value store core.

These run without ``langgraph`` installed (loaded via conftest by file path)
and exercise the real SuperLocalMemory engine against a temp database.
"""

import time


def test_put_and_get(kv_store):
    kv_store.put(("users", "1"), "profile", {"name": "Ada", "role": "eng"})
    env = kv_store.get(("users", "1"), "profile")
    assert env is not None
    assert env["value"] == {"name": "Ada", "role": "eng"}
    # A fresh insert stamps equal created/updated timestamps.
    assert env["created_at"] == env["updated_at"]


def test_get_missing_returns_none(kv_store):
    assert kv_store.get(("nope",), "missing") is None


def test_update_preserves_created_bumps_updated(kv_store):
    kv_store.put(("users", "1"), "p", {"v": 1})
    first = kv_store.get(("users", "1"), "p")
    time.sleep(0.01)
    kv_store.put(("users", "1"), "p", {"v": 2})
    second = kv_store.get(("users", "1"), "p")
    assert second["value"] == {"v": 2}
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] > first["updated_at"]


def test_search_prefix_is_tuple_wise(kv_store):
    kv_store.put(("users", "1"), "a", {"n": 1})
    kv_store.put(("users", "2"), "b", {"n": 2})
    kv_store.put(("users2",), "c", {"n": 3})  # sibling that must NOT match
    kv_store.put(("orgs",), "d", {"n": 4})

    hits = kv_store.search(("users",))
    namespaces = {tuple(h["namespace"]) for h in hits}
    assert namespaces == {("users", "1"), ("users", "2")}
    assert ("users2",) not in namespaces


def test_search_filter(kv_store):
    kv_store.put(("docs",), "d1", {"status": "draft"})
    kv_store.put(("docs",), "d2", {"status": "final"})
    hits = kv_store.search(("docs",), filter={"status": "final"})
    assert [h["key"] for h in hits] == ["d2"]


def test_search_limit_and_offset(kv_store):
    for i in range(5):
        kv_store.put(("n",), f"k{i}", {"i": i})
    page1 = kv_store.search(("n",), limit=2, offset=0)
    page2 = kv_store.search(("n",), limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 2
    assert {h["key"] for h in page1}.isdisjoint({h["key"] for h in page2})


def test_list_namespaces(kv_store):
    kv_store.put(("users", "1"), "a", {})
    kv_store.put(("users", "2"), "b", {})
    kv_store.put(("orgs",), "c", {})
    all_ns = kv_store.list_namespaces()
    assert ("users", "1") in all_ns
    assert ("orgs",) in all_ns


def test_list_namespaces_max_depth(kv_store):
    kv_store.put(("users", "1", "prefs"), "a", {})
    kv_store.put(("users", "2"), "b", {})
    depth1 = kv_store.list_namespaces(max_depth=1)
    assert ("users",) in depth1
    assert ("users", "1", "prefs") not in depth1


def test_list_namespaces_prefix_and_suffix(kv_store):
    kv_store.put(("a", "b"), "k", {})
    kv_store.put(("a", "c"), "k", {})
    kv_store.put(("x", "b"), "k", {})
    assert kv_store.list_namespaces(prefix=("a",)) == [("a", "b"), ("a", "c")]
    assert kv_store.list_namespaces(suffix=("b",)) == [("a", "b"), ("x", "b")]


def test_delete(kv_store):
    kv_store.put(("k",), "one", {"v": 1})
    assert kv_store.get(("k",), "one") is not None
    kv_store.delete(("k",), "one")
    assert kv_store.get(("k",), "one") is None


def test_delete_is_scoped_to_key(kv_store):
    kv_store.put(("k",), "one", {"v": 1})
    kv_store.put(("k",), "two", {"v": 2})
    kv_store.delete(("k",), "one")
    assert kv_store.get(("k",), "one") is None
    assert kv_store.get(("k",), "two") is not None


def test_reserved_separator_rejected(kv_store):
    import pytest
    # F4: a namespace element or key with the reserved separators must not be
    # allowed to collide onto another (namespace, key)'s session id.
    for ns, key in [(("a\x1fb",), "c"), (("a",), "c\x1ed"), (("x\x1e",), "y")]:
        with pytest.raises(ValueError, match="reserved separator"):
            kv_store.put(ns, key, {"v": 1})
    # The legitimate deeper namespace is unaffected.
    kv_store.put(("a", "b"), "c", {"v": 99})
    assert kv_store.get(("a", "b"), "c")["value"] == {"v": 99}


def test_empty_suffix_matches_all(kv_store):
    # F5: list_namespaces(suffix=()) must match every namespace, not filter all.
    kv_store.put(("users", "1"), "a", {})
    kv_store.put(("orgs",), "b", {})
    ns = kv_store.list_namespaces(suffix=())
    assert ("users", "1") in ns and ("orgs",) in ns
