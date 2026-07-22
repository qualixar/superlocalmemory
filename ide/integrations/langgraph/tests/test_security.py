"""Security tests for the V3 key-value store core.

SQL injection via namespace / key / value, oversized values, and edge-case
identifiers. These run without ``langgraph`` installed.
"""

import pytest


def test_sql_injection_in_namespace_and_key(kv_store):
    ns = ("'; DROP TABLE memories; --", "x")
    kv_store.put(ns, "'; DELETE FROM memories; --", {"ok": True})
    env = kv_store.get(ns, "'; DELETE FROM memories; --")
    assert env is not None and env["value"] == {"ok": True}
    # The table still exists and other rows are intact.
    kv_store.put(("safe",), "k", {"v": 1})
    assert kv_store.get(("safe",), "k")["value"] == {"v": 1}


def test_sql_injection_in_value(kv_store):
    payload = {"note": "'); DROP TABLE atomic_facts; --"}
    kv_store.put(("v",), "k", payload)
    assert kv_store.get(("v",), "k")["value"] == payload


def test_like_wildcards_in_namespace_are_literal(kv_store):
    # '%' and '_' must not act as LIKE wildcards during prefix search.
    kv_store.put(("a%b",), "k", {"n": 1})
    kv_store.put(("axb",), "k", {"n": 2})
    hits = kv_store.search(("a%b",))
    assert {tuple(h["namespace"]) for h in hits} == {("a%b",)}


def test_oversized_value_raises(kv_store):
    big = {"blob": "x" * 1_000_001}
    with pytest.raises(ValueError, match="exceeds maximum size"):
        kv_store.put(("big",), "k", big)


def test_empty_namespace_and_key(kv_store):
    kv_store.put((), "", {"edge": True})
    assert kv_store.get((), "")["value"] == {"edge": True}


def test_unicode_roundtrip(kv_store):
    value = {"text": "Ada — 日本語 — emoji 🎯"}
    kv_store.put(("u",), "k", value)
    assert kv_store.get(("u",), "k")["value"] == value
