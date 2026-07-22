"""Functional + security tests for the framework-free CrewAI store core.

Run without ``crewai`` installed (loaded via conftest by file path) against the
real SuperLocalMemory engine on a temp database.
"""

import math
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _vec(dim: int, value: float = 1.0) -> list:
    """Unit-like vector of given dimension."""
    return [value] * dim


def _unit(v: list) -> list:
    """Normalise vector to unit length."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-9:
        return v
    return [x / norm for x in v]


def _rec(rid: str = "r1", scope: str = "/", category: str = "general",
         content: str = "test content for the record store") -> dict:
    """Minimal record dict for the framework-free core."""
    return {
        "id": rid,
        "content": content,
        "embedding": _unit(_vec(4)),
        "scope": scope,
        "category": category,
        "metadata": {},
    }


# ── save and get ─────────────────────────────────────────────────────────────

def test_save_and_get_record(crewai_store):
    crewai_store.save(_rec("r1", "/project/alpha"))
    rec = crewai_store.get_record("r1")
    assert rec is not None
    assert rec["id"] == "r1"
    assert rec["scope"] == "/project/alpha"


def test_get_record_missing(crewai_store):
    assert crewai_store.get_record("nope") is None


def test_save_is_idempotent_on_same_id(crewai_store):
    crewai_store.save(_rec("r1", content="original"))
    crewai_store.save(_rec("r1", content="updated"))
    rec = crewai_store.get_record("r1")
    assert rec["content"] == "updated"


# ── update ───────────────────────────────────────────────────────────────────

def test_update_replaces_record(crewai_store):
    crewai_store.save(_rec("r1", content="v1"))
    updated = _rec("r1", content="v2")
    crewai_store.update(updated)
    assert crewai_store.get_record("r1")["content"] == "v2"


# ── delete ────────────────────────────────────────────────────────────────────

def test_delete_by_id(crewai_store):
    crewai_store.save(_rec("r1"))
    crewai_store.save(_rec("r2"))
    removed = crewai_store.delete_by_id("r1")
    assert removed == 1
    assert crewai_store.get_record("r1") is None
    assert crewai_store.get_record("r2") is not None


def test_delete_by_id_missing_returns_zero(crewai_store):
    assert crewai_store.delete_by_id("ghost") == 0


def test_delete_by_scope(crewai_store):
    crewai_store.save(_rec("r1", scope="/a"))
    crewai_store.save(_rec("r2", scope="/a/sub"))
    crewai_store.save(_rec("r3", scope="/b"))
    removed = crewai_store.delete_by_scope("/a")
    assert removed == 2
    assert crewai_store.get_record("r1") is None
    assert crewai_store.get_record("r2") is None
    assert crewai_store.get_record("r3") is not None


def test_delete_by_scope_root(crewai_store):
    crewai_store.save(_rec("r1", scope="/a"))
    crewai_store.save(_rec("r2", scope="/b"))
    removed = crewai_store.delete_by_scope("/")
    assert removed == 2


# ── list_records ─────────────────────────────────────────────────────────────

def test_list_records_all(crewai_store):
    crewai_store.save(_rec("r1", scope="/a"))
    crewai_store.save(_rec("r2", scope="/b"))
    recs = crewai_store.list_records()
    assert {r["id"] for r in recs} == {"r1", "r2"}


def test_list_records_scope_prefix(crewai_store):
    crewai_store.save(_rec("r1", scope="/project"))
    crewai_store.save(_rec("r2", scope="/project/alpha"))
    crewai_store.save(_rec("r3", scope="/projectx"))  # must NOT match
    recs = crewai_store.list_records(scope_prefix="/project")
    ids = {r["id"] for r in recs}
    assert ids == {"r1", "r2"}
    assert "r3" not in ids


def test_list_records_limit_and_offset(crewai_store):
    for i in range(6):
        crewai_store.save(_rec(f"r{i}", scope="/s"))
    page1 = crewai_store.list_records(limit=3, offset=0)
    page2 = crewai_store.list_records(limit=3, offset=3)
    assert len(page1) == 3 and len(page2) == 3
    ids1 = {r["id"] for r in page1}
    ids2 = {r["id"] for r in page2}
    assert ids1.isdisjoint(ids2)


# ── count ─────────────────────────────────────────────────────────────────────

def test_count_all(crewai_store):
    assert crewai_store.count() == 0
    crewai_store.save(_rec("r1"))
    crewai_store.save(_rec("r2"))
    assert crewai_store.count() == 2


def test_count_by_scope(crewai_store):
    crewai_store.save(_rec("r1", scope="/a"))
    crewai_store.save(_rec("r2", scope="/a/sub"))
    crewai_store.save(_rec("r3", scope="/b"))
    assert crewai_store.count(scope_prefix="/a") == 2
    assert crewai_store.count(scope_prefix="/b") == 1


# ── list_scopes and list_categories ──────────────────────────────────────────

def test_list_scopes(crewai_store):
    crewai_store.save(_rec("r1", scope="/a"))
    crewai_store.save(_rec("r2", scope="/a/sub"))
    crewai_store.save(_rec("r3", scope="/b"))
    scopes = crewai_store.list_scopes(parent="/")
    assert "/a" in scopes and "/b" in scopes


def test_list_categories(crewai_store):
    crewai_store.save(_rec("r1", category="ctx"))
    crewai_store.save(_rec("r2", category="ctx"))
    crewai_store.save(_rec("r3", category="short_term"))
    cats = crewai_store.list_categories()
    assert cats.get("ctx") == 2
    assert cats.get("short_term") == 1


# ── reset ─────────────────────────────────────────────────────────────────────

def test_reset_all(crewai_store):
    crewai_store.save(_rec("r1"))
    crewai_store.save(_rec("r2"))
    crewai_store.reset()
    assert crewai_store.count() == 0


def test_reset_by_scope(crewai_store):
    crewai_store.save(_rec("r1", scope="/a"))
    crewai_store.save(_rec("r2", scope="/b"))
    crewai_store.reset(scope_prefix="/a")
    assert crewai_store.count() == 1
    assert crewai_store.get_record("r2") is not None


# ── search (cosine similarity) ────────────────────────────────────────────────

def test_search_returns_sorted_by_score(crewai_store):
    # Store two records: r_close has embedding aligned with query, r_far is orthogonal.
    query_emb = _unit([1.0, 0.0, 0.0, 0.0])
    close_emb = _unit([0.9, 0.1, 0.0, 0.0])
    far_emb = _unit([0.0, 0.0, 1.0, 0.0])

    r_close = _rec("r_close", content="close to query vector")
    r_close["embedding"] = close_emb
    r_far = _rec("r_far", content="far from query vector")
    r_far["embedding"] = far_emb

    crewai_store.save(r_close)
    crewai_store.save(r_far)

    results = crewai_store.search(query_emb, limit=10)
    assert len(results) == 2
    # First result must have higher score
    assert results[0][1] >= results[1][1]
    assert results[0][0]["id"] == "r_close"


def test_search_min_score_filters(crewai_store):
    query_emb = _unit([1.0, 0.0, 0.0, 0.0])
    close_emb = _unit([0.99, 0.01, 0.0, 0.0])
    far_emb = _unit([0.0, 0.0, 0.0, 1.0])

    r_close = _rec("r_close")
    r_close["embedding"] = close_emb
    r_far = _rec("r_far")
    r_far["embedding"] = far_emb

    crewai_store.save(r_close)
    crewai_store.save(r_far)

    # High min_score: only the close one should pass
    results = crewai_store.search(query_emb, min_score=0.5, limit=10)
    ids = [r[0]["id"] for r in results]
    assert "r_close" in ids
    assert "r_far" not in ids


def test_search_empty_query_embedding_returns_empty(crewai_store):
    crewai_store.save(_rec("r1"))
    # Zero-vector query should not crash; may return 0.0 scores
    results = crewai_store.search([0.0, 0.0, 0.0, 0.0], limit=10)
    # Must not raise, must return list
    assert isinstance(results, list)
    # All scores must be 0.0 (zero-vector guard)
    for _, score in results:
        assert score == 0.0


def test_search_scope_prefix_filter(crewai_store):
    q = _unit([1.0, 0.0, 0.0, 0.0])
    r1 = _rec("r1", scope="/a")
    r1["embedding"] = _unit([1.0, 0.0, 0.0, 0.0])
    r2 = _rec("r2", scope="/b")
    r2["embedding"] = _unit([1.0, 0.0, 0.0, 0.0])

    crewai_store.save(r1)
    crewai_store.save(r2)

    results = crewai_store.search(q, scope_prefix="/a", limit=10)
    ids = [r[0]["id"] for r in results]
    assert "r1" in ids
    assert "r2" not in ids


def test_search_category_filter(crewai_store):
    q = _unit([1.0, 0.0, 0.0, 0.0])
    r1 = _rec("r1", category="ctx")
    r1["embedding"] = _unit([1.0, 0.0, 0.0, 0.0])
    r2 = _rec("r2", category="short_term")
    r2["embedding"] = _unit([1.0, 0.0, 0.0, 0.0])

    crewai_store.save(r1)
    crewai_store.save(r2)

    results = crewai_store.search(q, categories=["ctx"], limit=10)
    ids = [r[0]["id"] for r in results]
    assert "r1" in ids
    assert "r2" not in ids


def test_search_limit_respected(crewai_store):
    q = _unit([1.0, 0.0, 0.0, 0.0])
    for i in range(5):
        r = _rec(f"r{i}")
        r["embedding"] = _unit([1.0, 0.0, 0.0, 0.0])
        crewai_store.save(r)
    results = crewai_store.search(q, limit=3)
    assert len(results) <= 3


# ── security: SQL injection + LIKE wildcards ──────────────────────────────────

def test_sql_injection_in_record_id(crewai_store):
    rid = "'; DROP TABLE memories; --"
    crewai_store.save(_rec(rid))
    rec = crewai_store.get_record(rid)
    assert rec is not None
    assert rec["id"] == rid


def test_like_wildcards_in_scope_are_literal(crewai_store):
    crewai_store.save(_rec("r1", scope="/a%b"))
    crewai_store.save(_rec("r2", scope="/axb"))
    # Scope '/a%b' must NOT match '/axb' — % is literal, not wildcard.
    recs = crewai_store.list_records(scope_prefix="/a%b")
    ids = {r["id"] for r in recs}
    assert ids == {"r1"}


def test_unicode_roundtrip(crewai_store):
    content = "Ada — 日本語 — 🎯"
    r = _rec("u1", content=content)
    crewai_store.save(r)
    rec = crewai_store.get_record("u1")
    assert rec["content"] == content


def test_oversized_content_raises(crewai_store):
    r = _rec("big", content="x" * 1_000_001)
    with pytest.raises(ValueError, match="exceeds maximum size"):
        crewai_store.save(r)
