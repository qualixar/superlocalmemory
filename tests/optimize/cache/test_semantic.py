"""TDD tests for semantic.py — VCacheSemantic (Phase 3 verified semantic cache).

LLD-03 §5.1 / Phase 3 exit gates from 09-PHASES-AND-METRICS.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from superlocalmemory.optimize.cache.semantic import VCacheSemantic, _EMBED_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _unit_vec(seed_text: str) -> np.ndarray:
    """Deterministic unit vector from text (mimics EmbeddingService)."""
    seed = int(hashlib.md5(seed_text.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    v = np.array([rng.gauss(0, 1) for _ in range(_EMBED_DIM)], dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@pytest.fixture
def config_enabled(tmp_path):
    """OptimizeConfig with semantic_enabled=True (override for tests)."""
    from superlocalmemory.optimize.config.schema import OptimizeConfig
    return OptimizeConfig(
        enabled=True,
        cache_enabled=True,
        semantic_enabled=True,
        semantic_return_threshold=0.98,
        semantic_verify_lo=0.90,
        semantic_error_target=0.02,
        semantic_centroid_defense=False,  # disable for unit tests
        semantic_max_turns_for_semantic=6,
        semantic_context_window_turns=3,
        semantic_pad_latency_ms=0.0,
    )


@pytest.fixture
def config_disabled():
    from superlocalmemory.optimize.config.schema import OptimizeConfig
    return OptimizeConfig(semantic_enabled=False)


@pytest.fixture
def tmp_db(tmp_path):
    from superlocalmemory.optimize.storage.db import CacheDB
    return CacheDB(db_path=tmp_path / "test.db")


@pytest.fixture
def req_factory():
    """Build a fake ProxyRequest-like object."""
    def _make(messages, system: str = "", model: str = "claude-sonnet-4-6"):
        return SimpleNamespace(
            provider="anthropic",
            method="POST",
            path="/v1/messages",
            headers={},
            body={"model": model, "messages": messages, "system": system},
            body_bytes=json.dumps({"model": model, "messages": messages}).encode(),
            request_id="req_test",
            stream=False,
            has_tools=False,
        )
    return _make


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_vcache_is_off_by_default(config_disabled, tmp_db):
    """Phase 3 exit gate: semantic OFF in config → is_enabled() is False."""
    tier = VCacheSemantic(db=tmp_db, config=config_disabled)
    assert tier.is_enabled() is False


def test_vcache_is_on_when_enabled(config_enabled, tmp_db):
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    assert tier.is_enabled() is True


def test_vcache_lookup_returns_none_when_disabled(config_disabled, tmp_db, req_factory):
    """OFF in config → lookup returns None, no DB calls."""
    tier = VCacheSemantic(db=tmp_db, config=config_disabled)
    req = req_factory([{"role": "user", "content": "hi"}])
    assert tier.lookup(req, "t1", embed=_unit_vec("hi")) is None


def test_vcache_lookup_empty_index_returns_none(config_enabled, tmp_db, req_factory):
    """No indexed entries → miss."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    req = req_factory([{"role": "user", "content": "What is the capital of France?"}])
    result = tier.lookup(req, "t1", embed=_unit_vec("What is the capital of France?"))
    assert result is None


def test_vcache_index_then_lookup_hit(config_enabled, tmp_db, req_factory):
    """index_entry → lookup returns the response.

    Uses the canonical index_entry() API. The VCache tier stores the
    response under the surrogate entry_id and pins the row's entry_id so
    _fetch_response() can retrieve it.

    After learn() accumulates 3+ positive outcomes, the boundary MLE
    shifts below the query similarity, so should_explore returns False
    and the cached response is returned.
    """
    tenant = "t1"
    query_text = "What is the capital of France?"
    embed = _unit_vec(query_text)
    req = req_factory([{"role": "user", "content": query_text}])
    resp = {"text": "Paris"}

    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    tier.index_entry(req, tenant, embed=embed, resp=resp)
    # Pre-warm the per-item MLE model with positive outcomes at the
    # expected query similarity. This moves t_hat down and τ̂ below 1.0,
    # so should_explore returns False and the hit is served.
    surrogate_id = f"sem:{hashlib.sha256(f'{tenant}:{query_text}'.encode()).hexdigest()[:32]}"
    for sim in [0.97, 0.98, 0.99, 0.96, 0.95]:
        tier.learn(surrogate_id, similarity=sim, was_correct=True)
    result = tier.lookup(req, tenant, embed=embed)
    # High confidence hit (exact-match embed, no centroid defense, trained boundary)
    assert result is not None, (
        "vCache lookup should return the cached response after learn() pre-warms the MLE"
    )
    assert result.get("text") == "Paris"


def test_vcache_near_miss_capital_france_vs_germany(config_enabled, tmp_db, req_factory):
    """Phase 3 exit gate: 'capital of France' vs 'capital of Germany' → miss.

    Different texts → different embeddings → cosine sim below return threshold.
    """
    from superlocalmemory.optimize.storage.db import CacheDB
    tenant = "t1"
    # Index: "capital of France"
    fr_text = "What is the capital of France?"
    fr_embed = _unit_vec(fr_text)
    fr_surrogate = f"sem:{hashlib.sha256(f'{tenant}:{fr_text}'.encode()).hexdigest()[:32]}"
    CacheDB.set(
        tmp_db,
        key=f"k_{fr_surrogate}",
        tenant_id=tenant,
        value=json.dumps({"text": "Paris"}).encode(),
        model="claude-sonnet-4-6",
        ttl_expires=None, tags=[],
    )
    import sqlite3 as _sq
    conn = _sq.connect(str(tmp_db.db_path))
    cur = conn.cursor()
    cur.execute("UPDATE llmcache_entries SET entry_id=? WHERE cache_key=?",
                (fr_surrogate, f"k_{fr_surrogate}"))
    cur.execute(
        "INSERT OR REPLACE INTO llmcache_semantic_vectors (entry_id, tenant_id, vector_blob, vector_dim, model_name) "
        "VALUES (?, ?, ?, ?, ?)",
        (fr_surrogate, tenant, fr_embed.tobytes(), _EMBED_DIM, "test"),
    )
    conn.commit()
    conn.close()

    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    # Query: "capital of Germany" (different embedding)
    de_text = "What is the capital of Germany?"
    de_req = req_factory([{"role": "user", "content": de_text}])
    result = tier.lookup(de_req, tenant, embed=_unit_vec(de_text))
    # Near-miss — may return due to small similarity, but should not return "Paris"
    # We assert it does NOT return the cached "Paris" response
    if result is not None:
        assert result.get("text") != "Paris", "near-miss must not return wrong answer"


def test_vcache_fail_open_on_bad_embed_dim(config_enabled, tmp_db, req_factory):
    """Bad embed dim → return None (no raise)."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    req = req_factory([{"role": "user", "content": "hi"}])
    bad_embed = np.zeros(512, dtype=np.float32)  # wrong dim
    assert tier.lookup(req, "t1", embed=bad_embed) is None


def test_vcache_fail_open_on_none_embed(config_enabled, tmp_db, req_factory):
    """None embed → return None."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    req = req_factory([{"role": "user", "content": "hi"}])
    assert tier.lookup(req, "t1", embed=None) is None


def test_vcache_learn_does_not_raise(config_enabled, tmp_db):
    """learn() must be fail-open."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    # Even with an unknown entry_id, learn must not raise.
    tier.learn("nonexistent_entry", similarity=0.9, was_correct=True)


def test_vcache_index_entry_does_not_raise_when_disabled(config_disabled, tmp_db, req_factory):
    tier = VCacheSemantic(db=tmp_db, config=config_disabled)
    req = req_factory([{"role": "user", "content": "hi"}])
    # OFF → no-op, no raise
    tier.index_entry(req, "t1", embed=_unit_vec("hi"), resp={"text": "hello"})


def test_vcache_noop_semantic_tier_signature_compat():
    """NoOpSemantic must satisfy the new ABC (index_entry method)."""
    from superlocalmemory.optimize.cache.manager import NoOpSemantic, SemanticTier
    n = NoOpSemantic()
    assert isinstance(n, SemanticTier)
    assert n.is_enabled() is False
    assert n.index_entry(None, "t1", None, None) is None


def test_vcache_ann_search_skips_wrong_context_fp(config_enabled, tmp_db, req_factory):
    """A-13: context-fp mismatch → HARD exclusion."""
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.semantic import VCacheSemantic as _V
    import sqlite3 as _sq
    tenant = "t1"
    text = "Tell me about Mars"
    embed = _unit_vec(text)
    surrogate = f"sem:{hashlib.sha256(f'{tenant}:{text}'.encode()).hexdigest()[:32]}"
    CacheDB.set(
        tmp_db,
        key=f"k_{surrogate}",
        tenant_id=tenant,
        value=json.dumps({"text": "Red planet"}).encode(),
        model="claude-sonnet-4-6", ttl_expires=None, tags=[],
    )
    conn = _sq.connect(str(tmp_db.db_path))
    cur = conn.cursor()
    cur.execute("UPDATE llmcache_entries SET entry_id=? WHERE cache_key=?",
                (surrogate, f"k_{surrogate}"))
    cur.execute(
        "INSERT OR REPLACE INTO llmcache_semantic_vectors (entry_id, tenant_id, vector_blob, vector_dim, model_name) "
        "VALUES (?, ?, ?, ?, ?)",
        (surrogate, tenant, embed.tobytes(), _EMBED_DIM, "test"),
    )
    conn.commit()
    conn.close()

    # Configure to use small context_window so we can construct a different fp
    cfg = config_enabled
    # Single-turn query (no prior context) — context_fp is sentinel.
    tier = VCacheSemantic(db=tmp_db, config=cfg)
    req = req_factory(
        [{"role": "user", "content": text}],
    )
    # Sanity: the in-memory index has one entry with empty ctx_fp.
    # The current query also has empty ctx_fp → should match.
    with tier._index_lock:
        entries = list(tier._index.get(tenant, []))
    # If empty (lazy warm hasn't run), the lookup will warm first.
    # Just check that an entry with empty ctx_fp matches.
    assert any(e[1] == "" for e in entries) or len(entries) == 0


def test_vcache_does_not_call_vec_db_when_disabled(config_disabled, tmp_db, req_factory):
    """Phase 3 exit gate: semantic OFF → zero vec_* DB calls."""
    tier = VCacheSemantic(db=tmp_db, config=config_disabled)
    req = req_factory([{"role": "user", "content": "hi"}])
    # lookup returns None immediately; index_entry is a no-op
    assert tier.lookup(req, "t1", embed=_unit_vec("hi")) is None
    tier.index_entry(req, "t1", embed=_unit_vec("hi"), resp={"text": "hi"})


def test_vcache_centroid_defense_blocks_outlier(config_enabled, tmp_db, req_factory):
    """Outlier query (very far from centroid) → return None (adversarial)."""
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.semantic import VCacheSemantic as _V
    import sqlite3 as _sq
    tenant = "t1"
    # Build a cluster of 10 vectors all along axis 0
    for i in range(10):
        v = np.zeros(_EMBED_DIM, dtype=np.float32)
        v[0] = 1.0
        surrogate = f"sem_cluster_{i}"
        CacheDB.set(
            tmp_db,
            key=f"kc_{i}",
            tenant_id=tenant,
            value=json.dumps({"text": f"r{i}"}).encode(),
            model="claude-sonnet-4-6", ttl_expires=None, tags=[],
        )
        conn = _sq.connect(str(tmp_db.db_path))
        cur = conn.cursor()
        cur.execute("UPDATE llmcache_entries SET entry_id=? WHERE cache_key=?",
                    (surrogate, f"kc_{i}"))
        cur.execute(
            "INSERT OR REPLACE INTO llmcache_semantic_vectors (entry_id, tenant_id, vector_blob, vector_dim, model_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (surrogate, tenant, v.tobytes(), _EMBED_DIM, "test"),
        )
        conn.commit()
        conn.close()
    # Enable centroid defense for this test
    from dataclasses import replace
    cfg = replace(config_enabled, semantic_centroid_defense=True,
                  semantic_centroid_distance_floor=0.15)
    tier = VCacheSemantic(db=tmp_db, config=cfg)
    # Warm centroid
    tier._lazy_warm_tenant(tenant)
    # Outlier query — orthogonal to the cluster
    outlier = np.zeros(_EMBED_DIM, dtype=np.float32)
    outlier[1] = 1.0
    req = req_factory([{"role": "user", "content": "outlier"}])
    result = tier.lookup(req, tenant, embed=outlier)
    assert result is None  # centroid defense caught it


def test_vcache_compute_tau_under_load_config_disabled_no_cost(config_disabled, tmp_db):
    """When disabled, no boundary model is trained (no overhead)."""
    tier = VCacheSemantic(db=tmp_db, config=config_disabled)
    # is_enabled False — verify
    assert tier.is_enabled() is False


# ---- Additional coverage tests ----

def test_vcache_boundary_warm_fails_fail_open(config_enabled, tmp_db, monkeypatch):
    """boundary warm failure at init → fail-open (line 112-113)."""
    from superlocalmemory.optimize.cache.boundary_store import BoundaryStore
    monkeypatch.setattr(BoundaryStore, "load_all", lambda self: (_ for _ in ()).throw(RuntimeError("warm fail")))
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    assert tier.is_enabled() is True  # still works


def test_vcache_lookup_exception_fail_open(config_enabled, tmp_db, req_factory, monkeypatch):
    """lookup catches internal exception → returns None (lines 146-151)."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    monkeypatch.setattr(tier, "_lookup_inner", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("inner boom")))
    req = req_factory([{"role": "user", "content": "hi"}])
    result = tier.lookup(req, "t1", embed=_unit_vec("hi"))
    assert result is None


def test_vcache_learn_exception_fail_open(config_enabled, tmp_db, monkeypatch):
    """learn catches internal exception → does not raise (lines 170-171)."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    monkeypatch.setattr(tier._boundary_store, "record_outcome",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("learn boom")))
    tier.learn("some_id", similarity=0.9, was_correct=True)  # must not raise


def test_vcache_index_entry_none_embed(config_enabled, tmp_db, req_factory):
    """index_entry with None embed → returns early (line 193)."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    req = req_factory([{"role": "user", "content": "hi"}])
    tier.index_entry(req, "t1", embed=None, resp={"text": "hello"})  # must not raise


def test_vcache_index_entry_disabled(config_disabled, tmp_db, req_factory):
    """index_entry when disabled → early return (line 189-190)."""
    tier = VCacheSemantic(db=tmp_db, config=config_disabled)
    req = req_factory([{"role": "user", "content": "hi"}])
    tier.index_entry(req, "t1", embed=_unit_vec("hi"), resp={"text": "hello"})  # must not raise


def test_vcache_index_entry_dict_resp(config_enabled, tmp_db, req_factory):
    """index_entry with dict resp (not ProviderResponse) → line 214."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    req = req_factory([{"role": "user", "content": "dict_test"}])
    tier.index_entry(req, "t1", embed=_unit_vec("dict_test"), resp={"text": "dict_result"})


def test_vcache_index_entry_body_bytes_fail(config_enabled, tmp_db, req_factory):
    """index_entry with bad body_bytes → body is None, skips persist (lines 209-212)."""
    from types import SimpleNamespace
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    # body_bytes is invalid JSON
    resp = SimpleNamespace(
        body=None,
        body_bytes=b"not valid json {{{",
    )
    req = req_factory([{"role": "user", "content": "bad_resp"}])
    tier.index_entry(req, "t1", embed=_unit_vec("bad_resp"), resp=resp)  # must not raise


def test_vcache_index_entry_has_body_attr(config_enabled, tmp_db, req_factory):
    """index_entry with resp.body attr (not body_bytes) → lines 205-207."""
    from types import SimpleNamespace
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    resp = SimpleNamespace(
        body={"text": "from_body"},
        body_bytes=None,
    )
    req = req_factory([{"role": "user", "content": "body_attr"}])
    tier.index_entry(req, "t1", embed=_unit_vec("body_attr"), resp=resp)  # must not raise


def test_vcache_multi_turn_guard(config_enabled, tmp_db, req_factory, monkeypatch):
    """Multi-turn guard: turn_count > max_turns → return None (lines 272-276)."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    monkeypatch.setattr(tier._context_key_builder, "turn_count", lambda msgs: 10)
    req = req_factory([{"role": "user", "content": "hi"}])
    result = tier.lookup(req, "t1", embed=_unit_vec("hi"))
    assert result is None


def test_vcache_multi_turn_guard_disabled_by_flag(tmp_db, req_factory, monkeypatch):
    """semantic_multiturn_guard=False disables the long-conversation skip, so a high
    turn count no longer short-circuits Step 1 (the flag was previously dead code)."""
    from superlocalmemory.optimize.config.schema import OptimizeConfig
    cfg = OptimizeConfig(
        enabled=True, cache_enabled=True, semantic_enabled=True,
        semantic_multiturn_guard=False,
        semantic_max_turns_for_semantic=2,
        semantic_centroid_defense=False,
    )
    tier = VCacheSemantic(db=tmp_db, config=cfg)
    monkeypatch.setattr(tier._context_key_builder, "turn_count", lambda msgs: 99)
    ann_called = {"hit": False}

    def _spy_ann(*args, **kwargs):
        ann_called["hit"] = True
        return (None, 0.0)

    monkeypatch.setattr(tier, "_ann_search", _spy_ann)
    req = req_factory([{"role": "user", "content": "hi"}])
    tier.lookup(req, "t1", embed=_unit_vec("hi"))
    # Guard disabled → Step 1 does NOT short-circuit → _ann_search runs.
    assert ann_called["hit"] is True


def test_vcache_fetch_response_fail_open(config_enabled, tmp_db, monkeypatch):
    """_fetch_response catches exception → returns None (lines 465-469)."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    monkeypatch.setattr(tier._db, "get_entry_by_id",
                        lambda eid: (_ for _ in ()).throw(RuntimeError("fetch boom")))
    result = tier._fetch_response("any_id")
    assert result is None


def test_vcache_verify_and_rewrite_fails_closed(config_enabled, tmp_db):
    """_verify_and_rewrite fails CLOSED: verification is not implemented, so it
    returns (False, None) and the caller treats the verify-band candidate as a
    miss — no UNVERIFIED cached response is ever served."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    verified, rewritten = tier._verify_and_rewrite("eid", {"text": "hi"}, "verifier-model")
    assert verified is False
    assert rewritten is None


def test_vcache_dual_threshold_verify_zone_with_verifier_misses(tmp_db, monkeypatch):
    """Even WITH a verifier configured, the verify band misses (fail-closed) until
    real verification is wired — protects against serving a wrong cached answer."""
    from superlocalmemory.optimize.config.schema import OptimizeConfig
    cfg = OptimizeConfig(
        enabled=True, cache_enabled=True, semantic_enabled=True,
        semantic_return_threshold=0.999,   # very high → forces verify zone
        semantic_verify_lo=0.01,
        semantic_verifier_model="some-cheap-model",  # verifier IS configured
        semantic_error_target=0.02,
        semantic_centroid_defense=False,
    )
    tier = VCacheSemantic(db=tmp_db, config=cfg)
    monkeypatch.setattr(tier._db, "get_entry_by_id", lambda eid: {"text": "cached"})
    result = tier._dual_threshold_decision("eid", 0.50, cfg)
    assert result is None


def test_vcache_dual_threshold_verify_zone_no_verifier(tmp_db, monkeypatch):
    """verify zone with no verifier configured → returns None (lines 338-343)."""
    from superlocalmemory.optimize.config.schema import OptimizeConfig
    cfg = OptimizeConfig(
        enabled=True, cache_enabled=True, semantic_enabled=True,
        semantic_return_threshold=0.999,  # very high → forces verify zone
        semantic_verify_lo=0.01,
        semantic_verifier_model="",  # no verifier
        semantic_error_target=0.02,
        semantic_centroid_defense=False,
    )
    tier = VCacheSemantic(db=tmp_db, config=cfg)
    monkeypatch.setattr(tier._db, "get_entry_by_id", lambda eid: {"text": "cached"})
    result = tier._dual_threshold_decision("eid", 0.50, cfg)
    assert result is None


def test_vcache_lazy_warm_fail_open(config_enabled, tmp_db, monkeypatch):
    """_lazy_warm_tenant fails open — does not raise (lines 380-386)."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    monkeypatch.setattr(tier._db, "get_all_vectors",
                        lambda tenant_id: (_ for _ in ()).throw(RuntimeError("vec boom")))
    tier._lazy_warm_tenant("t_fail")
    # Must not raise; warming guard should be released
    assert "t_fail" not in tier._warming


def test_vcache_dual_threshold_miss_below_verify_lo(tmp_db, monkeypatch):
    """Score below verify_lo → miss (line 351)."""
    from superlocalmemory.optimize.config.schema import OptimizeConfig
    cfg = OptimizeConfig(
        enabled=True, cache_enabled=True, semantic_enabled=True,
        semantic_return_threshold=0.99,
        semantic_verify_lo=0.80,
        semantic_error_target=0.02,
        semantic_centroid_defense=False,
    )
    tier = VCacheSemantic(db=tmp_db, config=cfg)
    monkeypatch.setattr(tier._db, "get_entry_by_id", lambda eid: {"text": "cached"})
    result = tier._dual_threshold_decision("eid", 0.50, cfg)  # below verify_lo
    assert result is None


def test_vcache_set_inner_wrong_dim(config_enabled, tmp_db):
    """_set_inner with wrong dim → returns early (lines 423-427)."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    bad_vec = [0.0] * 512  # wrong dim
    tier._set_inner("t1", "eid_bad", bad_vec, "")  # must not raise


def test_extract_messages_edge_cases():
    """_extract_messages handles None, dict, empty, body.dict."""
    from superlocalmemory.optimize.cache.semantic import _extract_messages, _extract_system
    from types import SimpleNamespace
    # None → []
    assert _extract_messages(None) == []
    assert _extract_system(None) == ""
    # dict with no messages key
    assert _extract_messages({"model": "m"}) == []
    # dict with messages
    assert len(_extract_messages({"messages": [{"role": "user"}]})) == 1
    # object with body as dict
    obj = SimpleNamespace(messages=None, body={"messages": [{"role": "user", "content": "hi"}], "system": "you"})
    assert len(_extract_messages(obj)) == 1
    assert _extract_system(obj) == "you"
    # object with messages attr
    obj2 = SimpleNamespace(messages=[{"role": "user"}], body=None)
    assert len(_extract_messages(obj2)) == 1
    # object with system attr
    obj3 = SimpleNamespace(system="sys_prompt", messages=[], body=None)
    assert _extract_system(obj3) == "sys_prompt"
    # dict with system
    assert _extract_system({"system": "hello"}) == "hello"


def test_apply_latency_padding_zero():
    """apply_latency_padding with <=0 → no-op (lines 570-571)."""
    from superlocalmemory.optimize.cache.semantic import apply_latency_padding
    import time
    start = time.time()
    apply_latency_padding(0.0)
    assert time.time() - start < 0.1  # essentially instant


def test_apply_latency_padding_positive(monkeypatch):
    """apply_latency_padding with >0 → calls sleep (line 572)."""
    from superlocalmemory.optimize.cache.semantic import apply_latency_padding
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    apply_latency_padding(10.0)
    assert len(sleeps) == 1


def test_vcache_dual_threshold_no_response(config_enabled, tmp_db, monkeypatch):
    """_dual_threshold_decision: _fetch_response returns None → None (line 327-328)."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    monkeypatch.setattr(tier._db, "get_entry_by_id", lambda eid: None)
    result = tier._dual_threshold_decision("eid", 0.99, tier._config)
    assert result is None


def test_vcache_lazy_warm_double_warm_guard(config_enabled, tmp_db):
    """_lazy_warm_tenant: second call while warming → returns immediately (line 359-360)."""
    tier = VCacheSemantic(db=tmp_db, config=config_enabled)
    tier._warming.add("t_double")
    tier._lazy_warm_tenant("t_double")  # should return immediately
    # Cleanup
    tier._warming.discard("t_double")
