# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03

"""Supplementary coverage tests hitting branches not reached by the primary
TDD tests. Coverage-completion harness — no new behaviour."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from superlocalmemory.learning.arm_catalog import ARM_CATALOG
from superlocalmemory.learning.bandit import (
    BanditChoice,
    ContextualBandit,
    _conn_for,
    compute_stratum,
)
from superlocalmemory.learning.bandit_cache import (
    _BanditCache,
    get_shared_cache,
    reset_shared_cache,
)
from superlocalmemory.learning.ensemble import (
    EnsembleWeights,
    _apply_weights_score,
    _candidate_to_result,
    _parse_blend,
    _softmax_unit,
    ensemble_rerank,
)
from superlocalmemory.learning.reward_proxy import (
    _extract_query,
    _parse_iso,
    _tool_event_hit,
    settle_stale_plays,
)
from superlocalmemory.storage.migration_runner import apply_all


# ---------------------------------------------------------------------------
# bandit_cache
# ---------------------------------------------------------------------------


def test_cache_rejects_zero_max():
    with pytest.raises(ValueError):
        _BanditCache(max_entries=0)


def test_cache_size_reflects_entries():
    cache = _BanditCache(max_entries=8)
    cache.get("p", "s1", lambda p, s: {"a": (1.0, 1.0)})
    cache.get("p", "s2", lambda p, s: {"a": (1.0, 1.0)})
    assert cache.size() == 2


def test_cache_lru_eviction():
    cache = _BanditCache(max_entries=2)
    cache.get("p", "s1", lambda p, s: {"a": (1.0, 1.0)})
    cache.get("p", "s2", lambda p, s: {"a": (1.0, 1.0)})
    # Touch s1 so s2 becomes oldest.
    cache.get("p", "s1", lambda p, s: {"a": (1.0, 1.0)})
    cache.get("p", "s3", lambda p, s: {"a": (1.0, 1.0)})
    assert cache.size() == 2
    # s2 should be evicted; s1 + s3 remain.
    loaded = {"n": 0}
    def _loader(p, s):
        loaded["n"] += 1
        return {"b": (1.0, 1.0)}
    cache.get("p", "s1", _loader)  # hit, no load
    cache.get("p", "s3", _loader)  # hit, no load
    cache.get("p", "s2", _loader)  # miss → load
    assert loaded["n"] == 1


def test_cache_invalidate_absent_is_safe():
    cache = _BanditCache(max_entries=4)
    cache.invalidate("no", "such")  # no raise


def test_cache_clear_drops_everything():
    cache = _BanditCache(max_entries=4)
    cache.get("p", "s1", lambda p, s: {"a": (1.0, 1.0)})
    cache.clear()
    assert cache.size() == 0


def test_cache_concurrent_miss_double_load_safe():
    """Two threads racing on the same key: both may call loader, final value
    deterministic; size == 1."""
    cache = _BanditCache(max_entries=4)
    loads = {"n": 0}
    lock = threading.Lock()

    def _loader(p, s):
        with lock:
            loads["n"] += 1
        time.sleep(0.01)
        return {"a": (1.0, 1.0)}

    results = []
    def _worker():
        results.append(cache.get("p", "s", _loader))
    ts = [threading.Thread(target=_worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert cache.size() == 1
    assert all(r == {"a": (1.0, 1.0)} for r in results)


def test_shared_cache_singleton():
    reset_shared_cache()
    c1 = get_shared_cache()
    c2 = get_shared_cache()
    assert c1 is c2
    reset_shared_cache()
    c3 = get_shared_cache()
    assert c3 is not c1


# ---------------------------------------------------------------------------
# ensemble — helpers
# ---------------------------------------------------------------------------


def test_parse_blend_accepts_valid():
    assert _parse_blend("0.3:0.7", (0.0, 1.0)) == (0.3, 0.7)


def test_parse_blend_rejects_non_sum_to_one():
    # fallback
    assert _parse_blend("0.3:0.5", (0.4, 0.6)) == (0.4, 0.6)


def test_parse_blend_rejects_garbage():
    assert _parse_blend("foo:bar", (0.4, 0.6)) == (0.4, 0.6)
    assert _parse_blend("no-colon", (0.4, 0.6)) == (0.4, 0.6)
    assert _parse_blend(None, (0.4, 0.6)) == (0.4, 0.6)


def test_apply_weights_score_dict_with_score():
    assert _apply_weights_score({"score": 0.42}, {}) == pytest.approx(0.42)


def test_apply_weights_score_dict_no_score_builds_from_channels():
    d = {
        "channel_scores": {"semantic": 1.0, "bm25": 0.5},
        "cross_encoder_score": 0.8,
    }
    w = {"semantic": 2.0, "bm25": 1.0, "entity_graph": 1.0,
         "temporal": 1.0, "cross_encoder_bias": 1.5}
    # 1.0*2.0 + 0.5*1.0 + 0 + 0 + 0.8*1.5 = 2.0 + 0.5 + 1.2 = 3.7
    assert _apply_weights_score(d, w) == pytest.approx(3.7)


def test_apply_weights_score_object_no_score():
    @dataclass
    class C:
        channel_scores: dict = field(default_factory=dict)
        cross_encoder_score: float | None = None
    c = C(channel_scores={"semantic": 0.5}, cross_encoder_score=None)
    w = {"semantic": 1.0}
    assert _apply_weights_score(c, w) == pytest.approx(0.5)


def test_apply_weights_score_unparseable_returns_zero():
    @dataclass
    class BadScore:
        score: Any = "not a number"
        channel_scores: dict = field(default_factory=dict)
    assert _apply_weights_score(BadScore(), {}) == 0.0


def test_candidate_to_result_dict_passthrough():
    assert _candidate_to_result({"fact_id": "f"}) == {"fact_id": "f"}


def test_candidate_to_result_to_result_dict_method():
    class C:
        def to_result_dict(self):
            return {"fact_id": "via-method"}
    assert _candidate_to_result(C()) == {"fact_id": "via-method"}


def test_candidate_to_result_attr_fallback():
    class C:
        fact_id = "x"
        score = 1.5
        channel_scores = {"semantic": 1.0}
        cross_encoder_score = 0.4
    r = _candidate_to_result(C())
    assert r["fact_id"] == "x"
    assert r["score"] == 1.5


def test_ensemble_rerank_feature_build_exception_safe(monkeypatch):
    @dataclass
    class _Booster:
        def predict(self, X):  # pragma: no cover — should not reach
            return [0.0] * X.shape[0]

    @dataclass
    class _Model:
        booster: Any
    # Force FeatureExtractor.extract to raise.
    import superlocalmemory.learning.features as feat

    def _boom(self, *a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(feat.FeatureExtractor, "extract",
                        lambda result, ctx: _boom(None))
    candidates = [{"fact_id": "a"}, {"fact_id": "b"}]

    class _Choice:
        weights = {}
    out = ensemble_rerank(
        candidates, _Choice(), _Model(booster=_Booster()),
        EnsembleWeights(0.4, 0.6), {},
    )
    assert out == candidates


# ---------------------------------------------------------------------------
# reward_proxy helpers
# ---------------------------------------------------------------------------


def test_parse_iso_naive_becomes_utc():
    dt = _parse_iso("2026-04-18T12:00:00")
    assert dt is not None
    assert dt.tzinfo is timezone.utc


def test_parse_iso_malformed_returns_none():
    assert _parse_iso("not-a-date") is None
    assert _parse_iso("") is None


def test_parse_iso_accepts_fallback_format():
    # Already-ISO; passes through the main branch.
    dt = _parse_iso("2026-04-18T12:00:00+00:00")
    assert dt is not None
    assert dt.tzinfo is not None


def test_extract_query_handles_non_json():
    assert _extract_query("not json") == ""
    assert _extract_query(None) == ""


def test_extract_query_handles_non_dict_json():
    assert _extract_query("[1, 2, 3]") == ""


def test_extract_query_empty_payload_returns_empty():
    assert _extract_query(json.dumps({"other": "field"})) == ""


def test_extract_query_prefers_query_then_text_then_prompt():
    assert _extract_query(json.dumps({"query": "q"})) == "q"
    assert _extract_query(json.dumps({"text": "t"})) == "t"
    assert _extract_query(json.dumps({"prompt": "p"})) == "p"


def test_tool_event_hit_empty_fact_ids():
    """Empty fact_ids → False, no DB access."""
    conn = sqlite3.connect(":memory:")
    assert _tool_event_hit(conn, datetime.now(timezone.utc), []) is False
    conn.close()


def test_tool_event_hit_missing_table():
    conn = sqlite3.connect(":memory:")
    assert _tool_event_hit(conn, datetime.now(timezone.utc), ["f1"]) is False
    conn.close()


def test_top3_fact_ids_missing_learning_signals(tmp_path: Path):
    """_top3_fact_ids: table missing → [] gracefully."""
    from superlocalmemory.learning.reward_proxy import _top3_fact_ids
    learning = tmp_path / "l.db"
    memory = tmp_path / "m.db"
    apply_all(learning, memory)  # bandit tables but not learning_signals
    conn = sqlite3.connect(str(learning))
    try:
        assert _top3_fact_ids(conn, "qx") == []
    finally:
        conn.close()


def test_fetch_unsettled_missing_table(tmp_path: Path):
    """_fetch_unsettled: table missing → [] gracefully."""
    from superlocalmemory.learning.reward_proxy import _fetch_unsettled
    db = tmp_path / "e.db"
    conn = sqlite3.connect(str(db))
    try:
        assert _fetch_unsettled(conn, "p", datetime.now(timezone.utc)) == []
    finally:
        conn.close()


def test_requery_detected_no_seed(tmp_path: Path):
    """_requery_detected: no events at or before played_at → False."""
    from superlocalmemory.learning.reward_proxy import _requery_detected
    db = tmp_path / "m.db"
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute(
            "CREATE TABLE tool_events ("
            "id INTEGER PRIMARY KEY, occurred_at TEXT,"
            " tool_name TEXT, payload_json TEXT)"
        )
        # Insert a post-window event but no seed (prior recall).
        played = datetime(2026, 4, 18, tzinfo=timezone.utc)
        conn.execute(
            "INSERT INTO tool_events (occurred_at, tool_name, payload_json) "
            "VALUES (?, 'recall', ?)",
            ((played + timedelta(seconds=10)).isoformat(timespec="seconds"),
             json.dumps({"query": "some"})),
        )
        assert _requery_detected(conn, played, "qx") is False
    finally:
        conn.close()


def test_requery_detected_no_followup_rows(tmp_path: Path):
    """_requery_detected: no within-window recall events → False."""
    from superlocalmemory.learning.reward_proxy import _requery_detected
    db = tmp_path / "m.db"
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute(
            "CREATE TABLE tool_events ("
            "id INTEGER PRIMARY KEY, occurred_at TEXT,"
            " tool_name TEXT, payload_json TEXT)"
        )
        played = datetime(2026, 4, 18, tzinfo=timezone.utc)
        # No rows in tool_events at all.
        assert _requery_detected(conn, played, "qx") is False
    finally:
        conn.close()


def test_requery_detected_missing_tool_events_table(tmp_path: Path):
    from superlocalmemory.learning.reward_proxy import _requery_detected
    db = tmp_path / "m.db"
    conn = sqlite3.connect(str(db))
    try:
        played = datetime(2026, 4, 18, tzinfo=timezone.utc)
        assert _requery_detected(conn, played, "qx") is False
    finally:
        conn.close()


def test_settle_with_unsettleable_played_at_ignored(tmp_path: Path):
    """Rows with invalid played_at ISO strings are skipped."""
    learning = tmp_path / "learning.db"
    memory = tmp_path / "memory.db"
    stats = apply_all(learning, memory)
    assert "M005_bandit_tables" in stats["applied"]
    conn = sqlite3.connect(str(learning), isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO bandit_plays "
            "(profile_id, query_id, stratum, arm_id, played_at) "
            "VALUES ('p', 'q', 's', 'fallback_default', 'not-a-date')",
        )
    finally:
        conn.close()
    # Build a minimal memory.db
    sqlite3.connect(str(memory)).close()
    n = settle_stale_plays("p", learning, memory,
                           now=datetime.now(timezone.utc))
    assert n == 0


# ---------------------------------------------------------------------------
# bandit internals
# ---------------------------------------------------------------------------


def test_bandit_update_ignores_unknown_row(tmp_path: Path):
    """Covers the early return when bandit_plays has no matching row."""
    learning = tmp_path / "learning.db"
    memory = tmp_path / "memory.db"
    apply_all(learning, memory)
    b = ContextualBandit(learning, profile_id="p",
                         cache=_BanditCache(max_entries=4))
    assert b.update(999, reward=0.5) is False


def test_bandit_conn_reuses_threadlocal(tmp_path: Path):
    """Second call on same thread returns the same connection."""
    learning = tmp_path / "learning.db"
    memory = tmp_path / "memory.db"
    apply_all(learning, memory)
    c1 = _conn_for(learning)
    c2 = _conn_for(learning)
    assert c1 is c2


def test_bandit_choose_db_error_falls_back(tmp_path: Path, monkeypatch):
    """If the posterior load raises, choose returns a valid BanditChoice."""
    learning = tmp_path / "learning.db"
    memory = tmp_path / "memory.db"
    apply_all(learning, memory)
    cache = _BanditCache(max_entries=4)

    def _boom(profile, stratum, loader):
        raise sqlite3.OperationalError("mocked")
    cache.get = _boom  # type: ignore[method-assign]

    b = ContextualBandit(learning, profile_id="p", cache=cache)
    ch = b.choose({"query_type": "single_hop", "entity_count": 0,
                   "time_bucket": "morning"}, query_id="q")
    assert ch.arm_id in ARM_CATALOG


def test_bandit_update_sqlite_lookup_error_returns_false(
    tmp_path: Path, monkeypatch,
):
    """If the lookup SELECT raises, update returns False gracefully."""
    learning = tmp_path / "learning.db"
    memory = tmp_path / "memory.db"
    apply_all(learning, memory)
    b = ContextualBandit(learning, profile_id="p",
                         cache=_BanditCache(max_entries=4))
    ch = b.choose({"query_type": "single_hop", "entity_count": 0,
                   "time_bucket": "morning"}, query_id="q-err")

    # Drop bandit_plays so the SELECT raises.
    conn = sqlite3.connect(str(learning), isolation_level=None)
    try:
        conn.execute("DROP TABLE bandit_plays")
    finally:
        conn.close()
    # Force the ContextualBandit threadlocal conn to reopen by closing the
    # stale handle.
    from superlocalmemory.learning import bandit as _b
    if _b._holder.conn is not None:
        _b._holder.conn.close()
        _b._holder.conn = None
        _b._holder.path = None

    assert b.update(ch.play_id, reward=1.0) is False


def test_bandit_update_write_error_returns_false(tmp_path: Path):
    """If the UPDATE fails (corrupt table), update returns False."""
    learning = tmp_path / "learning.db"
    memory = tmp_path / "memory.db"
    apply_all(learning, memory)
    b = ContextualBandit(learning, profile_id="p",
                         cache=_BanditCache(max_entries=4))
    ch = b.choose({"query_type": "single_hop", "entity_count": 0,
                   "time_bucket": "morning"}, query_id="q-w")
    # Drop bandit_arms — the INSERT OR IGNORE will fail.
    from superlocalmemory.learning import bandit as _b
    if _b._holder.conn is not None:
        _b._holder.conn.close()
        _b._holder.conn = None
        _b._holder.path = None
    conn = sqlite3.connect(str(learning), isolation_level=None)
    try:
        conn.execute("DROP TABLE bandit_arms")
    finally:
        conn.close()
    assert b.update(ch.play_id, reward=1.0) is False


def test_bandit_snapshot_db_error_returns_empty(tmp_path: Path):
    """Missing bandit_arms → snapshot returns {}."""
    learning = tmp_path / "empty.db"
    sqlite3.connect(str(learning)).close()
    b = ContextualBandit(learning, profile_id="p",
                         cache=_BanditCache(max_entries=4))
    assert b.snapshot() == {}


def test_bandit_conn_switches_on_path_change(tmp_path: Path):
    learning1 = tmp_path / "l1.db"
    learning2 = tmp_path / "l2.db"
    memory = tmp_path / "m.db"
    apply_all(learning1, memory)
    apply_all(learning2, memory)
    c1 = _conn_for(learning1)
    c2 = _conn_for(learning2)
    assert c1 is not c2
