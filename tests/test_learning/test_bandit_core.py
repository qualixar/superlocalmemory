# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §7.2

"""Tests for ``learning/bandit.py`` — contextual Thompson bandit.

Covers hard rules B1, B2, B4, B5, B6, B7.
"""

from __future__ import annotations

import itertools
import secrets as _secrets
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.learning.arm_catalog import ARM_CATALOG
from superlocalmemory.learning.bandit import (
    BanditChoice,
    ContextualBandit,
    _ENTITY_BINS,
    _QUERY_TYPES,
    _TIME_BUCKETS,
    compute_stratum,
    current_time_bucket,
)
from superlocalmemory.learning.bandit_cache import _BanditCache
from superlocalmemory.storage.migration_runner import apply_all


@pytest.fixture()
def bandit_db(tmp_path: Path) -> Path:
    """Create a learning.db with M005 applied and return its path.

    Note: M001/M002 depend on tables owned by Wave 2B — those failures are
    expected in this Wave 3 Stream A isolation harness. We only require
    M003 (migration_log) and M005 (bandit tables) to succeed.
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    stats = apply_all(learning_db, memory_db)
    # Only require bandit-related migrations to be green.
    assert "M003_migration_log" in stats["applied"], stats
    assert "M005_bandit_tables" in stats["applied"], stats
    assert "M005_bandit_tables" not in stats["failed"], stats
    return learning_db


@pytest.fixture()
def fresh_cache() -> _BanditCache:
    """Dedicated cache per test — no cross-test leakage."""
    return _BanditCache(max_entries=64)


@pytest.fixture()
def bandit(bandit_db: Path, fresh_cache: _BanditCache) -> ContextualBandit:
    return ContextualBandit(
        bandit_db, profile_id="test_profile", cache=fresh_cache,
    )


def _ctx(qtype: str = "single_hop", ecount: int = 0,
         tbucket: str = "morning") -> dict:
    return {
        "query_type": qtype,
        "entity_count": ecount,
        "time_bucket": tbucket,
    }


# ---------------------------------------------------------------------------
# B4: stratum cardinality
# ---------------------------------------------------------------------------


def test_stratum_cardinality():
    """B4: 4 × 3 × 4 = 48 distinct strata, computed via compute_stratum."""
    strata = set()
    for qt, eb, tb in itertools.product(
        _QUERY_TYPES, _ENTITY_BINS, _TIME_BUCKETS,
    ):
        s = compute_stratum({
            "query_type": qt, "entity_count_bin": eb, "time_bucket": tb,
        })
        strata.add(s)
    assert len(strata) == 48, (
        f"Expected 48 distinct strata, got {len(strata)}"
    )


def test_stratum_unknown_query_type_falls_back_to_open_domain():
    s = compute_stratum({
        "query_type": "UNKNOWN_LABEL", "entity_count_bin": "0",
        "time_bucket": "morning",
    })
    assert s.startswith("open_domain|")


def test_stratum_entity_count_binning():
    assert compute_stratum({"query_type": "single_hop", "entity_count": 0,
                            "time_bucket": "morning"}).split("|")[1] == "0"
    assert compute_stratum({"query_type": "single_hop", "entity_count": 1,
                            "time_bucket": "morning"}).split("|")[1] == "1-2"
    assert compute_stratum({"query_type": "single_hop", "entity_count": 2,
                            "time_bucket": "morning"}).split("|")[1] == "1-2"
    assert compute_stratum({"query_type": "single_hop", "entity_count": 5,
                            "time_bucket": "morning"}).split("|")[1] == "3+"


def test_stratum_entity_count_negative_binned_as_zero():
    assert compute_stratum({"query_type": "single_hop", "entity_count": -3,
                            "time_bucket": "morning"}).split("|")[1] == "0"


def test_stratum_time_bucket_auto_when_absent():
    # Just verify it returns one of the 4 buckets (wall-clock-dependent).
    s = compute_stratum({"query_type": "single_hop", "entity_count": 0})
    assert s.split("|")[2] in set(_TIME_BUCKETS)


def test_stratum_invalid_entity_count_becomes_zero():
    s = compute_stratum({"query_type": "single_hop",
                         "entity_count": "not-a-number",
                         "time_bucket": "evening"})
    assert s.split("|")[1] == "0"


def test_current_time_bucket_hour_boundaries():
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    # Spot-check boundaries. Any fixed tz is fine — we just want hour.
    base = _dt(2026, 4, 18, 0, 0, tzinfo=_tz.utc)
    assert current_time_bucket(base.replace(hour=5)) == "morning"
    assert current_time_bucket(base.replace(hour=11)) == "morning"
    assert current_time_bucket(base.replace(hour=12)) == "afternoon"
    assert current_time_bucket(base.replace(hour=16)) == "afternoon"
    assert current_time_bucket(base.replace(hour=17)) == "evening"
    assert current_time_bucket(base.replace(hour=20)) == "evening"
    assert current_time_bucket(base.replace(hour=21)) == "night"
    assert current_time_bucket(base.replace(hour=2)) == "night"


# ---------------------------------------------------------------------------
# Choose / update
# ---------------------------------------------------------------------------


def test_cold_start_uniform_posteriors(bandit: ContextualBandit,
                                       bandit_db: Path):
    """No rows in bandit_arms → every arm has prior Beta(1,1)."""
    choice = bandit.choose(_ctx(), query_id="q-cold")
    # Expected: an arm from the catalog was selected, play row was inserted.
    assert choice.arm_id in ARM_CATALOG
    assert choice.play_id is not None
    # Verify persisted.
    conn = sqlite3.connect(str(bandit_db))
    try:
        row = conn.execute(
            "SELECT arm_id, profile_id, stratum FROM bandit_plays "
            "WHERE play_id = ?", (choice.play_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == choice.arm_id
    assert row[1] == "test_profile"


def test_choose_returns_catalog_weights(bandit: ContextualBandit):
    """choice.weights must equal ARM_CATALOG[choice.arm_id] (value-wise)."""
    choice = bandit.choose(_ctx(), query_id="q-weights")
    assert choice.weights == ARM_CATALOG[choice.arm_id]


def test_update_increments_posterior_on_reward_one(bandit: ContextualBandit,
                                                   bandit_db: Path):
    choice = bandit.choose(_ctx(), query_id="q-upd")
    assert bandit.update(choice.play_id, reward=1.0) is True
    conn = sqlite3.connect(str(bandit_db))
    try:
        row = conn.execute(
            "SELECT alpha, beta, plays FROM bandit_arms "
            "WHERE profile_id = ? AND stratum = ? AND arm_id = ?",
            ("test_profile", choice.stratum, choice.arm_id),
        ).fetchone()
    finally:
        conn.close()
    # prior (1,1) + reward 1.0 → (2,1).
    assert row is not None
    assert row[0] == pytest.approx(2.0)
    assert row[1] == pytest.approx(1.0)
    assert row[2] == 1


def test_update_zero_reward_grows_beta(bandit: ContextualBandit,
                                       bandit_db: Path):
    choice = bandit.choose(_ctx(), query_id="q-zero")
    assert bandit.update(choice.play_id, reward=0.0) is True
    conn = sqlite3.connect(str(bandit_db))
    try:
        row = conn.execute(
            "SELECT alpha, beta FROM bandit_arms "
            "WHERE profile_id = ? AND stratum = ? AND arm_id = ?",
            ("test_profile", choice.stratum, choice.arm_id),
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == pytest.approx(1.0)
    assert row[1] == pytest.approx(2.0)


def test_update_clamps_reward_out_of_range(bandit: ContextualBandit,
                                           bandit_db: Path):
    choice = bandit.choose(_ctx(), query_id="q-clamp")
    assert bandit.update(choice.play_id, reward=5.0) is True  # clamped to 1.0
    conn = sqlite3.connect(str(bandit_db))
    try:
        row = conn.execute(
            "SELECT reward FROM bandit_plays WHERE play_id = ?",
            (choice.play_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == pytest.approx(1.0)


def test_update_non_numeric_reward_rejected(bandit: ContextualBandit):
    choice = bandit.choose(_ctx(), query_id="q-bad")
    assert bandit.update(choice.play_id, reward="banana") is False


def test_update_unknown_play_id_logs_and_returns_false(
    bandit: ContextualBandit,
):
    assert bandit.update(999999, reward=1.0) is False


def test_update_already_settled_is_noop(bandit: ContextualBandit):
    choice = bandit.choose(_ctx(), query_id="q-idem")
    assert bandit.update(choice.play_id, reward=1.0) is True
    # Second attempt must not re-increment.
    assert bandit.update(choice.play_id, reward=1.0) is False


# ---------------------------------------------------------------------------
# B1: SystemRandom used (NOT random.betavariate)
# ---------------------------------------------------------------------------


def test_secure_rng_used(bandit_db: Path, monkeypatch):
    """B1: choose() must call into secrets.SystemRandom().betavariate."""
    call_counter = {"n": 0}
    real_system_random = _secrets.SystemRandom

    class _Spy(real_system_random):
        def betavariate(self, a, b):  # type: ignore[override]
            call_counter["n"] += 1
            return super().betavariate(a, b)

    monkeypatch.setattr("superlocalmemory.learning.bandit.secrets.SystemRandom",
                        _Spy)
    b = ContextualBandit(bandit_db, profile_id="rng",
                         cache=_BanditCache(max_entries=8))
    b.choose(_ctx(), query_id="rng-q")
    # 40 arms × 1 call each = 40 invocations.
    assert call_counter["n"] >= 40


# ---------------------------------------------------------------------------
# B2: alpha/beta cap
# ---------------------------------------------------------------------------


def test_alpha_beta_cap_1000(bandit_db: Path):
    """B2: alpha and beta clamp at 1000 regardless of update count."""
    cache = _BanditCache(max_entries=16)
    b = ContextualBandit(
        bandit_db, profile_id="cap", cache=cache, alpha_cap=1000.0,
    )
    # Force the same arm every time by seeding one arm row heavily via
    # driving updates on a freshly-created play. Easiest: 1500 plays, all
    # reward=1.0 — alpha builds, beta stays at 1.0.
    for i in range(1500):
        ch = b.choose(_ctx(), query_id=f"q-{i}")
        b.update(ch.play_id, reward=1.0)

    conn = sqlite3.connect(str(bandit_db))
    try:
        rows = conn.execute(
            "SELECT alpha, beta FROM bandit_arms WHERE profile_id = ?",
            ("cap",),
        ).fetchall()
    finally:
        conn.close()
    for alpha, beta in rows:
        assert alpha <= 1000.0
        assert beta <= 1000.0


# ---------------------------------------------------------------------------
# B5: cache invalidation on update
# ---------------------------------------------------------------------------


def test_choose_cache_invalidated_on_update(bandit_db: Path):
    """B5: next choose() after update re-reads posteriors from DB."""
    cache = _BanditCache(max_entries=16)
    loader_calls = {"n": 0}
    real = cache.get

    def _counting_get(profile, stratum, loader):
        def _wrapped(p, s):
            loader_calls["n"] += 1
            return loader(p, s)
        return real(profile, stratum, _wrapped)
    cache.get = _counting_get  # type: ignore[method-assign]

    b = ContextualBandit(bandit_db, profile_id="inv", cache=cache)
    # First choose — miss → 1 load call.
    ch1 = b.choose(_ctx(), query_id="q1")
    assert loader_calls["n"] == 1
    # Second choose without update — hit → still 1.
    b.choose(_ctx(), query_id="q2")
    assert loader_calls["n"] == 1
    # Update invalidates.
    assert b.update(ch1.play_id, reward=1.0) is True
    # Third choose — miss again → 2.
    b.choose(_ctx(), query_id="q3")
    assert loader_calls["n"] == 2


# ---------------------------------------------------------------------------
# B6: no raw query in bandit tables
# ---------------------------------------------------------------------------


def test_no_raw_query_in_bandit(bandit_db: Path):
    """B6: bandit_arms and bandit_plays must not have 'query' / 'query_text'."""
    conn = sqlite3.connect(str(bandit_db))
    try:
        arms_cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(bandit_arms)"
        ).fetchall()}
        plays_cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(bandit_plays)"
        ).fetchall()}
    finally:
        conn.close()
    forbidden = {"query", "query_text", "raw_query", "prompt"}
    assert arms_cols.isdisjoint(forbidden)
    assert plays_cols.isdisjoint(forbidden)


# ---------------------------------------------------------------------------
# Convergence (success criterion)
# ---------------------------------------------------------------------------


def test_choose_converges_to_rewarded_arm(bandit_db: Path):
    """With one arm always rewarding 1.0 and others 0.0, selection > 80 %.

    To keep the test deterministic without mocking the RNG, we directly
    seed bandit_arms rows with strong posteriors for the "winning" arm
    and weak posteriors for all others, then verify choose() picks it
    overwhelmingly.
    """
    # Seed: winning arm with (500, 1), others with (1, 500).
    cache = _BanditCache(max_entries=16)
    b = ContextualBandit(bandit_db, profile_id="conv", cache=cache)
    stratum = compute_stratum(_ctx())
    winning = "semantic_heavy_2"
    conn = sqlite3.connect(str(bandit_db), isolation_level=None)
    try:
        for arm_id in ARM_CATALOG:
            if arm_id == winning:
                alpha, beta = 500.0, 1.0
            else:
                alpha, beta = 1.0, 500.0
            conn.execute(
                "INSERT OR REPLACE INTO bandit_arms "
                "(profile_id, stratum, arm_id, alpha, beta, plays, "
                " last_played_at) VALUES (?, ?, ?, ?, ?, 0, '2026-01-01T00:00:00')",
                ("conv", stratum, arm_id, alpha, beta),
            )
    finally:
        conn.close()
    cache.clear()

    hits = 0
    trials = 100
    for i in range(trials):
        ch = b.choose(_ctx(), query_id=f"c-{i}")
        if ch.arm_id == winning:
            hits += 1
    assert hits / trials > 0.80, f"winning selected {hits}/{trials}"


# ---------------------------------------------------------------------------
# B7: latency p99
# ---------------------------------------------------------------------------


def test_bandit_latency_p99_under_10ms(bandit: ContextualBandit):
    """B7: choose() p99 ≤ 10 ms across 200 calls."""
    import time as _time
    samples = []
    for i in range(200):
        t0 = _time.perf_counter()
        bandit.choose(_ctx(), query_id=f"lat-{i}")
        samples.append((_time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p99 = samples[int(len(samples) * 0.99) - 1]
    assert p99 <= 10.0, f"p99 = {p99:.2f} ms"


# ---------------------------------------------------------------------------
# Defensive: DB issues
# ---------------------------------------------------------------------------


def test_choose_never_raises_on_missing_table(tmp_path: Path):
    """If bandit_arms is missing, choose() returns a valid BanditChoice."""
    db = tmp_path / "empty.db"
    # Don't run migrations → no bandit tables.
    cache = _BanditCache(max_entries=4)
    b = ContextualBandit(db, profile_id="p", cache=cache)
    choice = b.choose(_ctx(), query_id="noop")
    assert choice.arm_id in ARM_CATALOG
    # play_id may be None since insert will fail.


def test_snapshot_empty_profile_returns_empty_dict(bandit: ContextualBandit):
    assert bandit.snapshot() == {}


def test_snapshot_returns_top_n_by_plays(bandit: ContextualBandit,
                                         bandit_db: Path):
    for i in range(5):
        ch = bandit.choose(_ctx(), query_id=f"s-{i}")
        bandit.update(ch.play_id, reward=1.0)
    snap = bandit.snapshot(top_n=3)
    # At most 1 stratum appears (same context every call).
    assert len(snap) <= 1
    if snap:
        for stratum, arms in snap.items():
            assert len(arms) <= 3
            assert all("arm_id" in a and "plays" in a for a in arms)
