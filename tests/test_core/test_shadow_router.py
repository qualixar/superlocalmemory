# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track A.3 / Stage 8 SB-1

"""Tests for ``core/shadow_router.py`` — daemon-resident A/B router.

The router is the single in-process seam that wires LLD-10 Track A.3
pieces together at recall time:

  - ``ShadowTest`` (pre-promotion A/B accumulator).
  - ``ModelRollback`` (post-promotion 200-observation watch window).

Contract references:
  - LLD-00 §8    — two-phase shadow + auto-rollback.
  - LLD-10 §4.1  — deterministic A/B routing, tamper-proof via install_token.
  - LLD-10 §5    — atomic BEGIN IMMEDIATE promotion + rollback.
  - Stage 8 SB-1 — ship-blocker: router wires previously-dead code.

Stdlib-only. Exercises the pure state-machine; DB interaction is
monkey-patched where promotion / rollback are triggered so tests run
in milliseconds without lightgbm.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Shared schema — minimal subset needed for shadow_router promotion path
# ---------------------------------------------------------------------------

_LEARNING_SCHEMA = """
CREATE TABLE learning_model_state (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id        TEXT NOT NULL,
    model_version     TEXT NOT NULL DEFAULT '3.4.22',
    state_bytes       BLOB NOT NULL,
    bytes_sha256      TEXT NOT NULL DEFAULT '',
    trained_on_count  INTEGER NOT NULL DEFAULT 0,
    feature_names     TEXT NOT NULL DEFAULT '[]',
    metrics_json      TEXT NOT NULL DEFAULT '{}',
    is_active         INTEGER NOT NULL DEFAULT 0,
    trained_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    is_previous       INTEGER DEFAULT 0,
    is_rollback       INTEGER DEFAULT 0,
    is_candidate      INTEGER DEFAULT 0,
    shadow_results_json TEXT,
    promoted_at       TEXT,
    rollback_reason   TEXT,
    metadata_json     TEXT
);
CREATE UNIQUE INDEX idx_model_active_one
    ON learning_model_state(profile_id) WHERE is_active = 1;
CREATE UNIQUE INDEX idx_model_candidate_one
    ON learning_model_state(profile_id) WHERE is_candidate = 1;
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture()
def learning_db(tmp_path: Path) -> Path:
    db = tmp_path / "learning.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_LEARNING_SCHEMA)
    return db


@pytest.fixture()
def memory_db(tmp_path: Path) -> Path:
    db = tmp_path / "memory.db"
    # shadow_router does not DDL memory.db — a bare file is enough.
    db.write_bytes(b"")
    return db


@pytest.fixture(autouse=True)
def _reset_router_singleton():
    """Ensure each test starts with a fresh router singleton."""
    from superlocalmemory.core import shadow_router
    shadow_router.reset_for_testing()
    yield
    shadow_router.reset_for_testing()


@pytest.fixture(autouse=True)
def _stable_install_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Force a deterministic install token so route_query is reproducible."""
    from superlocalmemory.core import security_primitives as sp
    fake_path = tmp_path / ".install_token"
    fake_path.write_text("deadbeef" * 8, encoding="utf-8")
    monkeypatch.setattr(sp, "_install_token_path", lambda: fake_path)


# ---------------------------------------------------------------------------
# Route-query determinism + tamper resistance
# ---------------------------------------------------------------------------


def test_shadow_router_route_query_uses_install_token(
    learning_db: Path, memory_db: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """route_query MUST hash install_token + query_id, not query_id alone.

    Closes skeptic H-02 + H-03 — an attacker who controls query_id
    cannot bias the A/B split without also reading the install_token.
    """
    from superlocalmemory.core import shadow_router as sr_mod
    from superlocalmemory.core import security_primitives as sp

    router_a = sr_mod.ShadowRouter(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )
    arm_a = router_a.route_query("query-abc")

    # Rotate install token → arm for same query_id may flip.
    new_token_file = tmp_path / ".install_token_rotated"
    new_token_file.write_text("c0ffee" * 16, encoding="utf-8")
    monkeypatch.setattr(sp, "_install_token_path", lambda: new_token_file)

    router_b = sr_mod.ShadowRouter(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )
    # Deterministic verification: manual hash matches implementation.
    token_b = new_token_file.read_text().strip()
    expected_h = hashlib.sha256(
        (token_b + "query-abc").encode("utf-8"),
    ).hexdigest()[:8]
    expected_arm = "candidate" if int(expected_h, 16) % 2 == 1 else "baseline"
    assert router_b.route_query("query-abc") == expected_arm
    # And the route_query result uses the token — not query_id alone.
    query_only = hashlib.sha256("query-abc".encode("utf-8")).hexdigest()[:8]
    naive_arm = "candidate" if int(query_only, 16) % 2 == 1 else "baseline"
    # The two SHOULD often disagree; if they happen to agree for this
    # input, use another input to prove token-dependence.
    if router_b.route_query("query-abc") == naive_arm:
        # Find a query_id that distinguishes token vs naive — exists
        # in practice because SHA-256 distributes uniformly.
        for q in (f"q-{i}" for i in range(1000)):
            token_h = hashlib.sha256(
                (token_b + q).encode("utf-8"),
            ).hexdigest()[:8]
            naive_h = hashlib.sha256(q.encode("utf-8")).hexdigest()[:8]
            if (int(token_h, 16) % 2) != (int(naive_h, 16) % 2):
                assert router_b.route_query(q) != (
                    "candidate" if int(naive_h, 16) % 2 == 1 else "baseline"
                )
                return
        pytest.fail("route_query appears to ignore install_token")


def test_shadow_router_route_query_deterministic(
    learning_db: Path, memory_db: Path,
) -> None:
    """Same (install_token, query_id) → same arm, always.

    Deterministic re-routing survives daemon restart (LLD-10 §4.1).
    """
    from superlocalmemory.core.shadow_router import ShadowRouter

    r1 = ShadowRouter(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )
    r2 = ShadowRouter(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )
    qids = [f"q-{i:04d}" for i in range(32)]
    for qid in qids:
        assert r1.route_query(qid) == r2.route_query(qid)
    # And the returned value is strictly in the expected arm vocabulary.
    assert {r1.route_query(q) for q in qids} <= {"baseline", "candidate"}


# ---------------------------------------------------------------------------
# Promotion on ShadowTest.decide == 'promote'
# ---------------------------------------------------------------------------


def _seed_active_and_candidate(db_path: Path, profile_id: str) -> tuple[int, int]:
    now = _now_iso()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, bytes_sha256, is_active, "
            " trained_at, updated_at, metadata_json) "
            "VALUES (?, ?, ?, 1, ?, ?, ?)",
            (profile_id, b"active-bytes", "0" * 64, now, now, "{}"),
        )
        active_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, bytes_sha256, is_active, is_candidate,"
            " trained_at, updated_at, metadata_json) "
            "VALUES (?, ?, ?, 0, 1, ?, ?, ?)",
            (profile_id, b"cand-bytes", "0" * 64, now, now, "{}"),
        )
        cand_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    return active_id, cand_id


def test_shadow_router_promote_on_decide_promote(
    learning_db: Path, memory_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ShadowTest.decide() == 'promote', router fires
    ``_promote_candidate`` atomically and clears its shadow state.
    """
    from superlocalmemory.core import shadow_router as sr_mod

    _, cand_id = _seed_active_and_candidate(learning_db, "p")

    router = sr_mod.ShadowRouter(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )
    # Force the shadow test to return 'promote' after first settled pair.
    called = {"n": 0}

    class _FakeShadowTest:
        def __init__(self, *a, **kw):
            pass
        def record_recall_pair(self, **kw):
            called["n"] += 1
        def decide(self):
            return "promote", {"effect": 0.1}

    monkeypatch.setattr(sr_mod, "ShadowTest", _FakeShadowTest)
    # Re-attach a fresh shadow test that uses the fake class.
    router._shadow = _FakeShadowTest()
    router._candidate_id = cand_id

    router.on_recall_settled(
        query_id="q1", arm="candidate", ndcg_at_10=0.9,
    )

    # Promotion fired — candidate flag cleared on lineage_state row.
    with sqlite3.connect(learning_db) as conn:
        n_cand = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_candidate=1",
        ).fetchone()[0]
        assert n_cand == 0
        n_active = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()[0]
        assert n_active == 1
        promoted = conn.execute(
            "SELECT id FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()[0]
        assert promoted == cand_id


def test_shadow_router_rollback_on_200_regression(
    learning_db: Path, memory_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After promotion, 200 post-observation regressions trigger rollback."""
    from superlocalmemory.core import shadow_router as sr_mod

    # Seed an active + a is_previous row so rollback has something to restore.
    now = _now_iso()
    with sqlite3.connect(learning_db) as conn:
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, bytes_sha256, is_active, is_previous,"
            " trained_at, updated_at, metadata_json) "
            "VALUES (?, ?, ?, 0, 1, ?, ?, ?)",
            ("p", b"prev", "0" * 64, now, now, "{}"),
        )
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, bytes_sha256, is_active, "
            " trained_at, updated_at, metadata_json) "
            "VALUES (?, ?, ?, 1, ?, ?, ?)",
            ("p", b"active", "0" * 64, now, now, "{}"),
        )
        conn.commit()

    router = sr_mod.ShadowRouter(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )
    # Arm post-promotion watcher with high baseline, feed 200 low NDCG samples.
    router.arm_post_promotion_watch(baseline_ndcg=0.8)
    for i in range(200):
        router.on_recall_settled(
            query_id=f"q-{i}", arm="baseline", ndcg_at_10=0.1,
        )

    with sqlite3.connect(learning_db) as conn:
        row = conn.execute(
            "SELECT is_rollback FROM learning_model_state "
            "WHERE profile_id='p' AND state_bytes=?",
            (b"active",),
        ).fetchone()
        assert row is not None and row[0] == 1


# ---------------------------------------------------------------------------
# Singleton factory + reset
# ---------------------------------------------------------------------------


def test_shadow_router_singleton_factory_returns_same_instance(
    learning_db: Path, memory_db: Path,
) -> None:
    """``get_shadow_router`` returns the same instance for the same key."""
    from superlocalmemory.core.shadow_router import get_shadow_router

    r1 = get_shadow_router(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )
    r2 = get_shadow_router(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )
    assert r1 is r2

    # Different profile_id → different instance.
    r3 = get_shadow_router(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="q",
    )
    assert r3 is not r1


def test_shadow_router_reset_for_testing(
    learning_db: Path, memory_db: Path,
) -> None:
    """``reset_for_testing`` wipes the singleton cache."""
    from superlocalmemory.core import shadow_router as sr_mod

    r1 = sr_mod.get_shadow_router(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )
    sr_mod.reset_for_testing()
    r2 = sr_mod.get_shadow_router(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )
    assert r1 is not r2


# ---------------------------------------------------------------------------
# Stage 8 F4.B — H-19 RankerLineageState equivalent / full end-to-end flow
# ---------------------------------------------------------------------------
#
# H-19 asked for an owner of the 200-observation accumulator. ShadowRouter
# already owns ShadowTest + ModelRollback across the promotion boundary,
# but there was no integration test that drove the FULL flow:
#   1. Pre-promotion A/B routing.
#   2. ShadowTest decides 'promote' → atomic promotion.
#   3. Post-promotion watcher armed with baseline_ndcg.
#   4. 200 post-promotion regressions → auto-rollback fires.
# and the success-path counterpart:
#   post-promotion window sees stable NDCG → no rollback; watch clears.
#
# These tests exercise that contract.


def _seed_active_previous_candidate(
    db_path: Path, profile_id: str,
) -> tuple[int, int, int]:
    """Seed 3 rows: a `previous` (for rollback destination), an `active`
    (current), and a fresh `candidate` (to promote). Return IDs."""
    now = _now_iso()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, bytes_sha256, is_active, is_previous,"
            " trained_at, updated_at, metadata_json) "
            "VALUES (?, ?, ?, 0, 1, ?, ?, ?)",
            (profile_id, b"prev-bytes", "0" * 64, now, now, "{}"),
        )
        prev_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, bytes_sha256, is_active, "
            " trained_at, updated_at, metadata_json) "
            "VALUES (?, ?, ?, 1, ?, ?, ?)",
            (profile_id, b"active-bytes", "0" * 64, now, now, "{}"),
        )
        active_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, state_bytes, bytes_sha256, is_active, is_candidate,"
            " trained_at, updated_at, metadata_json) "
            "VALUES (?, ?, ?, 0, 1, ?, ?, ?)",
            (profile_id, b"cand-bytes", "0" * 64, now, now, "{}"),
        )
        cand_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    return prev_id, active_id, cand_id


def test_auto_rollback_after_promote_on_regression_window(
    learning_db: Path, memory_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 8 H-19: route → promote → 200 regressions → auto-rollback.

    Uses a fake ShadowTest that deterministically returns 'promote' after
    one recorded pair. Arms the post-promotion watch with baseline=0.8,
    feeds 200 observations at 0.1 (huge regression), verifies:
      * The promoted row was demoted via execute_rollback.
      * The pre-promotion 'is_previous' row is active again.
    """
    from superlocalmemory.core import shadow_router as sr_mod

    prev_id, active_id, cand_id = _seed_active_previous_candidate(
        learning_db, "p",
    )

    router = sr_mod.ShadowRouter(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )

    # Force the shadow decide to 'promote' after any recorded pair.
    class _PromoteImmediately:
        def __init__(self, *a, **kw): pass
        def record_recall_pair(self, **kw): pass
        def decide(self): return "promote", {"effect": 0.2}

    monkeypatch.setattr(sr_mod, "ShadowTest", _PromoteImmediately)
    router._shadow = _PromoteImmediately()
    router._candidate_id = cand_id

    # Step 1 — fire promotion.
    router.on_recall_settled(
        query_id="q-promote", arm="candidate", ndcg_at_10=0.9,
    )

    # After promotion, candidate flag should clear; cand_id is now active.
    with sqlite3.connect(learning_db) as conn:
        active_after_promote = conn.execute(
            "SELECT id FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()
        assert active_after_promote[0] == cand_id
        # previous row was active_id (demoted) after _promote_candidate.
        is_prev_active = conn.execute(
            "SELECT is_previous FROM learning_model_state "
            "WHERE id=?", (active_id,),
        ).fetchone()[0]
        assert is_prev_active == 1

    # Step 2 — arm post-promotion watch; feed 200 regressions.
    router.arm_post_promotion_watch(baseline_ndcg=0.8)
    for i in range(200):
        router.on_recall_settled(
            query_id=f"q-{i}", arm="baseline", ndcg_at_10=0.1,
        )

    # Step 3 — verify rollback fired: cand_id (promoted) now is_rollback=1,
    # active_id (former active) back to is_active=1.
    with sqlite3.connect(learning_db) as conn:
        cand_row = conn.execute(
            "SELECT is_active, is_rollback FROM learning_model_state "
            "WHERE id=?", (cand_id,),
        ).fetchone()
        assert cand_row[0] == 0
        assert cand_row[1] == 1

        active_now = conn.execute(
            "SELECT id FROM learning_model_state "
            "WHERE profile_id='p' AND is_active=1",
        ).fetchone()
        assert active_now[0] == active_id


def test_promote_then_clear_watch_window_on_success(
    learning_db: Path, memory_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H-19 counterpart: post-promotion stable NDCG → no rollback.

    200 post-promotion observations at the baseline → ``should_rollback``
    stays False. No flip on learning_model_state.
    """
    from superlocalmemory.core import shadow_router as sr_mod

    prev_id, active_id, cand_id = _seed_active_previous_candidate(
        learning_db, "p",
    )

    router = sr_mod.ShadowRouter(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )

    class _PromoteImmediately:
        def __init__(self, *a, **kw): pass
        def record_recall_pair(self, **kw): pass
        def decide(self): return "promote", {"effect": 0.2}

    monkeypatch.setattr(sr_mod, "ShadowTest", _PromoteImmediately)
    router._shadow = _PromoteImmediately()
    router._candidate_id = cand_id

    router.on_recall_settled(
        query_id="q-promote", arm="candidate", ndcg_at_10=0.9,
    )
    router.arm_post_promotion_watch(baseline_ndcg=0.8)
    # 200 observations matching baseline — no regression.
    for i in range(200):
        router.on_recall_settled(
            query_id=f"q-{i}", arm="baseline", ndcg_at_10=0.8,
        )

    with sqlite3.connect(learning_db) as conn:
        cand_row = conn.execute(
            "SELECT is_active, is_rollback FROM learning_model_state "
            "WHERE id=?", (cand_id,),
        ).fetchone()
        # Still promoted, no rollback fired.
        assert cand_row[0] == 1
        assert cand_row[1] == 0


def test_watch_window_needs_200_observations_before_rollback(
    learning_db: Path, memory_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-promotion watcher MUST accumulate 200 observations before
    any rollback decision — a strong regression at n=199 is still
    ``should_rollback is False``.
    """
    from superlocalmemory.core import shadow_router as sr_mod

    _, active_id, cand_id = _seed_active_previous_candidate(learning_db, "p")

    router = sr_mod.ShadowRouter(
        memory_db=str(memory_db),
        learning_db=str(learning_db),
        profile_id="p",
    )

    class _PromoteImmediately:
        def __init__(self, *a, **kw): pass
        def record_recall_pair(self, **kw): pass
        def decide(self): return "promote", {"effect": 0.2}

    monkeypatch.setattr(sr_mod, "ShadowTest", _PromoteImmediately)
    router._shadow = _PromoteImmediately()
    router._candidate_id = cand_id
    router.on_recall_settled(
        query_id="q-promote", arm="candidate", ndcg_at_10=0.9,
    )
    router.arm_post_promotion_watch(baseline_ndcg=0.8)

    # Feed 199 regressions — below watch window. Rollback must NOT fire.
    for i in range(199):
        router.on_recall_settled(
            query_id=f"q-{i}", arm="baseline", ndcg_at_10=0.1,
        )
    with sqlite3.connect(learning_db) as conn:
        cand_row = conn.execute(
            "SELECT is_rollback FROM learning_model_state WHERE id=?",
            (cand_id,),
        ).fetchone()
        assert cand_row[0] == 0, (
            "rollback fired before watch window filled — H-19 guard broken"
        )
