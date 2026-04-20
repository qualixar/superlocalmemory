# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §7.4

"""Tests for ``learning/reward_proxy.py`` — proxy settlement.

Covers hard rules P1 (60–120 s window) and P2 (NFC topic sig).
"""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from superlocalmemory.core.topic_signature import compute_topic_signature
from superlocalmemory.learning.bandit import ContextualBandit
from superlocalmemory.learning.bandit_cache import _BanditCache
from superlocalmemory.learning.reward_proxy import settle_stale_plays
from superlocalmemory.storage.migration_runner import apply_all


PROFILE = "rp"


def _bootstrap_learning_db(tmp_path: Path) -> Path:
    """Apply M003+M005 migrations AND add a minimal learning_signals table."""
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    stats = apply_all(learning_db, memory_db)
    assert "M005_bandit_tables" in stats["applied"], stats

    # Minimal learning_signals schema — just enough for the proxy lookup.
    conn = sqlite3.connect(str(learning_db), isolation_level=None)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS learning_signals ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " profile_id TEXT, query TEXT, fact_id TEXT,"
            " signal_type TEXT, value REAL, created_at TEXT,"
            " query_id TEXT, query_text_hash TEXT, position INTEGER,"
            " channel_scores TEXT, cross_encoder REAL)"
        )
    finally:
        conn.close()
    return learning_db


def _bootstrap_memory_db(tmp_path: Path) -> Path:
    """Create memory.db with a minimal tool_events table."""
    memory_db = tmp_path / "tool_events.db"
    conn = sqlite3.connect(str(memory_db), isolation_level=None)
    try:
        conn.execute(
            "CREATE TABLE tool_events ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " occurred_at TEXT NOT NULL,"
            " tool_name TEXT NOT NULL,"
            " payload_json TEXT)"
        )
    finally:
        conn.close()
    return memory_db


@pytest.fixture()
def env(tmp_path: Path):
    learning = _bootstrap_learning_db(tmp_path)
    memory = _bootstrap_memory_db(tmp_path)
    # Use a private cache so each test is isolated.
    cache = _BanditCache(max_entries=16)
    bandit = ContextualBandit(learning, profile_id=PROFILE, cache=cache)
    return {
        "learning": learning,
        "memory": memory,
        "bandit": bandit,
        "cache": cache,
    }


def _seed_play(db: Path, query_id: str, played_at: datetime) -> int:
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        cur = conn.execute(
            "INSERT INTO bandit_plays "
            "(profile_id, query_id, stratum, arm_id, played_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (PROFILE, query_id, "single_hop|0|morning",
             "fallback_default", played_at.isoformat(timespec="seconds")),
        )
        return int(cur.lastrowid)
    finally:
        conn.close()


def _seed_signals(db: Path, query_id: str, fact_ids: list[str]) -> None:
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        for i, fid in enumerate(fact_ids):
            conn.execute(
                "INSERT INTO learning_signals "
                "(profile_id, query, fact_id, signal_type, value, "
                " created_at, query_id, query_text_hash, position, "
                " channel_scores, cross_encoder) "
                "VALUES (?, '', ?, 'candidate', 1.0, ?, ?, '', ?, '{}', NULL)",
                (PROFILE, fid, "2026-04-18T00:00:00",
                 query_id, i),
            )
    finally:
        conn.close()


def _seed_tool_event(
    db: Path, occurred_at: datetime, tool_name: str,
    payload: dict,
) -> None:
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO tool_events "
            "(occurred_at, tool_name, payload_json) VALUES (?, ?, ?)",
            (occurred_at.isoformat(timespec="seconds"), tool_name,
             json.dumps(payload)),
        )
    finally:
        conn.close()


def _read_play(db: Path, play_id: int) -> tuple[float | None, str | None]:
    conn = sqlite3.connect(str(db))
    try:
        r = conn.execute(
            "SELECT reward, settlement_type FROM bandit_plays "
            "WHERE play_id = ?", (play_id,),
        ).fetchone()
    finally:
        conn.close()
    return (r[0], r[1]) if r else (None, None)


# ---------------------------------------------------------------------------
# P1 + proxy_position hit
# ---------------------------------------------------------------------------


def test_proxy_position_hit(env):
    """Tool event references a top-3 fact within 30 s → reward=1.0."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    played = now - timedelta(seconds=90)  # 90 s old → in window
    play_id = _seed_play(env["learning"], "q1", played)
    _seed_signals(env["learning"], "q1", ["factA", "factB", "factC"])
    _seed_tool_event(
        env["memory"], played + timedelta(seconds=15), "Read",
        {"path": "/home/foo/factB.md"},
    )

    settled = settle_stale_plays(
        PROFILE, env["learning"], env["memory"],
        now=now, bandit=env["bandit"],
    )
    assert settled == 1
    reward, kind = _read_play(env["learning"], play_id)
    assert reward == pytest.approx(1.0)
    assert kind == "proxy_position"


def test_proxy_no_signals_and_young_skips(env):
    """< 60 s old → not yet settleable."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    played = now - timedelta(seconds=30)
    play_id = _seed_play(env["learning"], "q-young", played)
    settled = settle_stale_plays(
        PROFILE, env["learning"], env["memory"],
        now=now, bandit=env["bandit"],
    )
    assert settled == 0
    reward, kind = _read_play(env["learning"], play_id)
    assert reward is None
    assert kind is None


def test_default_uncertain_after_120s(env):
    """No hit, no requery, > 120 s → reward=0.5, kind=default."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    played = now - timedelta(seconds=200)
    play_id = _seed_play(env["learning"], "q-default", played)
    _seed_signals(env["learning"], "q-default", ["f1"])

    settled = settle_stale_plays(
        PROFILE, env["learning"], env["memory"],
        now=now, bandit=env["bandit"],
    )
    assert settled == 1
    reward, kind = _read_play(env["learning"], play_id)
    assert reward == pytest.approx(0.5)
    assert kind == "default"


def test_proxy_in_window_no_evidence_waits(env):
    """Between 60 and 120 s with no evidence → wait (not settled)."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    played = now - timedelta(seconds=90)
    play_id = _seed_play(env["learning"], "q-wait", played)
    settled = settle_stale_plays(
        PROFILE, env["learning"], env["memory"],
        now=now, bandit=env["bandit"],
    )
    assert settled == 0
    reward, _ = _read_play(env["learning"], play_id)
    assert reward is None


# ---------------------------------------------------------------------------
# P2: requery via NFC topic sig
# ---------------------------------------------------------------------------


def test_requery_uses_nfc_topic_signature(env):
    """A follow-up query with equivalent NFC form triggers proxy_requery."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    played = now - timedelta(seconds=90)
    play_id = _seed_play(env["learning"], "q-req", played)
    _seed_signals(env["learning"], "q-req", ["factX"])

    # Original query (composed) and re-query (decomposed) — NFC normalises
    # them to the same codepoints.
    original = "café latté order"
    requery = unicodedata.normalize("NFD", "café latté order")
    assert compute_topic_signature(original) == compute_topic_signature(requery)

    # Seed the original recall event BEFORE played_at.
    _seed_tool_event(
        env["memory"], played - timedelta(seconds=1), "recall",
        {"query": original},
    )
    # Seed the requery AFTER played_at within 30 s.
    _seed_tool_event(
        env["memory"], played + timedelta(seconds=10), "recall",
        {"query": requery},
    )

    settled = settle_stale_plays(
        PROFILE, env["learning"], env["memory"],
        now=now, bandit=env["bandit"],
    )
    assert settled == 1
    reward, kind = _read_play(env["learning"], play_id)
    assert reward == pytest.approx(0.0)
    assert kind == "proxy_requery"


def test_requery_not_detected_when_topic_differs(env):
    """Different topic → NOT a requery → default after 120 s."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    played = now - timedelta(seconds=200)
    play_id = _seed_play(env["learning"], "q-other", played)
    _seed_signals(env["learning"], "q-other", ["factY"])

    _seed_tool_event(
        env["memory"], played - timedelta(seconds=1), "recall",
        {"query": "alpha beta gamma"},
    )
    _seed_tool_event(
        env["memory"], played + timedelta(seconds=10), "recall",
        {"query": "completely unrelated topic here"},
    )

    settled = settle_stale_plays(
        PROFILE, env["learning"], env["memory"],
        now=now, bandit=env["bandit"],
    )
    assert settled == 1
    reward, kind = _read_play(env["learning"], play_id)
    assert reward == pytest.approx(0.5)
    assert kind == "default"


# ---------------------------------------------------------------------------
# Missing tool_events DB — graceful degradation
# ---------------------------------------------------------------------------


def test_missing_tool_events_db_defaults_safely(tmp_path: Path, env):
    """tool_events DB absent → settlement still works on the 120 s default."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    played = now - timedelta(seconds=200)
    play_id = _seed_play(env["learning"], "q-no-te", played)

    ghost = tmp_path / "nonexistent.db"
    settled = settle_stale_plays(
        PROFILE, env["learning"], ghost,
        now=now, bandit=env["bandit"],
    )
    # With no evidence DB, age > 120 s → default reward.
    assert settled == 1
    reward, kind = _read_play(env["learning"], play_id)
    assert reward == pytest.approx(0.5)
    assert kind == "default"


def test_settle_stale_plays_on_missing_learning_db(tmp_path: Path):
    """Bad learning path → returns 0 without raising."""
    ghost = tmp_path / "nope.db"
    memory = _bootstrap_memory_db(tmp_path)
    # Opening a missing file creates an empty DB — so we use a directory
    # path that sqlite cannot open.
    bad = tmp_path  # a directory, not a file
    n = settle_stale_plays(PROFILE, bad, memory, now=datetime.now(timezone.utc))
    assert n == 0
