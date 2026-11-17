# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-6e (D-45): bandit plays are settled by what agents used of what they surfaced."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.learning import reward_from_ledger as rl
from superlocalmemory.learning.reward_proxy import settle_stale_plays
from superlocalmemory.storage.migrations import M075_injection_receipts as M075

NOW = datetime.now(UTC)
OLD = (NOW - timedelta(days=2)).isoformat(timespec="microseconds")
FRESH = (NOW - timedelta(hours=2)).isoformat(timespec="microseconds")


@pytest.fixture
def stores(tmp_path):
    from tests.test_learning._signal_fixtures import make_db_with_migrations

    make_db_with_migrations(tmp_path)
    memory = tmp_path / "memory.db"
    conn = sqlite3.connect(str(memory))
    M075.apply(conn)
    conn.commit()
    conn.close()
    return tmp_path / "learning.db", memory


def _play(learning, play_id, query_id, *, at=OLD, shown=("a", "b"), profile="default"):
    conn = sqlite3.connect(str(learning))
    conn.execute("INSERT INTO bandit_plays (play_id, profile_id, query_id, stratum, arm_id,"
                 " played_at, shown_fact_ids) VALUES (?, ?, ?, 's', 'arm-1', ?, ?)",
                 (play_id, profile, query_id, at,
                  None if shown is None else json.dumps(list(shown))))
    conn.commit()
    conn.close()


def _delivery(memory, did, query_id, facts, *, at=OLD, profile="default"):
    conn = sqlite3.connect(str(memory))
    conn.execute("INSERT INTO injection_deliveries (delivery_id, profile_id, path, design,"
                 " holdout_state, epsilon, eligible_count, candidates, delivered_at,"
                 " recall_query_id) VALUES (?, ?, 'mcp_recall', 'x', 'off', 0, 0, ?, ?, ?)",
                 (did, profile, len(facts), at, query_id))
    for i, f in enumerate(facts, start=1):
        conn.execute("INSERT INTO injection_receipts VALUES (?, ?, ?, 'recall', ?, NULL, 0,"
                     " 'path_not_randomised', 1.0, 1, 1)", (did, f, profile, i))
    conn.commit()
    conn.close()


def _use(memory, did, fact, *, signal="marker_edit", after_h=1, at=OLD):
    when = (datetime.fromisoformat(at) + timedelta(hours=after_h)).isoformat()
    conn = sqlite3.connect(str(memory))
    conn.execute("INSERT INTO injection_usage VALUES (?, ?, 'default', ?, NULL, ?)",
                 (did, fact, signal, when))
    conn.commit()
    conn.close()


def _settled(learning, play_id):
    conn = sqlite3.connect(str(learning))
    row = conn.execute("SELECT reward, settlement_type, settled_at FROM bandit_plays"
                       " WHERE play_id = ?", (play_id,)).fetchone()
    conn.close()
    return row


def test_a_play_whose_shown_memory_was_used_is_rewarded(stores) -> None:
    learning, memory = stores
    _play(learning, 1, "rq-1")
    _delivery(memory, "d1", "rq-1", ["a", "b", "c"])
    _use(memory, "d1", "b")
    assert rl.settle_from_ledger("default", learning, memory) == 1
    reward, kind, settled_at = _settled(learning, 1)
    assert (reward, kind) == (1.0, rl.SETTLEMENT) and settled_at


def test_unused_or_a_use_of_something_it_did_not_surface_is_zero(stores) -> None:
    learning, memory = stores
    _play(learning, 1, "rq-1", shown=("a",))
    _delivery(memory, "d1", "rq-1", ["a", "b"])
    _use(memory, "d1", "b")                         # used, but the play surfaced only a
    _use(memory, "d1", "a", signal="overlap")       # not a use
    _use(memory, "d1", "a", after_h=30)             # after the window
    assert rl.settle_from_ledger("default", learning, memory) == 1
    assert _settled(learning, 1)[:2] == (0.0, rl.SETTLEMENT)


def test_a_play_that_recorded_nothing_uses_its_answers_top_five(stores) -> None:
    learning, memory = stores
    _play(learning, 1, "rq-1", shown=None)
    _play(learning, 2, "rq-2", shown=None)
    _delivery(memory, "d1", "rq-1", ["a", "b", "c", "d", "e", "f"])
    _delivery(memory, "d2", "rq-2", ["a", "b", "c", "d", "e", "f"])
    _use(memory, "d1", "e")                          # position 5
    _use(memory, "d2", "f")                          # position 6: outside
    rl.settle_from_ledger("default", learning, memory)
    assert _settled(learning, 1)[0] == 1.0 and _settled(learning, 2)[0] == 0.0


def test_an_open_window_waits_and_the_proxy_leaves_the_play_alone(stores) -> None:
    learning, memory = stores
    _play(learning, 1, "rq-1", at=(NOW - timedelta(hours=3)).isoformat())
    _delivery(memory, "d1", "rq-1", ["a"], at=FRESH)
    assert rl.settle_from_ledger("default", learning, memory) == 0
    settle_stale_plays("default", learning, memory)  # would expire a 3 h old play
    assert _settled(learning, 1) == (None, None, None)
    conn = sqlite3.connect(f"file:{memory}?mode=ro", uri=True)
    assert rl.awaiting_ledger(conn, "default", "rq-1", NOW)
    assert not rl.awaiting_ledger(conn, "default", "rq-unknown", NOW)
    assert not rl.awaiting_ledger(None, "default", "rq-1", NOW)


def test_a_play_no_delivery_names_is_left_to_the_proxy(stores) -> None:
    learning, memory = stores
    _play(learning, 1, "rq-1", at=(NOW - timedelta(hours=3)).isoformat())
    assert rl.settle_from_ledger("default", learning, memory) == 0
    settle_stale_plays("default", learning, memory)
    reward, kind, settled_at = _settled(learning, 1)
    assert kind != rl.SETTLEMENT and settled_at      # the proxy closed it as before


def test_profiles_do_not_mix(stores) -> None:
    learning, memory = stores
    _play(learning, 1, "rq-1")
    _delivery(memory, "d1", "rq-1", ["a"], profile="work")
    assert rl.settle_from_ledger("default", learning, memory) == 0


def test_a_store_without_the_ledger_settles_nothing(tmp_path, stores) -> None:
    learning, _memory = stores
    _play(learning, 1, "rq-1")
    bare = tmp_path / "bare.db"
    sqlite3.connect(str(bare)).close()
    assert rl.settle_from_ledger("default", learning, bare) == 0
    assert rl.settle_from_ledger("default", learning, tmp_path / "missing.db") == 0


def test_the_loop_settles_outcomes_then_usage_then_the_proxy(monkeypatch, tmp_path) -> None:
    from superlocalmemory.learning import reward_from_outcomes, reward_proxy
    from superlocalmemory.server import bandit_loops

    order = []
    monkeypatch.setattr(reward_from_outcomes, "settle_from_outcomes",
                        lambda *a: order.append("outcomes") or 0)
    monkeypatch.setattr(rl, "settle_from_ledger", lambda *a: order.append("usage") or 0)

    def proxy(*a):
        order.append("proxy")
        raise asyncio.CancelledError

    monkeypatch.setattr(reward_proxy, "settle_stale_plays", proxy)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(bandit_loops._reward_proxy_loop(tmp_path / "l.db", tmp_path / "m.db",
                                                    "default", 0.0))
    assert order == ["outcomes", "usage", "proxy"]
