# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-7: the thematic-context lookup checks erasures against the few tombstones, not every member.

Profile (store copy, 43,888 facts): ``build_community_context`` was 88 ms at the median of a
recall (127 ms in isolation) on the serial path after retrieval. The raw read of the 41 summaries
takes 1.6 ms and parsing their member lists 3 ms; the rest was the erasure predicate, a
``json_each`` over all 28,800 members probed against ``projection_tombstones`` (136 rows). The
check now reads the tombstones of the profile once and tests the member lists the lookup parses
anyway. The answer is the same: a summary written from a member that is being erased is withheld
whole, and a tombstone of another profile withholds nothing.
"""

from __future__ import annotations

import json
import time
import types

import pytest

from superlocalmemory.retrieval import community_context as cc
from tests.test_core.test_coverage_backfill import _db


def _result(fid: str):
    return types.SimpleNamespace(fact=types.SimpleNamespace(fact_id=fid))


def _community(db, cid, members, *, profile="default", summary="A summary."):
    db.execute(
        "INSERT INTO community_summaries (profile_id, community_id, summary, keywords, "
        "fact_ids_json, fact_count) VALUES (?, ?, ?, 'k', ?, ?)",
        (profile, cid, summary, json.dumps(members), len(members)))


def _tombstone(db, fact_id, *, profile="default"):
    db.execute("INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, memory_id, "
               "created_at) VALUES (?, ?, 'e1', NULL, 1.0)", (profile, fact_id))


@pytest.fixture
def db(tmp_path):
    return _db(tmp_path / "memory.db")


def _ask(db, ids):
    return cc.build_community_context(db, [_result(i) for i in ids], "default")


def test_a_summary_written_from_an_erasing_member_is_withheld(db) -> None:
    _community(db, 0, ["a", "b", "c", "gone"])
    assert _ask(db, ["a", "b", "c"])["community_id"] == 0
    _tombstone(db, "gone")
    assert _ask(db, ["a", "b", "c"]) is None


def test_only_the_summary_with_the_erasing_member_is_withheld(db) -> None:
    _community(db, 0, ["a", "b", "gone"], summary="First.")
    _community(db, 1, ["x", "y", "z"], summary="Second.")
    _tombstone(db, "gone")
    assert _ask(db, ["a", "b"]) is None
    assert _ask(db, ["x", "y", "z"])["summary"] == "Second."


def test_a_tombstone_of_another_profile_withholds_nothing(db) -> None:
    _community(db, 0, ["a", "b", "c"])
    _tombstone(db, "a", profile="someone-else")
    assert _ask(db, ["a", "b", "c"])["community_id"] == 0


def test_a_withheld_summary_does_not_borrow_members_for_another(db) -> None:
    """A fact in two communities: the erased one is dropped whole, the other still answers."""
    _community(db, 0, ["a", "b", "gone"])
    _community(db, 1, ["a", "b", "c"])
    _tombstone(db, "gone")
    assert _ask(db, ["a", "b", "c"])["community_id"] == 1


def test_a_store_without_tombstones_answers(db) -> None:
    import sqlite3

    conn = sqlite3.connect(str(db.db_path))
    conn.execute("DROP TABLE projection_tombstones")
    conn.commit()
    conn.close()
    _community(db, 0, ["a", "b"])
    assert _ask(db, ["a", "b"])["community_id"] == 0


def test_a_malformed_member_list_is_skipped_not_fatal(db) -> None:
    db.execute("INSERT INTO community_summaries (profile_id, community_id, summary, keywords, "
               "fact_ids_json, fact_count) VALUES ('default', 5, 's', 'k', 'not json', 1)")
    _community(db, 0, ["a", "b"])
    _tombstone(db, "elsewhere")
    assert _ask(db, ["a", "b"])["community_id"] == 0


def test_the_lookup_no_longer_expands_every_member_in_sql(db, monkeypatch) -> None:
    _community(db, 0, ["a", "b"])
    _tombstone(db, "gone")
    seen: list[str] = []
    real = db.execute

    def spy(sql, params=()):
        seen.append(sql)
        return real(sql, params)

    monkeypatch.setattr(db, "execute", spy)
    _ask(db, ["a", "b"])
    assert seen and not any("json_each" in s for s in seen)


def test_the_lookup_is_fast_on_a_store_sized_like_the_real_one(db) -> None:
    """41 communities of 700 members and 136 tombstones: the real store's shape."""
    for cid in range(41):
        _community(db, cid, [f"c{cid}-m{i}" for i in range(700)])
    for i in range(136):
        _tombstone(db, f"elsewhere-{i}")
    started = time.perf_counter()
    for _ in range(5):
        assert _ask(db, [f"c3-m{i}" for i in range(6)])["community_id"] == 3
    per_call_ms = (time.perf_counter() - started) / 5 * 1000
    assert per_call_ms < 40, f"{per_call_ms:.0f} ms per lookup"
