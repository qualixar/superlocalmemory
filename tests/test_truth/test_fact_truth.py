# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""One memory's truth view for the dashboard detail (§8.10.3 b)."""

from __future__ import annotations

import sqlite3

from superlocalmemory.truth.fact_truth import for_fact
from tests.test_truth.test_legacy_cases import _case, _pass
from tests.test_truth.test_slot_pass import _fact


def _view(db, fact_id, **kw):
    with db.raw_connection() as conn:
        return for_fact(conn, "default", fact_id, **kw)


def test_a_chain_reads_newest_first_whatever_the_index_order(pair) -> None:
    db, old, new, tmp = pair
    newest = _fact(db, "Varun works from a co-working space", "2026-11-20T00:00:00Z")
    _case(db, "case-1", old, new)
    _case(db, "case-2", new, newest)
    _pass(db, tmp)
    (slot,) = _view(db, new)["slots"]
    assert slot["legacy"] is True and slot["subject"] is None
    assert [v["fact_id"] for v in slot["values"]] == [newest, new, old]
    assert [v["current"] for v in slot["values"]] == [True, False, False]
    assert [v["this_memory"] for v in slot["values"]] == [False, True, False]


def test_a_value_the_caller_may_not_see_keeps_its_place_but_not_its_text(pair) -> None:
    db, old, new, tmp = pair
    _case(db, "case-1", old, new)
    _pass(db, tmp)
    (slot,) = _view(db, new, hidden=lambda ids: set(ids))["slots"]
    other = next(v for v in slot["values"] if v["fact_id"] == old)
    assert other["visible"] is False and other["content"] is None
    mine = next(v for v in slot["values"] if v["fact_id"] == new)
    assert mine["visible"] and mine["content"]


def test_a_store_without_slot_tables_has_an_empty_view() -> None:
    conn = sqlite3.connect(":memory:")
    assert for_fact(conn, "default", "f") == {"fact_id": "f", "slots": [], "disputes": [],
                                              "source": None}
