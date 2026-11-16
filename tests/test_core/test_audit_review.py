# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The audit queue a person answers: blind, one item per pair, labels on every drawn receipt."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core import audit_review
from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_ladder import JudgeLadder
from superlocalmemory.truth.dispute_pass import DisputePass
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_dispute_pass import Fixed
from tests.test_truth.test_slot_pass import _fact, _flags


class Other(Fixed):
    name = "jev"


@pytest.fixture
def judged(tmp_path):
    """A disputed pair judged by two tiers in shadow, both receipts drawn for audit."""
    db = _db(tmp_path / "memory.db")
    old = _fact(db, "The release cadence of SLM is one release every week", "2026-03-01T00:00:00Z")
    new = _fact(db, "The release cadence of SLM is one big release", "2026-11-14T00:00:00Z")
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    flags = FeatureFlags(file_values={"judge.ladder": "shadow"}, environ={})
    DisputePass(db, flags, JudgeLadder([Fixed("same", 0.4), Other("replaces", 0.95)])).run()
    db.execute("UPDATE judge_receipts SET audit_selected = 1 WHERE verdict NOT IN "
               "('no_answer', 'tier_failed', 'abstain')")
    return db, old, new


def _items(db):
    with db.raw_connection() as conn:
        return audit_review.pending(conn, "default")


def test_one_item_per_pair_in_time_order_without_any_verdict(judged) -> None:
    db, old, new = judged
    (item,) = _items(db)
    assert item["question"].startswith("Memory A was saved before memory B")
    assert [m["fact_id"] for m in item["memories"]] == [old, new]
    assert [m["name"] for m in item["memories"]] == ["A", "B"]
    flat = repr(item)
    for leak in ("verdict", "confidence", "model", "laya", "jev", "0.95"):
        assert leak not in flat
    assert [o["label"] for o in item["options"]][-1] == "cannot_tell"


def test_an_answer_labels_every_drawn_receipt_of_the_pair_blind(judged) -> None:
    db, _old, _new = judged
    (item,) = _items(db)
    drawn = db.execute("SELECT COUNT(*) AS n FROM judge_receipts WHERE audit_selected = 1")[0]["n"]
    with db.raw_connection() as conn:
        written = audit_review.answer(conn, profile_id="default", item_id=item["item_id"],
                                      label="replaces", channel="dashboard", actor="me")
    assert written == drawn >= 2
    rows = db.execute("SELECT label, source, labeller, blind FROM judge_labels")
    assert {(r["label"], r["source"], r["labeller"], r["blind"]) for r in rows} == {
        ("replaces", "audit", "person:dashboard:me", 1)}
    assert _items(db) == []
    with db.raw_connection() as conn:
        assert audit_review.pending_count(conn, "default") == 0


def test_a_wrong_answer_or_channel_writes_nothing(judged) -> None:
    db, _old, _new = judged
    (item,) = _items(db)
    with db.raw_connection() as conn:
        with pytest.raises(ValueError):
            audit_review.answer(conn, profile_id="default", item_id=item["item_id"],
                                label="maybe", channel="cli")
        with pytest.raises(ValueError):
            audit_review.answer(conn, profile_id="default", item_id=item["item_id"],
                                label="same", channel="mcp")
    assert db.execute("SELECT COUNT(*) AS n FROM judge_labels")[0]["n"] == 0


def test_another_profile_or_an_unknown_item_is_not_found(judged) -> None:
    db, _old, _new = judged
    (item,) = _items(db)
    with db.raw_connection() as conn:
        assert audit_review.pending(conn, "someone-else") == []
        with pytest.raises(LookupError):
            audit_review.answer(conn, profile_id="someone-else", item_id=item["item_id"],
                                label="same", channel="cli")
        with pytest.raises(LookupError):
            audit_review.answer(conn, profile_id="default", item_id="nope", label="same",
                                channel="cli")


def test_a_receipt_not_drawn_is_never_asked(judged) -> None:
    db, _old, _new = judged
    db.execute("UPDATE judge_receipts SET audit_selected = 0, inbox_selected = 0")
    assert _items(db) == []


def test_cannot_tell_is_an_allowed_answer(judged) -> None:
    db, _old, _new = judged
    (item,) = _items(db)
    with db.raw_connection() as conn:
        assert audit_review.answer(conn, profile_id="default", item_id=item["item_id"],
                                   label="cannot_tell", channel="cli") >= 2


def test_a_store_without_receipts_has_no_queue() -> None:
    conn = sqlite3.connect(":memory:")
    assert audit_review.pending(conn, "default") == [] and audit_review.pending_count(conn, "p") == 0
