# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The audit queue any caller answers: blind, one item per pair, labels on every drawn receipt (D-49)."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core import audit_review
from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_ladder import JudgeLadder
from superlocalmemory.truth.dispute_pass import DisputePass
from superlocalmemory.truth.slot_pass import SlotPass
from tests.helpers.flag_baseline import off_except
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
    flags = FeatureFlags(file_values=off_except({"judge.ladder": "shadow"}), environ={})
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
        out = audit_review.answer(conn, profile_id="default", item_id=item["item_id"],
                                  label="replaces", channel="dashboard", actor="dashboard",
                                  kind="person")
    assert out == {"labelled": drawn, "counted": drawn} and drawn >= 2
    rows = db.execute("SELECT label, source, labeller, blind, labeller_kind FROM judge_labels")
    assert {(r["label"], r["source"], r["labeller"], r["blind"], r["labeller_kind"])
            for r in rows} == {("replaces", "audit", "dashboard@dashboard", 1, "person")}
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
                                label="same", channel="agent")
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
                                   label="cannot_tell", channel="cli")["labelled"] >= 2


def test_a_store_without_receipts_has_no_queue() -> None:
    conn = sqlite3.connect(":memory:")
    assert audit_review.pending(conn, "default") == [] and audit_review.pending_count(conn, "p") == 0


# -- D-49: agent labellers, counted only when their model differs from the judged tier's ----------

def _models(db):
    """Give the two tiers' receipts distinct models (the fixture's fake tiers share one)."""
    ids = [r["receipt_id"] for r in db.execute(
        "SELECT receipt_id FROM judge_receipts WHERE audit_selected = 1 ORDER BY receipt_id")]
    for receipt_id, model in zip(ids, ("model-a", "model-b")):
        db.execute("UPDATE judge_receipts SET model = ? WHERE receipt_id = ?", (model, receipt_id))
    return ["model-a", "model-b"]


def _drawn_labels(db):
    """What the certificate gate reads for each drawn receipt: its counting label, if any."""
    from superlocalmemory.core.judge_calibration import COUNTING_LABEL

    return {r["receipt_id"]: r["label"] for r in db.execute(
        "SELECT r.receipt_id, l.label FROM judge_receipts r LEFT JOIN judge_labels l "
        f"ON l.receipt_id = r.receipt_id AND l.source = 'audit' AND l.blind = 1 AND {COUNTING_LABEL} "
        "WHERE r.audit_selected = 1")}


def test_an_agent_label_from_the_judged_model_is_recorded_but_not_counted(judged) -> None:
    db, _old, _new = judged
    (item,) = _items(db)
    same = _models(db)[0]
    with db.raw_connection() as conn:
        out = audit_review.answer(conn, profile_id="default", item_id=item["item_id"],
                                  label="replaces", channel="mcp", actor="agent:a", model=same)
    drawn = len(_drawn_labels(db))
    # the labeller's model equals one tier's model: that receipt does not count, the other does
    assert out["labelled"] == drawn and out["counted"] == drawn - 1
    counted = [lab for lab in _drawn_labels(db).values() if lab]
    assert len(counted) == drawn - 1
    row = db.execute("SELECT labeller, labeller_kind, labeller_model FROM judge_labels "
                     "ORDER BY receipt_id")[0]
    assert (row["labeller"], row["labeller_kind"], row["labeller_model"]) == (
        "agent:a@mcp", "agent", same)


def test_an_agent_that_names_no_model_never_counts(judged) -> None:
    db, _old, _new = judged
    _models(db)
    (item,) = _items(db)
    with db.raw_connection() as conn:
        out = audit_review.answer(conn, profile_id="default", item_id=item["item_id"],
                                  label="replaces", channel="http", actor="agent:a")
    assert out["counted"] == 0 and not any(_drawn_labels(db).values())
    with db.raw_connection() as conn:
        assert audit_review.pending_count(conn, "default") == 1  # still waiting for a labeller


def test_an_agent_with_another_model_counts_on_every_receipt(judged) -> None:
    db, _old, _new = judged
    _models(db)
    (item,) = _items(db)
    with db.raw_connection() as conn:
        out = audit_review.answer(conn, profile_id="default", item_id=item["item_id"],
                                  label="replaces", channel="cli", actor="agent:a",
                                  model="some-other-model")
    assert out["counted"] == out["labelled"] and all(_drawn_labels(db).values())
    assert _items(db) == []


def test_a_persons_label_always_counts(judged) -> None:
    db, _old, _new = judged
    (item,) = _items(db)
    with db.raw_connection() as conn:
        out = audit_review.answer(conn, profile_id="default", item_id=item["item_id"],
                                  label="replaces", channel="dashboard", actor="dashboard",
                                  kind="person")
    assert out["counted"] == out["labelled"]


def test_the_queue_does_not_offer_a_pair_only_the_labellers_own_model_judged(judged) -> None:
    db, _old, _new = judged
    a, b = _models(db)
    with db.raw_connection() as conn:
        assert len(audit_review.pending(conn, "default", labeller_model="")) == 1
        assert len(audit_review.pending(conn, "default", labeller_model=a)) == 1  # b can count
        db.execute("DELETE FROM judge_receipts WHERE model = ?", (b,))
        assert audit_review.pending(conn, "default", labeller_model=a) == []
        assert len(audit_review.pending(conn, "default", labeller_model="another")) == 1


def test_one_label_per_receipt_per_labeller(judged) -> None:
    db, _old, _new = judged
    (item,) = _items(db)
    with db.raw_connection() as conn:
        audit_review.answer(conn, profile_id="default", item_id=item["item_id"], label="same",
                            channel="mcp", actor="agent:a", model="")
        with pytest.raises(LookupError):  # the same labeller cannot add a second one
            audit_review.answer(conn, profile_id="default", item_id=item["item_id"],
                                label="replaces", channel="mcp", actor="agent:a", model="")


def test_the_label_is_recorded_in_the_decision_log_and_can_be_retracted(judged) -> None:
    db, _old, _new = judged
    (item,) = _items(db)
    with db.raw_connection() as conn:
        audit_review.answer(conn, profile_id="default", item_id=item["item_id"], label="same",
                            channel="mcp", actor="agent:a", model="m1")
        out = audit_review.retract(conn, profile_id="default", item_id=item["item_id"],
                                   channel="dashboard", actor="dashboard")
    assert out["retracted"] >= 2
    assert db.execute("SELECT COUNT(*) AS n FROM judge_labels")[0]["n"] == 0
    assert len(_items(db)) == 1  # waiting again
    rows = [tuple(r) for r in db.execute(
        "SELECT action, actor, channel, reverted_by FROM decision_log ORDER BY rowid")]
    assert rows == [("label", "agent:a", "mcp", "dashboard"),
                    ("retract", "dashboard", "dashboard", None)]


def test_there_is_nothing_to_retract_twice(judged) -> None:
    db, _old, _new = judged
    (item,) = _items(db)
    with db.raw_connection() as conn:
        audit_review.answer(conn, profile_id="default", item_id=item["item_id"], label="same",
                            channel="cli", actor="a")
        audit_review.retract(conn, profile_id="default", item_id=item["item_id"], channel="cli")
        with pytest.raises(LookupError):
            audit_review.retract(conn, profile_id="default", item_id=item["item_id"],
                                 channel="cli")
