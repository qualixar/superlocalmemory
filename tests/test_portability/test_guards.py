# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-9, D-19, M-15: imported text never moves a slot head; a contradiction opens a dispute that
no judge answers; one run lists at most DISPUTES_PER_RUN of them."""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.portability import constants as C
from superlocalmemory.truth.dispute_pass import DisputePass
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_portability import fixtures as fx
from tests.test_truth.test_slot_pass import NEW, OLD, _fact

NEWER = [{"id": f"m{n}", "memory": text, "created_at": "2026-11-10T00:00:00Z"} for n, text in
         enumerate([NEW, "The release cadence of SLM is two releases a month",
                    "The release cadence of SLM is daily"])]


def _flags(taint: str = "off") -> FeatureFlags:
    return FeatureFlags(file_values={"truth.slots": "on", "trust.taint": taint, "judge.ladder": "on"},
                        environ={})


def _slots(world, tmp_path, taint="off"):
    return SlotPass(world.db, _flags(taint), state_dir=tmp_path / "slot-state").run()


def _disputes(world):
    return [dict(r) for r in world.db.execute(
        "SELECT d.dispute_id, d.state, d.reason_code, d.resolved_by FROM slot_disputes d "
        "ORDER BY d.opened_at")]


def _head_fact(world):
    return dict(world.db.execute(
        "SELECT r.fact_id FROM memory_slots s JOIN slot_revisions r ON r.revision_id = "
        "s.head_revision_id")[0])["fact_id"]


@pytest.mark.parametrize("taint", ["off", "shadow", "on"])
def test_a_contradiction_from_an_import_opens_a_dispute_and_never_moves_the_head(world, tmp_path,
                                                                                taint) -> None:
    old = _fact(world.db, OLD, "2026-03-01T00:00:00+00:00")
    run = world.run("m.json", fx.dumps(NEWER[:1]))
    assert run["saved_items"] == 1
    counts = _slots(world, tmp_path, taint)
    assert counts["dispute"] == 1
    [dispute] = _disputes(world)
    assert (dispute["state"], dispute["reason_code"]) == ("open", "origin_below_head")
    assert _head_fact(world) == old  # the head did not move
    assert world.count("atomic_facts", "lifecycle = 'active'") >= 2  # both stay current
    from superlocalmemory.portability import service

    assert service.get_run(world.engine, world.profile, run["run_id"])["disputes_opened"] == 1


def test_no_judge_is_asked_about_an_imported_contradiction(world, tmp_path) -> None:
    _fact(world.db, OLD, "2026-03-01T00:00:00+00:00")
    world.run("m.json", fx.dumps(NEWER[:1]))
    _slots(world, tmp_path)

    class Ladder:
        def available_model_tiers(self, _question):
            return ["a model"]

        def decide(self, *_a, **_k):
            raise AssertionError("an import must cause no judge call")

    assert DisputePass(world.db, _flags(), Ladder()).run() == {}
    assert C.JUDGE_CALLS_PER_RUN == 0


def test_a_run_lists_at_most_the_cap_and_counts_the_rest(world, tmp_path, monkeypatch) -> None:
    from superlocalmemory.portability import service

    monkeypatch.setattr(C, "DISPUTES_PER_RUN", 1)
    old = _fact(world.db, OLD, "2026-03-01T00:00:00+00:00")
    run = world.run("m.json", fx.dumps(NEWER))
    assert run["saved_items"] == 3 and run["dispute_cap"] == 1
    _slots(world, tmp_path)
    states = sorted(d["state"] for d in _disputes(world))
    assert states == ["dismissed", "dismissed", "open"]
    assert {d["resolved_by"] for d in _disputes(world) if d["state"] == "dismissed"} == {"import:cap"}
    final = service.get_run(world.engine, world.profile, run["run_id"])
    assert (final["disputes_opened"], final["disputes_unlisted"]) == (1, 2)
    assert _head_fact(world) == old
    assert world.count("memories") == 4  # the text of every imported item is kept
    state = dict(world.db.execute("SELECT state FROM memory_slots")[0])["state"]
    assert state == "disputed"  # one is still open


def test_an_ordinary_save_is_not_held_by_the_import_guard(world, tmp_path) -> None:
    _fact(world.db, OLD, "2026-03-01T00:00:00+00:00")
    _fact(world.db, NEW, "2026-11-14T00:00:00+00:00")  # a user's own contradiction
    _slots(world, tmp_path)
    [dispute] = _disputes(world)
    assert dispute["reason_code"] != "origin_below_head"  # the normal path, unchanged


def test_an_import_into_an_empty_slot_becomes_its_head_and_displaces_nothing(world, tmp_path) -> None:
    run = world.run("m.json", fx.dumps(NEWER[:1]))
    counts = _slots(world, tmp_path)
    assert counts["head"] == 1 and "dispute" not in counts
    assert world.count("slot_disputes") == 0 and run["disputes_opened"] == 0


def test_an_older_imported_value_leaves_its_ordinary_reason(world, tmp_path) -> None:
    _fact(world.db, NEW, "2026-11-14T00:00:00+00:00")
    world.run("m.json", fx.dumps([{"id": "old", "memory": OLD, "created_at": "2025-01-01T00:00:00Z"}]))
    _slots(world, tmp_path)
    [dispute] = _disputes(world)
    assert dispute["reason_code"] == "older_than_head"  # the head stays; a judge may retire the import
