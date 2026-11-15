# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Open slot disputes settled by the ladder: shadow records, on acts, ledger retires."""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_ladder import JudgeLadder, TierAnswer
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.truth.dispute_pass import DisputePass
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_slot_pass import _fact, _flags, _retired


class Fixed:
    name = "laya"

    def __init__(self, verdict, confidence=0.95):
        self.verdict, self.confidence, self.calls = verdict, confidence, 0

    def answer(self, question, items):
        self.calls += 1
        return [TierAnswer(self.verdict, self.confidence, model="fake")] * len(items)


@pytest.fixture
def disputed(tmp_path):
    db = _db(tmp_path / "memory.db")
    old = _fact(db, "The release cadence of SLM is one release every week", "2026-03-01T00:00:00Z")
    new = _fact(db, "The release cadence of SLM is one big release", "2026-11-14T00:00:00Z")
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    assert db.execute("SELECT COUNT(*) AS n FROM slot_disputes WHERE state='open'")[0]["n"] == 1
    return db, old, new


def _open(db) -> int:
    return db.execute("SELECT COUNT(*) AS n FROM slot_disputes WHERE state='open'")[0]["n"]


def _calibrate(db, verdict, *, n=None, wrong=0, tier="laya", blind=1, source="audit"):
    """Seed labelled receipts, as a person's reviews would, so the gate can decide."""
    n = n if n is not None else (80 if verdict == "replaces" else 400)
    with db.raw_connection() as conn:
        for i in range(n):
            rid = f"seed-{verdict}-{i}"
            conn.execute(
                "INSERT INTO judge_receipts (receipt_id, profile_id, question_id, "
                "question_version, subject_kind, subject_id, population, tier, model, verdict, "
                "confidence, threshold, accepted, calibration_id, inputs_hash, decided_at) VALUES "
                "(?, 'default', 'SLOT_RELATION', ?, 'seed', ?, 'slot_dispute', ?, 'fake', ?, 0.95, "
                "0.7, 1, 'c', ?, '2026-11-15T00:00:00+00:00')",
                (rid, SLOT_RELATION.version, rid, tier, verdict, "seed-" + rid))
            truth = ("replaces" if verdict != "replaces" else "unrelated") if i < wrong \
                else verdict
            conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labeller, blind, "
                         "labelled_at) VALUES (?, ?, ?, 'auditor', ?, ?)",
                         (rid, truth, source, blind,
                          f"2026-11-15T{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}+00:00"))


def _run(db, verdict, *, confidence=0.95, calibrated=True, **flags):
    if calibrated:
        _calibrate(db, verdict)
    tier = Fixed(verdict, confidence)
    counts = DisputePass(db, FeatureFlags(file_values=flags, environ={}),
                         JudgeLadder([tier])).run()
    return counts, tier


def test_off_asks_nothing(disputed) -> None:
    db, _old, _new = disputed
    counts, tier = _run(db, "replaces")
    assert counts == {} and tier.calls == 0


def test_shadow_records_a_receipt_and_changes_nothing(disputed) -> None:
    db, old, _new = disputed
    counts, _ = _run(db, "replaces", calibrated=False, **{"judge.ladder": "shadow"})
    assert counts == {"laya:replaces": 1}
    assert _open(db) == 1 and _retired(db, old) == (None, None, None)
    assert db.execute("SELECT COUNT(*) AS n FROM judge_receipts")[0]["n"] == 1


def test_a_pair_already_asked_is_not_asked_again(disputed) -> None:
    db, _old, _new = disputed
    _run(db, "replaces", confidence=0.5, calibrated=False, **{"judge.ladder": "shadow"})
    counts, tier = _run(db, "replaces", calibrated=False, **{"judge.ladder": "shadow"})
    assert counts == {} and tier.calls == 0


def test_replaces_without_auto_apply_is_held_for_review(disputed) -> None:
    db, old, _new = disputed
    counts, _ = _run(db, "replaces", **{"judge.ladder": "on"})
    assert counts["held_for_review"] == 1 and _open(db) == 1
    assert _retired(db, old) == (None, None, None)


def test_replaces_with_auto_apply_retires_through_the_ledger(disputed) -> None:
    db, old, new = disputed
    counts, _ = _run(db, "replaces", **{"judge.ladder": "on", "truth.slots": "on",
                                        "truth.slots.auto_apply": "on"})
    assert counts["resolved_replaces"] == 1 and _open(db) == 0
    _until, expired, by = _retired(db, old)
    assert by == new and expired is not None
    head = db.execute("SELECT r.fact_id FROM memory_slots s JOIN slot_revisions r "
                      "ON r.revision_id = s.head_revision_id")[0]["fact_id"]
    assert head == new


@pytest.mark.parametrize("verdict,state", [("same", "duplicate"), ("both_true", "compatible"),
                                           ("unrelated", "compatible")])
def test_non_retiring_answers_settle_without_touching_facts(disputed, verdict, state) -> None:
    db, old, new = disputed
    counts, _ = _run(db, verdict, **{"judge.ladder": "on"})
    assert counts[f"resolved_{verdict}"] == 1 and _open(db) == 0
    assert _retired(db, old) == (None, None, None)
    assert db.execute("SELECT state FROM slot_revisions WHERE fact_id = ?",
                      (new,))[0]["state"] == state


def test_an_unsure_judge_leaves_the_dispute_open(disputed) -> None:
    db, old, _new = disputed
    counts, _ = _run(db, "replaces", confidence=0.6,
                     **{"judge.ladder": "on", "truth.slots": "on", "truth.slots.auto_apply": "on"})
    assert counts == {"none:abstain": 1} and _open(db) == 1
    assert _retired(db, old) == (None, None, None)


def test_without_labels_nothing_acts(disputed) -> None:
    db, old, _new = disputed
    counts, _ = _run(db, "replaces", calibrated=False,
                     **{"judge.ladder": "on", "truth.slots.auto_apply": "on"})
    assert counts["held_uncalibrated"] == 1 and _open(db) == 1
    assert _retired(db, old) == (None, None, None)


def test_labels_with_too_many_errors_keep_it_held(disputed) -> None:
    db, old, _new = disputed
    _calibrate(db, "replaces", n=200, wrong=20)  # an error rate near 10%
    counts, _ = _run(db, "replaces", calibrated=False,
                     **{"judge.ladder": "on", "truth.slots.auto_apply": "on"})
    assert counts["held_uncalibrated"] == 1 and _retired(db, old) == (None, None, None)


def test_too_few_clean_labels_are_not_enough(disputed) -> None:
    db, _old, _new = disputed
    _calibrate(db, "same", n=300)  # alpha_close 0.01 needs ~373 clean
    counts, _ = _run(db, "same", calibrated=False, **{"judge.ladder": "on"})
    assert counts["held_uncalibrated"] == 1


def test_auto_apply_needs_slots_on_not_shadow(disputed) -> None:
    db, old, _new = disputed
    counts, _ = _run(db, "replaces", **{"judge.ladder": "on", "truth.slots": "shadow",
                                        "truth.slots.auto_apply": "on"})
    assert counts["held_for_review"] == 1 and _retired(db, old) == (None, None, None)


def test_review_labels_never_certify(disputed) -> None:
    db, old, _new = disputed
    _calibrate(db, "replaces", n=200, source="review")
    counts, _ = _run(db, "replaces", calibrated=False,
                     **{"judge.ladder": "on", "truth.slots": "on", "truth.slots.auto_apply": "on"})
    assert counts["held_uncalibrated"] == 1 and _retired(db, old) == (None, None, None)


def test_a_decision_made_in_shadow_is_acted_on_later_without_re_asking(disputed) -> None:
    db, old, new = disputed
    _run(db, "replaces", calibrated=False, **{"judge.ladder": "shadow"})
    counts, tier = _run(db, "replaces",
                        **{"judge.ladder": "on", "truth.slots": "on", "truth.slots.auto_apply": "on"})
    assert tier.calls == 0 and counts["resolved_replaces"] == 1
    assert _retired(db, old)[2] == new


# -- R-35 / R-41: which earlier receipts make a pair "already judged" --------------------------

class FailingTier:
    name = "laya"

    def __init__(self):
        self.calls = 0

    def answer(self, question, items):
        from superlocalmemory.core.judge_ladder import TierFailure

        self.calls += 1
        raise TierFailure("timeout")


def _pass(db, *tiers, limit=32, **flags):
    flags = flags or {"judge.ladder": "shadow"}
    return DisputePass(db, FeatureFlags(file_values=flags, environ={}), JudgeLadder(list(tiers)),
                       limit=limit).run()


def test_a_failed_ask_is_receipted_and_the_pair_is_asked_again(disputed) -> None:
    db, _old, _new = disputed
    failing = FailingTier()
    _pass(db, failing)
    rows = db.execute("SELECT verdict, failure_kind FROM judge_receipts")
    assert [(r["verdict"], r["failure_kind"]) for r in rows] == [("tier_failed", "timeout")]
    retry = Fixed("both_true", confidence=0.5)
    _pass(db, retry)
    assert retry.calls == 1  # a failed ask judged nothing, so it did not park the pair


def test_a_pair_abstained_on_in_shadow_reaches_a_tier_that_is_newly_available(disputed) -> None:
    db, _old, _new = disputed
    laya = Fixed("both_true", confidence=0.5)
    _pass(db, laya)
    _pass(db, laya)
    assert laya.calls == 1  # abstained by every available tier: paged past

    class Jev(Fixed):
        name = "jev"

    jev = Jev("both_true", confidence=0.5)
    _pass(db, laya, jev)
    assert jev.calls == 1  # the ladder grew (shadow -> on): the pair is judged again


def test_with_no_model_tier_the_page_limit_does_not_block_the_rules(disputed) -> None:
    from superlocalmemory.core.judge_ladder import RulesTier

    db, _old, _new = disputed

    class CountingRules(RulesTier):
        calls = 0

        def answer(self, question, items):
            CountingRules.calls += len(items)
            return super().answer(question, items)

    _pass(db, CountingRules(), limit=0)
    assert CountingRules.calls == 1
    assert _pass(db, CountingRules(), Fixed("same"), limit=0) == {}  # a model tier: limit holds
