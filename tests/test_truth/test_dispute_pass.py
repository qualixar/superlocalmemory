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
from tests.helpers.flag_baseline import off_except
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


def _calibrate(db, verdict, *, n=None, wrong=0, tier="laya", blind=1, source="audit",
               regime="single"):
    """Seed labelled receipts, as a person's reviews would, so the gate can decide.

    A model's closing verdict needs 509 clean audits (alpha_close at delta/3, W0-07)
    and a tier inbox with real replacements in it (M-02), seeded under another
    model; a rule's needs 373 (alpha_close at delta, D-35).
    """
    n = n if n is not None else (80 if verdict == "replaces" else 560)
    if verdict != "replaces":
        _seed(db, "replaces", 300, tier=tier, model="inbox-model", label="replaces", tag="inbox",
              regime=regime)
    _seed(db, verdict, n, tier=tier, model="fake", wrong=wrong, blind=blind, source=source,
          regime=regime)


def _seed(db, verdict, n, *, tier, model, wrong=0, blind=1, source="audit", label=None, tag=None,
          regime="single"):
    tag = tag or verdict
    with db.raw_connection() as conn:
        for i in range(n):
            rid = f"seed-{tag}-{i}"
            conn.execute(
                "INSERT INTO judge_receipts (receipt_id, profile_id, question_id, "
                "question_version, subject_kind, subject_id, population, tier, model, verdict, "
                "confidence, threshold, accepted, calibration_id, inputs_hash, decided_at, "
                "audit_selected, audit_probability, batching_regime, inbox_selected) VALUES "
                "(?, 'default', 'SLOT_RELATION', ?, 'seed', ?, 'slot_dispute', ?, ?, ?, 0.95, "
                "0.7, 1, 'c', ?, ?, 1, 0.1, ?, 1)",
                (rid, SLOT_RELATION.version, rid, tier, model, verdict, "seed-" + rid,
                 f"2026-11-15T{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}+00:00", regime))
            truth = label or (("replaces" if verdict != "replaces" else "unrelated") if i < wrong
                              else verdict)
            conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labeller, blind, "
                         "labelled_at) VALUES (?, ?, ?, 'auditor', ?, ?)",
                         (rid, truth, source, blind,
                          f"2026-11-15T{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}+00:00"))


def _run(db, verdict, *, confidence=0.95, calibrated=True, **flags):
    if calibrated:
        _calibrate(db, verdict)
    tier = Fixed(verdict, confidence)
    counts = DisputePass(db, FeatureFlags(file_values=off_except(flags), environ={}),
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


def test_the_auto_apply_flag_alone_retires_nothing_without_a_certificate(disputed) -> None:
    db, old, _new = disputed  # W0-18 lifted by D-36: the gate, not a guard, holds it
    counts, _ = _run(db, "replaces", calibrated=False,
                     **{"judge.ladder": "on", "truth.slots": "on", "truth.slots.auto_apply": "on"})
    assert counts["held_uncalibrated"] == 1 and _open(db) == 1
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
                                           ("unrelated", "unbound")])
def test_non_retiring_answers_settle_without_touching_facts(disputed, verdict, state) -> None:
    db, old, new = disputed
    counts, _ = _run(db, verdict, **{"judge.ladder": "on"})
    assert counts[f"resolved_{verdict}"] == 1 and _open(db) == 0
    assert _retired(db, old) == (None, None, None)
    assert db.execute("SELECT state FROM slot_revisions WHERE fact_id = ?",
                      (new,))[0]["state"] == state


def test_a_head_already_retired_for_another_dismisses_the_dispute_and_changes_nothing(
        disputed) -> None:
    from superlocalmemory.truth.slot_resolver import retire_superseded

    db, old, new = disputed
    other = _fact(db, "A different memory that already retired the old one", "2026-05-01T00:00:00Z")
    with db.raw_connection() as conn:
        retire_superseded(conn, profile_id="default", scope="personal", predecessor=old,
                          successor=other)
    counts, _ = _run(db, "replaces", **{"judge.ladder": "on", "truth.slots": "on",
                                        "truth.slots.auto_apply": "on"})
    assert counts["ledger_retired_for_another"] == 1 and _open(db) == 0
    assert db.execute("SELECT state, resolved_by FROM slot_disputes")[0]["resolved_by"] == (
        "judge_ladder:retired_for_another")
    assert _retired(db, old)[2] == other  # the earlier retirement stands
    head = db.execute("SELECT r.fact_id FROM memory_slots s JOIN slot_revisions r "
                      "ON r.revision_id = s.head_revision_id")[0]["fact_id"]
    assert head == old  # rolled back: the slot is not moved on a retirement that did not happen


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
    return DisputePass(db, FeatureFlags(file_values=off_except(flags), environ={}), JudgeLadder(list(tiers)),
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


# -- R-04 / M-18 / R-36: rules act only on an audit of their own decisions --------------------

class RuleSame:
    """A deterministic tier answering "same", as Tier 0 does for identical statements."""
    name = "rules"
    is_model = False
    log_abstentions = False

    def answer(self, question, items):
        return [TierAnswer("same", 1.0, {"same": 1.0}, model="rules-v3",
                           calibration_id="rules-v3")] * len(items)


def test_a_rule_receipt_has_no_made_up_score(disputed) -> None:
    db, _old, _new = disputed
    _pass(db, RuleSame())
    rows = db.execute("SELECT tier, verdict, confidence, raw_score FROM judge_receipts")
    assert [(r["tier"], r["verdict"], r["confidence"], r["raw_score"]) for r in rows] == [
        ("rules", "same", None, None)]


def test_rules_do_not_close_a_dispute_without_an_audit_of_their_own(disputed) -> None:
    db, _old, _new = disputed
    counts = _pass(db, RuleSame(), **{"judge.ladder": "on"})
    assert counts["held_uncalibrated"] == 1 and _open(db) == 1


def test_audited_rules_close_a_dispute(disputed) -> None:
    db, _old, _new = disputed
    _calibrate(db, "same", tier="rules", regime="unbatched")  # alpha_close at delta/2: 459 clean audits
    with db.raw_connection() as conn:
        conn.execute("UPDATE judge_receipts SET model = 'rules-v3', confidence = NULL "
                     "WHERE tier = 'rules'")
    counts = _pass(db, RuleSame(), **{"judge.ladder": "on"})
    assert counts["resolved_same"] == 1 and _open(db) == 0


# -- D-36: an explicit key's new stated value is answered by the rule explicit-v1 -------------

@pytest.fixture
def explicit(tmp_path):
    from superlocalmemory.truth.slot_resolver import resolve_fact

    db = _db(tmp_path / "memory.db")
    old = _fact(db, "We ship SLM every week", "2026-03-01T00:00:00Z")
    new = _fact(db, "From now on SLM ships one big release", "2026-11-14T00:00:00Z")
    with db.raw_connection() as conn:
        for fact, value, when in ((old, "weekly", "2026-03-01"),
                                  (new, "one big release", "2026-11-14T00:00:00+00:00")):
            resolve_fact(conn, fact_id=fact, profile_id="default", scope="personal",
                         content="", valid_from=when, explicit_key="SLM/release cadence",
                         explicit_value=value)
    assert db.execute("SELECT reason_code FROM slot_disputes")[0][0] == "explicit_value_changed"
    return db, old, new


def _certify_explicit(db, n=80, wrong=0):
    _seed(db, "replaces", n, tier="rules", model="explicit-v1", wrong=wrong, regime="unbatched")
    with db.raw_connection() as conn:
        conn.execute("UPDATE judge_receipts SET confidence = NULL WHERE model = 'explicit-v1'")


def test_explicit_v1_answers_replaces_and_its_receipt_is_always_audited(explicit) -> None:
    from superlocalmemory.core.judge_ladder import RulesTier

    db, _old, _new = explicit
    assert _pass(db, RulesTier()) == {"rules:replaces": 1}
    rows = db.execute("SELECT tier, model, verdict, confidence, batching_regime, "
                      "audit_probability, audit_selected FROM judge_receipts")
    assert [tuple(r) for r in rows] == [
        ("rules", "explicit-v1", "replaces", None, "unbatched", 1.0, 1)]  # D-35: pi = 1


def test_explicit_v1_without_its_own_certificate_retires_nothing(explicit) -> None:
    from superlocalmemory.core.judge_ladder import RulesTier

    db, old, _new = explicit
    counts = _pass(db, RulesTier(), **{"judge.ladder": "on", "truth.slots": "on",
                                       "truth.slots.auto_apply": "on"})
    assert counts["held_uncalibrated"] == 1 and _open(db) == 1
    assert _retired(db, old) == (None, None, None)


def test_a_certified_explicit_v1_retires_the_old_value_through_the_ledger(explicit) -> None:
    from superlocalmemory.core.judge_ladder import RulesTier

    db, old, new = explicit
    _certify_explicit(db)
    held = _pass(db, RulesTier(), **{"judge.ladder": "on"})
    assert held["held_for_review"] == 1 and _retired(db, old) == (None, None, None)
    with db.raw_connection() as conn:  # the same decision, acted on once auto_apply is on
        conn.execute("DELETE FROM judge_receipts WHERE subject_kind = 'slot_dispute'")
    counts = _pass(db, RulesTier(), **{"judge.ladder": "on", "truth.slots": "on",
                                       "truth.slots.auto_apply": "on"})
    assert counts["resolved_replaces"] == 1 and _open(db) == 0
    assert _retired(db, old)[2] == new
    case = db.execute("SELECT status, reason_code FROM correction_cases")[0]
    assert (case["status"], case["reason_code"]) == ("applied", "slot_supersede")
    acted = db.execute("SELECT receipt_id FROM judge_receipts WHERE action_taken = "
                       "'resolved_replaces'")[0]["receipt_id"]
    linked = db.execute("SELECT receipt_id FROM slot_revisions WHERE fact_id = ?", (new,))
    assert linked[0]["receipt_id"] == acted  # R-09: a rollback can label what acted


def test_an_audit_that_finds_explicit_v1_wrong_stops_it(explicit) -> None:
    from superlocalmemory.core.judge_ladder import RulesTier

    db, old, _new = explicit
    _certify_explicit(db, n=120, wrong=6)
    counts = _pass(db, RulesTier(), **{"judge.ladder": "on", "truth.slots": "on",
                                       "truth.slots.auto_apply": "on"})
    assert counts["held_uncalibrated"] == 1 and _retired(db, old) == (None, None, None)


def test_explicit_v1_never_answers_a_dispute_it_did_not_open(disputed) -> None:
    from superlocalmemory.core.judge_ladder import RulesTier

    db, _old, _new = disputed  # pattern keys: telling a change from a paraphrase is a model's job
    assert _pass(db, RulesTier()) == {"none:abstain": 1}
    assert db.execute("SELECT COUNT(*) AS n FROM judge_receipts")[0]["n"] == 0


# -- R-10: a value older than the head is history, settled through the ladder and the ledger ---

@pytest.fixture
def older(tmp_path):
    db = _db(tmp_path / "memory.db")
    new = _fact(db, "The release cadence of SLM is one big release", "2026-11-14T00:00:00Z")
    old = _fact(db, "The release cadence of SLM is one release every week", "2026-03-01T00:00:00Z")
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    assert db.execute("SELECT reason_code FROM slot_disputes")[0][0] == "older_than_head"
    return db, old, new


class Recording(Fixed):
    def answer(self, question, items):
        self.seen = [dict(i.texts) for i in items]
        return super().answer(question, items)


def test_an_older_value_is_asked_in_time_order(older) -> None:
    db, _old, _new = older
    tier = Recording("replaces")
    _pass(db, tier)
    assert tier.seen == [{"older": "The release cadence of SLM is one release every week",
                          "newer": "The release cadence of SLM is one big release"}]


def test_replaces_on_an_older_value_retires_it_and_keeps_the_head(older) -> None:
    db, old, new = older
    _calibrate(db, "replaces")
    counts = DisputePass(db, FeatureFlags(file_values=off_except({
        "judge.ladder": "on", "truth.slots": "on", "truth.slots.auto_apply": "on"}), environ={}),
        JudgeLadder([Fixed("replaces")])).run()
    assert counts["resolved_replaces"] == 1 and _open(db) == 0
    until, expired, by = _retired(db, old)
    assert by == new and expired is not None  # no longer current: R-10's stale memory
    head = db.execute("SELECT r.fact_id FROM memory_slots s JOIN slot_revisions r "
                      "ON r.revision_id = s.head_revision_id")[0]["fact_id"]
    assert head == new


def test_explicit_v1_answers_a_stated_older_value(tmp_path) -> None:
    from superlocalmemory.core.judge_ladder import RulesTier
    from superlocalmemory.truth.slot_resolver import resolve_fact

    db = _db(tmp_path / "memory.db")
    new = _fact(db, "SLM ships one big release", "2026-11-14T00:00:00Z")
    old = _fact(db, "SLM shipped weekly", "2026-03-01T00:00:00Z")
    with db.raw_connection() as conn:
        for fact, value, when in ((new, "one big release", "2026-11-14"),
                                  (old, "weekly", "2026-03-01")):
            resolve_fact(conn, fact_id=fact, profile_id="default", scope="personal", content="",
                         valid_from=when, explicit_key="SLM/release cadence",
                         explicit_value=value)
    assert db.execute("SELECT reason_code FROM slot_disputes")[0][0] == "explicit_older_value"
    assert _pass(db, RulesTier()) == {"rules:replaces": 1}
