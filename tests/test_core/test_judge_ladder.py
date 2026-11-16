# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The judge ladder: routing by confidence, abstention, strictness, receipts."""

from __future__ import annotations

import json
import sqlite3

import pytest

from superlocalmemory.core.judge_ladder import ABSTAIN, Item, JudgeLadder, RulesTier, TierAnswer, inputs_hash
from superlocalmemory.core.judge_questions import QUESTIONS, SLOT_RELATION
from superlocalmemory.storage.migrations import M072_judge_receipts as M072


class Scripted:
    def __init__(self, name, answers, calls=None):
        self.name, self._answers, self.calls = name, answers, calls if calls is not None else []

    def answer(self, question, items):
        self.calls.append(len(items))
        return self._answers(items) if callable(self._answers) else self._answers


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    M072.apply(c)
    return c


def _items(n=1):
    return [Item("slot_dispute", f"d{i}", "p", {"older": f"old {i}", "newer": f"new {i}"})
            for i in range(n)]


def _receipts(conn):
    return conn.execute("SELECT tier, verdict, accepted, threshold FROM judge_receipts "
                        "ORDER BY decided_at, tier").fetchall()


def test_a_sure_first_tier_decides_and_later_tiers_are_not_asked(conn) -> None:
    second = Scripted("jev", [TierAnswer("same", 0.99)])
    ladder = JudgeLadder([Scripted("laya", [TierAnswer("both_true", 0.9)]), second])
    [d] = ladder.decide_and_record(conn, SLOT_RELATION, _items())
    assert (d.verdict, d.tier, len(d.receipts)) == ("both_true", "laya", 1)
    assert second.calls == []


def test_an_unsure_answer_escalates_and_both_are_receipted(conn) -> None:
    ladder = JudgeLadder([Scripted("laya", [TierAnswer("replaces", 0.8)]),
                          Scripted("jev", [TierAnswer("replaces", 0.9)])])
    [d] = ladder.decide_and_record(conn, SLOT_RELATION, _items())
    assert (d.verdict, d.tier) == ("replaces", "jev")
    assert sorted(_receipts(conn)) == [("jev", "replaces", 1, 0.85), ("laya", "replaces", 0, 0.85)]


def test_replacing_needs_more_confidence_than_other_labels() -> None:
    assert SLOT_RELATION.threshold("replaces") == 0.85
    assert SLOT_RELATION.threshold("same") == 0.7


def test_nobody_sure_means_abstain(conn) -> None:
    ladder = JudgeLadder([Scripted("laya", [TierAnswer("replaces", 0.6)])])
    [d] = ladder.decide_and_record(conn, SLOT_RELATION, _items())
    assert d.abstained and d.verdict == ABSTAIN and len(d.receipts) == 1


@pytest.mark.parametrize("bad", [
    None, [], [TierAnswer("maybe", 0.99)], [TierAnswer("same", 1.5)], [TierAnswer("same", True)],
    ["not an answer"], RuntimeError("boom")])
def test_malformed_or_failing_tiers_are_skipped(conn, bad) -> None:
    def broken(items):
        if isinstance(bad, Exception):
            raise bad
        return bad

    ladder = JudgeLadder([Scripted("laya", broken), Scripted("jev", [TierAnswer("same", 0.9)])])
    [d] = ladder.decide_and_record(conn, SLOT_RELATION, _items())
    assert d.tier == "jev"


def test_batches_are_split_and_answers_stay_with_their_items(conn) -> None:
    calls = []
    tier = Scripted("laya", lambda items: [
        TierAnswer("same" if it.subject_id.endswith(("0", "2", "4", "6", "8")) else "both_true",
                   0.9) for it in items], calls)
    decisions = JudgeLadder([tier]).decide_and_record(conn, SLOT_RELATION, _items(20))
    assert calls == [16, 4]
    assert [d.verdict for d in decisions[:4]] == ["same", "both_true", "same", "both_true"]


def test_receipts_hold_ids_numbers_and_a_hash_never_text(conn) -> None:
    item = Item("slot_dispute", "d1", "p", {"older": "secret older", "newer": "secret newer"})
    JudgeLadder([Scripted("laya", [TierAnswer("same", 0.9, {"same": 0.9}, "m", "cal")])]
                ).decide_and_record(conn, SLOT_RELATION, [item])
    row = conn.execute("SELECT * FROM judge_receipts").fetchone()
    assert "secret" not in json.dumps(row)
    names = [d[0] for d in conn.execute("SELECT * FROM judge_receipts").description]
    rec = dict(zip(names, row))
    assert rec["inputs_hash"] == inputs_hash(SLOT_RELATION, item)
    assert (rec["model"], rec["calibration_id"], rec["question_version"]) == ("m", "cal", "2")


def test_rules_tier_only_settles_near_identical_wording() -> None:
    rules = RulesTier()
    same = Item("x", "1", "p", {"older": "Set the flag to zero until the release.",
                                "newer": "set  the flag to ZERO until the release"})
    changed = Item("x", "2", "p", {"older": "The cadence is weekly", "newer": "The cadence is monthly"})
    a, b = rules.answer(SLOT_RELATION, [same, changed])
    assert a is not None and a.verdict == "same" and b is None


def test_every_question_fits_the_worker_limits() -> None:
    for q in QUESTIONS.values():
        assert len(q.instructions) <= 300 and 2 <= len(q.criteria) <= 10


def test_laya_tier_returns_every_score_and_routes_by_its_floor(monkeypatch) -> None:
    from superlocalmemory.core import judge_tiers
    from superlocalmemory.retrieval import judge_ask

    class Laya:
        _proc = object()
        ready = True

    monkeypatch.setattr(judge_ask, "ask_laya_statement", lambda *a, **k: [0.93, 0.4])
    out = judge_tiers.LayaTier(lambda: Laya()).answer(SLOT_RELATION, _items(2))
    assert [(a.verdict, a.confidence) for a in out] == [("same", 0.93), ("same", 0.4)]
    assert judge_tiers.LayaTier.floors == {"same": 0.8}


def test_laya_tier_skips_a_jev_judge_and_unmeasured_questions() -> None:
    from dataclasses import replace

    from superlocalmemory.core import judge_tiers

    class Jev:
        ready = True

    assert judge_tiers.LayaTier(lambda: Jev()).answer(SLOT_RELATION, _items()) is None
    other = replace(SLOT_RELATION, question_id="OTHER")
    assert judge_tiers.LayaTier(lambda: object()).answer(other, _items()) is None


def test_jev_tier_needs_consent(monkeypatch) -> None:
    from superlocalmemory.core import judge_tiers

    seen = {}

    class Jev:
        def ask_ladder(self, question, items, *, consent):
            seen["consent"] = consent
            return None

    judge_tiers.JevTier(lambda: Jev(), consent_ref=lambda: None, live=lambda: True).answer(
        SLOT_RELATION, _items())
    assert seen == {}  # without consent the judge is never asked
    judge_tiers.JevTier(lambda: Jev(), consent_ref=lambda: "judge_ladder_hosted@t", live=lambda: True).answer(
        SLOT_RELATION, _items())
    assert seen == {"consent": True}


class _CountingJev:
    """A hosted judge that records every time anything could have been sent."""

    def __init__(self):
        self.supplied = 0
        self.asked = 0

    def supplier(self):
        self.supplied += 1
        return self

    def ask_ladder(self, question, items, *, consent):
        self.asked += 1
        return [TierAnswer("replaces", 0.99, model="jev")] * len(items)


def test_a_ladder_in_shadow_never_reaches_the_hosted_judge() -> None:
    """D-27 / R-41: shadow sends nothing off the machine, even with consent."""
    from superlocalmemory.core import judge_tiers

    jev = _CountingJev()
    ladder = judge_tiers.build_ladder(jev.supplier, hosted_consent_ref=lambda: "judge_ladder_hosted@t",
                                      hosted_live=lambda: False)
    assert "jev" not in ladder.tier_names
    decisions, rows = ladder.decide(SLOT_RELATION, _items(3))
    assert jev.asked == 0 and all(r.tier != "jev" for r in rows)
    assert all(d.tier != "jev" for d in decisions)


def test_a_ladder_that_is_on_with_consent_asks_the_hosted_judge() -> None:
    from superlocalmemory.core import judge_tiers

    jev = _CountingJev()
    ladder = judge_tiers.build_ladder(jev.supplier, hosted_consent_ref=lambda: "judge_ladder_hosted@t",
                                      hosted_live=lambda: True)
    assert ladder.tier_names == ("rules", "laya", "jev")
    decisions, _ = ladder.decide(SLOT_RELATION, _items(2))
    # SLOT_RELATION is consequential: one pair per hosted request (D-43).
    assert jev.asked == 2 and [d.tier for d in decisions] == ["jev", "jev"]


def test_switching_to_shadow_mid_pass_stops_hosted_calls_before_the_next_ask() -> None:
    """The flag is read at every ask, not only when the ladder is built."""
    from superlocalmemory.core import judge_tiers

    jev = _CountingJev()
    live = {"on": True}
    tier = judge_tiers.JevTier(jev.supplier, consent_ref=lambda: "judge_ladder_hosted@t", live=lambda: live["on"])
    assert tier.answer(SLOT_RELATION, _items()) is not None
    live["on"] = False
    assert tier.answer(SLOT_RELATION, _items()) is None
    assert jev.asked == 1 and jev.supplied == 1  # not even the judge object was fetched


@pytest.mark.parametrize("live,consent", [(False, "ref"), (True, None), (None, "ref"), (1, "ref"),
                                          (True, ""), (True, True)])
def test_the_hosted_tier_needs_a_live_flag_and_a_consent_record(live, consent) -> None:
    from superlocalmemory.core import judge_tiers

    jev = _CountingJev()
    tier = judge_tiers.JevTier(jev.supplier, consent_ref=lambda: consent, live=lambda: live)
    assert tier.answer(SLOT_RELATION, _items()) is None and jev.asked == 0


def test_a_failing_flag_read_counts_as_shadow() -> None:
    from superlocalmemory.core import judge_tiers

    def broken():
        raise OSError("features.json unreadable")

    jev = _CountingJev()
    tier = judge_tiers.JevTier(jev.supplier, consent_ref=lambda: "judge_ladder_hosted@t", live=broken)
    assert tier.answer(SLOT_RELATION, _items()) is None and jev.asked == 0
    ladder = judge_tiers.build_ladder(jev.supplier, hosted_consent_ref=lambda: "ref",
                                      hosted_live=broken)
    assert "jev" not in ladder.tier_names


@pytest.mark.parametrize("older,newer", [
    ("[Agentic_official] session ended 2026-10-26 17:06", "[Agentic_official] session ended 14:01"),
    ("The conversation ended at 2026-09-04 13:32", "The conversation ended at 2026-09-04 15:41"),
    ("2026-08-16 is the end date of Alex's session", "2026-08-17 is the end date of Alex's session"),
    ("File changed: p3_assemble.py", "File changed: assemble.py"),
])
def test_records_of_moments_never_replace_each_other(older, newer) -> None:
    [a] = RulesTier().answer(SLOT_RELATION, [Item("x", "1", "p", {"older": older, "newer": newer})])
    assert a is not None and a.verdict == "separate_events" and a.confidence >= 0.99


def test_a_real_value_change_is_left_to_the_models() -> None:
    item = Item("x", "1", "p", {"older": "Call stage_wait (timeout_s 240, detail true).",
                                "newer": "Call stage_wait (timeout_s 180)."})
    assert RulesTier().answer(SLOT_RELATION, [item]) == [None]


def test_slot_relation_v2_has_an_event_label() -> None:
    assert SLOT_RELATION.version == "2" and "separate_events" in SLOT_RELATION.criteria


def test_decide_touches_no_database_and_returns_rows() -> None:
    decisions, rows = JudgeLadder([Scripted("laya", [TierAnswer("same", 0.9, model="m")])]).decide(
        SLOT_RELATION, _items())
    assert decisions[0].verdict == "same" and decisions[0].model == "m"
    assert [r.receipt_id for r in rows] == list(decisions[0].receipts)


def test_every_model_ask_is_receipted_including_misses(conn) -> None:
    tier = Scripted("laya", lambda items: [None, TierAnswer("same", 0.1)])
    JudgeLadder([tier]).decide_and_record(conn, SLOT_RELATION, _items(2))
    rows = conn.execute("SELECT verdict, accepted FROM judge_receipts ORDER BY verdict").fetchall()
    assert rows == [("no_answer", 0), ("same", 0)]


def test_rule_misses_are_not_receipted(conn) -> None:
    JudgeLadder([RulesTier()]).decide_and_record(conn, SLOT_RELATION, _items())
    assert conn.execute("SELECT COUNT(*) FROM judge_receipts").fetchone()[0] == 0


@pytest.mark.parametrize("older,newer", [
    ("The server listens on port 8080", "The server listens on port 8443"),
    ("The cache is enabled", "The cache is not enabled"),
    ("Alex owns the repo", "Sam owns the repo"),
    ("The build runs on CI", "CI runs the build"),
    # Fable's v2 probes:
    ("The offset is -5", "The offset is 5"), ("Use a > b", "Use a < b"),
    ("Check x != y", "Check x = y"), ("It costs $20", "It costs €20"),
    ("The cache was enabled", "The cache is enabled"), ("Signed on Monday", "Signed by Monday"),
    ("Use A and B", "Use A or B"), ("Größe ist 5", "Größe ist 6"),
    ("The build runs on CI", "The build runs on the CI"),
])
def test_rules_never_call_a_changed_statement_same(older, newer) -> None:
    assert RulesTier().answer(SLOT_RELATION, [Item("x", "1", "p", {"older": older,
                                                                  "newer": newer})]) == [None]


def test_a_tier_floor_routes_but_never_lowers_the_question_threshold(conn) -> None:
    class Floored(Scripted):
        floors = {"same": 0.8}

    tier = Floored("laya", [TierAnswer("same", 0.75)])
    [d] = JudgeLadder([tier]).decide_and_record(conn, SLOT_RELATION, _items())
    assert d.abstained


# -- R-35 / I-09: a tier that was asked and failed as a whole is receipted --------------------

class Failing:
    def __init__(self, name, exc):
        self.name, self._exc, self.calls = name, exc, 0

    def answer(self, question, items):
        self.calls += 1
        raise self._exc


@pytest.mark.parametrize("exc,kind", [
    (None, "timeout"), (RuntimeError("boom"), "error"), (None, "made-up-kind")])
def test_a_tier_that_fails_as_a_whole_is_receipted_for_every_item(conn, exc, kind) -> None:
    from superlocalmemory.core.judge_ladder import TIER_FAILED, TierFailure

    err = exc if exc is not None else TierFailure(kind)
    expected = kind if kind in {"timeout"} else "error"
    decisions, rows = JudgeLadder([Failing("jev", err)]).decide(SLOT_RELATION, _items(3))
    assert all(d.abstained for d in decisions)
    assert [(r.tier, r.verdict, r.accepted, r.failure_kind) for r in rows] == [
        ("jev", TIER_FAILED, False, expected)] * 3
    assert all(d.receipts == (r.receipt_id,) for d, r in zip(decisions, rows))
    from superlocalmemory.core.judge_ladder import write_receipts
    write_receipts(conn, rows)  # the schema accepts them (CHECK keeps verdict and kind in step)
    assert conn.execute("SELECT COUNT(*) FROM judge_receipts WHERE failure_kind IS NOT NULL"
                        ).fetchone()[0] == 3


def test_a_malformed_batch_is_a_receipted_failure_not_a_silent_skip() -> None:
    decisions, rows = JudgeLadder([Scripted("laya", [TierAnswer("same", 0.9)])]).decide(
        SLOT_RELATION, _items(2))  # one answer for two items
    assert [(r.verdict, r.failure_kind) for r in rows] == [("tier_failed", "malformed")] * 2


def test_a_declining_tier_writes_nothing_and_the_next_tier_is_asked() -> None:
    nxt = Scripted("jev", [TierAnswer("same", 0.95, model="m")])
    decisions, rows = JudgeLadder([Scripted("laya", None), nxt]).decide(SLOT_RELATION, _items())
    assert [r.tier for r in rows] == ["jev"] and decisions[0].tier == "jev"


def test_after_a_failed_tier_the_next_tier_is_still_asked() -> None:
    from superlocalmemory.core.judge_ladder import TierFailure

    nxt = Scripted("jev", [TierAnswer("same", 0.95, model="m")])
    decisions, rows = JudgeLadder([Failing("laya", TierFailure("worker")), nxt]).decide(
        SLOT_RELATION, _items())
    assert [(r.tier, r.verdict) for r in rows] == [("laya", "tier_failed"), ("jev", "same")]
    assert decisions[0].receipts == tuple(r.receipt_id for r in rows)


def test_the_schema_refuses_a_failure_kind_without_a_failed_verdict(conn) -> None:
    from dataclasses import replace

    from superlocalmemory.core.judge_ladder import write_receipts

    _, rows = JudgeLadder([Scripted("laya", [TierAnswer("same", 0.9)])]).decide(
        SLOT_RELATION, _items())
    with pytest.raises(sqlite3.IntegrityError):
        write_receipts(conn, [replace(rows[0], failure_kind="timeout")])
    with pytest.raises(sqlite3.IntegrityError):
        write_receipts(conn, [replace(rows[0], verdict="tier_failed")])


def test_rules_are_not_a_model_tier_and_availability_never_asks() -> None:
    from superlocalmemory.core.judge_ladder import is_model_tier

    probe = Scripted("laya", [TierAnswer("same", 0.9)])
    probe.available = lambda q: True
    broken = Scripted("jev", None)
    broken.available = lambda q: (_ for _ in ()).throw(OSError("x"))
    ladder = JudgeLadder([RulesTier(), probe, broken])
    assert not is_model_tier(RulesTier())
    assert ladder.available_model_tiers(SLOT_RELATION) == frozenset({"laya"})
    assert probe.calls == [] and broken.calls == []


# -- R-42 / I-15: a hosted receipt names the consent record it relied on ----------------------

def _hosted(answers_or_exc):
    from superlocalmemory.core import judge_tiers

    class Jev:
        def ask_ladder(self, question, items, *, consent):
            if isinstance(answers_or_exc, BaseException):
                raise answers_or_exc
            return answers_or_exc(items)

    return judge_tiers.JevTier(lambda: Jev(), consent_ref=lambda: "judge_ladder_hosted@2026-11-15",
                               live=lambda: True)


@pytest.mark.parametrize("make", [
    lambda: _hosted(lambda items: [TierAnswer("replaces", 0.99, model="jev")] * len(items)),
    lambda: _hosted(lambda items: [TierAnswer("not-a-label", 0.99)] * len(items)),  # misses
    lambda: _hosted(__import__("superlocalmemory.core.judge_ladder",
                               fromlist=["TierFailure"]).TierFailure("timeout")),
    lambda: _hosted(RuntimeError("crashed after sending")),
])
def test_every_receipt_of_a_hosted_ask_carries_its_consent_record(conn, make) -> None:
    from superlocalmemory.core.judge_ladder import write_receipts

    _, rows = JudgeLadder([RulesTier(), make()]).decide(SLOT_RELATION, _items(2))
    assert rows and {r.consent_ref for r in rows} == {"judge_ladder_hosted@2026-11-15"}
    write_receipts(conn, rows)
    assert conn.execute("SELECT COUNT(*) FROM judge_receipts WHERE consent_ref IS NULL"
                        ).fetchone()[0] == 0


def test_local_receipts_carry_no_consent_record() -> None:
    _, rows = JudgeLadder([Scripted("laya", [TierAnswer("same", 0.9)])]).decide(
        SLOT_RELATION, _items())
    assert [r.consent_ref for r in rows] == [None]
