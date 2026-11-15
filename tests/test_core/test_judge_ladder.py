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
    [d] = ladder.decide(conn, SLOT_RELATION, _items())
    assert (d.verdict, d.tier, len(d.receipts)) == ("both_true", "laya", 1)
    assert second.calls == []


def test_an_unsure_answer_escalates_and_both_are_receipted(conn) -> None:
    ladder = JudgeLadder([Scripted("laya", [TierAnswer("replaces", 0.8)]),
                          Scripted("jev", [TierAnswer("replaces", 0.9)])])
    [d] = ladder.decide(conn, SLOT_RELATION, _items())
    assert (d.verdict, d.tier) == ("replaces", "jev")
    assert sorted(_receipts(conn)) == [("jev", "replaces", 1, 0.85), ("laya", "replaces", 0, 0.85)]


def test_replacing_needs_more_confidence_than_other_labels() -> None:
    assert SLOT_RELATION.threshold("replaces") == 0.85
    assert SLOT_RELATION.threshold("same") == 0.7


def test_nobody_sure_means_abstain(conn) -> None:
    ladder = JudgeLadder([Scripted("laya", [TierAnswer("replaces", 0.6)])])
    [d] = ladder.decide(conn, SLOT_RELATION, _items())
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
    [d] = ladder.decide(conn, SLOT_RELATION, _items())
    assert d.tier == "jev"


def test_batches_are_split_and_answers_stay_with_their_items(conn) -> None:
    calls = []
    tier = Scripted("laya", lambda items: [
        TierAnswer("same" if it.subject_id.endswith(("0", "2", "4", "6", "8")) else "both_true",
                   0.9) for it in items], calls)
    decisions = JudgeLadder([tier]).decide(conn, SLOT_RELATION, _items(20))
    assert calls == [16, 4]
    assert [d.verdict for d in decisions[:4]] == ["same", "both_true", "same", "both_true"]


def test_receipts_hold_ids_numbers_and_a_hash_never_text(conn) -> None:
    item = Item("slot_dispute", "d1", "p", {"older": "secret older", "newer": "secret newer"})
    JudgeLadder([Scripted("laya", [TierAnswer("same", 0.9, {"same": 0.9}, "m", "cal")])]).decide(
        conn, SLOT_RELATION, [item])
    row = conn.execute("SELECT * FROM judge_receipts").fetchone()
    assert "secret" not in json.dumps(row)
    names = [d[0] for d in conn.execute("SELECT * FROM judge_receipts").description]
    rec = dict(zip(names, row))
    assert rec["inputs_hash"] == inputs_hash(SLOT_RELATION, item)
    assert (rec["model"], rec["calibration_id"], rec["question_version"]) == ("m", "cal", "2")


def test_rules_tier_only_settles_near_identical_wording() -> None:
    rules = RulesTier()
    same = Item("x", "1", "p", {"older": "Set the flag to zero until the release.",
                                "newer": "set the flag to zero until the release"})
    changed = Item("x", "2", "p", {"older": "The cadence is weekly", "newer": "The cadence is monthly"})
    a, b = rules.answer(SLOT_RELATION, [same, changed])
    assert a is not None and a.verdict == "same" and b is None


def test_every_question_fits_the_worker_limits() -> None:
    for q in QUESTIONS.values():
        assert len(q.instructions) <= 300 and 2 <= len(q.criteria) <= 10


def test_laya_tier_answers_only_a_sure_same(monkeypatch) -> None:
    from superlocalmemory.core import judge_tiers
    from superlocalmemory.retrieval import judge_ask

    class Laya:
        _proc = object()
        ready = True

    monkeypatch.setattr(judge_ask, "ask_laya_statement", lambda *a, **k: [0.93, 0.4])
    out = judge_tiers.LayaTier(lambda: Laya()).answer(SLOT_RELATION, _items(2))
    assert out[0] is not None and out[0].verdict == "same" and out[1] is None


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

    judge_tiers.JevTier(lambda: Jev(), consent=lambda: False).answer(SLOT_RELATION, _items())
    assert seen["consent"] is False


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
