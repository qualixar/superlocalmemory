# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W2: every further question is typed, within the workers' limits, and evaluated."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.core import judge_questions as jq
from superlocalmemory.core.audit_review import PLAIN
from superlocalmemory.core.judge_calibration import CANNOT_TELL
from superlocalmemory.core.judge_ladder import TierAnswer
from superlocalmemory.evaluation import question_eval as qe

SETS = Path(__file__).resolve().parents[1] / "fixtures" / "question_evals"
NEW = ("DUPLICATE", "ADMIT", "CITATION_SUPPORT", "INSTRUCTION_RISK", "INJECT_VALUE")


def test_every_new_question_is_registered_consequential_and_hosted_only_with_consent() -> None:
    for qid in NEW:
        q = jq.QUESTIONS[qid]
        assert q.version == "1" and q.consequential is True
        assert q.privacy == jq.HOSTED_WITH_CONSENT
        assert all(label in q.criteria for label in q.thresholds)
        assert "data" in q.instructions  # every question says the texts are data


@pytest.mark.parametrize("qid", NEW)
def test_each_question_has_an_eval_set_covering_every_label(qid) -> None:
    q = jq.QUESTIONS[qid]
    cases = qe.load_set(q, SETS / f"{qid}.jsonl")
    counts = {label: sum(c.label == label for c in cases) for label in q.criteria}
    assert min(counts.values()) >= 3, counts


def test_audit_wording_for_duplicate_uses_the_judges_labels() -> None:
    labels = [k for k, _ in PLAIN["DUPLICATE"]["options"]]
    assert set(labels) == set(jq.DUPLICATE.criteria) | {CANNOT_TELL}


def _set(tmp_path, rows):
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_a_set_that_does_not_match_its_question_is_refused(tmp_path) -> None:
    ok = {"id": "x", "texts": {"memory": "m"}, "label": "keep"}
    with pytest.raises(qe.EvalSetError):
        qe.load_set(jq.ADMIT, _set(tmp_path, [ok, ok]))  # repeated id
    with pytest.raises(qe.EvalSetError):
        qe.load_set(jq.ADMIT, _set(tmp_path, [{**ok, "texts": {"a": "m"}}]))
    with pytest.raises(qe.EvalSetError):
        qe.load_set(jq.ADMIT, _set(tmp_path, [{**ok, "label": "maybe"}]))


def _cases(tmp_path):
    rows = [{"id": f"k{i}", "texts": {"memory": f"fact {i}"}, "label": "keep"} for i in range(6)]
    rows += [{"id": f"b{i}", "texts": {"memory": f"log {i}"}, "label": "bookkeeping"}
             for i in range(4)]
    return qe.load_set(jq.ADMIT, _set(tmp_path, rows))


def test_scores_sure_answers_against_gold_with_exact_bounds(tmp_path) -> None:
    def ask(question, items):
        out = []
        for it in items:
            if it.subject_id == "k0":
                out.append(TierAnswer("bookkeeping", 0.95, model="m"))  # a confident mistake
            elif it.subject_id == "k1":
                out.append(TierAnswer("keep", 0.3, model="m"))  # unsure: abstains
            else:
                out.append(TierAnswer("keep" if it.subject_id[0] == "k" else "bookkeeping",
                                      0.95, model="m"))
        return out

    r = qe.evaluate(ask, jq.ADMIT, _cases(tmp_path))
    by = {x.label: x for x in r.labels}
    assert (r.cases, r.answered, r.sure, r.abstained) == (10, 10, 9, 1)
    assert (by["keep"].sure, by["keep"].sure_right, by["keep"].recall) == (4, 4, 4 / 6)
    assert (by["bookkeeping"].sure, by["bookkeeping"].sure_right) == (5, 4)
    assert by["bookkeeping"].precision == 0.8 and 0 < by["bookkeeping"].precision_lower < 0.8
    assert r.confusion["keep"] == {"bookkeeping": 1, "keep?": 1, "keep": 4}
    assert r.accuracy_sure == 8 / 9 and r.model == "m"


def test_asks_one_item_per_request_by_default(tmp_path) -> None:
    sizes = []

    def ask(question, items):
        sizes.append(len(items))
        return [TierAnswer("keep", 0.9)] * len(items)

    qe.evaluate(ask, jq.ADMIT, _cases(tmp_path))
    assert sizes == [1] * 10


def test_a_failed_request_is_counted_and_abstains(tmp_path) -> None:
    def ask(question, items):
        if items[0].subject_id == "k2":
            raise RuntimeError("down")
        return None if items[0].subject_id == "k3" else [TierAnswer("keep", 0.9)]

    r = qe.evaluate(ask, jq.ADMIT, _cases(tmp_path))
    assert r.failed_requests == 2 and r.confusion["keep"]["(none)"] == 2
    assert r.as_dict()["labels"][0]["label"] == "keep"
