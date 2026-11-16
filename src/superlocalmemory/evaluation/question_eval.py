# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Evaluate a judge on one typed question before anything relies on it (W2).

Every question in ``core/judge_questions.QUESTIONS`` other than the one the
ladder already settles gets an eval here before it is asked anywhere it could
act. An eval set is JSON Lines, one case per line::

    {"id": "dup-01", "texts": {"a": "...", "b": "..."}, "label": "duplicate"}

``texts`` must name exactly the question's fields and ``label`` must be one of
its labels. ``evaluate`` asks a tier (any ``ask(question, items)`` callable,
one item per request by default, as D-43 requires for consequential
questions) and reports, per label: how often a confident answer of that label
was right, with the exact one-sided 95% lower bound (Clopper-Pearson), the
recall, and the full confusion table, plus abstentions (no answer, or below
the question's routing threshold).

An eval is evaluation data, never certificate input (M-01): a judge acts only
on blind audits of its own decisions on the user's store (D-09). Sets written
by hand are synthetic; they bound obvious failures, not a store's error rate.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from superlocalmemory.core.judge_calibration import clopper_pearson_lower
from superlocalmemory.core.judge_ladder import Item, TierAnswer
from superlocalmemory.core.judge_questions import Question

DELTA = 0.05
Ask = Callable[[Question, Sequence[Item]], "list[TierAnswer | None] | None"]


class EvalSetError(ValueError):
    """An eval set does not match its question."""


@dataclass(frozen=True)
class Case:
    case_id: str
    item: Item
    label: str


def load_set(question: Question, path: Path | str) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        cid, texts, label = str(row.get("id", "")), row.get("texts"), row.get("label")
        if not cid or cid in seen:
            raise EvalSetError(f"line {n}: a missing or repeated id")
        if not isinstance(texts, dict) or set(texts) != set(question.fields):
            raise EvalSetError(f"line {n}: texts must name exactly {question.fields}")
        if label not in question.criteria:
            raise EvalSetError(f"line {n}: {label!r} is not a {question.question_id} label")
        seen.add(cid)
        cases.append(Case(cid, Item("eval", cid, "eval", {k: str(v) for k, v in texts.items()}),
                          str(label)))
    return cases


@dataclass(frozen=True)
class LabelResult:
    label: str
    gold: int
    sure: int
    sure_right: int
    precision: float | None
    precision_lower: float
    recall: float | None


@dataclass(frozen=True)
class Report:
    question_id: str
    version: str
    model: str
    cases: int
    answered: int
    sure: int
    abstained: int
    failed_requests: int
    accuracy_sure: float | None
    labels: tuple[LabelResult, ...]
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {**{k: getattr(self, k) for k in (
            "question_id", "version", "model", "cases", "answered", "sure", "abstained",
            "failed_requests", "accuracy_sure")},
            "labels": [vars(x) for x in self.labels], "confusion": self.confusion}


def evaluate(ask: Ask, question: Question, cases: Sequence[Case], *, batch: int = 1) -> Report:
    """Ask every case and score the answers against the gold labels."""
    answers: list[TierAnswer | None] = []
    failed = 0
    for start in range(0, len(cases), max(1, batch)):
        part = cases[start:start + max(1, batch)]
        try:
            got = ask(question, [c.item for c in part])
        except Exception:  # noqa: BLE001 - a failed request is an abstention, counted
            got = None
        if not isinstance(got, list) or len(got) != len(part):
            failed += 1
            got = [None] * len(part)
        answers += [a if isinstance(a, TierAnswer) else None for a in got]
    confusion: dict[str, Counter] = {g: Counter() for g in question.criteria}
    sure_hits: Counter = Counter()
    sure_all: Counter = Counter()
    model = next((a.model for a in answers if a is not None and a.model), "")
    answered = sure = 0
    for case, answer in zip(cases, answers):
        if answer is None or answer.verdict not in question.criteria:
            confusion[case.label]["(none)"] += 1
            continue
        answered += 1
        confident = answer.confidence >= question.threshold(answer.verdict)
        confusion[case.label][answer.verdict if confident else answer.verdict + "?"] += 1
        if confident:
            sure += 1
            sure_all[answer.verdict] += 1
            sure_hits[answer.verdict] += answer.verdict == case.label
    gold = Counter(c.label for c in cases)
    labels = tuple(
        LabelResult(label, gold[label], sure_all[label], sure_hits[label],
                    sure_hits[label] / sure_all[label] if sure_all[label] else None,
                    clopper_pearson_lower(sure_hits[label], sure_all[label], DELTA),
                    sure_hits[label] / gold[label] if gold[label] else None)
        for label in question.criteria)
    right = sum(sure_hits.values())
    return Report(question.question_id, question.version, model, len(cases), answered, sure,
                  len(cases) - sure, failed, right / sure if sure else None, labels,
                  {g: dict(c) for g, c in confusion.items()})


__all__ = ["Case", "EvalSetError", "LabelResult", "Report", "evaluate", "load_set"]
