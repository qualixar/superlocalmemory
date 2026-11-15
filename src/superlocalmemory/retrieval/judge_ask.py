# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Typed judge-ladder questions on the Laya worker (``ask``) and on Jev.

Both judges already answer small typed questions in the background (memory
kinds); this asks the ladder's questions the same way and reads the replies as
strictly: every item answered, every choice one of the question's labels,
every number a probability, or nothing at all.

Laya: the worker's ``ask`` command, two items per request so a recall that
arrives mid-question waits for one short exchange, the worker borrowed only
when free (``laya_background.exchange``), never started or warmed from here.

Jev: one request for a whole batch, every text screened by the hosted-judge
redaction first. A text that is nothing but a credential aborts the request:
refusing to send beats asking about a placeholder.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from superlocalmemory.core import recall_gate
from superlocalmemory.core.judge_ladder import Item, TierAnswer
from superlocalmemory.core.judge_questions import Question
from superlocalmemory.retrieval.hosted_redaction import redact_or_none

logger = logging.getLogger(__name__)

#: Items per worker request (as for memory kinds: ~55 ms an item).
CHUNK_ITEMS = 2
#: The worker cuts every text here; Jev gets the same so both see the same input.
MAX_TEXT_CHARS = 1800
QID = "q"
_YIELD_S = 0.01


def _probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) and 0.0 <= f <= 1.0 else None


def read_choice(raw: object, question: Question, model: str,
                calibration_id: str) -> TierAnswer | None:
    """One choice answer, strictly; None when anything is off."""
    if not isinstance(raw, Mapping) or raw.get("type", "choice") != "choice":
        return None
    choice = raw.get("choice")
    probabilities = raw.get("probabilities")
    if choice not in question.criteria or not isinstance(probabilities, Mapping):
        return None
    clean = {}
    for label, value in probabilities.items():
        p = _probability(value)
        if label not in question.criteria or p is None:
            return None
        clean[label] = p
    confidence = _probability(raw.get("confidence", clean.get(choice)))
    if confidence is None:
        return None
    return TierAnswer(verdict=choice, confidence=confidence, probabilities=clean, model=model,
                      calibration_id=calibration_id)


def _question_spec(question: Question) -> dict[str, Any]:
    return {"type": "choice", "instructions": question.instructions,
            "criteria": dict(question.criteria)}


def laya_requests(question: Question, items: Sequence[Item]) -> list[dict[str, Any]]:
    spec = _question_spec(question)
    reqs = []
    for start in range(0, len(items), CHUNK_ITEMS):
        part = items[start:start + CHUNK_ITEMS]
        reqs.append({"cmd": "ask", "items": [
            {"state": {f: item.texts.get(f, "")[:MAX_TEXT_CHARS] for f in question.fields},
             "questions": {QID: spec}} for item in part]})
    return reqs


def ask_laya(judge: Any, question: Question, items: Sequence[Item], *,
             model: str, calibration_id: str) -> list[TierAnswer | None] | None:
    """Answers from the running Laya worker, or None. Background threads only."""
    from superlocalmemory.retrieval.laya_background import exchange

    if not recall_gate.is_background_work() or getattr(judge, "ready", False) is not True:
        return None
    out: list[TierAnswer | None] = []
    for req in laya_requests(question, items):
        if recall_gate.background_preempt_requested():
            return None
        reply = exchange(judge, req, min(3.0, 0.25 + 0.15 * len(req["items"])))
        answers = reply.get("answers") if isinstance(reply, Mapping) and reply.get("ok") else None
        if not isinstance(answers, list) or len(answers) != len(req["items"]):
            return None
        for raw in answers:
            parsed = read_choice(raw.get(QID) if isinstance(raw, Mapping) else None,
                                 question, model, calibration_id)
            if parsed is None:
                return None
            out.append(parsed)
        time.sleep(_YIELD_S)
    return out


#: The statement Laya is asked for ``SLOT_RELATION``. Measured 15 Nov 2026 on 25
#: labelled pairs: as a true/false statement Laya separates "says the same
#: thing" from everything else at AUC 0.90, but "replaces" and "both true" at
#: 0.39-0.40 (below chance): the shipped model reads paraphrase, not
#: supersession. So Laya answers only "same"; the rest goes up the ladder.
SAME_STATEMENT = ("Read `older` and `newer`. Is this statement true: `newer` and `older` say "
                  "the same thing.")


def ask_laya_statement(judge: Any, statement: str, fields: Sequence[str],
                       items: Sequence[Item]) -> list[float] | None:
    """One true/false score per item from the running worker, or None."""
    from superlocalmemory.retrieval.laya_background import exchange

    if not recall_gate.is_background_work() or getattr(judge, "ready", False) is not True:
        return None
    spec = {"type": "noul", "instructions": statement}
    out: list[float] = []
    for start in range(0, len(items), CHUNK_ITEMS):
        part = items[start:start + CHUNK_ITEMS]
        if recall_gate.background_preempt_requested():
            return None
        req = {"cmd": "ask", "items": [
            {"state": {f: it.texts.get(f, "")[:MAX_TEXT_CHARS] for f in fields},
             "questions": {QID: spec}} for it in part]}
        reply = exchange(judge, req, min(3.0, 0.25 + 0.15 * len(part)))
        answers = reply.get("answers") if isinstance(reply, Mapping) and reply.get("ok") else None
        if not isinstance(answers, list) or len(answers) != len(part):
            return None
        for raw in answers:
            q = raw.get(QID) if isinstance(raw, Mapping) else None
            score = _probability(q.get("noul")) if isinstance(q, Mapping) else None
            if score is None:
                return None
            out.append(score)
        time.sleep(_YIELD_S)
    return out


def jev_body(model: str, question: Question, items: Sequence[Item]) -> dict[str, Any] | None:
    """The ``{model, state, questions}`` body, or None when it must not be sent."""
    state: dict[str, dict[str, str]] = {}
    questions: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items):
        key = f"i{index}"
        texts = {}
        for f in question.fields:
            text = redact_or_none(item.texts.get(f, ""))
            if text is None:
                logger.warning("judge ladder: refusing to send; a text is empty after redaction")
                return None
            texts[f] = text[:MAX_TEXT_CHARS]
        state[key] = texts
        spec = _question_spec(question)
        spec["instructions"] = (f"Judge ONLY item {key} (state.items.{key}); the other items "
                                "must not change this answer.\n" + question.instructions)
        questions[key] = spec
    return {"model": model, "state": {"items": state}, "questions": questions}


def read_jev(raw: bytes, body: Mapping[str, Any], question: Question, model: str,
             calibration_id: str, is_model: Any) -> list[TierAnswer | None] | None:
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, Mapping) or not is_model(payload.get("model")):
        return None
    answers = payload.get("answers")
    if not isinstance(answers, Mapping) or set(answers) != set(body["questions"]):
        return None
    out = [read_choice(answers[key], question, model, calibration_id)
           for key in body["state"]["items"]]
    return None if any(a is None for a in out) else out


__all__ = ["CHUNK_ITEMS", "MAX_TEXT_CHARS", "SAME_STATEMENT", "ask_laya", "ask_laya_statement",
           "jev_body", "laya_requests", "read_choice", "read_jev"]
