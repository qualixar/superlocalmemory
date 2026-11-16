# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Tier 0.5: a small local scorer for "does the newer memory replace the older?" (W2, innovation 4).

Trained only on people's blind audit labels on this store (M-01), never on a
model's answers, and asked only in shadow: the ladder receipts its score under
``<population>:shadow`` and never stops on it (``shadow_only``). It can act only
after it holds a certificate of its own, like any judge (D-09).

Features come from the pair's text alone, because that is all a judge is given:
word overlap, whether the numbers differ, a negation on one side only, a shared
opening (same subject), and the length ratio. The blueprint's other features
(embedding cosine, sheaf residual, time gap, origin) need data the ladder's
items do not carry yet; they arrive when items do (recorded in the blueprint).

The model is an L2-regularised logistic regression on standardised features,
fitted by gradient descent in plain Python, and exists only once there are at
least ``MIN_PER_CLASS`` audited "replaces" and as many "not replaces" pairs. Its
name carries a hash of the labelled receipts it was fitted on, so new labels
make a new model and a new certificate cell (R-26).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from superlocalmemory.core.judge_calibration import CANNOT_TELL, RETIRING
from superlocalmemory.core.judge_ladder import Item, TierAnswer
from superlocalmemory.core.judge_questions import SLOT_RELATION, Question

MIN_PER_CLASS = 20
L2 = 0.1
STEPS = 400
RATE = 0.5
FEATURES = ("overlap", "numbers_differ", "negation_flip", "same_opening", "length_ratio")
_WORD = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_NEGATION = frozenset({"not", "no", "never", "none", "without", "stopped"})


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def features(older: str, newer: str) -> list[float]:
    a, b = _words(older), _words(newer)
    sa, sb = set(a), set(b)
    overlap = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    na, nb = set(_NUMBER.findall(older)), set(_NUMBER.findall(newer))
    numbers_differ = 1.0 if na and nb and na != nb else 0.0
    neg_a, neg_b = bool(sa & _NEGATION), bool(sb & _NEGATION)
    head_a, head_b = set(a[:4]), set(b[:4])
    same_opening = len(head_a & head_b) / len(head_a | head_b) if head_a | head_b else 0.0
    la, lb = len(a), len(b)
    length_ratio = min(la, lb) / max(la, lb) if max(la, lb) else 1.0
    return [overlap, numbers_differ, 1.0 if neg_a != neg_b else 0.0, same_opening, length_ratio]


@dataclass(frozen=True)
class Model:
    name: str
    weights: tuple[float, ...]
    bias: float
    means: tuple[float, ...]
    scales: tuple[float, ...]
    positives: int
    negatives: int

    def probability(self, x: Sequence[float]) -> float:
        z = self.bias + sum(w * (v - m) / s for w, v, m, s in
                            zip(self.weights, x, self.means, self.scales))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


_LABELLED = """
SELECT r.receipt_id, r.fact_ids_json, l.label FROM judge_receipts r
  JOIN judge_labels l ON l.receipt_id = r.receipt_id AND l.source = 'audit' AND l.blind = 1
 WHERE r.question_id = ? AND r.question_version = ? AND r.fact_ids_json IS NOT NULL
 ORDER BY r.receipt_id
"""


def _pairs(conn: sqlite3.Connection) -> list[tuple[str, list[float], int]]:
    """(receipt id, features, 1 if replaces) per unanimously labelled pair, once per pair."""
    labels: dict[str, set[str]] = {}
    facts: dict[str, tuple[str, str]] = {}
    for rid, fids, label in conn.execute(_LABELLED, (SLOT_RELATION.question_id,
                                                     SLOT_RELATION.version)):
        ids = tuple(json.loads(fids or "[]"))
        if len(ids) != 2:
            continue
        labels.setdefault("|".join(ids), set()).add(label)
        facts.setdefault("|".join(ids), (rid, ids))
    out = []
    for key, got in sorted(labels.items()):
        if len(got) != 1 or CANNOT_TELL in got:
            continue  # labellers disagreed, or could not tell: not training data
        rid, (older, newer) = facts[key]
        rows = dict(conn.execute("SELECT fact_id, content FROM atomic_facts WHERE fact_id IN "
                                 "(?, ?)", (older, newer)).fetchall())
        if older not in rows or newer not in rows:
            continue  # erased since: its label goes with it
        out.append((rid, features(rows[older] or "", rows[newer] or ""),
                    1 if got == {RETIRING} else 0))
    return out


def train(conn: sqlite3.Connection) -> Model | None:
    """Fit on this store's audited pairs, or None while either class is short."""
    try:
        data = _pairs(conn)
    except sqlite3.OperationalError:
        return None
    pos = sum(y for _, _, y in data)
    neg = len(data) - pos
    if pos < MIN_PER_CLASS or neg < MIN_PER_CLASS:
        return None
    k = len(FEATURES)
    means = [sum(x[i] for _, x, _ in data) / len(data) for i in range(k)]
    scales = [math.sqrt(sum((x[i] - means[i]) ** 2 for _, x, _ in data) / len(data)) or 1.0
              for i in range(k)]
    xs = [[(x[i] - means[i]) / scales[i] for i in range(k)] for _, x, _ in data]
    ys = [y for _, _, y in data]
    w, b = [0.0] * k, 0.0
    for _ in range(STEPS):
        gw, gb = [L2 * wi for wi in w], 0.0
        for x, y in zip(xs, ys):
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, b + sum(a * c for a, c in zip(w, x))))))
            for i in range(k):
                gw[i] += (p - y) * x[i] / len(xs)
            gb += (p - y) / len(xs)
        w = [wi - RATE * g for wi, g in zip(w, gw)]
        b -= RATE * gb
    digest = hashlib.sha256("\n".join(r for r, _, _ in data).encode()).hexdigest()[:10]
    return Model(f"tier05-lr-{digest}", tuple(w), b, tuple(means), tuple(scales), pos, neg)


class RelationScorerTier:
    """The ladder's shadow tier: answers SLOT_RELATION "replaces" with a score."""

    name = "tier05"
    per_item = True  # a function of the pair alone (features of its own texts)
    shadow_only = True

    def __init__(self, model: Model | None) -> None:
        self._model = model

    def available(self, question: Question) -> bool:
        return self._model is not None and question.question_id == SLOT_RELATION.question_id

    def answer(self, question: Question, items: Sequence[Item]) -> list[TierAnswer | None] | None:
        if not self.available(question):
            return None
        m = self._model
        return [TierAnswer(RETIRING, m.probability(features(it.texts.get("older", ""),
                                                           it.texts.get("newer", ""))),
                           model=m.name, calibration_id=f"{m.name}:uncalibrated")
                for it in items]


__all__ = ["FEATURES", "MIN_PER_CLASS", "Model", "RelationScorerTier", "features", "train"]
