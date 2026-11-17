# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The offline gate a ranker candidate must pass before anyone is served it (D-45).

The question: would the candidate's order lead agents to act on more memories
than the order recall actually served? It is answered on logged deliveries the
candidate never trained on (the query-disjoint holdout), scored by what agents
used, with position attention measured from logged swaps (D-46), so the score
does not simply favour whatever was on top.

**Metric: expected uses in the top ``SWAP_DEPTH``.** Under the examination model
a delivered memory's relevance is estimated by its use divided by the attention
of the position it was shown at (inverse propensity). The candidate's gain on
one delivery is

    delta = sum over memories shown in the top five of
            used x (p(candidate position) / p(shown position) - 1),

with p = 0 below the top five. The served order scores exactly 0. A used
memory shown below the top five, which the candidate lifts into it, is credited
used x p(candidate position) / p(5): attention never rises going down the list,
so p(shown) <= p(5) and this under-states its relevance in expectation (the
gate gives partial credit, never too much).

**Bound.** The mean delta over held-out queries, with a session-clustered
standard error and the anytime-valid multiplier (``effects.cs_multiplier``) at
``GATE_ALPHA``, computed under three attention curves: the estimate and the
flattest and steepest ends of its joint interval (``position_bias``). The
candidate is certified only if all three lower bounds are above 0. With no
measured curve nothing is certified: without it the gate would be guessing.

The certificate admits the candidate to the live sequential test; it is not a
promotion. The live test (mixture SPRT, rollback) is the unconditional check.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from superlocalmemory.learning.position_bias import MODEL_ALPHA, PositionModel
from superlocalmemory.outcomes import effects
from superlocalmemory.outcomes.swaps import SWAP_DEPTH

#: Policy: the half of the overall 0.05 that is the gate's own (the other half
#: is the attention curve's, ``position_bias.MODEL_ALPHA``).
GATE_ALPHA = effects.ALPHA - MODEL_ALPHA
CURVES = ("estimate", "low", "high")


@dataclass(frozen=True)
class Verdict:
    certified: bool
    reason: str
    queries: int = 0
    clusters: int = 0
    gain: float | None = None              # mean delta under the estimated curve
    lower: dict | None = None              # curve -> anytime lower bound

    def as_dict(self) -> dict:
        return asdict(self)


def _ledger(rows: Sequence[dict]) -> list[dict]:
    return [r for r in rows if str(r.get("label_source", "")).startswith("ledger")
            and r.get("shown_position") is not None]


def _deltas(groups: dict[str, list[dict]], scores: dict[int, float],
            attention: list[float]) -> list[float]:
    out = []
    p = lambda k: attention[k - 1] if 1 <= k <= SWAP_DEPTH else 0.0  # noqa: E731
    for rows in groups.values():
        ranked = sorted(rows, key=lambda r: (-scores[id(r)], int(r["shown_position"])))
        new = {id(r): i + 1 for i, r in enumerate(ranked)}
        d = 0.0
        for r in rows:
            shown = int(r["shown_position"])
            if float(r["outcome_reward"]) <= 0:
                continue
            if shown <= SWAP_DEPTH:
                d += p(new[id(r)]) / p(shown) - 1.0
            else:  # a lower bound on its relevance: p(shown) <= p(SWAP_DEPTH)
                d += p(new[id(r)]) / p(SWAP_DEPTH)
        out.append(d)
    return out


def _bound(deltas: list[float], clusters: list[str]) -> tuple[float, float | None, int]:
    """(mean, anytime lower bound, number of clusters)."""
    n = len(deltas)
    mean = sum(deltas) / n
    totals: dict[str, float] = defaultdict(float)
    for d, c in zip(deltas, clusters, strict=True):
        totals[c] += d - mean
    s = len(totals)
    if s < 2:
        return mean, None, s
    se = math.sqrt(sum(t * t for t in totals.values()) * s / (s - 1)) / n
    return mean, mean - effects.cs_multiplier(s, s, GATE_ALPHA) * se, s


def evaluate(score, rows: Sequence[dict], model: PositionModel) -> Verdict:
    """Certify or not. ``score(rows) -> list[float]`` is the candidate's scorer."""
    rows = _ledger(rows)
    if not model.measured:
        return Verdict(False, "position attention is not measured yet (needs logged swaps)")
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["query_id"])].append(r)
    if len(groups) < 2:
        return Verdict(False, "too few held-out deliveries", queries=len(groups))
    flat = [r for g in groups.values() for r in g]
    scores = {id(r): float(s) for r, s in zip(flat, score(flat), strict=True)}
    clusters = [str(g[0].get("session_id") or f"query:{q}") for q, g in groups.items()]
    lower, gain, s = {}, None, 0
    for curve in CURVES:
        mean, lb, s = _bound(_deltas(groups, scores, model.attention(curve)), clusters)
        gain = mean if curve == "estimate" else gain
        lower[curve] = lb
    if any(v is None for v in lower.values()):
        return Verdict(False, "too few independent sessions", len(groups), s, gain, lower)
    ok = all(v > 0 for v in lower.values())
    return Verdict(ok, "certified: more uses expected than the served order"
                   if ok else "not better than the served order with confidence",
                   len(groups), s, gain, lower)


__all__ = ["CURVES", "GATE_ALPHA", "Verdict", "evaluate"]
