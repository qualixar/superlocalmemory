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
standard error and the anytime-valid multiplier (``effects.cs_multiplier``),
computed under three attention curves: the estimate and the flattest and
steepest ends of its joint interval (``position_bias``). The candidate is
certified only if all three lower bounds are above 0. With no measured curve
nothing is certified: without it the gate would be guessing.

**Anytime, truly (A-05a).** The boundary is tuned once, at the fixed planned size
``GATE_PLANNED_CLUSTERS`` sessions, whatever the current size. (It was tuned at the
current size on every look, which makes the multiplier the constant 3.279 for every
sample size: a fixed z-threshold read repeatedly, not an anytime sequence; the null
simulation certified 4.05% against a one-sided 1.25%.) A test also needs
``GATE_MIN_CLUSTERS`` sessions before a bound is computed.

**Many candidates on one holdout (A-05b).** Every retrain produces a new candidate
tested on the same hash-fixed holdout, so error must be spent across candidates.
Chosen: alpha spending, not a fresh holdout slice per candidate (a slice would cost
a fifth of the labelled queries again for each candidate, and the labels are the
scarce thing). Candidate ``j`` (1, 2, ...) is tested at ``GATE_ALPHA / (j (j + 1))``:
these sum to at most ``GATE_ALPHA`` however many candidates are tried (the series
telescopes to ``GATE_ALPHA (1 - 1 / (J + 1))``), so the chance that any candidate
ever tested is certified while no better than the served order stays within
``GATE_ALPHA`` (a union bound; the three curve bounds are an intersection and cost
nothing more). The price is that later candidates face a stricter bound: the 10th
is tested at ``GATE_ALPHA / 110``. A candidate counts once a bound has been
computed for it (``Verdict.tested``); the count is kept in a small file next to
learning.db (``next_index`` / ``record_test``), and an unreadable file fails
closed (nothing is certified).

The certificate admits the candidate to the live sequential test; it is not a
promotion. The live test (mixture SPRT, rollback) is the unconditional check.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from superlocalmemory.learning.position_bias import MODEL_ALPHA, PositionModel
from superlocalmemory.outcomes import effects
from superlocalmemory.outcomes.swaps import SWAP_DEPTH

#: Policy: the half of the overall 0.05 that is the gate's own (the other half
#: is the attention curve's, ``position_bias.MODEL_ALPHA``).
GATE_ALPHA = effects.ALPHA - MODEL_ALPHA
CURVES = ("estimate", "low", "high")
#: The boundary's tuning size, fixed (routing: where the bound is sharpest; validity
#: holds for any value, as long as it is not a function of the data). A-05a.
GATE_PLANNED_CLUSTERS = 200
#: Sessions needed before a bound is computed. The standard error is a plug-in
#: estimate over session totals of skewed gains; with fewer than this the
#: normal approximation behind the boundary is not trusted. Set by the null
#: simulation in ``tests/test_learning/test_offline_gate_anytime.py``.
GATE_MIN_CLUSTERS = 30
LEDGER_FILE = "offline_gate_ledger.json"


def candidate_alpha(index: int) -> float:
    """Level for the ``index``-th candidate tested on this holdout (A-05b):
    GATE_ALPHA / (j (j + 1)), which sums to at most GATE_ALPHA over all candidates."""
    if index < 1:
        raise ValueError("candidate index starts at 1")
    return GATE_ALPHA / (index * (index + 1))


@dataclass(frozen=True)
class Verdict:
    certified: bool
    reason: str
    queries: int = 0
    clusters: int = 0
    gain: float | None = None              # mean delta under the estimated curve
    lower: dict | None = None              # curve -> anytime lower bound
    candidate_index: int = 1               # j: which candidate on this holdout (A-05b)
    alpha: float | None = None             # the level it was tested at, candidate_alpha(j)
    tested: bool = False                   # a bound was computed: this candidate spent its alpha

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


def boundary_multiplier(clusters: int, alpha: float) -> float:
    """Width of the lower bound in standard errors after ``clusters`` sessions: the
    mixture boundary tuned at the FIXED ``GATE_PLANNED_CLUSTERS`` (A-05a)."""
    return effects.cs_multiplier(clusters, GATE_PLANNED_CLUSTERS, alpha)


def _bound(deltas: list[float], clusters: list[str],
           alpha: float) -> tuple[float, float | None, int]:
    """(mean, anytime lower bound, number of clusters), boundary tuned at the fixed
    ``GATE_PLANNED_CLUSTERS``; no bound below ``GATE_MIN_CLUSTERS`` sessions."""
    n = len(deltas)
    mean = sum(deltas) / n
    totals: dict[str, float] = defaultdict(float)
    for d, c in zip(deltas, clusters, strict=True):
        totals[c] += d - mean
    s = len(totals)
    if s < GATE_MIN_CLUSTERS:
        return mean, None, s
    se = math.sqrt(sum(t * t for t in totals.values()) * s / (s - 1)) / n
    return mean, mean - boundary_multiplier(s, alpha) * se, s


def evaluate(score, rows: Sequence[dict], model: PositionModel, *,
             candidate_index: int = 1) -> Verdict:
    """Certify or not. ``score(rows) -> list[float]`` is the candidate's scorer;
    ``candidate_index`` is its place among the candidates tested on this holdout (A-05b)."""
    alpha = candidate_alpha(candidate_index)
    rows = _ledger(rows)
    if not model.measured:
        return Verdict(False, "position attention is not measured yet (needs logged swaps)",
                       candidate_index=candidate_index, alpha=alpha)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["query_id"])].append(r)
    if len(groups) < 2:
        return Verdict(False, "too few held-out deliveries", queries=len(groups),
                       candidate_index=candidate_index, alpha=alpha)
    flat = [r for g in groups.values() for r in g]
    scores = {id(r): float(s) for r, s in zip(flat, score(flat), strict=True)}
    clusters = [str(g[0].get("session_id") or f"query:{q}") for q, g in groups.items()]
    lower, gain, s = {}, None, 0
    for curve in CURVES:
        mean, lb, s = _bound(_deltas(groups, scores, model.attention(curve)), clusters, alpha)
        gain = mean if curve == "estimate" else gain
        lower[curve] = lb
    if any(v is None for v in lower.values()):
        return Verdict(False, f"too few independent sessions (needs {GATE_MIN_CLUSTERS})",
                       len(groups), s, gain, lower, candidate_index, alpha, False)
    ok = all(v > 0 for v in lower.values())
    return Verdict(ok, "certified: more uses expected than the served order"
                   if ok else "not better than the served order with confidence",
                   len(groups), s, gain, lower, candidate_index, alpha, True)


# -- the candidate counter (A-05b) -----------------------------------------------------------------

def _counts(folder: Path) -> dict[str, int]:
    try:
        raw = (Path(folder) / LEDGER_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    data = json.loads(raw)
    return {str(k): int(v) for k, v in data.items()}


def next_index(folder: Path, profile_id: str) -> int:
    """The index of the next candidate to be tested for this profile. Raises on an
    unreadable counter: the caller must then certify nothing."""
    return _counts(folder).get(profile_id, 0) + 1


def record_test(folder: Path, profile_id: str, index: int) -> None:
    """Record that candidate ``index`` has been tested. Atomic (write, then rename)."""
    counts = _counts(folder)
    counts[profile_id] = max(counts.get(profile_id, 0), int(index))
    path = Path(folder) / LEDGER_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(counts, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


__all__ = ["CURVES", "GATE_ALPHA", "GATE_MIN_CLUSTERS", "GATE_PLANNED_CLUSTERS", "LEDGER_FILE",
           "Verdict", "boundary_multiplier", "candidate_alpha", "evaluate", "next_index", "record_test"]
