# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The harm gate, in shadow: which memories make the agent worse? (WS-4, D-15, M-14)

A memory can hurt: out of date, beside the point, or pulling the agent off
course. The randomised withholding of WS-3 (``one-out-v2``) leaves one
eligible memory out of a small logged share of session-start deliveries, so
for each memory there are deliveries with it and without it, each with its
logged inclusion probability. That is enough to ask, per memory, whether
including it changes what happens next.

**Estimand, per fact f and outcome Y** (the M-12 outcomes, 24 h window):
``Δ_f = E[Y | f included] − E[Y | f left out]`` over the deliveries where f was
eligible. It is a **two-arm** contrast: a fact is tested only when it has
deliveries in *both* arms, at least ``MIN_ARM_CLUSTERS`` sessions in each
(A-01: with ε/k left out, most facts have never been left out, and a one-armed
sum read "never left out" as "re-asked in half the sessions" and flagged them
all). The estimate is the Hájek (self-normalised) form of inverse-propensity
weighting, which stays inside [0, 1] for a 0/1 outcome,

    Δ̂ = Σ (I/p)·Y / Σ (I/p)  −  Σ ((1−I)/(1−p))·Y / Σ ((1−I)/(1−p)),

with the cluster-robust (session) linearised standard error, floored by the
bounded-outcome standard error of ``effects.variance_floor`` over the arms'
effective session counts (the plug-in alone is far too small when a few units
agree). The uplift of including f is ``+Δ̂`` for ``used`` (the agent acted on a
delivered memory) and ``−Δ̂`` for ``reask`` (the agent had to ask memory again).

**Decision rule (D-15).** f is *harmful* when the upper bound of its uplift
is below 0 for an outcome, at one-sided level ``DELTA_H``, after
false-discovery control across every (fact, outcome) tested:

* per test, the anytime-valid boundary of M-12 (Robbins normal mixture,
  ``effects.anytime_half_width``), so the gate may be read at any time; its
  one-sided p-value is the smallest level whose bound excludes 0;
* across tests, Benjamini–Yekutieli at ``DELTA_H``, valid under any
  dependence (the tests share deliveries, so BH's independence or PRDS
  condition cannot be assumed).

Guarantee, checked by the null simulation in ``tests/test_core/test_harm_gate_null.py``:
when including a fact changes nothing, the share of repeated reads at which it
is flagged stays at or under ``DELTA_H`` (the simulation states its tolerance).
The anytime boundary is asymptotic in the number of sessions; the floor and the
two-arm minimum are what keep it honest while the left-out arm is small.

**Shadow only in 4.2.30.** The gate computes on read and writes nothing,
sends nothing (D-27, I-15). Acting would be a memory change (I-10) and is not
built: when it is, a withheld fact must keep inclusion probability ≥ ε so the
evidence against it keeps arriving.

At the measured rate (F-47) a single fact is rarely left out often enough, so
expect "not testable yet" for a long time; the report says how many facts had
both arms, and prints no "would withhold" list while none did.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from superlocalmemory.outcomes import effects, usage
from superlocalmemory.outcomes import randomiser as rz

#: One-sided level of each harm test and the false-discovery level (D-25's δ).
DELTA_H = 0.05
#: Routing only: the cluster count the anytime boundary is tuned at. It moves
#: power between early and late looks; validity holds for any value.
TUNED_CLUSTERS = 100
#: A fact is tested only with this many sessions in EACH arm (included, left out): the
#: same minimum as an effect claim (``effects.MIN_ARM_CLUSTERS``, A-04).
MIN_ARM_CLUSTERS = effects.MIN_ARM_CLUSTERS
OUTCOMES = (effects.USED, effects.REASK)
#: Routing only: the largest one-sided level the p-value search reports.
P_CAP = 0.25

_UNITS = """
SELECT r.fact_id, r.included, r.propensity, d.session_id, d.delivery_id, {outcome} AS y
  FROM injection_receipts r
  JOIN injection_deliveries d ON d.delivery_id = r.delivery_id
 WHERE d.profile_id = ? AND d.path IN ({paths}) AND r.eligible = 1
   AND r.propensity < 1
   AND julianday(d.delivered_at) + ? / 24.0 <= julianday(?)
   AND {evidence}
"""


@dataclass(frozen=True)
class HarmTest:
    fact_id: str
    outcome: str
    deliveries: int
    clusters: int
    uplift: float | None
    upper: float | None          # anytime upper bound at one-sided DELTA_H (uncorrected)
    p_value: float               # anytime-valid, one-sided, for "uplift < 0"
    harmful: bool = False        # after Benjamini–Yekutieli across all tests
    clusters_included: int = 0   # sessions with the fact in
    clusters_left_out: int = 0   # sessions with the fact left out


def _units(conn: sqlite3.Connection, profile_id: str, outcome: str,
           as_of: datetime) -> dict[str, list[tuple[int, float, str, float]]]:
    expr, params = effects._outcome_sql(outcome)  # noqa: SLF001 - the one outcome definition
    paths = sorted(rz.RANDOMISED_PATHS)
    clause, pulls = effects.evidence_clause()  # W0-17b: pilot deliveries are not evidence
    sql = _UNITS.format(outcome=expr, paths=", ".join("?" * len(paths)), evidence=clause)
    out: dict[str, list[tuple[int, float, str, float]]] = defaultdict(list)
    try:
        rows = conn.execute(sql, [*params, profile_id, *paths, effects.OUTCOME_WINDOW_H,
                                  usage.stamp(as_of), *pulls])
    except sqlite3.OperationalError:
        return {}
    for fact, included, p, session, did, y in rows:
        if outcome == effects.REASK and not session:
            continue
        p = float(p)
        if not 0 < p < 1:
            continue
        out[fact].append((int(included), p, session or f"delivery:{did}", float(y)))
    return out


def _p_value(uplift: float, se: float, n_eff: float) -> float:
    """Smallest one-sided level whose anytime upper bound is below 0."""
    if uplift >= 0 or se <= 0:
        return 1.0 if uplift >= 0 else 0.0

    def excludes(a: float) -> bool:
        return uplift + effects.cs_multiplier(n_eff, TUNED_CLUSTERS, 2 * a) * se < 0

    # The two-sided boundary is defined below level 1; levels past 0.25 are
    # reported as 1 (no test here runs at a level anywhere near that).
    if not excludes(P_CAP):
        return 1.0
    lo, hi = 1e-12, P_CAP
    if excludes(lo):
        return lo
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if excludes(mid) else (mid, hi)
    return hi


def _kish(weights_by_cluster: dict[str, float]) -> float:
    """Effective number of clusters when clusters carry unequal total weight."""
    total = sum(weights_by_cluster.values())
    squares = sum(w * w for w in weights_by_cluster.values())
    return total * total / squares if squares > 0 else 0.0


def contrast(units: list[tuple[int, float, str, float]]) -> tuple[float, float, float, float] | None:
    """(Δ̂, plug-in SE, effective sessions included, effective sessions left out), or None.

    Hájek two-arm contrast with its session-clustered linearised standard error.
    None when either arm is empty. No minimum is applied here (``check_fact`` does)."""
    a = [i / p for i, p, _s, _y in units]
    b = [(1 - i) / (1 - p) for i, p, _s, _y in units]
    big_a, big_b = sum(a), sum(b)
    if big_a <= 0 or big_b <= 0:
        return None
    mu1 = sum(w * y for w, (_i, _p, _s, y) in zip(a, units, strict=True)) / big_a
    mu0 = sum(w * y for w, (_i, _p, _s, y) in zip(b, units, strict=True)) / big_b
    totals: dict[str, float] = defaultdict(float)
    w_in: dict[str, float] = defaultdict(float)
    w_out: dict[str, float] = defaultdict(float)
    for wa, wb, (_i, _p, session, y) in zip(a, b, units, strict=True):
        totals[session] += wa * (y - mu1) / big_a - wb * (y - mu0) / big_b
        if wa:
            w_in[session] += wa
        if wb:
            w_out[session] += wb
    s = len(totals)
    var = sum(t * t for t in totals.values()) * s / (s - 1) if s > 1 else math.inf
    return mu1 - mu0, math.sqrt(var), _kish(w_in), _kish(w_out)


def check_fact(fact_id: str, outcome: str, units: list[tuple[int, float, str, float]]) -> HarmTest:
    """One fact's test. Untestable (``upper`` None) unless both arms have
    ``MIN_ARM_CLUSTERS`` sessions (A-01)."""
    sign = 1.0 if outcome == effects.USED else -1.0
    sessions = {s for _i, _p, s, _y in units}
    s_in = len({s for i, _p, s, _y in units if i})
    s_out = len({s for i, _p, s, _y in units if not i})
    base = (fact_id, outcome, len(units), len(sessions))
    if min(s_in, s_out) < MIN_ARM_CLUSTERS:
        return HarmTest(*base, None, None, 1.0, False, s_in, s_out)
    got = contrast(units)
    if got is None:
        return HarmTest(*base, None, None, 1.0, False, s_in, s_out)
    delta, se_plugin, k_in, k_out = got
    uplift = sign * delta
    se = max(se_plugin, effects.variance_floor(k_in, k_out))
    n_eff = effects.VARIANCE_BOUND / (se * se)
    upper = uplift + effects.cs_multiplier(n_eff, TUNED_CLUSTERS, 2 * DELTA_H) * se
    return HarmTest(*base, uplift, upper, _p_value(uplift, se, n_eff), False, s_in, s_out)


def benjamini_yekutieli(p_values: list[float], level: float = DELTA_H) -> list[bool]:
    """Rejections at false-discovery ``level`` under arbitrary dependence."""
    m = len(p_values)
    if not m:
        return []
    harmonic = sum(1.0 / i for i in range(1, m + 1))
    order = sorted(range(m), key=lambda i: p_values[i])
    k = 0
    for rank, i in enumerate(order, 1):
        if p_values[i] <= rank * level / (m * harmonic):
            k = rank
    rejected = set(order[:k])
    return [i in rejected for i in range(m)]


def evaluate(conn: sqlite3.Connection, profile_id: str, *,
             as_of: datetime | None = None) -> list[HarmTest]:
    """Every (fact, outcome) with enough sessions to test, BY-corrected."""
    when = as_of or datetime.now(UTC)
    tests = [check_fact(fact, outcome, units)
             for outcome in OUTCOMES
             for fact, units in sorted(_units(conn, profile_id, outcome, when).items())]
    tested = [t for t in tests if t.upper is not None]
    flags = benjamini_yekutieli([t.p_value for t in tested])
    marked = {(t.fact_id, t.outcome): f for t, f in zip(tested, flags, strict=True)}
    return [HarmTest(**{**asdict(t), "harmful": marked.get((t.fact_id, t.outcome), False)})
            for t in tests]


def report(conn: sqlite3.Connection, profile_id: str, *, as_of: datetime | None = None,
           state: str = "off") -> dict:
    """The shadow gate for the Memory Health report. Writes nothing.

    ``facts_both_arms`` counts the facts that had deliveries with and without them in at
    least ``min_arm_clusters`` sessions each: the only ones that can be tested (A-01)."""
    tests = evaluate(conn, profile_id, as_of=as_of)
    tested = [t for t in tests if t.upper is not None]
    harmful = sorted({t.fact_id for t in tested if t.harmful})
    left_out = {t.fact_id for t in tests if t.clusters_left_out}
    return {"state": state, "acting": False, "delta": DELTA_H,
            "facts_seen": len({t.fact_id for t in tests}),
            "facts_ever_left_out": len(left_out),
            "facts_both_arms": len({t.fact_id for t in tested}),
            "min_arm_clusters": MIN_ARM_CLUSTERS,
            "tests": len(tested), "would_withhold": harmful,
            "method": "two-arm Hajek IPW per fact over randomised withholding, session-clustered "
                      "with a bounded-outcome floor, anytime-valid (M-12 boundary), "
                      "Benjamini-Yekutieli across facts and outcomes"}


def describe(section: dict) -> list[str]:
    if not section:
        return []
    seen = section["facts_seen"]
    both = section.get("facts_both_arms", 0)
    need = section.get("min_arm_clusters", MIN_ARM_CLUSTERS)
    head = (f"Harm gate (shadow, never acts in 4.2.30): {seen} memories seen in randomised "
            f"deliveries; {both} of them with both arms (shown and left out in at least {need} "
            f"sessions each), {section['tests']} tests")
    if not section["tests"]:
        return [head + ". Not testable yet: no memory has been left out often enough to say "
                "whether it helps or hurts."]
    would = section.get("would_withhold") or []
    tail = (f"; would withhold {len(would)}: {', '.join(would[:5])}" if would
            else "; none shown to make the agent worse")
    return [head + tail + "."]


__all__ = ["DELTA_H", "MIN_ARM_CLUSTERS", "OUTCOMES", "TUNED_CLUSTERS", "HarmTest",
           "benjamini_yekutieli", "check_fact", "contrast", "describe", "evaluate", "report"]
