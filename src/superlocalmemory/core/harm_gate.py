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
``Δ_f = E[Y | f included] − E[Y | f left out]``, estimated by inverse
propensity weighting over the deliveries where f was eligible,

    φ_d = Y_d · (I_d / p_d − (1 − I_d) / (1 − p_d)),

clustered by session. The uplift of including f is ``+Δ`` for ``used`` (the
agent acted on a delivered memory) and ``−Δ`` for ``reask`` (the agent had to
ask memory again).

**Decision rule (D-15).** f is *harmful* when the upper bound of its uplift
is below 0 for an outcome, at one-sided level ``DELTA_H``, after
false-discovery control across every (fact, outcome) tested:

* per test, the anytime-valid boundary of M-12 (Robbins normal mixture,
  ``effects.cs_multiplier``), so the gate may be read at any time; its
  one-sided p-value is the smallest level whose bound excludes 0;
* across tests, Benjamini–Yekutieli at ``DELTA_H``, valid under any
  dependence (the tests share deliveries, so BH's independence or PRDS
  condition cannot be assumed).

**Shadow only in 4.2.30.** The gate computes on read and writes nothing,
sends nothing (D-27, I-15). Acting would be a memory change (I-10) and is not
built: when it is, a withheld fact must keep inclusion probability ≥ ε so the
evidence against it keeps arriving.

At the measured rate (F-47) a single fact gathers few randomised deliveries,
so expect "not enough data" for a long time; the report says how many facts
could be tested at all.
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
#: A test needs at least this many sessions for a clustered standard error.
MIN_CLUSTERS = 2
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


def _p_value(uplift: float, se: float, clusters: int) -> float:
    """Smallest one-sided level whose anytime upper bound is below 0."""
    if uplift >= 0 or se <= 0:
        return 1.0 if uplift >= 0 else 0.0

    def excludes(a: float) -> bool:
        return uplift + effects.cs_multiplier(clusters, TUNED_CLUSTERS, 2 * a) * se < 0

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


def check_fact(fact_id: str, outcome: str, units: list[tuple[int, float, str, float]]) -> HarmTest:
    sign = 1.0 if outcome == effects.USED else -1.0
    phi = [sign * y * (i / p - (1 - i) / (1 - p)) for i, p, _s, y in units]
    clusters = [s for _i, _p, s, _y in units]
    s = len(set(clusters))
    if not phi or s < MIN_CLUSTERS:
        return HarmTest(fact_id, outcome, len(phi), s, None, None, 1.0)
    tau, se = effects._clustered(phi, clusters)  # noqa: SLF001 - the same clustered SE as M-12
    if se is None:
        return HarmTest(fact_id, outcome, len(phi), s, tau, None, 1.0)
    upper = tau + effects.cs_multiplier(s, TUNED_CLUSTERS, 2 * DELTA_H) * se
    return HarmTest(fact_id, outcome, len(phi), s, tau, upper, _p_value(tau, se, s))


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
    """The shadow gate for the Memory Health report. Writes nothing."""
    tests = evaluate(conn, profile_id, as_of=as_of)
    tested = [t for t in tests if t.upper is not None]
    harmful = sorted({t.fact_id for t in tested if t.harmful})
    return {"state": state, "acting": False, "delta": DELTA_H,
            "facts_seen": len({t.fact_id for t in tests}),
            "tests": len(tested), "would_withhold": harmful,
            "method": "IPW per fact over randomised withholding, session-clustered, anytime-valid "
                      "(M-12 boundary), Benjamini-Yekutieli across facts and outcomes"}


def describe(section: dict) -> list[str]:
    if not section:
        return []
    head = (f"Harm gate (shadow, never acts in 4.2.30): {section['facts_seen']} memories "
            f"seen in randomised deliveries, {section['tests']} tests with enough sessions")
    would = section.get("would_withhold") or []
    tail = (f"; would withhold {len(would)}: {', '.join(would[:5])}" if would
            else "; none shown to make the agent worse")
    return [head + tail + "."]


__all__ = ["DELTA_H", "MIN_CLUSTERS", "OUTCOMES", "TUNED_CLUSTERS", "HarmTest",
           "benjamini_yekutieli", "describe", "evaluate", "report", "check_fact"]
