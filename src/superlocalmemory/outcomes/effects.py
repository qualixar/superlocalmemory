# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Does delivered memory help? IPW and doubly robust estimates with intervals (W3-4, M-12).

Everything here reads only the injection ledger (M075) and is design-based:
the randomiser logged each delivery's exact assignment probabilities (D-14,
D-33, D-44), so the inverse-propensity estimate is unbiased with no model at
all, and the doubly robust one stays unbiased whatever its outcome model.

**Estimand (W0-17).** Per randomised path (``session_init``, ``session_context``)
and outcome, the average effect on a delivery of

* ``memory``: leaving out one eligible memory drawn uniformly (the one-out
  arm), among deliveries with at least one eligible candidate. Reported as
  E[Y | all included] - E[Y | one left out], which equals the mean, over a
  delivery's eligible candidates, of the marginal effect of including that one
  candidate (D-14's estimand);
* ``channel`` / ``channel:<name>``: switching off any one / that one retrieval
  channel for the fetch (``session_init`` only), as E[Y | full fetch] -
  E[Y | channel off]: the channel's marginal value.

A contrast compares two arms of the same randomisation; deliveries in the other
arms are left out and the two arms' probabilities are taken conditional on
being in one of them, which keeps assignment random and the weights exact.

**Outcomes and window.** Each is observed within ``OUTCOME_WINDOW_H`` hours of
the delivery, and a delivery is scored only once its window has closed
(an open window would count missing signals as zeros):

* ``used`` (higher is better): a memory of this delivery was acted on
  (marker echoed in an edit or a tool call, cited, or named in an explicit
  success report). Overlap scores are not used: they have no calibrated
  threshold yet (W3-2b);
* ``reask`` (lower is better): the agent pulled memory again (an explicit
  recall from MCP, CLI, HTTP, hook or view) in the same conversation. Needs a
  session id, so deliveries without one are left out of this outcome.

**Intervals.** Deliveries of one conversation share its outcomes, so variance
is clustered by session (a delivery without a session is its own cluster).
Two intervals are reported at ``ALPHA``: a fixed-sample one (valid at one
planned look) and an anytime-valid confidence sequence (valid however often
the report is read): Robbins' two-sided normal mixture, applied to the cluster
contributions as in Waudby-Smith et al.'s asymptotic confidence sequences.
**A claim (helps / hurts) is made only when the anytime interval excludes 0.**

**Power line (D-14).** ``planned_deliveries`` gives the deliveries a contrast
needs before an effect of ``PLANNED_MDE`` is found with probability ``POWER``
by the anytime interval, under the worst-case variance of a 0/1 outcome. The
confidence sequence is tuned to be tightest at that size; any tuning keeps its
validity, so the constants below set only where it is sharpest (routing, not a
threshold on data).
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from superlocalmemory.outcomes import randomiser as rz
from superlocalmemory.outcomes import usage

#: Outcome window: signals later than this after a delivery are not its outcome.
OUTCOME_WINDOW_H = 24.0
#: Two-sided error level of every interval.
ALPHA = 0.05
#: Effect size (absolute, on a 0/1 outcome) the power line plans for.
PLANNED_MDE = 0.10
#: Power the plan aims at.
POWER = 0.80
#: Largest variance of a 0/1 outcome, p(1 - p) <= 1/4: the plan assumes the worst.
VARIANCE_BOUND = 0.25

USED = "used"
REASK = "reask"
#: Each outcome and the sign that means "memory helps".
OUTCOMES = {USED: +1, REASK: -1}
USED_SIGNALS = (usage.MARKER_EDIT, usage.MARKER_TOOL, "cite", usage.EXPLICIT + "success")

MEMORY = "memory"
CHANNEL = "channel"
CONTROL = "control"


@dataclass(frozen=True)
class Unit:
    """One randomised delivery, its assignment, its probabilities and outcome."""

    delivery_id: str
    cluster: str
    path: str
    arm: str                 # control, memory, or channel:<name>
    p_control: float
    p_memory: float
    p_channel_each: float
    eligible_count: int
    candidates: int
    y: float


@dataclass(frozen=True)
class Estimate:
    path: str
    contrast: str
    outcome: str
    n: int
    clusters: int
    n_treated: int
    n_control: int
    ipw: float | None
    ipw_se: float | None
    aipw: float | None
    aipw_se: float | None
    interval: tuple[float, float] | None
    anytime: tuple[float, float] | None
    planned: int
    claim: str | None


# -- the power line -------------------------------------------------------------------------------

def _z(q: float) -> float:
    """Standard normal quantile."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(q)


def mixture_x(alpha: float = ALPHA) -> float:
    """rho^2 * n at which the normal-mixture boundary is tightest relative to sqrt(n)."""
    return -2.0 * math.log(alpha) + math.log(-2.0 * math.log(alpha) + 1.0)


def cs_multiplier(clusters: int, planned: int, alpha: float = ALPHA) -> float:
    """Width of the anytime interval in standard errors after ``clusters`` units,
    for a boundary tuned at ``planned`` units (Robbins' two-sided normal mixture:
    |S_t| < sqrt(2 (1 + rho^2 V_t) / rho^2 * log(sqrt(1 + rho^2 V_t) / alpha))
    for every t with probability >= 1 - alpha, V_t = t sigma^2)."""
    if clusters < 1:
        return math.inf
    rho2 = mixture_x(alpha) / max(planned, 1)
    v = clusters * rho2
    return math.sqrt(2.0 * (1.0 + v) / v * math.log(math.sqrt(1.0 + v) / alpha))


def contrast_variance(p_treated: float, p_control: float,
                      variance: float = VARIANCE_BOUND) -> float:
    """Per-delivery variance of the doubly robust contrast for a 0/1 outcome
    of variance ``variance``, arms drawn with these (conditional) probabilities."""
    return variance * (1.0 / p_treated + 1.0 / p_control)


def planned_deliveries(p_treated: float, p_control: float, *, mde: float = PLANNED_MDE,
                       alpha: float = ALPHA, power: float = POWER,
                       variance: float = VARIANCE_BOUND, design_effect: float = 1.0,
                       anytime: bool = True) -> int:
    """Deliveries needed to find an effect of ``mde`` with probability ``power``.

    Fixed sample: n = sigma^2 (z_{1-alpha/2} + z_power)^2 / mde^2. Anytime: the
    boundary tuned at n has width g* sigma / sqrt(n) with g* = cs_multiplier at
    its own tuning point (a constant, 3.04 at alpha 0.05), so n = sigma^2
    (g* + z_power)^2 / mde^2: about 1.9 times the fixed-sample size."""
    sigma2 = contrast_variance(p_treated, p_control, variance) * design_effect
    first = cs_multiplier(1, 1, alpha) if anytime else _z(1.0 - alpha / 2.0)
    return math.ceil(sigma2 * (first + _z(power)) ** 2 / mde ** 2)


def design_probabilities(path: str, contrast: str,
                         epsilon: float = rz.EPSILON) -> tuple[float, float]:
    """(P(treated), P(control)) of a contrast under the nominal design, each
    conditional on the delivery being in one of the two arms."""
    arms = rz.ARMS[path]
    share = arms.count(rz.MEMORY_ARM) / len(arms)
    p_control = 1.0 - epsilon
    if contrast == MEMORY:
        p_t = epsilon * share
    elif contrast == CHANNEL:
        p_t = epsilon * (1.0 - share)
    else:
        p_t = epsilon * (1.0 - share) / len(rz.DROPPABLE_CHANNELS)
    if p_t <= 0:
        raise ValueError(f"{path!r} has no {contrast!r} arm")
    return p_t / (p_t + p_control), p_control / (p_t + p_control)


def planned_for(path: str, contrast: str) -> int:
    return planned_deliveries(*design_probabilities(path, contrast))


# -- reading the ledger ---------------------------------------------------------------------------

_UNITS = """
SELECT d.delivery_id, d.session_id, d.path, d.epsilon, d.eligible_count, d.candidates,
       d.withhold_draw, d.arm, d.dropped_channel, d.memory_share,
       {outcome} AS y
  FROM injection_deliveries d
 WHERE d.profile_id = ? AND d.path IN ({paths}) AND d.withhold_draw IS NOT NULL
   AND julianday(d.delivered_at) + ? / 24.0 <= julianday(?)
"""

_USED = """EXISTS (SELECT 1 FROM injection_usage u
 WHERE u.delivery_id = d.delivery_id AND u.signal IN ({signals})
   AND julianday(u.observed_at) <= julianday(d.delivered_at) + ? / 24.0)"""

_REASK = """EXISTS (SELECT 1 FROM injection_deliveries p
 WHERE p.profile_id = d.profile_id AND p.session_id = d.session_id
   AND p.path IN ({pulls})
   AND julianday(p.delivered_at) > julianday(d.delivered_at)
   AND julianday(p.delivered_at) <= julianday(d.delivered_at) + ? / 24.0)"""


def _marks(n: int) -> str:
    return ", ".join("?" * n)


def _outcome_sql(outcome: str) -> tuple[str, list]:
    if outcome == USED:
        return _USED.format(signals=_marks(len(USED_SIGNALS))), [*USED_SIGNALS,
                                                                  OUTCOME_WINDOW_H]
    if outcome == REASK:
        pulls = sorted(rz.ASKED_PATHS)
        return _REASK.format(pulls=_marks(len(pulls))), [*pulls, OUTCOME_WINDOW_H]
    raise ValueError(f"unknown outcome: {outcome!r}")


def _arm(withhold_draw: float, epsilon: float, arm: str | None, channel: str | None) -> str:
    if withhold_draw >= epsilon or arm is None:
        return CONTROL
    if arm == rz.CHANNEL_ARM and channel:
        return f"{CHANNEL}:{channel}"
    return MEMORY


def load_units(conn: sqlite3.Connection, profile_id: str, outcome: str, *,
               as_of: datetime | None = None) -> list[Unit]:
    """Every randomised delivery whose outcome window has closed, with its outcome."""
    expr, params = _outcome_sql(outcome)
    paths = sorted(rz.RANDOMISED_PATHS)
    sql = _UNITS.format(outcome=expr, paths=_marks(len(paths)))
    when = usage.stamp(as_of or datetime.now(UTC))
    try:
        rows = conn.execute(sql, [*params, profile_id, *paths, OUTCOME_WINDOW_H, when])
    except sqlite3.OperationalError:
        return []
    units = []
    for (did, session, path, eps, k, cands, u, arm, channel, share, y) in rows:
        if outcome == REASK and not session:
            continue
        share = 1.0 if share is None else float(share)
        units.append(Unit(
            did, session or f"delivery:{did}", path, _arm(u, eps, arm, channel),
            1.0 - eps, eps * share, eps * (1.0 - share) / len(rz.DROPPABLE_CHANNELS),
            int(k), int(cands), float(y)))
    return units


# -- the estimators -------------------------------------------------------------------------------

def _in_contrast(unit: Unit, contrast: str) -> str | None:
    """``treated``, ``control`` or None (outside this contrast)."""
    if unit.arm == CONTROL:
        return CONTROL if contrast != MEMORY or unit.eligible_count >= 1 else None
    if contrast == MEMORY:
        return "treated" if unit.arm == MEMORY and unit.eligible_count >= 1 else None
    if contrast == CHANNEL:
        return "treated" if unit.arm.startswith(CHANNEL + ":") else None
    return "treated" if unit.arm == contrast else None


def _p_treated(unit: Unit, contrast: str) -> float:
    if contrast == MEMORY:
        return unit.p_memory
    if contrast == CHANNEL:
        return unit.p_channel_each * len(rz.DROPPABLE_CHANNELS)
    return unit.p_channel_each


def _covariates(unit: Unit, contrast: str) -> list[float]:
    """Pre-treatment covariates only: a channel drop changes the fetch, so the
    candidate counts are outcomes of it and a channel contrast uses none."""
    if contrast == MEMORY:
        return [1.0, float(unit.eligible_count), float(unit.candidates)]
    return [1.0]


def _fold(cluster: str) -> int:
    return hashlib.sha256(cluster.encode()).digest()[0] & 1


def _fit(xs: list[list[float]], ys: list[float]) -> np.ndarray | None:
    """Least squares on the other fold; its mean when too few points; None if empty."""
    if not ys:
        return None
    x, y = np.asarray(xs), np.asarray(ys)
    if len(ys) <= x.shape[1]:
        coef = np.zeros(x.shape[1])
        coef[0] = float(y.mean())
        return coef
    return _solve_normal(x, y)


#: Ridge added to the normal equations' diagonal: keeps a rank-deficient design
#: solvable (routing only: far below any coefficient that matters).
_RIDGE = 1e-9


def _solve_normal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares by the normal equations, solved in Python with partial
    pivoting. A handful of covariates, so this is exact enough and keeps the
    estimate off native LAPACK (Accelerate has crashed this process on macOS;
    ``tests/test_math/test_no_native_lapack_in_source.py``)."""
    k = x.shape[1]
    a = [[float(v) for v in row] for row in (x.T @ x)]
    b = [float(v) for v in (x.T @ y)]
    for i in range(k):
        a[i][i] += _RIDGE
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        if abs(a[col][col]) < 1e-300:
            continue
        for r in range(col + 1, k):
            f = a[r][col] / a[col][col]
            for c in range(col, k):
                a[r][c] -= f * a[col][c]
            b[r] -= f * b[col]
    out = [0.0] * k
    for i in range(k - 1, -1, -1):
        if abs(a[i][i]) < 1e-300:
            continue
        out[i] = (b[i] - sum(a[i][j] * out[j] for j in range(i + 1, k))) / a[i][i]
    return np.asarray(out)


def _predict(coef: np.ndarray | None, x: list[float]) -> float:
    return 0.0 if coef is None else float(np.dot(coef, x))


def _cross_fit(rows: list[tuple[Unit, str, float, float]],
               contrast: str) -> list[tuple[float, float]]:
    """(m_treated(x), m_control(x)) for each row, fitted on the other fold."""
    models: dict[tuple[int, str], np.ndarray | None] = {}
    for fold in (0, 1):
        for side in ("treated", CONTROL):
            pts = [(_covariates(u, contrast), u.y) for u, s, _, _ in rows
                   if s == side and _fold(u.cluster) != fold]
            models[(fold, side)] = _fit([p[0] for p in pts], [p[1] for p in pts])
    out = []
    for u, _, _, _ in rows:
        f, x = _fold(u.cluster), _covariates(u, contrast)
        out.append((_predict(models[(f, "treated")], x), _predict(models[(f, CONTROL)], x)))
    return out


def _clustered(phi: Sequence[float], clusters: Sequence[str]) -> tuple[float, float | None]:
    """Mean of the per-delivery terms and its cluster-robust standard error."""
    n = len(phi)
    tau = sum(phi) / n
    totals: dict[str, float] = {}
    for value, cluster in zip(phi, clusters, strict=True):
        totals[cluster] = totals.get(cluster, 0.0) + value - tau
    s = len(totals)
    if s < 2:
        return tau, None
    var = sum(t * t for t in totals.values()) * s / (s - 1) / n ** 2
    return tau, math.sqrt(var)


def _rows(units: Sequence[Unit], contrast: str) -> list[tuple[Unit, str, float, float]]:
    """Units in this contrast with their side and conditional probabilities."""
    out = []
    for u in units:
        side = _in_contrast(u, contrast)
        if side is None:
            continue
        p_t = _p_treated(u, contrast)
        total = p_t + u.p_control
        out.append((u, side, p_t / total, u.p_control / total))
    return out


def estimate(units: Sequence[Unit], path: str, contrast: str, outcome: str) -> Estimate:
    """IPW and doubly robust estimates of one contrast on one path, with intervals."""
    planned = planned_for(path, contrast)
    rows = [r for r in _rows([u for u in units if u.path == path], contrast)
            if r[2] > 0 and r[3] > 0]
    n_t = sum(1 for r in rows if r[1] == "treated")
    n_c = len(rows) - n_t
    clusters = [r[0].cluster for r in rows]
    if not n_t or not n_c:
        return Estimate(path, contrast, outcome, len(rows), len(set(clusters)), n_t, n_c,
                        None, None, None, None, None, None, planned, None)
    # The effect of including: E[Y | control] - E[Y | treated (left out / off)].
    ipw_terms = [(-u.y / e_t if s == "treated" else u.y / e_c) for u, s, e_t, e_c in rows]
    fits = _cross_fit(rows, contrast)
    dr_terms = [m_c - m_t + (-(u.y - m_t) / e_t if s == "treated" else (u.y - m_c) / e_c)
                for (u, s, e_t, e_c), (m_t, m_c) in zip(rows, fits, strict=True)]
    ipw, ipw_se = _clustered(ipw_terms, clusters)
    aipw, aipw_se = _clustered(dr_terms, clusters)
    s = len(set(clusters))
    interval = anytime = None
    claim = None
    if aipw_se is not None:
        z = _z(1.0 - ALPHA / 2.0)
        interval = (aipw - z * aipw_se, aipw + z * aipw_se)
        g = cs_multiplier(s, planned)
        anytime = (aipw - g * aipw_se, aipw + g * aipw_se)
        claim = _claim(anytime, OUTCOMES[outcome])
    return Estimate(path, contrast, outcome, len(rows), s, n_t, n_c, ipw, ipw_se, aipw,
                    aipw_se, interval, anytime, planned, claim)


def _claim(anytime: tuple[float, float], better: int) -> str | None:
    """Including (memory or channel) helps or hurts, only when 0 is outside the interval."""
    lo, hi = anytime
    if lo > 0:
        return "helps" if better > 0 else "hurts"
    if hi < 0:
        return "hurts" if better > 0 else "helps"
    return None


def contrasts(path: str) -> list[str]:
    if rz.CHANNEL_ARM not in rz.ARMS[path]:
        return [MEMORY]
    return [MEMORY, CHANNEL, *(f"{CHANNEL}:{c}" for c in rz.DROPPABLE_CHANNELS)]


def report(conn: sqlite3.Connection, profile_id: str, *,
           as_of: datetime | None = None) -> list[Estimate]:
    """Every contrast on every randomised path, for every outcome."""
    out = []
    for outcome in OUTCOMES:
        units = load_units(conn, profile_id, outcome, as_of=as_of)
        for path in sorted(rz.RANDOMISED_PATHS):
            out.extend(estimate(units, path, c, outcome) for c in contrasts(path))
    return out


__all__ = ["ALPHA", "CHANNEL", "CONTROL", "MEMORY", "OUTCOMES", "OUTCOME_WINDOW_H",
           "PLANNED_MDE", "POWER", "REASK", "USED", "USED_SIGNALS", "VARIANCE_BOUND",
           "Estimate", "Unit", "contrast_variance", "contrasts", "cs_multiplier",
           "design_probabilities", "estimate", "load_units", "mixture_x", "planned_deliveries",
           "planned_for", "report"]
