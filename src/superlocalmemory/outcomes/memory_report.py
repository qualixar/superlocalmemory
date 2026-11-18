# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Memory Health report: what SLM delivered, what was used, what it can claim (W3-5).

One implementation for ``slm health --memory``, ``GET /api/outcomes/health``,
the MCP ``health`` tool (``memory=true``) and the dashboard, so they cannot
disagree. Read-only, ids and numbers only (no memory text, no query text):

* **ledger**: whether receipts and withholding are on, and deliveries in the
  last ``RECENT_DAYS`` days by path;
* **effects** (M-12): every contrast with its estimate, both intervals, the
  sample it has and the sample its power line needs, and a claim only when
  the anytime interval excludes 0;
* **use** (M-13): the store's use rate, per kind, the most used and the
  confidently rarely used memories, each with a credible interval;
* **certificates** and **second opinions** (WS-2): which judge may act, and
  where a higher tier disagreed (R-26, R-30);
* **learning** (D-45): every learner's state (learning, stalled, off, waiting,
  not wired), so a stalled learner is never silent again.

Every section fails soft: a store without a table reports that section empty.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from superlocalmemory.outcomes import effects, utility

logger = logging.getLogger(__name__)

#: Descriptive: the window of the "recent deliveries" counts.
RECENT_DAYS = 30
#: How many memories each list shows.
LIST_SIZE = 10


def _ledger(conn: sqlite3.Connection, profile_id: str, as_of: datetime,
            slm_home: Path | None) -> dict:
    out: dict = {"receipts": None, "withholding": None, "recent_days": RECENT_DAYS,
                 "deliveries": {}}
    if slm_home is not None:
        try:
            from superlocalmemory.core.feature_flags import FeatureFlags

            flags = FeatureFlags(slm_home)
            out["receipts"] = flags.state("outcomes.receipts").value
            out["withholding"] = flags.state("outcomes.holdout").value
        except Exception:  # noqa: BLE001 - a flag file never breaks the report
            logger.debug("flags unreadable", exc_info=True)
    since = (as_of - timedelta(days=RECENT_DAYS)).isoformat(timespec="microseconds")
    try:
        rows = conn.execute(
            "SELECT path, COUNT(*) FROM injection_deliveries WHERE profile_id = ?"
            " AND julianday(delivered_at) >= julianday(?) GROUP BY path ORDER BY path",
            (profile_id, since)).fetchall()
        out["deliveries"] = {p: n for p, n in rows}
    except sqlite3.OperationalError:
        pass
    return out


def _effect(e: effects.Estimate) -> dict:
    d = asdict(e)
    d["enough_data"] = e.n >= e.planned
    return d


def _fact(f: utility.FactUtility) -> dict:
    return {"fact_id": f.fact_id, "kind": f.group, "deliveries": f.deliveries,
            "uses": f.uses, "rate": f.mean, "low": f.low, "high": f.high}


def _use(u: utility.Utility) -> dict:
    return {"rate": u.mean, "concentration": u.concentration,
            "facts_delivered": len(u.facts), "never_used": u.never_used(),
            "kinds": [asdict(g) for g in u.groups],
            "most_used": [_fact(f) for f in u.most_used(LIST_SIZE)],
            "rarely_used": [_fact(f) for f in u.rarely_used(LIST_SIZE)]}


def _judges(conn: sqlite3.Connection, profile_id: str) -> tuple[list, list]:
    from superlocalmemory.core import judge_certificates as certs
    from superlocalmemory.core import shadow_escalation as shadow

    try:
        records = [{k: c[k] for k in ("tier", "model", "verdict", "population", "regime", "n",
                                      "errors", "issued_at", "voided_at", "void_reason")}
                   for c in certs.records(conn, profile_id)]
    except (sqlite3.OperationalError, KeyError):
        records = []
    try:
        second = [s.as_dict() for s in shadow.disagreements(conn, profile_id)]
    except sqlite3.OperationalError:
        second = []
    return records, second


def _learning(conn: sqlite3.Connection, profile_id: str, slm_home: Path | None) -> dict:
    from superlocalmemory.learning import learning_health

    path = None if slm_home is None else slm_home / "learning.db"
    learning = None
    try:
        if path is not None and path.is_file():
            from superlocalmemory.truth.invariants import open_read_only

            learning = open_read_only(str(path))
        learners = learning_health.measure(learning, conn, profile_id)
    except Exception:  # noqa: BLE001 - the learning section never breaks the report
        logger.debug("learning health unreadable", exc_info=True)
        learners = []
    finally:
        if learning is not None:
            learning.close()
    return {"learners": [x.as_dict() for x in learners],
            "states": learning_health.summary(learners)}


def build(conn: sqlite3.Connection, profile_id: str, *, as_of: datetime | None = None,
          slm_home: Path | None = None) -> dict:
    """The whole report for one profile (JSON-ready)."""
    when = as_of or datetime.now(UTC)
    certificates, second = _judges(conn, profile_id)
    return {
        "profile_id": profile_id,
        "as_of": when.isoformat(timespec="seconds"),
        "outcome_window_hours": effects.OUTCOME_WINDOW_H,
        "alpha": effects.ALPHA,
        "ledger": _ledger(conn, profile_id, when, slm_home),
        "effects": [_effect(e) for e in effects.report(conn, profile_id, as_of=when)],
        "use": _use(utility.measure(conn, profile_id, as_of=when)),
        "certificates": certificates,
        "second_opinions": second,
        "learning": _learning(conn, profile_id, slm_home),
        "need_gate": _need_gate(conn, profile_id, when),
        "harm_gate": _harm_gate(conn, profile_id, when, slm_home),
        "trust": _trust(conn, profile_id, slm_home),
    }


def _harm_gate(conn: sqlite3.Connection, profile_id: str, when: datetime,
               slm_home: Path | None) -> dict:
    """W3-9 (D-15, M-14): memories shown to make the agent worse, in shadow."""
    from superlocalmemory.core import harm_gate
    from superlocalmemory.core.feature_flags import FeatureFlags

    try:
        state = FeatureFlags(slm_home).state("delivery.harm_gate").value
        return harm_gate.report(conn, profile_id, as_of=when, state=state)
    except Exception:  # noqa: BLE001 - this section never breaks the report
        logger.debug("harm gate unreadable", exc_info=True)
        return {}


def _trust(conn: sqlite3.Connection, profile_id: str, slm_home: Path | None) -> dict:
    """WS-5: origins, what waits for a person, credentials held as placeholders (counts only)."""
    from superlocalmemory.trust import report

    return report.build(conn, profile_id, slm_home=slm_home)  # fail-open inside


def _need_gate(conn: sqlite3.Connection, profile_id: str, when: datetime) -> dict:
    """W3-7d (M-16): the per-prompt need gate's log, threshold and rule bound."""
    from superlocalmemory.core import need_gate

    try:
        return need_gate.report(conn, profile_id, now=when)
    except Exception:  # noqa: BLE001 - this section never breaks the report
        logger.debug("need gate unreadable", exc_info=True)
        return {"turns": None}


# -- plain words, for people ------------------------------------------------------------------------

def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _signed(x: float) -> str:
    return f"{100 * x:+.1f} points"


_CONTRAST = {"memory": "one more memory at session start", "channel": "any retrieval channel"}
_OUTCOME = {"used": "the agent acting on a memory", "reask": "the agent asking memory again"}


def describe_effect(e: dict) -> str:
    what = _CONTRAST.get(e["contrast"], "the " + e["contrast"].split(":", 1)[-1] + " channel")
    head = f"{e['path']}: {what}, on {_OUTCOME[e['outcome']]}"
    if e["aipw"] is None:
        return f"{head}: no estimate yet ({e['n']} deliveries; needs both arms)"
    lo, hi = e["anytime"] or (None, None)
    rng = "" if lo is None else f" (anytime-valid range {_signed(lo)} to {_signed(hi)})"
    verdict = {"helps": "it helps", "hurts": "it hurts"}.get(e["claim"], "no claim yet")
    return (f"{head}: {_signed(e['aipw'])}{rng}; {verdict}; "
            f"{e['n']} of the {e['planned']} deliveries its plan needs")


def describe(report: dict) -> list[str]:
    """The report as short lines for a terminal."""
    led = report["ledger"]
    lines = [f"Memory Health (profile {report['profile_id']}, as of {report['as_of']})",
             f"  Receipts: {led['receipts'] or 'unknown'}; withholding: "
             f"{led['withholding'] or 'unknown'}"]
    recent = led["deliveries"]
    total = sum(recent.values())
    lines.append(f"  Deliveries in the last {led['recent_days']} days: {total}"
                 + (" (" + ", ".join(f"{p} {n}" for p, n in recent.items()) + ")"
                    if recent else ""))
    use = report["use"]
    lines.append(f"  Use rate when delivered: {_pct(use['rate'])} over "
                 f"{use['facts_delivered']} memories; never used: {use['never_used']}")
    for f in use["most_used"][:5]:
        lines.append(f"    most used {f['fact_id']}: {_pct(f['rate'])} "
                     f"({_pct(f['low'])}-{_pct(f['high'])}), {f['uses']}/{f['deliveries']}")
    for f in use["rarely_used"][:5]:
        lines.append(f"    rarely used {f['fact_id']}: under {_pct(f['high'])}, "
                     f"{f['uses']}/{f['deliveries']}")
    lines.append("  Does memory help? (randomised; a claim needs the anytime-valid range "
                 "to exclude 0)")
    for e in report["effects"]:
        if e["contrast"] in ("memory", "channel") or e["aipw"] is not None:
            lines.append("    " + describe_effect(e))
    certs = report["certificates"]
    active = [c for c in certs if not c.get("voided_at")]
    lines.append(f"  Judge certificates: {len(active)} active, {len(certs) - len(active)} voided")
    lines.append(f"  Second opinions that disagreed: {len(report['second_opinions'])}")
    learning = report.get("learning") or {}
    states = learning.get("states") or {}
    if learning.get("learners"):
        lines.append("  Is SLM learning? " + ", ".join(f"{n} {s.replace('_', ' ')}"
                                                       for s, n in sorted(states.items())))
        for item in learning["learners"]:
            lines.append(f"    {item['name']}: {item['state'].replace('_', ' ')} "
                         f"({item['reason']})")
    from superlocalmemory.core import need_gate

    lines.extend("  " + line for line in need_gate.describe(report.get("need_gate") or {}))
    from superlocalmemory.core import harm_gate

    lines.extend("  " + line for line in harm_gate.describe(report.get("harm_gate") or {}))
    from superlocalmemory.trust import report as trust_report

    lines.extend("  " + line for line in trust_report.describe(report.get("trust") or {}))
    return lines


__all__ = ["LIST_SIZE", "RECENT_DAYS", "build", "describe", "describe_effect"]
