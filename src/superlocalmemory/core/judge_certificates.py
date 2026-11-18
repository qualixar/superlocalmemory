# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Judge certificates as records: issued, voided, and voided on drift (I-14, R-26, M-07).

``ActingGate`` computes from blind audit labels whether a verdict may act.
This module keeps the record of that (M077 ``judge_certificates``):

* **issued** the first time a cell's certificate holds, with its method,
  alpha, delta, the n and errors at the chosen threshold, the threshold and a
  hash of the exact receipts it rests on;
* **voided** with reason ``labels`` when new labels no longer support it;
* **voided** with reason ``drift`` when the judge's recent scores stop looking
  like the audited ones: a two-sample Kolmogorov-Smirnov test of the scores of
  the cell's receipts decided since issue against the scores of its audited
  receipts, at ``DRIFT_LEVEL`` split over the profile's open certificates
  (Bonferroni), and only once ``MIN_RECENT`` recent decisions exist.

A certificate is bound to its question version and model by its key: a new
version or model is a new cell with no labels. After a drift void only labels
on decisions made after the void count (``labels_since``), so the old labels
cannot re-issue it at once. Counts and ids only, never memory text.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

#: Family-wise level of the drift tests over one profile's open certificates.
DRIFT_LEVEL = 0.01
#: Recent decisions needed before a drift test is run at all.
MIN_RECENT = 30
ISSUED, OPEN, VOIDED, NONE = "issued", "open", "voided", "none"
LABELS, DRIFT = "labels", "drift"

_KEY = ("profile_id = ? AND question_id = ? AND question_version = ? AND tier = ? AND model = ? "
        "AND verdict = ? AND population = ? AND regime = ?")


@dataclass(frozen=True)
class Cell:
    profile_id: str
    question_id: str
    version: str
    tier: str
    model: str
    verdict: str
    population: str
    regime: str

    def args(self) -> tuple[str, ...]:
        return (self.profile_id, self.question_id, self.version, self.tier, self.model,
                self.verdict, self.population, self.regime)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _table(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                             "AND name='judge_certificates'").fetchone())


def label_set_hash(receipt_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(receipt_ids)).encode("utf-8")).hexdigest()


def open_record(conn: sqlite3.Connection, cell: Cell) -> tuple | None:
    if not _table(conn):
        return None
    return conn.execute(f"SELECT cert_id, issued_at FROM judge_certificates WHERE {_KEY} "  # noqa: S608
                        "AND voided_at IS NULL", cell.args()).fetchone()


def labels_since(conn: sqlite3.Connection, cell: Cell) -> str:
    """Labels count only on decisions after the cell's last drift void ('' = all)."""
    if not _table(conn):
        return ""
    row = conn.execute(f"SELECT MAX(voided_at) FROM judge_certificates WHERE {_KEY} "  # noqa: S608
                       "AND void_reason = ?", (*cell.args(), DRIFT)).fetchone()
    return row[0] or ""


def _starts(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                             "AND name='judge_certificate_starts'").fetchone())


def start_of(conn: sqlite3.Connection, cell: Cell) -> float | None:
    """The threshold the cell's fixed sequence begins at, once it has been fixed (W0-11)."""
    if not _starts(conn):
        return None
    row = conn.execute(f"SELECT start_threshold FROM judge_certificate_starts WHERE {_KEY}",  # noqa: S608
                       cell.args()).fetchone()
    return None if row is None else float(row[0])


def fix_start(conn: sqlite3.Connection, cell: Cell, start: float) -> float:
    """Store the cell's start the first time it is tested; returns the start that stands.

    The first claim wins and a trigger refuses any later change, so the start depends on
    the scores at the first test and never moves with later scores or labels.
    """
    if not _starts(conn):
        return start
    conn.execute("INSERT OR IGNORE INTO judge_certificate_starts (profile_id, question_id, "
                 "question_version, tier, model, verdict, population, regime, start_threshold, "
                 "fixed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 (*cell.args(), float(start), _now()))
    stored = start_of(conn, cell)
    return start if stored is None else stored


def _void(conn: sqlite3.Connection, cert_id: str, reason: str) -> None:
    conn.execute("UPDATE judge_certificates SET voided_at = ?, void_reason = ? "
                 "WHERE cert_id = ? AND voided_at IS NULL", (_now(), reason, cert_id))


def reconcile(conn: sqlite3.Connection, cell: Cell, cert, *, method: str,
              receipt_ids: Sequence[str]) -> str:
    """Bring the stored record in line with the certificate just computed."""
    if not _table(conn):
        return NONE
    row = open_record(conn, cell)
    if not cert.acts:
        if row is None:
            return NONE
        _void(conn, row[0], LABELS)
        return VOIDED
    if row is not None:
        return OPEN
    n, errors = next(((n, e) for t, n, e, _ok in cert.tested if t == cert.threshold), (0, 0))
    conn.execute(
        "INSERT INTO judge_certificates (cert_id, profile_id, question_id, question_version, tier, "
        "model, verdict, population, regime, method, alpha, delta, n, errors, threshold, "
        "label_set_hash, issued_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("cert_" + uuid.uuid4().hex, *cell.args(), method, float(cert.alpha), float(cert.delta),
         int(n), int(errors), float(cert.threshold), label_set_hash(receipt_ids), _now()))
    return ISSUED


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """The two-sample Kolmogorov-Smirnov distance sup |F_a - F_b|."""
    xs, ys = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(xs) and j < len(ys):
        v = min(xs[i], ys[j])
        while i < len(xs) and xs[i] <= v:
            i += 1
        while j < len(ys) and ys[j] <= v:
            j += 1
        d = max(d, abs(i / len(xs) - j / len(ys)))
    return d


def ks_pvalue(d: float, n: int, m: int) -> float:
    """Asymptotic p-value of the two-sample KS statistic (Kolmogorov distribution),
    with the usual small-sample correction to the effective size."""
    if n == 0 or m == 0:
        return 1.0
    ne = n * m / (n + m)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    if lam < 1e-3:
        return 1.0
    total = sum(2 * (-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return min(1.0, max(0.0, total))


@dataclass(frozen=True)
class DriftResult:
    tested: bool
    statistic: float
    p_value: float
    level: float
    n_audited: int
    n_recent: int

    @property
    def shifted(self) -> bool:
        return self.tested and self.p_value < self.level


def check_drift(conn: sqlite3.Connection, cell: Cell, audited_scores: Sequence[float],
                *, score_of) -> DriftResult:
    """M-07: do the scores decided since issue still look like the audited ones?"""
    row = open_record(conn, cell)
    open_count = int(conn.execute("SELECT COUNT(*) FROM judge_certificates WHERE profile_id = ? "
                                  "AND voided_at IS NULL", (cell.profile_id,)).fetchone()[0]) \
        if row is not None else 0
    level = DRIFT_LEVEL / max(open_count, 1)
    if row is None:
        return DriftResult(False, 0.0, 1.0, level, len(audited_scores), 0)
    recent = [score_of(c) for (c,) in conn.execute(
        "SELECT confidence FROM judge_receipts WHERE profile_id = ? AND question_id = ? "
        "AND question_version = ? AND tier = ? AND model = ? AND verdict = ? AND population = ? "
        "AND batching_regime = ? AND decided_at > ?", (*cell.args(), row[1]))]
    if len(recent) < MIN_RECENT or not audited_scores:
        return DriftResult(False, 0.0, 1.0, level, len(audited_scores), len(recent))
    d = ks_statistic(audited_scores, recent)
    return DriftResult(True, d, ks_pvalue(d, len(audited_scores), len(recent)), level,
                       len(audited_scores), len(recent))


def void_for_drift(conn: sqlite3.Connection, cell: Cell) -> bool:
    row = open_record(conn, cell)
    if row is None:
        return False
    _void(conn, row[0], DRIFT)
    return True


def records(conn: sqlite3.Connection, profile_id: str | None, limit: int = 100) -> list[dict]:
    """A profile's certificates (every profile's when ``None``), newest first."""
    if not _table(conn):
        return []
    cols = ("cert_id", "question_id", "question_version", "tier", "model", "verdict",
            "population", "regime", "method", "alpha", "delta", "n", "errors", "threshold",
            "issued_at", "voided_at", "void_reason")
    where, args = ("WHERE profile_id = ? ", (profile_id,)) if profile_id is not None else ("", ())
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM judge_certificates {where}"  # noqa: S608
                        "ORDER BY issued_at DESC LIMIT ?", (*args, int(limit))).fetchall()
    return [dict(zip(cols, r)) for r in rows]


__all__ = ["Cell", "DRIFT", "DRIFT_LEVEL", "DriftResult", "LABELS", "MIN_RECENT", "check_drift",
           "fix_start", "ks_pvalue", "ks_statistic", "label_set_hash", "labels_since", "open_record",
           "reconcile", "records", "start_of", "void_for_drift"]
