# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The truth layer's invariants, checked against a store (R-27, D-24, blueprint I-01..I-16).

Read-only: every check is a query, and ``slm doctor --truth`` opens the store
with ``mode=ro``. Each result names the invariant, how many rows break it and up
to ``SAMPLE`` ids, never a memory's text. A store without the slot tables has
nothing to check (``not_applicable``). Invariants that need evidence this file
cannot see (receipts per ask, certificates, erasure across every table, shadow
writes) are listed as ``not_checked``, so a clean report never claims them.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from superlocalmemory.truth.validity import AFTER, BEFORE, SAME, order

SAMPLE = 5
OK, VIOLATED, NOT_APPLICABLE, NOT_CHECKED = "ok", "violated", "not_applicable", "not_checked"


@dataclass(frozen=True)
class Result:
    invariant: str
    title: str
    status: str
    count: int = 0
    sample: tuple[str, ...] = field(default_factory=tuple)


_LIVE_FACT = """(f.fact_id IS NOT NULL AND f.quarantined = 0 AND f.lifecycle != 'archived'
    AND f.merged_into IS NULL AND v.system_expired_at IS NULL)"""

_SQL: dict[str, tuple[str, str]] = {
    "I-01": ("A slot head points at a live fact", f"""
        SELECT s.slot_id FROM memory_slots s
          JOIN slot_revisions r ON r.revision_id = s.head_revision_id
          LEFT JOIN atomic_facts f ON f.fact_id = r.fact_id
          LEFT JOIN fact_temporal_validity v ON v.fact_id = r.fact_id
         WHERE NOT {_LIVE_FACT}"""),
    "I-02": ("A single-valued slot has at most one revision in state head", """
        SELECT s.slot_id FROM memory_slots s JOIN slot_revisions r ON r.slot_id = s.slot_id
         WHERE s.cardinality = 'single' AND r.state = 'head'
         GROUP BY s.slot_id HAVING COUNT(*) > 1"""),
    "I-03": ("No open dispute's side a is a revision other than the head", """
        SELECT d.dispute_id FROM slot_disputes d JOIN memory_slots s ON s.slot_id = d.slot_id
         WHERE d.state = 'open' AND d.revision_a IS NOT s.head_revision_id"""),
    "I-04": ("A disputed slot has an open dispute; an empty slot has no head", """
        SELECT s.slot_id FROM memory_slots s
         WHERE (s.state = 'disputed' AND NOT EXISTS (SELECT 1 FROM slot_disputes d
                 WHERE d.slot_id = s.slot_id AND d.state = 'open'))
            OR (s.state = 'empty' AND s.head_revision_id IS NOT NULL)"""),
    "I-06": ("A superseded or historical revision has its fact retired or a pending action", """
        SELECT r.revision_id FROM slot_revisions r
          LEFT JOIN fact_temporal_validity v ON v.fact_id = r.fact_id
          LEFT JOIN correction_cases c ON c.predecessor_fact_id = r.fact_id
               AND c.status = 'proposed'
         WHERE r.state IN ('superseded', 'historical') AND v.system_expired_at IS NULL
           AND c.case_id IS NULL"""),
    "I-07": ("Every applied slot-originated case has a revision with that case id", """
        SELECT c.case_id FROM correction_cases c
         WHERE c.reason_code = 'slot_supersede' AND c.status = 'applied'
           AND NOT EXISTS (SELECT 1 FROM slot_revisions r WHERE r.case_id = c.case_id)"""),
    "I-08": ("A rolled-back correction has restored the slot as well as the fact", """
        SELECT r.revision_id FROM slot_revisions r
          JOIN memory_slots s ON s.head_revision_id = r.revision_id
          JOIN correction_cases c ON c.case_id = r.case_id
         WHERE c.status = 'rolled_back'"""),
    "I-16": ("A current slot has a head and no open dispute", """
        SELECT s.slot_id FROM memory_slots s
         WHERE s.state = 'current' AND (s.head_revision_id IS NULL OR EXISTS (
               SELECT 1 FROM slot_disputes d WHERE d.slot_id = s.slot_id AND d.state = 'open'))"""),
}

#: Needed by each query beyond the slot tables; a store without one skips the check.
_NEEDS = {"I-01": ("atomic_facts", "fact_temporal_validity"),
          "I-06": ("fact_temporal_validity", "correction_cases"),
          "I-07": ("correction_cases",), "I-08": ("correction_cases",)}

NOT_CHECKED_HERE = {
    "I-09": "every model-tier ask has a receipt (checked by the ladder's tests, not the store)",
    "I-10": "no memory change without a certificate or approval (a property of the code paths)",
    "I-11": "erasure removes everything derived (checked by the erasure suite)",
    "I-13": "a verdict acts only on the pair it was computed on (checked at action time)",
    "I-14": "every certificate has a stored record (WS-2)",
    "I-15": "shadow writes only its own tables (checked by the flag tests)",
}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _ids(conn: sqlite3.Connection, sql: str) -> list[str]:
    return [str(r[0]) for r in conn.execute(sql)]


def _when(valid_from: str | None, recorded_at: str | None) -> str | None:
    return valid_from or recorded_at


def _i05(conn: sqlite3.Connection) -> list[str]:
    """I-05: a superseded value began before the value that superseded it."""
    bad = []
    for rid, a_from, a_rec, b_from, b_rec in conn.execute(
            "SELECT r.revision_id, r.valid_from, r.recorded_at, w.valid_from, w.recorded_at "
            "FROM slot_revisions r JOIN slot_revisions w ON w.revision_id = r.superseded_by "
            "JOIN memory_slots s ON s.slot_id = r.slot_id "
            "WHERE s.cardinality = 'single' AND r.state = 'superseded'"):
        if order(_when(a_from, a_rec), _when(b_from, b_rec)) in (AFTER, SAME):
            bad.append(str(rid))
    return bad


def _i12(conn: sqlite3.Connection) -> list[str]:
    """I-12: an end never before its start (interval-aware), and every end has a source."""
    columns = {r[1] for r in conn.execute("PRAGMA table_info(fact_temporal_validity)")}
    if "valid_until_source" not in columns:  # a store before M073: the end has no source column
        return [str(r[0]) for r in conn.execute(
            "SELECT fact_id FROM fact_temporal_validity WHERE valid_until IS NOT NULL")]
    return [str(fid) for fid, start, end, source in conn.execute(
        "SELECT fact_id, valid_from, valid_until, valid_until_source FROM fact_temporal_validity "
        "WHERE valid_until IS NOT NULL")
        if source is None or (start is not None and order(end, start) == BEFORE)]


def check(conn: sqlite3.Connection) -> list[Result]:
    """Every invariant this module can check, in order, then the ones it cannot."""
    present = _tables(conn)
    if not {"memory_slots", "slot_revisions", "slot_disputes"} <= present:
        return [Result("I-01..I-16", "the store has no slot tables", NOT_APPLICABLE)]
    runs: list[tuple[str, str, Callable[[], list[str]] | None]] = []
    for inv, (title, sql) in _SQL.items():
        ready = set(_NEEDS.get(inv, ())) <= present
        runs.append((inv, title, (lambda sql=sql: _ids(conn, sql)) if ready else None))
    runs.append(("I-05", "Single-valued slot revisions are ordered in valid time", lambda: _i05(conn)))
    runs.append(("I-12", "valid_until is not before valid_from, and every valid_until has a source",
                 (lambda: _i12(conn)) if "fact_temporal_validity" in present else None))
    results = []
    for inv, title, run in sorted(runs, key=lambda r: r[0]):
        if run is None:
            results.append(Result(inv, title, NOT_APPLICABLE))
            continue
        ids = run()
        results.append(Result(inv, title, VIOLATED if ids else OK, len(ids), tuple(ids[:SAMPLE])))
    results += [Result(inv, why, NOT_CHECKED) for inv, why in NOT_CHECKED_HERE.items()]
    return results


def open_read_only(path: str) -> sqlite3.Connection:
    """The store at ``path``, opened so nothing can be written through it."""
    from pathlib import Path

    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)


__all__ = ["NOT_APPLICABLE", "NOT_CHECKED", "NOT_CHECKED_HERE", "OK", "Result", "SAMPLE",
           "VIOLATED", "check", "open_read_only"]
