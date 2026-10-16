# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Take the made-up dates of builds before 4.1.6 out of a store (M065).

WHICH FACTS
A fact qualifies only when every one of these holds; anything else is left
exactly as it is:

* it was saved before 4.1.6 shipped (:data:`CUTOFF`). Later facts that mention
  those days (release notes about this very bug) are never read for change;
* it has a reachable original (its source memory, or the raw text of the
  ingestion record linked by ``derivation_lineage`` or by the operation's
  fact lists) and no original names the day in any written form or was dated
  that day (``fabricated_dates.source_names_the_day``). A fact with no
  reachable original is left as it is: nothing can show its date was made up;
* it was not saved within :data:`NEAR_DAYS` of the day, when "yesterday" could
  truly have been 15 January;
* it is not being erased.

WHAT CHANGES
The clause carrying the date leaves the text when the sentence reads without it
(``fabricated_dates.remove_date_clauses``); a statement that is about the date
keeps its words. Either way the fact keeps no made-up date: the fact's own date
columns lose it, a temporal event left with no other date is removed (one with
another date keeps that), and a validity row left with nothing is removed. No
fact is deleted.

INDEXES
The keyword index follows the text through the ``atomic_facts_fts`` triggers;
the persisted BM25 tokens, temporal event descriptions and the rules-made
context line are rewritten here. The embedding is set to NULL, which is how the
maintenance pass's bounded self-heal knows to embed the fact again
(``embedding_migrator.backfill_missing_embeddings``); until then the old vector
(the same sentence plus a date) keeps answering. The graph/vector projections
are queued through ``projection_outbox`` in the same transaction.

Every change is written to ``data_repair_undo`` first; ``data_repair_undo.undo``
puts it all back.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from superlocalmemory.storage.data_repair_undo import UndoWriter
from superlocalmemory.storage.fabricated_dates import (
    FABRICATED_DAYS,
    contains_fabricated_date,
    is_fabricated_date_value,
    remove_date_clauses,
    source_names_the_day,
)

logger = logging.getLogger(__name__)

#: 4.1.6 release commit 9d30aa4c (2026-08-24 16:42:24 +05:30): the prompt that
#: leaked the example date was replaced in this build.
CUTOFF = datetime(2026, 8, 24, 11, 12, 24, tzinfo=UTC)
#: A fact saved this close to a fabricated day may hold a true relative date.
NEAR_DAYS = 7

_FACT_DATES = ("observation_date", "referenced_date", "interval_start", "interval_end",
               "valid_from", "valid_until")
_EVENT_DATES = ("observation_date", "referenced_date", "interval_start", "interval_end")
_VALIDITY_DATES = ("valid_from", "valid_until")
_VALIDITY_KEEP = ("valid_from", "valid_until", "system_expired_at", "invalidated_by",
                  "invalidation_reason")
#: Cheap SQL prefilter; Python decides. Spelled forms are matched by LIKE too.
_TEXT_LIKES = ("%2024-01-15%", "%2026-01-15%", "%15, 2024%", "%15, 2026%",
               "%15th, 2024%", "%15th, 2026%", "%15 January 202%", "%15 Jan 202%")


@dataclass(frozen=True)
class FactPlan:
    fact_id: str
    profile_id: str
    content: str
    new_content: str | None
    fact_dates: tuple[str, ...]
    events: tuple[dict[str, Any], ...] = field(default=())
    validity: dict[str, Any] | None = None
    #: Undated-by-us events that quote the fact's text (their description follows it).
    quoting: tuple[dict[str, Any], ...] = field(default=())

    @property
    def text_changes(self) -> bool:
        return self.new_content is not None


def _cols(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _parse(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _saved_before_cutoff_and_not_near(created_at: object) -> bool:
    created = _parse(created_at)
    if created is None or created >= CUTOFF:
        return False
    days = (datetime.fromisoformat(d).replace(tzinfo=UTC) for d in FABRICATED_DAYS)
    return all(abs(created - day) > timedelta(days=NEAR_DAYS) for day in days)


def _candidate_ids(conn: sqlite3.Connection, fact_cols: list[str]) -> list[str]:
    days = tuple(f"{d}%" for d in FABRICATED_DAYS)
    text = " OR ".join("content LIKE ?" for _ in _TEXT_LIKES)
    dated = " OR ".join(f"{c} LIKE ? OR {c} LIKE ?" for c in _FACT_DATES if c in fact_cols)
    sql = f"SELECT fact_id FROM atomic_facts WHERE {text}" + (f" OR {dated}" if dated else "")
    params: list[str] = [*_TEXT_LIKES, *(days * sum(c in fact_cols for c in _FACT_DATES))]
    ids = {r[0] for r in conn.execute(sql, params)}  # noqa: S608
    for table, cols in (("temporal_events", _EVENT_DATES),
                        ("fact_temporal_validity", _VALIDITY_DATES)):
        present = [c for c in cols if c in _cols(conn, table)]
        if present:
            where = " OR ".join(f"{c} LIKE ? OR {c} LIKE ?" for c in present)
            ids |= {r[0] for r in conn.execute(f"SELECT fact_id FROM {table} WHERE {where}",  # noqa: S608
                                               days * len(present))}
    return sorted(ids)


def _rows(conn: sqlite3.Connection, table: str, fact_id: str) -> list[dict[str, Any]]:
    cols = _cols(conn, table)
    if not cols:
        return []
    return [dict(zip(cols, r)) for r in conn.execute(
        f"SELECT {','.join(cols)} FROM {table} WHERE fact_id = ?", (fact_id,))]  # noqa: S608


def _rows_by_fact(conn: sqlite3.Connection, table: str,
                  fact_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Rows of ``table`` for many facts in one statement (temporal_events has no
    index led by fact_id, so one query per fact was one table scan per fact)."""
    cols = _cols(conn, table)
    out: dict[str, list[dict[str, Any]]] = {}
    if not cols or not fact_ids:
        return out
    for row in conn.execute(
            f"SELECT {','.join(cols)} FROM {table} WHERE fact_id IN "  # noqa: S608
            "(SELECT value FROM json_each(?))", (json.dumps(fact_ids),)):
        record = dict(zip(cols, row))
        out.setdefault(record["fact_id"], []).append(record)
    return out


def _ingestion_sources(conn: sqlite3.Connection,
                       fact_ids: list[str]) -> dict[str, list[tuple[Any, Any]]]:
    """Raw text and session date of the ingestion records each fact came from:
    linked by derivation lineage, or named in the operation's own fact lists."""
    out: dict[str, list[tuple[Any, Any]]] = {}
    if not fact_ids or not _cols(conn, "ingestion_operations"):
        return out
    ids = json.dumps(fact_ids)
    queries = [f"SELECT je.value, io.raw_content, io.session_date FROM ingestion_operations io, "  # noqa: S608
               f"json_each(io.{column}) je WHERE je.value IN (SELECT value FROM json_each(?))"
               for column in ("final_fact_ids_json", "queryable_fact_ids_json")
               if column in _cols(conn, "ingestion_operations")]
    if _cols(conn, "derivation_lineage"):
        queries.append("SELECT dl.object_id, io.raw_content, io.session_date "
                       "FROM derivation_lineage dl JOIN ingestion_operations io "
                       "ON io.operation_id = dl.operation_id WHERE dl.object_type = 'fact' "
                       "AND dl.object_id IN (SELECT value FROM json_each(?))")
    for sql in queries:
        for fact_id, raw, session_date in conn.execute(sql, (ids,)):
            out.setdefault(str(fact_id), []).append((raw, session_date))
    return out


def _sources(conn: sqlite3.Connection, fact: dict[str, Any],
             ingested: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """Every original text the fact was extracted from, with its session date."""
    row = conn.execute("SELECT content, session_date FROM memories WHERE memory_id = ?",
                       (fact["memory_id"],)).fetchone()
    found = [tuple(row)] if row is not None else []
    return [s for s in (*found, *ingested) if s[0] is not None and str(s[0]).strip()]


def _source_names_it(sources: list[tuple[Any, Any]]) -> bool:
    """A source names the day (any written form) or was dated that day: real."""
    return any(source_names_the_day(str(text)) or is_fabricated_date_value(day)
               for text, day in sources)


def _being_erased(conn: sqlite3.Connection, fact_id: str) -> bool:
    if not _cols(conn, "projection_tombstones"):
        return False
    return conn.execute("SELECT 1 FROM projection_tombstones WHERE fact_id = ?",
                        (fact_id,)).fetchone() is not None


#: Why a candidate fact was left as it is (see :class:`Survey`).
KEPT_IN_SOURCE = "in_source"
KEPT_NO_SOURCE = "no_source"
_KEPT_OTHER = "other"


def _classify(conn: sqlite3.Connection, fact: dict[str, Any], events: list[dict[str, Any]],
              validity: list[dict[str, Any]], ingested: list[tuple[Any, Any]],
              ) -> tuple[str, FactPlan | None]:
    """``("repair", plan)`` or ``(reason kept, None)``.

    A date is made up only when no original the fact came from names that day,
    so a fact with no reachable original is never changed: nothing can show its
    date was invented.
    """
    if not _saved_before_cutoff_and_not_near(fact.get("created_at")):
        return _KEPT_OTHER, None
    if _being_erased(conn, fact["fact_id"]):
        return _KEPT_OTHER, None
    sources = _sources(conn, fact, ingested)
    if not sources:
        return KEPT_NO_SOURCE, None
    if _source_names_it(sources):
        return KEPT_IN_SOURCE, None
    dated = tuple(e for e in events
                  if any(is_fabricated_date_value(e.get(c)) for c in _EVENT_DATES))
    plan_ = FactPlan(
        fact_id=fact["fact_id"], profile_id=fact["profile_id"], content=fact["content"],
        new_content=remove_date_clauses(fact["content"]),
        fact_dates=tuple(c for c in _FACT_DATES if is_fabricated_date_value(fact.get(c))),
        events=dated,
        validity=next((v for v in validity if any(
            is_fabricated_date_value(v.get(c)) for c in _VALIDITY_DATES)), None),
        quoting=tuple(e for e in events
                      if e not in dated and e.get("description") == fact["content"]))
    if not (plan_.text_changes or plan_.fact_dates or plan_.events or plan_.validity):
        return _KEPT_OTHER, None
    return "repair", plan_


@dataclass(frozen=True)
class Survey:
    """Candidates by outcome: repaired, or kept because a source names the day,
    or kept because no source is reachable. Fact ids for the kept groups."""
    repairs: tuple[FactPlan, ...]
    in_source: tuple[str, ...]
    no_source: tuple[str, ...]


def survey(conn: sqlite3.Connection, fact_ids: list[str] | None = None) -> Survey:
    """Classify every candidate. Reads only. ``fact_ids`` limits the look."""
    fact_cols = _cols(conn, "atomic_facts")
    if not fact_cols:
        return Survey((), (), ())
    wanted = ["fact_id", "memory_id", "profile_id", "content", "created_at",
              *[c for c in _FACT_DATES if c in fact_cols]]
    ids = _candidate_ids(conn, fact_cols) if fact_ids is None else fact_ids
    events = _rows_by_fact(conn, "temporal_events", ids)
    validity = _rows_by_fact(conn, "fact_temporal_validity", ids)
    ingested = _ingestion_sources(conn, ids)
    groups: dict[str, list[Any]] = {"repair": [], KEPT_IN_SOURCE: [], KEPT_NO_SOURCE: [],
                                    _KEPT_OTHER: []}
    for fact_id in ids:
        row = conn.execute(f"SELECT {','.join(wanted)} FROM atomic_facts WHERE fact_id = ?",  # noqa: S608
                           (fact_id,)).fetchone()
        if row is None:
            continue
        reason, found = _classify(conn, dict(zip(wanted, row)), events.get(fact_id, []),
                                  validity.get(fact_id, []), ingested.get(fact_id, []))
        groups[reason].append(found if found is not None else fact_id)
    return Survey(tuple(groups["repair"]), tuple(groups[KEPT_IN_SOURCE]),
                  tuple(groups[KEPT_NO_SOURCE]))


def plan(conn: sqlite3.Connection, fact_ids: list[str] | None = None) -> list[FactPlan]:
    """What the repair would change. Reads only. ``fact_ids`` limits the look."""
    return list(survey(conn, fact_ids).repairs)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _update(conn: sqlite3.Connection, writer: UndoWriter, plan_: FactPlan, table: str,
            key_col: str, row: dict[str, Any], changes: dict[str, Any]) -> None:
    changes = {c: v for c, v in changes.items() if row.get(c) != v}
    if not changes:
        return
    writer.record(table, str(row[key_col]), plan_.fact_id, plan_.profile_id,
                  {c: row.get(c) for c in changes}, changes)
    sets = ",".join(f"{c} = ?" for c in changes)
    conn.execute(f"UPDATE {table} SET {sets} WHERE {key_col} = ?",  # noqa: S608
                 (*changes.values(), row[key_col]))


def _remove(conn: sqlite3.Connection, writer: UndoWriter, plan_: FactPlan, table: str,
            key_col: str, row: dict[str, Any]) -> None:
    writer.record(table, str(row[key_col]), plan_.fact_id, plan_.profile_id, row, None)
    conn.execute(f"DELETE FROM {table} WHERE {key_col} = ?", (row[key_col],))  # noqa: S608


def _repair_events(conn, plan_: FactPlan, writer: UndoWriter) -> None:
    for event in plan_.events:
        cleared: dict[str, Any] = {
            c: None for c in _EVENT_DATES if is_fabricated_date_value(event.get(c))}
        if all(event.get(c) is None or c in cleared for c in _EVENT_DATES):
            _remove(conn, writer, plan_, "temporal_events", "event_id", event)
            continue
        if plan_.text_changes and event.get("description") == plan_.content:
            cleared["description"] = plan_.new_content
        _update(conn, writer, plan_, "temporal_events", "event_id", event, cleared)
    if not plan_.text_changes:
        return
    for event in plan_.quoting:
        _update(conn, writer, plan_, "temporal_events", "event_id", event,
                {"description": plan_.new_content})


def _repair_validity(conn, plan_: FactPlan, writer: UndoWriter) -> None:
    row = plan_.validity
    if row is None:
        return
    cleared = {c: None for c in _VALIDITY_DATES if is_fabricated_date_value(row.get(c))}
    if all(row.get(c) is None or c in cleared for c in _VALIDITY_KEEP if c in row):
        _remove(conn, writer, plan_, "fact_temporal_validity", "fact_id", row)
    else:
        _update(conn, writer, plan_, "fact_temporal_validity", "fact_id", row, cleared)


def _context_line(row: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    """The rules-made line quotes the fact: swap the quote, drop the date keywords.
    Anything else that still names the date is cleared, as a correction does."""
    description = str(row.get("contextual_description") or "")
    try:
        keywords = json.loads(row.get("keywords") or "[]")
    except (TypeError, ValueError):
        keywords = []
    kept = [k for k in keywords if not (contains_fabricated_date(str(k))
                                        or is_fabricated_date_value(str(k)))]
    if old in description:
        description = description.replace(old, new)
    if contains_fabricated_date(description):
        return {"contextual_description": "", "keywords": ""}
    return {"contextual_description": description,
            "keywords": json.dumps(kept) if isinstance(keywords, list) else row.get("keywords")}


class _Rows:
    """``execute`` returning rows, as the storage helpers expect of a manager."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple = ()) -> list:
        return self._conn.execute(sql, params).fetchall()


def _repair_text_indexes(conn, plan_: FactPlan, writer: UndoWriter) -> None:
    from superlocalmemory.retrieval.bm25_channel import tokenize

    new = str(plan_.new_content)
    for row in _rows(conn, "bm25_tokens", plan_.fact_id):
        _update(conn, writer, plan_, "bm25_tokens", "fact_id", row,
                {"tokens": json.dumps(tokenize(new))})
    for row in _rows(conn, "fact_context", plan_.fact_id):
        _update(conn, writer, plan_, "fact_context", "fact_id", row,
                _context_line(row, plan_.content, new))
    from superlocalmemory.storage.projection_outbox import enqueue

    enqueue(_Rows(conn), plan_.fact_id, plan_.profile_id)  # no-op without the queue


def _repair_fact(conn: sqlite3.Connection, plan_: FactPlan, writer: UndoWriter) -> None:
    """Every change of one fact; the caller's transaction holds them all."""
    changes: dict[str, Any] = {c: None for c in plan_.fact_dates}
    if plan_.text_changes:
        changes["content"] = plan_.new_content
    cols = ["fact_id", "content", *plan_.fact_dates]
    current = conn.execute(f"SELECT {','.join(cols)} FROM atomic_facts WHERE fact_id = ?",  # noqa: S608
                           (plan_.fact_id,)).fetchone()
    _update(conn, writer, plan_, "atomic_facts", "fact_id", dict(zip(cols, current)), changes)
    if plan_.text_changes:
        # NULL embedding = "embed me again" for the maintenance self-heal pass.
        conn.execute("UPDATE atomic_facts SET embedding = NULL WHERE fact_id = ?",
                     (plan_.fact_id,))
        _repair_text_indexes(conn, plan_, writer)
    _repair_events(conn, plan_, writer)
    _repair_validity(conn, plan_, writer)


@dataclass(frozen=True)
class RepairResult:
    facts: int
    text_changed: int
    dates_removed: int
    kept_in_source: int = 0
    kept_no_source: int = 0


def apply(conn: sqlite3.Connection, repair: str) -> RepairResult:
    """Find the facts without the write lock, then plan and apply them again
    inside one ``BEGIN IMMEDIATE`` transaction.

    The full scan (seconds on a large store) runs before the lock, so other
    writers wait only for the short write. The plan is re-read inside the
    transaction for the facts found, so it acts on what is there then. A crash
    part-way leaves the store exactly as it was; the runner sees the migration
    unfinished and runs it again at the next start.
    """
    looked = survey(conn)
    kept = {"kept_in_source": len(looked.in_source), "kept_no_source": len(looked.no_source)}
    found = [p.fact_id for p in looked.repairs]
    if not found:
        return RepairResult(0, 0, 0, **kept)
    conn.execute("BEGIN IMMEDIATE")
    try:
        plans = plan(conn, found)
        writer = UndoWriter(conn, repair)
        for plan_ in plans:
            _repair_fact(conn, plan_, writer)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    changed = [p.fact_id for p in plans if p.text_changes]
    _drop_cached_context(conn, changed)
    return RepairResult(len(plans), len(changed),
                        sum(1 for p in plans if p.fact_dates or p.events or p.validity), **kept)


def _drop_cached_context(conn: sqlite3.Connection, fact_ids: list[str]) -> None:
    """The context cache (its own file) may quote the old text; drop those entries."""
    if not fact_ids:
        return
    from types import SimpleNamespace

    from superlocalmemory.core.mutations import _invalidate_context_cache_for_fact

    row = conn.execute("PRAGMA database_list").fetchone()
    if not row or not row[2]:
        return
    engine = SimpleNamespace(_db=SimpleNamespace(db_path=row[2]))
    for fact_id in fact_ids:
        _invalidate_context_cache_for_fact(engine, fact_id)


__all__ = ["CUTOFF", "NEAR_DAYS", "FactPlan", "RepairResult", "apply", "plan"]
