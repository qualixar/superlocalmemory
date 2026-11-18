# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Deterministic templates over slot values, and the check every page sentence passes (D-18).

A template sentence has a fixed frame; what fills it comes from the slot tables
(subject, relation, dates, which side of a dispute) and from the verbatim text of
cited live facts. ``render`` reads those values *now* from the store and returns
the sentence, or ``None`` when its inputs are gone or no longer say the same
thing. ``verify`` is the span check of D-18 for both kinds:

* a **span** passes when its text appears word for word in the content of the
  fact it cites, and that fact is live (not retired, merged, archived, quarantined,
  erased or being erased);
* a **template** passes when rendering it again from the current store gives
  exactly its text, so a slot that moved on or a fact whose text changed fails it.

No model writes any sentence: a model-written sentence would need its own
CITATION_SUPPORT certificate, and none exists in 4.2.30. Reads only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from superlocalmemory.compiler.page_text import (
    MAX_SENTENCE_CHARS,
    SPAN,
    TEMPLATE,
    day,
    humanize,
    one_line,
    sentences_of,
)
from superlocalmemory.compiler.versions import select_rows as _select

SLOT_VALUE, SLOT_SINCE, SLOT_DISPUTED, STORY_STEP = (
    "slot_value", "slot_since", "slot_disputed", "story_step")

LIVE = ("f.profile_id = ? AND COALESCE(f.quarantined, 0) = 0 AND COALESCE(f.lifecycle, '') != "
         "'archived' AND f.merged_into IS NULL AND v.system_expired_at IS NULL")
FACT_COLUMNS = ("f.fact_id AS fact_id, f.content AS content, f.created_at AS created_at, "
                 "f.referenced_date AS referenced_date, f.observation_date AS observation_date, "
                 "f.canonical_entities_json AS entities_json, f.memory_id AS memory_id, "
                 "f.memory_kind AS memory_kind")


def _has_table(conn: Any, name: str) -> bool:
    return bool(_select(conn, "SELECT 1 AS p FROM sqlite_master WHERE type = 'table' AND name = ?",
                        (name,)))


def hidden_clause(conn: Any) -> str:
    """What a page never cites, beyond the live test: a fact whose erasure has begun, and a
    correction's successor that is not applied (a proposal is no one's say-so)."""
    clause = ""
    if _has_table(conn, "projection_tombstones"):
        clause += (" AND NOT EXISTS (SELECT 1 FROM projection_tombstones t WHERE t.fact_id = "
                   "f.fact_id AND t.profile_id = f.profile_id)")
    if _has_table(conn, "correction_cases"):
        clause += (" AND NOT EXISTS (SELECT 1 FROM correction_cases c WHERE c.successor_fact_id = "
                   "f.fact_id AND c.status != 'applied')")
    return clause


def live_facts(conn: Any, profile_id: str, fact_ids: Sequence[str]) -> dict[str, dict]:
    """``{fact_id: row}`` for the cited facts that are live right now; others are absent."""
    ids = list(dict.fromkeys(str(i) for i in fact_ids if i))
    found: dict[str, dict] = {}
    for start in range(0, len(ids), 400):
        chunk = ids[start:start + 400]
        marks = ",".join("?" for _ in chunk)
        rows = _select(
            conn, f"SELECT {FACT_COLUMNS} FROM atomic_facts f LEFT JOIN fact_temporal_validity v "  # noqa: S608
            f"ON v.fact_id = f.fact_id WHERE {LIVE} AND f.fact_id IN ({marks})"
            + hidden_clause(conn), (profile_id, *chunk))
        found.update({r["fact_id"]: r for r in rows})
    return found


def _slot(conn: Any, profile_id: str, slot_id: str) -> dict | None:
    rows = _select(conn, "SELECT s.slot_id, s.subject_key, s.relation_key, s.qualifier, s.state, "
                   "s.version, s.head_revision_id, r.fact_id AS head_fact, "
                   "COALESCE(r.valid_from, r.recorded_at) AS since FROM memory_slots s "
                   "LEFT JOIN slot_revisions r ON r.revision_id = s.head_revision_id "
                   "WHERE s.slot_id = ? AND s.profile_id = ?", (slot_id, profile_id))
    return rows[0] if rows else None


def subject_words(conn: Any, profile_id: str, subject_key: str) -> str:
    """A slot subject in words; ``entity:<id>`` subjects are named by their entity."""
    if subject_key.startswith("entity:"):
        rows = _select(conn, "SELECT canonical_name FROM canonical_entities WHERE entity_id = ? "
                       "AND profile_id = ?", (subject_key[7:], profile_id))
        if rows and rows[0]["canonical_name"]:
            return str(rows[0]["canonical_name"])
    return humanize(subject_key)


def _what(conn: Any, profile_id: str, slot: Mapping[str, Any]) -> str:
    subject = subject_words(conn, profile_id, str(slot["subject_key"]))
    relation = humanize(str(slot["relation_key"]))
    qualifier = humanize(str(slot["qualifier"] or ""))
    return f"{subject} {relation}" + (f" ({qualifier})" if qualifier else "")


def _short(text: str) -> str | None:
    line = one_line(text)
    return line if 0 < len(line) <= MAX_SENTENCE_CHARS else None


# -- the templates: each returns (text, cited fact ids) or None ----------------------------------

def _slot_value(conn: Any, profile_id: str, p: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    slot = _slot(conn, profile_id, str(p.get("slot_id")))
    fact_id = str(p.get("fact_id") or "")
    if not slot or slot["state"] != "current" or slot["head_fact"] != fact_id:
        return None
    fact = live_facts(conn, profile_id, [fact_id]).get(fact_id)
    value = _short(fact["content"]) if fact else None
    return (f"{_what(conn, profile_id, slot)}: {value}", [fact_id]) if value else None


def _slot_since(conn: Any, profile_id: str, p: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    slot = _slot(conn, profile_id, str(p.get("slot_id")))
    fact_id = str(p.get("fact_id") or "")
    if not slot or slot["state"] != "current" or slot["head_fact"] != fact_id:
        return None
    since = day(slot["since"])
    if not since or fact_id not in live_facts(conn, profile_id, [fact_id]):
        return None
    return f"{_what(conn, profile_id, slot)} has had this value since {since}.", [fact_id]


def _slot_disputed(conn: Any, profile_id: str, p: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    slot = _slot(conn, profile_id, str(p.get("slot_id")))
    rows = _select(conn, "SELECT d.revision_a, d.revision_b, ra.fact_id AS fa, rb.fact_id AS fb "
                   "FROM slot_disputes d JOIN slot_revisions ra ON ra.revision_id = d.revision_a "
                   "JOIN slot_revisions rb ON rb.revision_id = d.revision_b "
                   "WHERE d.dispute_id = ? AND d.profile_id = ? AND d.state = 'open'",
                   (str(p.get("dispute_id")), profile_id))
    if not slot or not rows or (rows[0]["fa"], rows[0]["fb"]) != (p.get("fact_a"), p.get("fact_b")):
        return None
    live = live_facts(conn, profile_id, [rows[0]["fa"], rows[0]["fb"]])
    a, b = live.get(rows[0]["fa"]), live.get(rows[0]["fb"])
    one, two = (_short(a["content"]) if a else None), (_short(b["content"]) if b else None)
    if not one or not two:
        return None
    text = (f"Disputed, not settled: {_what(conn, profile_id, slot)} is “{one}” on one side "
            f"and “{two}” on the other.")
    return text, [rows[0]["fa"], rows[0]["fb"]]


def _story_step(conn: Any, profile_id: str, p: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    fact_id = str(p.get("fact_id") or "")
    fact = live_facts(conn, profile_id, [fact_id]).get(fact_id)
    if not fact:
        return None
    when = day(fact["referenced_date"]) or day(fact["observation_date"]) or day(fact["created_at"])
    first = next(iter(sentences_of(fact["content"])), "")
    value = _short(first)
    return (f"{when}: {value}", [fact_id]) if when and value else None


TEMPLATES: Mapping[str, Callable[[Any, str, Mapping[str, Any]], tuple[str, list[str]] | None]] = {
    SLOT_VALUE: _slot_value, SLOT_SINCE: _slot_since, SLOT_DISPUTED: _slot_disputed,
    STORY_STEP: _story_step,
}


def render(conn: Any, profile_id: str, template: str, params: Mapping[str, Any]
           ) -> tuple[str, list[str]] | None:
    """The sentence this template gives now, with the facts it cites; ``None`` if it cannot."""
    fn = TEMPLATES.get(template)
    return fn(conn, profile_id, params) if fn else None


def verify(conn: Any, profile_id: str, item: Mapping[str, Any],
           live: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[bool, str]:
    """The span check (D-18) for one stored sentence: ``(passed, reason)``.

    ``live`` is an optional ``live_facts`` result for the page's facts, so a page of
    many spans is checked with one read."""
    text, facts = str(item.get("text") or ""), [str(f) for f in item.get("facts") or []]
    if not text.strip() or not facts:
        return False, "no words or no citation"
    if item.get("type") == SPAN:
        if len(facts) != 1:
            return False, "a span cites exactly one memory"
        fact = (live if live is not None else live_facts(conn, profile_id, facts)).get(facts[0])
        if fact is None:
            return False, "the cited memory is gone or no longer current"
        if text not in fact["content"]:
            return False, "the sentence is no longer found in the memory it cites"
        return True, ""
    if item.get("type") != TEMPLATE:
        return False, "unknown kind of sentence"
    got = render(conn, profile_id, str(item.get("template") or ""), item.get("params") or {})
    if got is None:
        return False, "the values it was built from changed"
    if got[0] != text or sorted(got[1]) != sorted(facts):
        return False, "the values it was built from changed"
    return True, ""


def verify_all(conn: Any, profile_id: str, sentences: Sequence[Mapping[str, Any]]
               ) -> tuple[list[dict], int]:
    """The sentences that pass, in order, and how many did not."""
    cited = [str(f) for s in sentences if s.get("type") == SPAN for f in s.get("facts") or []]
    live = live_facts(conn, profile_id, cited)
    kept = [dict(s) for s in sentences if verify(conn, profile_id, s, live)[0]]
    return kept, len(sentences) - len(kept)


__all__ = ["FACT_COLUMNS", "LIVE", "SLOT_DISPUTED", "SLOT_SINCE", "SLOT_VALUE", "STORY_STEP", "TEMPLATES",
           "hidden_clause", "live_facts", "render", "subject_words", "verify", "verify_all"]
