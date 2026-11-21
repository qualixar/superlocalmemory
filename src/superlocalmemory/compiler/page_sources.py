# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Drafting pages from what the store knows: slots, decision families, entities and tags (D-18).

A draft is the page as it would be stored, built only from verbatim spans of live
facts and the deterministic templates of ``page_templates``; nothing is written
and no model is asked. Each draft says how many candidate sentences it considered
and how many it left out, and why (``withheld``), so the page can state it.

* **current_state**: one page per slot subject: the value each current slot
  holds (head) and since when, and every open dispute as a dispute, never as a value.
* **story**: one page per decision family (memory kind ``decision`` facts that
  share a slot subject, or else an entity), one dated sentence per decision, in time order.
* **topic**: one page per entity or tag with enough live facts, the first sentences
  of those facts as spans.

A fact whose source file changed or went missing (a stale lease) is left out
(``source_changed``): a page must not repeat what the file may no longer say.
Reads only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from superlocalmemory.compiler import page_templates as tpl
from superlocalmemory.compiler.page_text import (
    MAX_SENTENCE_CHARS,
    MAX_SENTENCES,
    MAX_SPANS_PER_FACT,
    MIN_STORY_FACTS,
    MIN_TOPIC_FACTS,
    OVER_LIMIT,
    SOURCE_CHANGED,
    SPAN,
    TEMPLATE,
    TOO_LONG,
    UNVERIFIABLE,
    content_hash,
    day,
    page_id,
    renumber,
    sentence,
    sentences_of,
)
from superlocalmemory.compiler.versions import select_rows as _select
from superlocalmemory.storage.json_columns import load_json_column

#: Routing only: how many facts one draft run reads for story and topic pages.
SCAN_LIMIT = 5000
LEGACY = "legacy"


def _tally(withheld: dict[str, int], reason: str, n: int = 1) -> None:
    withheld[reason] = withheld.get(reason, 0) + n


def _draft(profile_id: str, kind: str, subject: str, title: str, sentences: list[dict],
           candidates: int, withheld: Mapping[str, int]) -> dict:
    shown = renumber(sentences[:MAX_SENTENCES])
    left = dict(withheld)
    if len(sentences) > MAX_SENTENCES:
        _tally(left, OVER_LIMIT, len(sentences) - MAX_SENTENCES)
    return {"page_id": page_id(profile_id, kind, subject), "kind": kind, "subject_key": subject,
            "title": title, "sentences": shown, "candidates": candidates,
            "withheld": {k: v for k, v in left.items() if v},
            "content_hash": content_hash(title, shown)}


def stale_fact_ids(conn: Any, profile_id: str) -> set[str]:
    """Facts whose source file changed or is missing (leases, D-38)."""
    try:
        rows = _select(conn, "SELECT DISTINCT fact_id FROM memory_leases WHERE profile_id = ? "
                       "AND state IN ('changed', 'missing')", (profile_id,))
    except Exception as exc:  # noqa: BLE001 - a store before M070 has no leases
        if "no such table" in str(exc).lower():
            return set()
        raise
    return {r["fact_id"] for r in rows}


# -- current state -------------------------------------------------------------------------------

def _slot_rows(conn: Any, profile_id: str) -> list[dict]:
    return _select(
        conn, "SELECT s.slot_id, s.subject_key, s.state, r.fact_id AS head_fact, "
        "COALESCE(r.valid_from, r.recorded_at) AS since FROM memory_slots s "
        "JOIN slot_revisions r ON r.revision_id = s.head_revision_id "
        "WHERE s.profile_id = ? AND s.state IN ('current', 'disputed') AND s.key_source != ? "
        "ORDER BY s.subject_key, s.relation_key, s.qualifier, s.slot_id LIMIT ?",
        (profile_id, LEGACY, SCAN_LIMIT))


def _open_disputes(conn: Any, profile_id: str, slot_id: str) -> list[dict]:
    return _select(conn, "SELECT d.dispute_id, ra.fact_id AS fact_a, rb.fact_id AS fact_b "
                   "FROM slot_disputes d JOIN slot_revisions ra ON ra.revision_id = d.revision_a "
                   "JOIN slot_revisions rb ON rb.revision_id = d.revision_b "
                   "WHERE d.slot_id = ? AND d.profile_id = ? AND d.state = 'open' "
                   "ORDER BY d.opened_at, d.dispute_id", (slot_id, profile_id))


def _add(conn: Any, profile_id: str, template: str, params: dict, out: list[dict],
         withheld: dict[str, int]) -> None:
    got = tpl.render(conn, profile_id, template, params)
    if got is None:  # the one reason a fresh read cannot render is a value too long to be a sentence
        _tally(withheld, TOO_LONG if template != tpl.SLOT_SINCE else UNVERIFIABLE)
        return
    out.append(sentence(0, TEMPLATE, got[0], got[1], template=template, params=params))


def current_state_pages(conn: Any, profile_id: str, stale: set[str]) -> list[dict]:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in _slot_rows(conn, profile_id):
        by_subject[row["subject_key"]].append(row)
    drafts = []
    for subject, slots in by_subject.items():
        sentences: list[dict] = []
        withheld: dict[str, int] = {}
        candidates = 0
        for slot in slots:
            if slot["state"] == "disputed":
                for dispute in _open_disputes(conn, profile_id, slot["slot_id"]):
                    candidates += 1
                    if {dispute["fact_a"], dispute["fact_b"]} & stale:
                        _tally(withheld, SOURCE_CHANGED)
                        continue
                    _add(conn, profile_id, tpl.SLOT_DISPUTED,
                         {"slot_id": slot["slot_id"], "dispute_id": dispute["dispute_id"],
                          "fact_a": dispute["fact_a"], "fact_b": dispute["fact_b"]},
                         sentences, withheld)
                continue
            params = {"slot_id": slot["slot_id"], "fact_id": slot["head_fact"]}
            candidates += 1 + (1 if day(slot["since"]) else 0)
            if slot["head_fact"] in stale:
                _tally(withheld, SOURCE_CHANGED, 1 + (1 if day(slot["since"]) else 0))
                continue
            _add(conn, profile_id, tpl.SLOT_VALUE, params, sentences, withheld)
            if day(slot["since"]):
                _add(conn, profile_id, tpl.SLOT_SINCE, params, sentences, withheld)
        if sentences:
            name = tpl.subject_words(conn, profile_id, subject)
            drafts.append(_draft(profile_id, "current_state", subject, f"{name}: current state",
                                 sentences, candidates, withheld))
    return drafts


# -- decision families and topics ----------------------------------------------------------------

def _live_rows(conn: Any, profile_id: str, extra: str = "") -> list[dict]:
    return _select(
        conn, f"SELECT {tpl.FACT_COLUMNS} FROM atomic_facts f LEFT JOIN fact_temporal_validity v "  # noqa: S608
        f"ON v.fact_id = f.fact_id WHERE {tpl.LIVE}{extra}{tpl.hidden_clause(conn)} "
        "ORDER BY f.created_at, f.fact_id LIMIT ?", (profile_id, SCAN_LIMIT))


def _entities(row: Mapping[str, Any]) -> list[str]:
    try:
        value = load_json_column(row.get("entities_json"), [], column="atomic_facts.entities_json")
    except (TypeError, ValueError):
        return []
    return [str(e) for e in value] if isinstance(value, list) else []


def _entity_names(conn: Any, profile_id: str) -> dict[str, str]:
    return {r["entity_id"]: r["canonical_name"] for r in _select(
        conn, "SELECT entity_id, canonical_name FROM canonical_entities WHERE profile_id = ?",
        (profile_id,)) if r["canonical_name"]}


def _when(row: Mapping[str, Any]) -> str:
    return day(row["referenced_date"]) or day(row["observation_date"]) or day(row["created_at"])


def story_pages(conn: Any, profile_id: str, stale: set[str]) -> list[dict]:
    names = _entity_names(conn, profile_id)
    subject_of = {r["fact_id"]: r["subject_key"] for r in _select(
        conn, "SELECT r.fact_id, s.subject_key FROM slot_revisions r JOIN memory_slots s "
        "ON s.slot_id = r.slot_id WHERE s.profile_id = ? AND s.key_source != ?",
        (profile_id, LEGACY))}
    families: dict[str, list[dict]] = defaultdict(list)
    for row in _live_rows(conn, profile_id, " AND f.memory_kind = 'decision'"):
        ents = _entities(row)
        key = (f"slot:{subject_of[row['fact_id']]}" if row["fact_id"] in subject_of
               else (f"entity:{ents[0]}" if ents else None))
        if key:
            families[key].append(row)
    drafts = []
    for key, rows in sorted(families.items()):
        rows.sort(key=lambda r: (_when(r), r["fact_id"]))
        withheld: dict[str, int] = {}
        sentences: list[dict] = []
        for row in rows:
            if row["fact_id"] in stale:
                _tally(withheld, SOURCE_CHANGED)
            else:
                _add(conn, profile_id, tpl.STORY_STEP, {"fact_id": row["fact_id"]},
                     sentences, withheld)
        if len(sentences) >= MIN_STORY_FACTS:
            label = (tpl.subject_words(conn, profile_id, key[5:]) if key.startswith("slot:")
                     else names.get(key[7:], key[7:]))
            drafts.append(_draft(profile_id, "story", key, f"{label}: how it was decided",
                                 sentences, len(rows), withheld))
    return drafts


def _spans(row: Mapping[str, Any], withheld: dict[str, int]) -> tuple[list[dict], int]:
    """The spans one fact gives a topic page and how many sentences it offered."""
    spans, all_sentences = [], sentences_of(row["content"])
    for text in all_sentences:
        if len(spans) >= MAX_SPANS_PER_FACT:
            _tally(withheld, OVER_LIMIT)
        elif len(text) > MAX_SENTENCE_CHARS:
            _tally(withheld, TOO_LONG)
        else:
            spans.append(sentence(0, SPAN, text, [row["fact_id"]]))
    return spans, len(all_sentences)


def _tags(conn: Any, profile_id: str) -> dict[str, list[str]]:
    """``{fact_id: [tag, ...]}`` from the saving memory's metadata (bounded scan)."""
    out: dict[str, list[str]] = {}
    for r in _select(
            conn, "SELECT f.fact_id AS fact_id, m.metadata_json AS meta FROM atomic_facts f "
            "JOIN memories m ON m.memory_id = f.memory_id AND m.profile_id = f.profile_id "
            "WHERE f.profile_id = ? AND m.metadata_json LIKE '%tags%' "
            "ORDER BY f.created_at DESC LIMIT ?", (profile_id, SCAN_LIMIT)):
        try:
            raw = (json.loads(r["meta"] or "{}") or {}).get("tags")
        except (TypeError, ValueError, AttributeError):
            continue
        items = raw.split(",") if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
        tags = sorted({str(t).strip().lower() for t in items if str(t).strip()})
        if tags:
            out[r["fact_id"]] = tags
    return out


def topic_pages(conn: Any, profile_id: str, stale: set[str]) -> list[dict]:
    names = _entity_names(conn, profile_id)
    groups: dict[str, list[dict]] = defaultdict(list)
    rows = _live_rows(conn, profile_id)
    tags = _tags(conn, profile_id)
    for row in rows:
        for ent in _entities(row):
            groups[f"entity:{ent}"].append(row)
        for tag in tags.get(row["fact_id"], ()):
            groups[f"tag:{tag}"].append(row)
    drafts = []
    for key, members in sorted(groups.items()):
        if len(members) < MIN_TOPIC_FACTS:
            continue
        members.sort(key=lambda r: (r["created_at"] or "", r["fact_id"]), reverse=True)
        withheld: dict[str, int] = {}
        sentences: list[dict] = []
        candidates = 0
        for row in members:
            if row["fact_id"] in stale:
                _tally(withheld, SOURCE_CHANGED, max(1, len(sentences_of(row["content"]))))
                candidates += max(1, len(sentences_of(row["content"])))
                continue
            spans, offered = _spans(row, withheld)
            sentences.extend(spans)
            candidates += offered
        if sentences:
            label = names.get(key[7:], key[7:]) if key.startswith("entity:") else f"#{key[4:]}"
            drafts.append(_draft(profile_id, "topic", key, f"{label}: what is known", sentences,
                                 candidates, withheld))
    return drafts


def build_drafts(conn: Any, profile_id: str, kinds: tuple[str, ...] | None = None) -> list[dict]:
    """Every draft the store supports, in a fixed order (kind, then subject)."""
    wanted = set(kinds or ("current_state", "story", "topic"))
    stale = stale_fact_ids(conn, profile_id)
    drafts: list[dict] = []
    if "current_state" in wanted:
        drafts += current_state_pages(conn, profile_id, stale)
    if "story" in wanted:
        drafts += story_pages(conn, profile_id, stale)
    if "topic" in wanted:
        drafts += topic_pages(conn, profile_id, stale)
    return drafts


__all__ = ["SCAN_LIMIT", "UNVERIFIABLE", "build_drafts", "current_state_pages", "stale_fact_ids",
           "story_pages", "topic_pages"]
