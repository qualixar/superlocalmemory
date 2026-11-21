# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Engagement observed from what an agent did, not from what it was asked to say.

WHY THIS EXISTS
---------------
The reward ladder in ``reward_proxy`` asks one question — did a recalled
``fact_id`` appear verbatim in a later tool event — and defaults to ``0.5`` when
the answer is no. Both halves of that are broken.

The question requires the caller to copy an opaque marker into its next tool
call. Nothing makes it: no tool description asks for it and no rule requires
it. A design that depends on a behaviour nothing produces has no signal, and
in practice no signal was ever registered.

The default is worse than no answer. ``alpha += 0.5`` and ``beta += 0.5`` move
together, so a Beta posterior keeps its mean at exactly 0.5 while its variance
*shrinks*. Every neutral settlement makes an arm more confident that it is
average and harder for real evidence to move later. Neutral is not a safe
default; it is a slow commitment to knowing nothing.

WHAT REPLACES IT
----------------
Features computed from rows the system already writes. An agent that uses a
recalled memory leaves traces whether or not it cooperates: it reads or edits
the files the memory names, its next actions stay on the memory's subject, it
writes a follow-up memory that overlaps. None of that requires it to quote an
identifier.

Every feature here is observable, and each is reported separately so a reward
can say *why*. When nothing is observable the answer is ``None`` — abstain —
never a number.

NOT SELF-REFERENTIAL
--------------------
A signal derived from the mechanism it evaluates cannot detect that mechanism's
failure. Ranking position is therefore not a feature: the bandit chose the
ranking, so scoring it by what the bandit ranked first would confirm the bandit
to itself. Position enters only in ``propensity.py``, as a correction applied
*against* the observation, never as evidence for it.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from superlocalmemory.storage.json_columns import load_json_column

__all__ = [
    "EngagementFeatures",
    "OBSERVATION_WINDOW_SEC",
    "extract_features",
    "tokenize",
]

#: How long after a recall an action may still be counted as caused by it.
#: Wider than the old 30 s hit window: an agent reads a memory, then thinks,
#: then acts, and 30 s discarded most of the acting.
OBSERVATION_WINDOW_SEC = 300

#: Tools whose payload naming a recalled memory's content is strong evidence
#: the memory was actually used, not merely returned.
_ARTIFACT_TOOLS = frozenset({"Write", "Edit", "NotebookEdit", "MultiEdit"})

#: Words carrying no topical signal; overlap on these is noise, and without
#: this filter every pair of payloads overlaps.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those is are was were be been
being to of in on at by for with from as it its into over under about after
before not no nor so such can will would should could may might must do does
did done have has had i you he she we they them his her their our your my me
""".split())

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]{2,}")


def tokenize(text: str) -> set[str]:
    """Content-bearing lowercase tokens, stopwords and short words removed."""
    if not text:
        return set()
    return {
        tok for tok in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
        if tok not in _STOPWORDS and len(tok) > 2
    }


@dataclass
class EngagementFeatures:
    """What was observed after one recall. Every field is measured, not inferred.

    ``observed`` is the honest summary: False means nothing happened that this
    module can see, which is a reason to abstain rather than a reason to
    penalise. An agent may have used a memory perfectly and left no trace.
    """

    #: Best Jaccard-style overlap between any recalled fact and any following
    #: tool payload, in [0, 1].
    peak_overlap: float = 0.0
    #: Overlap restricted to file-writing tools — the memory reached an artifact.
    artifact_overlap: float = 0.0
    #: A later remember/update_memory overlapping a recalled fact.
    followup_write_overlap: float = 0.0
    #: The same question asked again inside the requery window: the answer did
    #: not satisfy. The one unambiguous negative available.
    requeried: bool = False
    #: Seconds from recall to the first following action, when there was one.
    dwell_sec: float | None = None
    #: Tool events seen in the window at all.
    action_count: int = 0
    #: Which fact ids the overlap landed on, so a reward can be explained.
    matched_fact_ids: list[str] = field(default_factory=list)
    #: A recalled fact's own id appeared verbatim in a later tool event. Rare,
    #: because nothing makes an agent echo it — but unambiguous when it does
    #: happen, so it is kept as the strongest single piece of evidence rather
    #: than discarded along with the ladder that relied on it alone.
    marker_hit: bool = False

    @property
    def observed(self) -> bool:
        """Whether anything at all was seen. Drives abstention."""
        return bool(
            self.requeried
            or self.marker_hit
            or self.action_count > 0
            and (
                self.peak_overlap > 0.0
                or self.artifact_overlap > 0.0
                or self.followup_write_overlap > 0.0
            )
        )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
        ).fetchone() is not None
    except sqlite3.Error:
        return False


def _fact_tokens(conn: sqlite3.Connection, fact_ids: list[str]) -> dict[str, set[str]]:
    """Content tokens per fact. Entities are folded in when the column exists,
    because a memory's entities are what a later action is most likely to name.
    """
    if not fact_ids or not _table_exists(conn, "atomic_facts"):
        return {}
    placeholders = ",".join("?" * len(fact_ids))
    try:
        rows = conn.execute(
            f"SELECT fact_id, content, COALESCE(entities_json,'') "  # noqa: S608
            f"FROM atomic_facts WHERE fact_id IN ({placeholders})",
            tuple(fact_ids),
        ).fetchall()
    except sqlite3.Error:
        return {}

    out: dict[str, set[str]] = {}
    for fact_id, content, entities_json in rows:
        tokens = tokenize(content or "")
        if entities_json:
            try:
                parsed = load_json_column(entities_json, None, column="atomic_facts.entities_json")
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                for ent in parsed:
                    tokens |= tokenize(ent if isinstance(ent, str) else str(ent))
        if tokens:
            out[str(fact_id)] = tokens
    return out


def _following_events(
    conn: sqlite3.Connection,
    session_id: str,
    profile_id: str,
    recalled_at: datetime,
) -> list[tuple[str, str]]:
    """(tool_name, payload) for actions in this conversation after the recall.

    Scoped to the conversation. Without that predicate a busy machine's
    unrelated activity in the same five minutes would read as engagement.
    """
    if not _table_exists(conn, "tool_events"):
        return []
    start = recalled_at.isoformat()
    end = (recalled_at + timedelta(seconds=OBSERVATION_WINDOW_SEC)).isoformat()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tool_events)")}
    except sqlite3.Error:
        return []
    if not {"session_id", "created_at", "tool_name"} <= cols:
        return []

    sql = (
        "SELECT tool_name, COALESCE(input_summary,'') || ' ' || "
        "COALESCE(output_summary,'') FROM tool_events "
        "WHERE session_id = ? AND created_at > ? AND created_at <= ?"
    )
    params: tuple = (session_id, start, end)
    if "profile_id" in cols:
        sql += " AND (profile_id = ? OR profile_id IS NULL)"
        params += (profile_id,)
    sql += " ORDER BY created_at LIMIT 200"
    try:
        return [(str(r[0]), str(r[1])) for r in conn.execute(sql, params)]
    except sqlite3.Error:
        return []


#: A single word in common is coincidence, not evidence. Any two texts about
#: software share one token eventually, and containment makes that worse: a
#: memory that reduces to one content word scores a perfect 1.0 against every
#: payload containing it. Two independent tokens is the cheapest threshold that
#: distinguishes a shared subject from a shared word.
_MIN_OVERLAP_TOKENS = 2


def _overlap(fact_tokens: set[str], payload_tokens: set[str]) -> float:
    """Containment of the memory in the action, not symmetric Jaccard.

    A tool payload is often far larger than a fact, and Jaccard would divide
    that signal away precisely when the evidence is strongest.
    """
    if not fact_tokens or not payload_tokens:
        return 0.0
    shared = fact_tokens & payload_tokens
    if len(shared) < _MIN_OVERLAP_TOKENS:
        return 0.0
    return len(shared) / len(fact_tokens)


def extract_features(
    memory_conn: sqlite3.Connection,
    *,
    session_id: str,
    profile_id: str,
    fact_ids: list[str],
    recalled_at: datetime,
    requeried: bool = False,
    marker_hit: bool = False,
) -> EngagementFeatures:
    """Observe what followed one recall. Never raises; returns empty on error."""
    features = EngagementFeatures(
        requeried=bool(requeried), marker_hit=bool(marker_hit),
    )
    try:
        by_fact = _fact_tokens(memory_conn, [str(f) for f in fact_ids])
        events = _following_events(memory_conn, session_id, profile_id, recalled_at)
    except sqlite3.Error:
        return features

    features.action_count = len(events)
    if not by_fact or not events:
        return features

    matched: set[str] = set()
    for tool_name, payload in events:
        payload_tokens = tokenize(payload)
        if not payload_tokens:
            continue
        for fact_id, tokens in by_fact.items():
            score = _overlap(tokens, payload_tokens)
            if score <= 0.0:
                continue
            matched.add(fact_id)
            features.peak_overlap = max(features.peak_overlap, score)
            if tool_name in _ARTIFACT_TOOLS:
                features.artifact_overlap = max(features.artifact_overlap, score)
            if tool_name.endswith(("remember", "update_memory")):
                features.followup_write_overlap = max(
                    features.followup_write_overlap, score,
                )
    features.matched_fact_ids = sorted(matched)
    return features
