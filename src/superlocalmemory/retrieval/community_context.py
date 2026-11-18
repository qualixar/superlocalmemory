# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Thematic community context attached to a recall (Wave Q2b).

Extracted out of ``retrieval/engine.py`` (Q9, 2026-10-06) rather than grown
in place: that module is already over the project's 800-line file cap, so
new logic goes in its own small module instead
(``RetrievalEngine._community_context`` is now a thin delegator — see that
docstring for the gating this module does not own).

On-device-safe (market CRIT-1): a single read of the ≤N precomputed
``community_summaries`` rows + a membership tally — never a per-query LLM
fan-out.

Q9: ``fact_ids_json`` legitimately holds every member of the community (a few
thousand on a large store) — written once at compute time for durable
drill-down (see ``core/community_summary.py``, ``storage/schema.py``).
Echoing the whole array into every recall/session_init response made a
single field 73% of a 104KB ``session_init`` payload on a 3,816-member
community. ``build_community_context`` returns a bounded, query-relevant
SAMPLE (the ids this query's own top results actually matched) plus an
honest ``member_count`` and ``member_fact_ids_truncated`` flag. The full
list stays reachable via ``get_memory_summary(kind="community")`` (explicit
drill-down, not on the recall hot path).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from superlocalmemory.storage.erasure_visibility import aggregate_erasure_clause_for_db

logger = logging.getLogger(__name__)

#: ``thematic_context.member_fact_ids`` is a relevance sample, not the
#: membership list. Full membership is reachable through
#: ``get_memory_summary(kind="community")``.
MAX_MEMBER_SAMPLE = 10


def _tombstoned(db: Any, profile_id: str) -> frozenset[str]:
    """Fact ids of this profile that are being erased (none when the store has no tombstones)."""
    if not aggregate_erasure_clause_for_db(db, "community_summaries"):
        return frozenset()  # the manager's cached probe: the table is absent
    return frozenset(
        str(dict(r)["fact_id"]) for r in db.execute(
            "SELECT fact_id FROM projection_tombstones WHERE profile_id = ?", (profile_id,)))


def build_community_context(
    db: Any, results: list[Any], profile_id: str, top_k: int = 8,
) -> dict | None:
    """The precomputed community summary the top ``results`` cluster in.

    Gated: fires only when >=2 of the top results AND >=40% of them belong
    to one community, so precise factual queries are untouched. Returns
    ``None`` on any error or when the gate does not clear — the caller
    treats both as "no thematic context for this recall", never as a
    failure.
    """
    rows = [
        dict(r) for r in db.execute(
            "SELECT community_id, summary, keywords, fact_ids_json, "
            "fact_count FROM community_summaries WHERE profile_id = ?",
            (profile_id,),
        )
    ]
    if not rows:
        return None
    # A summary written from a member that is being erased is withheld whole. The check used
    # to be SQL: a json_each over every member of every summary, probed against the
    # tombstones (WS-7: 88 ms of every recall on a 44k-fact store). The tombstones are few
    # and this function parses the member lists anyway, so the check is a set lookup here.
    # Read after the summaries: a tombstone written in between still withholds.
    tombstoned = _tombstoned(db, profile_id)

    fact_to_cid: dict[str, int] = {}
    summ_by_cid: dict[int, dict] = {}
    for r in rows:
        cid = int(r["community_id"])
        try:
            members = [str(fid) for fid in json.loads(r.get("fact_ids_json") or "[]")]
        except (ValueError, TypeError):
            continue
        if tombstoned and not tombstoned.isdisjoint(members):
            continue
        summ_by_cid[cid] = r
        for fid in members:
            fact_to_cid[fid] = cid

    top_ids = [
        res.fact.fact_id
        for res in results[:top_k]
        if getattr(res, "fact", None) is not None
    ]
    tally = Counter(fact_to_cid[fid] for fid in top_ids if fid in fact_to_cid)
    if not tally:
        return None
    best_cid, count = tally.most_common(1)[0]
    coverage = count / len(top_ids) if top_ids else 0.0
    if count < 2 or coverage < 0.4:
        return None

    row = summ_by_cid[best_cid]
    from superlocalmemory.retrieval import visibility

    if not visibility.is_empty():
        # The summary was written from every member, so one the caller may not
        # see taints it: say nothing rather than echo what it drew from.
        try:
            members = [str(f) for f in json.loads(row.get("fact_ids_json") or "[]")]
        except (ValueError, TypeError):
            return None
        if visibility.hidden_among(db, profile_id, members):
            return None
    member_count = int(row.get("fact_count") or 0)
    sample = [fid for fid in top_ids if fact_to_cid.get(fid) == best_cid]
    sample = sample[:MAX_MEMBER_SAMPLE]
    if not member_count:
        # fact_count missing on an older row — the one time a full reparse
        # is worth it, since sample size alone understates it.
        try:
            member_count = len(json.loads(row.get("fact_ids_json") or "[]"))
        except (ValueError, TypeError):
            member_count = len(sample)
    return {
        "community_id": best_cid,
        "summary": row.get("summary", ""),
        "keywords": row.get("keywords", ""),
        "member_fact_ids": sample,
        "member_count": member_count,
        "member_fact_ids_truncated": member_count > len(sample),
        "coverage": round(coverage, 3),
        "matched_results": count,
    }


__all__ = ["MAX_MEMBER_SAMPLE", "build_community_context"]
