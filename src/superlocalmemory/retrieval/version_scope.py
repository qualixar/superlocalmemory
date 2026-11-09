# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""A question that names a version is not answered by another version.

"SLM 4.2.15 release" and "SLM 4.2.0 released" are near neighbours to an
embedding, to the Hopfield channel and to the entity graph: same product, same
verb, digits that look alike. The BM25 channel already refuses a near-miss
version any credit (``bm25_channel._without_partial_version_credit``); every
other channel still ranked it as an answer. On the author's store (9 Nov 2026)
seven other-version release notes outranked current memories naming 4.2.15.

This is recall's ordering rule for that, applied twice: on the candidate pool
before it is cut to the caller's limit (``retrieval/engine``), and on the
final answer after learned ranking (``core/recall_pipeline``), which can
reorder. When the question names one or more dotted versions, a result whose text names only OTHER versions moves below
every result that does not, keeping the order inside each group. It never
drops a result, never raises one above another of its own group, and runs
before the exact-hit pin, so an exact lexical hit still comes first. Versions
are read with the store's one definition (``storage.fts_terms.version_terms``),
so "v4.2.15" is 4.2.15 and "4.2.1" or "4.2.150" are not.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from superlocalmemory.storage.fts_terms import version_terms


def names_only_other_versions(content: str, wanted: Iterable[str]) -> bool:
    """True when ``content`` names a version and none of ``wanted``."""
    named = version_terms(content or "")
    return bool(named) and not (set(named) & set(wanted))


def prefer_named_versions(results: Sequence, query: str) -> list:
    """``results`` with other-version-only results after the rest (stable)."""
    wanted = version_terms(query or "")
    if not wanted or not results:
        return results  # type: ignore[return-value]
    keep, later = [], []
    for result in results:
        content = getattr(getattr(result, "fact", None), "content", "") or ""
        (later if names_only_other_versions(content, wanted) else keep).append(result)
    if not later or not keep:
        return results  # type: ignore[return-value]
    return [*keep, *later]


def prefer_named_versions_in_pool(pool: Sequence, facts: dict, query: str) -> list:
    """The same rule on the candidate pool, BEFORE recall cuts it to the limit.

    Applied to the finished answer alone, the rule can only reorder what is
    already there: on the author's store the top 30 for "SLM 4.2.15 release"
    held no current memory naming 4.2.15 but the session summary, while three
    such memories existed. ``pool`` holds fusion results (``fact_id``); ``facts`` maps ids to
    loaded facts. Same stable partition, nothing dropped.
    """
    wanted = version_terms(query or "")
    if not wanted or not pool:
        return pool  # type: ignore[return-value]
    keep, later = [], []
    for item in pool:
        fact = facts.get(getattr(item, "fact_id", None))
        content = getattr(fact, "content", "") or ""
        (later if names_only_other_versions(content, wanted) else keep).append(item)
    if not later or not keep:
        return pool  # type: ignore[return-value]
    return [*keep, *later]


__all__ = ["names_only_other_versions", "prefer_named_versions", "prefer_named_versions_in_pool"]
