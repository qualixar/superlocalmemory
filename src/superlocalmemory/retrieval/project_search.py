# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A project-filtered recall searches inside the project, not only filters after.

WHY
---
``project`` narrows the fused candidate pool (``retrieval.project_scope``). The
pool is built without knowing the project, so a project's memory that closer
matches from other projects pushed below the pool was never a candidate: the
filter then fell back to unfiltered results and said "nothing found was saved
under this project", which was untrue, and a strict filter would have returned
nothing although the memory was there. The same defect, and the same remedy,
as kind-filtered recall (``retrieval.kind_scope``).

WHAT THIS DOES
--------------
Before fusion, the meaning and words searches run once more restricted to the
facts of memories saved under that project (``core.project_identity`` decides
what "the same project" means), and their hits are merged into the semantic and
BM25 channel lists (same ids keep their best score). The post-fusion project
filter still runs and still decides what is shown: this only makes sure the
memories it looks for are in front of it.

Members are read fresh for each recall that names a project (one indexed pass
over the profile's memories that carry a project); nothing is cached, so a
memory saved or moved a moment ago counts at once. Personal scope only, like
kind scoping: shared or global memories of other profiles are still found by
the ordinary channels and filtered after fusion. Never raises.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from superlocalmemory.core.project_identity import project_key
from superlocalmemory.retrieval.kind_scope import _merge, _Within

logger = logging.getLogger(__name__)

#: A project bigger than this is searched by the ordinary channels only: the
#: restricted searches would cost more than they can add.
MAX_MEMBERS = 50_000


def member_fact_ids(db: Any, profile_id: str, project: str) -> frozenset[str] | None:
    """Visible fact ids of ``profile_id``'s memories saved under ``project``,
    or None when they cannot be read."""
    want = project_key(project)
    if want is None:
        return frozenset()
    try:
        rows = db.execute(
            "SELECT af.fact_id AS fact_id, json_extract(m.metadata_json, '$.project') "
            "AS project FROM atomic_facts af JOIN memories m ON m.memory_id = af.memory_id "
            "WHERE af.profile_id = ? AND json_valid(m.metadata_json) "
            "AND json_extract(m.metadata_json, '$.project') IS NOT NULL"
            f"{db.visible_fact_clause('af')}",
            (profile_id,),
        )
        return frozenset(str(dict(r)["fact_id"]) for r in rows
                         if project_key(dict(r)["project"]) == want)
    except Exception as exc:  # noqa: BLE001 - reported; the filter still runs
        logger.warning("project members unreadable (%s)", type(exc).__name__)
        return None


def _within_clause(members: frozenset[str]) -> tuple[str, list[str]]:
    # One JSON parameter instead of one per id: no variable limit to hit.
    return " AND af.fact_id IN (SELECT value FROM json_each(?))", [
        json.dumps(sorted(members))]


def supplement(
    engine: Any, ch_results: dict[str, list[tuple[str, float]]], *, query: str,
    query_embedding: list[float] | None, profile_id: str, project: str,
    stage_ms: dict[str, float] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """``ch_results`` with in-project semantic and BM25 hits merged in."""
    t0 = time.monotonic()
    out = dict(ch_results)
    try:
        members = member_fact_ids(engine._db, profile_id, project)
        if not members or len(members) > MAX_MEMBERS:
            if members is not None and len(members) > MAX_MEMBERS:
                logger.info("project %r has %d facts; searched by the ordinary "
                            "channels only", project, len(members))
            return out
        semantic = engine._semantic
        index = getattr(engine, "_kind_vectors", None)
        if semantic is not None and query_embedding is not None and index is not None:
            import numpy as np

            q_vec = np.array(query_embedding, dtype=np.float32)
            hits = semantic._search_via_vector_store(
                query_embedding, q_vec, profile_id, engine._config.semantic_top_k,
                source=_Within(index, members),
            )
            if hits:
                out["semantic"] = _merge(out.get("semantic", []), hits)
        bm25 = engine._bm25
        if bm25 is not None and hasattr(bm25, "_fts5_search"):
            k = engine._config.bm25_top_k
            hits = bm25._fts5_search(query, profile_id, k,
                                     extra_where=_within_clause(members))
            hits = [h for h in hits if h[0] in members][:k]
            if hits:
                out["bm25"] = _merge(out.get("bm25", []), hits)
    except Exception as exc:  # noqa: BLE001 - fall back to the ordinary answer
        logger.warning("project-scoped search failed (%s); project=%r is filtered "
                       "after fusion only", type(exc).__name__, project)
        return dict(ch_results)
    finally:
        if stage_ms is not None:
            stage_ms["project_scope"] = round((time.monotonic() - t0) * 1000.0, 1)
    return out


__all__ = ["MAX_MEMBERS", "member_fact_ids", "supplement"]
