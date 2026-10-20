# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Notice when the graph copy has drifted from the store, and rebuild it.

The Cozo graph is a copy of SQLite's logical edges (``storage/logical_edges``),
kept in step by the projection queue. A queue cannot repair what it never saw:
an edge an older version failed to remove, or one whose fact has since been
hard-deleted, stays in the copy for good, because nothing will ever queue it
again. Issue #296 is that state at scale -- 4.3 M edges in the copy against
2.9 M in the store, on a store promoted while empty and never rebuilt.

So the daemon checks. A copy whose queue is drained and whose edge count still
differs from the store's, on a second look a moment later, is rebuilt for that
profile from SQLite (``BackendOrchestrator.rebuild_graph_profile``). Recall
reads SQLite while it runs. A copy that is still catching up is left alone:
the queue is the cheaper repair and it is already running.

Only the daemon runs this. It owns the Cozo store (RocksDB takes an exclusive
process lock), so ``slm db regraph`` asks it over HTTP.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["GraphCheck", "check_graph_projection", "regraph"]

#: How long to wait before looking again, so a write landing between the two
#: counts is not mistaken for drift.
SETTLE_S = 2.0


@dataclass(frozen=True)
class GraphCheck:
    profile_id: str
    stored: int = 0
    copied: int = 0
    pending: int = 0
    rebuilt: bool = False
    after: int | None = None
    skipped: str = ""
    error: str = ""

    @property
    def in_step(self) -> bool:
        return not self.skipped and not self.error and self.stored == self.copied

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id, "stored": self.stored, "copied": self.copied,
            "pending": self.pending, "in_step": self.in_step, "rebuilt": self.rebuilt,
            "after": self.after, "skipped": self.skipped, "error": self.error,
        }


def _counts(db: Any, cozo: Any, profile_id: str) -> tuple[int, int, int]:
    """(edges in the store, edges in the copy, facts still queued) for one profile."""
    from superlocalmemory.storage.logical_edges import count_logical_edges

    with db.raw_connection() as conn:
        stored = count_logical_edges(conn, profile_id)
    rows = cozo._db.run(
        "?[count(a)] := *edge{from_id: a, profile_id: $pid}", {"pid": profile_id},
    )
    copied = int(rows.values.tolist()[0][0]) if len(rows) else 0
    pending_rows = db.execute(
        "SELECT COUNT(*) AS c FROM projection_outbox WHERE profile_id = ?", (profile_id,),
    )
    pending = int(dict(pending_rows[0])["c"]) if pending_rows else 0
    return stored, copied, pending


def check_graph_projection(
    db: Any,
    orchestrator: Any,
    profile_id: str,
    *,
    heal: bool,
    settle_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> GraphCheck:
    """Compare one profile's graph copy with the store; with ``heal``, rebuild a drifted one."""
    cozo = orchestrator.graph_backend_for_check() if orchestrator is not None else None
    if cozo is None:
        return GraphCheck(profile_id, skipped="no graph copy on this store")
    try:
        stored, copied, pending = _counts(db, cozo, profile_id)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return GraphCheck(profile_id, error=f"could not count: {exc}")
    report = GraphCheck(profile_id, stored, copied, pending)
    if stored == copied or not heal:
        return report
    if pending:
        return GraphCheck(profile_id, stored, copied, pending,
                          skipped="the projection queue is still catching up")
    sleep(SETTLE_S if settle_s is None else settle_s)
    try:
        stored, copied, pending = _counts(db, cozo, profile_id)
    except Exception as exc:  # noqa: BLE001
        return GraphCheck(profile_id, error=f"could not count: {exc}")
    if stored == copied or pending:
        return GraphCheck(profile_id, stored, copied, pending)
    logger.warning(
        "graph copy for %s has drifted (%d edges in the store, %d in the copy); "
        "rebuilding it from the store", profile_id, stored, copied,
    )
    try:
        orchestrator.rebuild_graph_profile(profile_id)
        _stored, after, _pending = _counts(db, cozo, profile_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("graph copy rebuild for %s failed: %s", profile_id, exc)
        return GraphCheck(profile_id, stored, copied, pending, error=str(exc))
    logger.info("graph copy for %s rebuilt: %d edges", profile_id, after)
    return GraphCheck(profile_id, stored, copied, pending, rebuilt=True, after=after)


def regraph(db: Any, orchestrator: Any, *, profile: str = "", heal: bool,
            settle_s: float | None = None) -> list[dict[str, Any]]:
    """``check_graph_projection`` for one named profile, or every profile with memories.

    Shared by the daemon's ``/maintenance/regraph`` and the offline ``slm db regraph``.
    """
    rows = db.execute("SELECT DISTINCT profile_id FROM atomic_facts ORDER BY profile_id")
    profiles = [str(dict(r)["profile_id"]) for r in rows]
    if profile:
        profiles = [p for p in profiles if p == profile]
    return [
        check_graph_projection(db, orchestrator, p, heal=heal, settle_s=settle_s).as_dict()
        for p in profiles
    ]
