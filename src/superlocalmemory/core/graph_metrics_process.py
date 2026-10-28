# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The daemon's graph metrics, computed in a separate low-priority process.

WHY
---
PageRank and the Louvain partition over a whole store are pure-Python graph
work: on a copy of a 22k-fact store (480k edges) they held the interpreter for
seconds and set off garbage collections of 75-163 ms every few hundred
milliseconds. The daemon runs the pass shortly after every start that follows
new memories (``maintenance_scheduler._initial_graph_metrics``) and on its
maintenance cycle, and recalls that landed meanwhile took 5-9 s. Waiting for
"no recall in flight" would not help: a recall arriving mid-pass still waits.

WHAT THIS DOES
--------------
A spawned child process (``nice`` +10) opens the store read-only, reads the
graph and computes the rows with the same functions the in-process pass uses
(``graph_metrics.read_graph``, ``_networkx_metrics``, ``metrics_rows``), and
hands the rows back. The daemon writes them through the usual chunked writer
(``graph_metrics._write``): one writer, short transactions, the same rows. The
child never writes, so the store keeps a single writing process.

A child that fails or exceeds ``TIMEOUT_SECONDS`` is reported as an error (the
metrics stay as they were and the next cycle retries); it is never re-run on
the daemon's own interpreter, which is the stall this exists to remove.
"""

from __future__ import annotations

import concurrent.futures
import sqlite3
from typing import Any

from superlocalmemory.core.background_process import NICE_INCREMENT, run_in_child

#: Generous: the 22k-fact store takes seconds; a pathological one must not hang a cycle.
TIMEOUT_SECONDS = 900.0


def child_compute(db_path: str, profile_id: str, damping: float) -> dict[str, Any]:
    """Runs in the child: read-only snapshot in, ``fact_importance`` rows out."""
    from superlocalmemory.core import graph_metrics as gm

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    try:
        conn.execute("PRAGMA query_only=ON")
        nodes, edges = gm.read_graph(conn, profile_id)
    finally:
        conn.close()
    if not nodes:
        return {"empty": True, "engine": "none", "notes": ["no visible facts"],
                "facts": 0, "edges": len(edges)}
    pagerank, communities = gm._networkx_metrics(edges, damping)
    computed = gm.metrics_rows(profile_id, nodes, edges, damping, pagerank, communities)
    computed["engine"] = "networkx"
    computed["notes"] = ["computed in a separate process", *computed["notes"]]
    return computed


#: The watchdog's per-worker default (``SLM_MAX_WORKER_MB``); the helper never gets less.
_DEFAULT_LIMIT_MB = 2500
#: networkx holds the whole graph: the helper reached 2,927 MB on 3.6 M edges when
#: the watchdog killed it (#304), so this allows ~1.2 KB per edge over a fixed base.
_BASE_MB = 800
_MB_PER_EDGE = 0.0012


def _memory_budget_mb() -> int:
    """The daemon's whole-process memory budget: 40% of RAM (``health_monitor``)."""
    try:
        import psutil

        return max(_DEFAULT_LIMIT_MB, int(psutil.virtual_memory().total / 1048576 * 0.40))
    except Exception:  # noqa: BLE001 -- the same fallback the health monitor uses
        return 8000


def helper_limit_mb(edge_count: int) -> int:
    """The helper's memory limit for a graph this size, within the daemon's budget (#304)."""
    import os

    try:
        default = int(os.environ.get("SLM_MAX_WORKER_MB", _DEFAULT_LIMIT_MB))
    except ValueError:
        default = _DEFAULT_LIMIT_MB
    wanted = int(_BASE_MB + max(0, edge_count) * _MB_PER_EDGE)
    return max(default, min(wanted, _memory_budget_mb()))


def _edge_count(db: Any, profile_id: str) -> int:
    try:
        rows = db.execute("SELECT COUNT(*) AS n FROM graph_edges WHERE profile_id = ?",
                          (profile_id,))
        return int(dict(rows[0])["n"]) if rows else 0
    except Exception:  # noqa: BLE001 -- unknown size keeps the default limit
        return 0


def compute_rows_in_child(db: Any, profile_id: str, damping: float) -> dict[str, Any]:
    """The child's result, or ``{"error": ...}``. Never computes in this process."""
    db_path = getattr(db, "db_path", None)
    if db_path is None:
        return {"error": "no store path to compute from", "engine": "networkx"}
    try:
        return run_in_child(child_compute, str(db_path), profile_id, float(damping),
                            timeout=TIMEOUT_SECONDS,
                            memory_limit_mb=helper_limit_mb(_edge_count(db, profile_id)))
    except (concurrent.futures.TimeoutError, TimeoutError):
        return {"error": f"graph metrics process exceeded {TIMEOUT_SECONDS:.0f}s",
                "engine": "networkx"}
    except Exception as exc:  # noqa: BLE001 -- reported, retried next cycle
        return {"error": f"graph metrics process failed: {type(exc).__name__}: {exc}",
                "engine": "networkx"}


__all__ = ["NICE_INCREMENT", "TIMEOUT_SECONDS", "child_compute", "compute_rows_in_child",
           "helper_limit_mb"]
