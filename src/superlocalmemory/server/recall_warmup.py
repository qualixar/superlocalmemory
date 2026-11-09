# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The daemon's start-up recalls: they warm caches, and nobody asked them.

v3.4.62: the first user query used to take 15-24 s because it read the graph
tables from disk; v3.8.5: the first FULL recall paid an 8-13 s cold cost for the
ranking model and graph metrics. These recalls pay both at boot instead.

Each holds its own short ``operation_nowait()`` lease, so a profile switch
issued at daemon start is never blocked behind them; a pending transition
preempts the remaining ones (they complete on the next boot).

They are the system's own recalls, so they never ask the answer check: no
verdict, nothing sent to a hosted provider, nothing billed, and no memory
leaves the machine because the daemon started (``skip_answer_check()``). The
on-device check's model is warmed separately — locally, without a recall.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from superlocalmemory.core.answer_check_scope import not_a_read, skip_answer_check

logger = logging.getLogger(__name__)

#: 4.1.22: what the warm-up is doing, for /health. ``ready`` only says the
#: daemon can serve; until this reads ``warm`` a recall may still find a cache
#: being built (and say so: an incomplete channel reports ``warming``).
#: pending -> indexes -> entity_graph -> recalls -> warm | failed
_state_lock = threading.Lock()
_state = {"phase": "pending"}
_WARMING_PHASES = ("pending", "indexes", "entity_graph", "recalls")


def _set_phase(phase: str) -> None:
    with _state_lock:
        _state["phase"] = phase


def begin(engine: Any) -> threading.Thread | None:
    """Daemon start, before the embedding model is loaded: build the graph now.

    The entity graph needs no model, so it starts at once instead of after the
    model load, and from here on a graph build never runs on a recall's clock
    (``adjacency_rcu.prefer_background_builds``): a recall that arrives first
    reports the channel ``warming`` instead of building it itself.
    """
    from superlocalmemory.retrieval import adjacency_rcu, entity_graph_warmup

    # The in-memory vector index decodes every stored vector: in a child
    # process for this daemon (retrieval/vector_index_build.py).
    retrieval = getattr(engine, "_retrieval_engine", engine)
    vectors = getattr(retrieval, "_kind_vectors", None)
    if vectors is not None and hasattr(vectors, "build_in_child"):
        vectors.build_in_child = True
    try:
        adjacency_rcu.prefer_background_builds(entity_graph_warmup.channel_of(engine))
        profile_id = str(getattr(engine, "profile_id", "") or "default")
        thread = threading.Thread(target=entity_graph_warmup.warm, args=(engine, profile_id),
                                  name="slm-graph-warmup", daemon=True)
        thread.start()
    except Exception as exc:  # noqa: BLE001 -- the warm-up builds it later instead
        logger.warning("early entity graph build not started (%s)", type(exc).__name__)
        return None
    with _state_lock:
        _state["graph_thread"] = thread
    return thread


def warmup_status() -> dict[str, Any]:
    """``{"phase": ..., "warm": bool, "warming": [what is still being built]}``."""
    with _state_lock:
        phase = _state["phase"]
    if phase in _WARMING_PHASES:
        still = list(_WARMING_PHASES[max(1, _WARMING_PHASES.index(phase)):])
    else:
        still = []
    return {"phase": phase, "warm": phase == "warm", "warming": still}

#: One to load the graph page cache, one to warm the reranker and producers.
WARMUP_QUERIES = ("memory recall performance", "context injection retrieval")
FULL_PATH_QUERY = "memory recall performance"


def run_warmup_recalls(engine: Any, profile_runtime: Any, *,
                       warm_spreading_activation: Callable[[Any, Any], Any]) -> None:
    """Fire the start-up recalls, then warm the on-device check's model."""
    try:
        with skip_answer_check(), not_a_read():
            _recalls(engine, profile_runtime, warm_spreading_activation)
    except BaseException:
        _set_phase("failed")
        raise
    _set_phase("warm")
    warm_answer_check(engine)


def _recalls(engine: Any, profile_runtime: Any,
             warm_spreading_activation: Callable[[Any, Any], Any]) -> None:
    # The in-memory vector and kind indexes first (retrieval/kind_scope): built
    # here, the warm-up recalls below and the first real one find them ready
    # instead of waiting behind the build past the channel guard.
    from superlocalmemory.retrieval import entity_graph_warmup, kind_scope

    profile_id = str(getattr(engine, "profile_id", "") or "default")
    _set_phase("indexes")
    kind_scope.warm(engine, profile_id)
    # The entity graph next, directly: built inside the first recall it cost
    # that recall ~6 s. It builds off the cache lock; a recall meanwhile reports
    # entity_graph ``warming`` instead of waiting (retrieval/adjacency_rcu).
    _set_phase("entity_graph")
    with _state_lock:
        early = _state.get("graph_thread")
    if early is not None:
        early.join()  # started by ``begin``; bounded by the build itself
    else:
        entity_graph_warmup.warm(engine, profile_id)
    # The answer-support gate reads the hub probe vectors; embed them now rather
    # than leave the first recalls to impute (retrieval/hub_baseline).
    warm_hub_probes(engine)
    _set_phase("recalls")
    for query in WARMUP_QUERIES:
        with profile_runtime.operation_nowait() as snapshot:
            if snapshot is None:
                logger.debug("Recall warmup preempted by profile transition "
                             "— skipping remaining warmup queries")
                break
            # Short lease, so a profile switch can drain within 5 s.
            engine.recall(query, limit=5, fast=True)
    # The fast recalls skip spreading activation; warm that channel directly so
    # the first FULL recall is not cold.
    warm_spreading_activation(engine, profile_runtime)
    # The fast recalls do not exercise the full ranking path or the agentic
    # round, so one full recall loads those at boot, not on a user's query.
    try:
        with profile_runtime.operation_nowait() as snapshot:
            if snapshot is not None:
                engine.recall(FULL_PATH_QUERY, limit=5, fast=False)
    except Exception as exc:  # noqa: BLE001 — best-effort; never blocks readiness
        logger.debug("Full-path warmup skipped (non-fatal): %s", exc)


def warm_hub_probes(engine: Any) -> bool:
    """Embed the hub probes for the engine's model (never raises)."""
    from superlocalmemory.retrieval import hub_baseline

    return hub_baseline.warm_probes(getattr(engine, "_embedder", None))


def warm_answer_check(engine: Any) -> bool:
    """Start loading the on-device check's model, without asking it anything.

    The check is built idle so a one-shot command never loads a model it will
    not use; the long-running daemon loads it here, at boot, so the first
    recall a person makes can be judged. A hosted check has nothing to warm.
    """
    retrieval = getattr(engine, "_retrieval_engine", None)
    judge = getattr(retrieval, "_sufficiency_judge", None) if retrieval else None
    start = getattr(judge, "start_warmup", None)
    if not callable(start):
        return False
    try:
        start()
    except Exception as exc:  # noqa: BLE001 — a judge never breaks start-up
        logger.debug("Answer check warm-up skipped: %s", type(exc).__name__)
        return False
    return True


__all__ = ["FULL_PATH_QUERY", "WARMUP_QUERIES", "run_warmup_recalls", "warm_answer_check",
           "warm_hub_probes", "begin", "warmup_status"]
