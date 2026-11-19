# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Projections and context caches that follow an authorized fact delete or update.

Split out of ``core/mutations.py`` (final audit F-15, 19 Nov 2026) so that module stays under
800 lines. ``core/mutations`` re-exports every name here, so callers and tests that use
``superlocalmemory.core.mutations.<name>`` keep working unchanged.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger("superlocalmemory.audit")


def _context(
    engine: Any,
    operation: str,
    fact_id: str,
    *,
    trusted_actor_id: str,
    source_agent_id: str,
    content_preview: str = "",
    profile_id: str | None = None,
) -> tuple[str, dict[str, str]]:
    if not trusted_actor_id:
        raise ValueError("trusted actor identity is required")
    # A routed caller names the profile (already authorized on it); every
    # other caller works on the engine's active profile.
    profile_id = profile_id or engine.profile_id
    context = {
        "operation": operation,
        "agent_id": trusted_actor_id,
        "source_agent_id": source_agent_id,
        "profile_id": profile_id,
        "fact_id": fact_id,
    }
    if content_preview:
        context["content_preview"] = content_preview[:100]
    engine._hooks.run_pre(operation, context)
    return profile_id, context


def _invalidate_context_cache_for_fact(
    engine: Any,
    fact_id: str,
) -> None:
    """Best-effort removal of context cache entries referencing fact_id.

    Opens the cache SQLite file directly (no ContextCache instance needed).
    Tries the two standard cache locations relative to the database directory:
    a ``context-cache`` subdirectory (isolation convention used in tests) and
    the database directory itself (production default). Fail-open: any error
    is silently discarded.
    """
    try:
        db_parent = Path(engine._db.db_path).parent
    except AttributeError:
        return
    pattern = f'%"{fact_id}"%'
    candidates = [
        db_parent / "context-cache" / "active_brain_cache.db",
        db_parent / "active_brain_cache.db",
    ]
    for cache_path in candidates:
        if not cache_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(cache_path), timeout=2.0)
            try:
                conn.execute(
                    "DELETE FROM context_entries WHERE fact_ids LIKE ?",
                    (pattern,),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass


def purge_profile_context_cache(engine: Any, profile_id: str) -> None:
    """Best-effort full-profile cache purge after a truth lifecycle change.

    Apply and rollback can flip which of two immutable facts is current.  A
    fact-ID substring delete is therefore insufficient: any cached prose for
    the profile could contain the old current truth.  Read-side admission
    remains the authoritative backstop if this post-commit cleanup is busy.
    """
    try:
        db_parent = Path(engine._db.db_path).parent
    except AttributeError:
        return
    try:
        from superlocalmemory.core.context_cache import purge_profile_from_cache_db

        for cache_path in (
            db_parent / "context-cache" / "active_brain_cache.db",
            db_parent / "active_brain_cache.db",
        ):
            purge_profile_from_cache_db(cache_path, profile_id)
    except Exception:
        logger.warning("Profile context cache purge failed after correction transition")


def _sync_vector_ann(
    retrieval: Any,
    fact_id: str,
    profile_id: str,
    embedding: list[float] | None,
    *,
    operation: str,
) -> None:
    """Replace or remove a fact's entry in the vector store and ANN index.

    ``operation`` is either ``"update"`` or ``"delete"``. For updates,
    ``embedding`` must be provided; the semantic representations are replaced
    in place so a correction is reflected by every semantic query. Fail-open:
    errors are logged and ignored so a projection hiccup never blocks the
    authoritative write.
    """
    vector_store = getattr(retrieval, "_vector_store", None)
    ann_index = getattr(retrieval, "_ann_index", None)
    if operation == "update" and embedding:
        if vector_store is not None:
            try:
                # upsert replaces the stored vector for an existing fact_id.
                vector_store.upsert(fact_id, profile_id, embedding)
            except Exception as exc:
                logger.warning("vector_store upsert failed for %s: %s", fact_id[:16], exc)
        if ann_index is not None:
            try:
                # add() is upsert-semantic: it overwrites an existing entry.
                ann_index.add(fact_id, embedding)
            except Exception as exc:
                logger.warning("ann_index add failed for %s: %s", fact_id[:16], exc)
    elif operation == "delete":
        if vector_store is not None:
            try:
                from superlocalmemory.retrieval.vector_store import classify_delete

                if classify_delete(vector_store.delete(fact_id)).failed:
                    # The erasure receipt (VectorOwner) reports the residue; say so here too.
                    logger.warning(
                        "vector_store.delete failed for %s: a vector may remain", fact_id[:16])
            except Exception as exc:
                logger.warning("vector_store.delete failed for %s: %s", fact_id[:16], exc)
        if ann_index is not None:
            try:
                ann_index.remove(fact_id)
            except Exception as exc:
                logger.warning("ann_index.remove failed for %s: %s", fact_id[:16], exc)


def _converge_update_projections(
    engine: Any,
    fact_id: str,
    content: str,
    profile_id: str,
    embedding: list[float] | None,
) -> None:
    """Fan out a fact correction to every derived projection.

    Projection obligations on correction:
    - temporal_events.description — updated to new content
    - fact_context.contextual_description — cleared of stale tokens (row kept)
    - fact_expansion_fts.alt_keys — cleared of stale tokens (row kept)
    - context cache — entry deleted so stale content is never served
    - BM25 live channel — replaced (one entry, no duplicate)
    - vector store + ANN index — update called
    """
    db = engine._db

    # Temporal events: description must match new content
    db.update_temporal_event_description(fact_id, content)

    # Fact context: keep row alive but clear stale description/keywords
    db.store_fact_context(fact_id, profile_id, "", "")

    # Expansion FTS: keep row alive but replace with empty alt_keys
    db.reset_fact_expansion(fact_id, "")

    # Context cache: delete stale entries so the hot path returns nothing stale
    _invalidate_context_cache_for_fact(engine, fact_id)

    # BM25 live channel: replace (remove then add) to guarantee exactly one entry
    retrieval = getattr(engine, "_retrieval_engine", None)
    bm25 = getattr(retrieval, "_bm25", None) if retrieval else None
    if bm25 is not None and hasattr(bm25, "update_fact"):
        bm25.update_fact(fact_id, content, profile_id)
    elif bm25 is not None:
        # Fallback: add without dedup (legacy path)
        bm25.add(fact_id, content, profile_id)

    # Vector store + ANN index
    if retrieval is not None:
        _sync_vector_ann(retrieval, fact_id, profile_id, embedding, operation="update")

    try:
        from superlocalmemory.core.backend_orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        if orchestrator is not None:
            orchestrator.sync_changed_fact(fact_id)
    except Exception:
        logger.warning("Derived projection update sync failed for %s", fact_id[:16])


def _project_correction_successor(
    engine: Any,
    fact_id: str,
    content: str,
    profile_id: str,
    embedding: list[float] | None,
) -> None:
    """Project a new successor without touching predecessor projections."""
    retrieval = getattr(engine, "_retrieval_engine", None)
    bm25 = getattr(retrieval, "_bm25", None) if retrieval else None
    if bm25 is not None:
        try:
            bm25.add(fact_id, content, profile_id)
        except Exception as exc:
            logger.warning("BM25 successor projection failed for %s: %s", fact_id[:16], exc)
    if retrieval is not None:
        _sync_vector_ann(retrieval, fact_id, profile_id, embedding, operation="update")
