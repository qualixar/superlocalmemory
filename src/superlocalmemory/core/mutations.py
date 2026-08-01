# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Canonical authorized fact delete and update operations."""

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
) -> tuple[str, dict[str, str]]:
    if not trusted_actor_id:
        raise ValueError("trusted actor identity is required")
    profile_id = engine.profile_id
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
                vector_store.delete(fact_id)
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


def _purge_delete_projections(
    engine: Any,
    fact_id: str,
    profile_id: str,
) -> None:
    """Purge every derived representation of a fact before the canonical row is gone.

    Projection obligations on deletion:
    - memories (raw memory record that sourced this fact)
    - bm25_tokens (persisted tokens in the DB table)
    - fact_expansion_fts (BM25 alt-key expansion entry)
    - graph_edges (any edge where fact_id is source or target)
    - memory_scenes (fact_id removed from JSON arrays; empty scenes deleted)
    - fact_context (contextual description entry)
    - context cache (entries referencing this fact_id)
    - BM25 live channel (evicted from in-memory corpus)
    - vector store + ANN index (delete / remove called)
    """
    db = engine._db

    # Raw memory — must be read before the atomic_facts row disappears
    db.delete_memory_for_fact(fact_id, profile_id)

    # Persistent BM25 tokens
    db.delete_bm25_tokens_for_fact(fact_id)

    # FTS expansion entry
    db.upsert_fact_expansion(fact_id, "")

    # Knowledge graph edges
    db.delete_graph_edges_for_fact(fact_id)

    # Memory scenes — remove fact_id from JSON arrays
    db.remove_fact_from_scenes(fact_id, profile_id)

    # Fact context description
    db.delete_fact_context(fact_id)

    # Context cache
    _invalidate_context_cache_for_fact(engine, fact_id)

    # BM25 live channel (in-memory eviction)
    retrieval = getattr(engine, "_retrieval_engine", None)
    bm25 = getattr(retrieval, "_bm25", None) if retrieval else None
    if bm25 is not None and hasattr(bm25, "remove_fact"):
        bm25.remove_fact(fact_id)

    # Vector store + ANN index
    if retrieval is not None:
        _sync_vector_ann(retrieval, fact_id, profile_id, None, operation="delete")

    try:
        from superlocalmemory.core.backend_orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        if orchestrator is not None:
            orchestrator.sync_deleted_fact(fact_id)
    except Exception:
        logger.warning("Derived projection deletion sync failed for %s", fact_id[:16])


def delete_fact_authorized(
    engine: Any,
    fact_id: str,
    *,
    trusted_actor_id: str,
    source_agent_id: str,
    canonical_runtime: Any | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Authorize, delete one profile-owned fact, then emit post hooks."""
    profile_id, context = _context(
        engine,
        "delete",
        fact_id,
        trusted_actor_id=trusted_actor_id,
        source_agent_id=source_agent_id,
    )
    if canonical_runtime is not None:
        result = dict(canonical_runtime.delete_fact(
            profile_id, fact_id, idempotency_key=idempotency_key,
        ))
        if not result.get("ok"):
            return {"ok": False, "error": f"Memory {fact_id} not found"}
        content_preview = str(result.get("content_preview", ""))
    else:
        rows = engine._db.execute(
            "SELECT content FROM atomic_facts "
            "WHERE fact_id = ? AND profile_id = ? LIMIT 1",
            (fact_id, profile_id),
        )
        if not rows:
            return {"ok": False, "error": f"Memory {fact_id} not found"}
        content_preview = dict(rows[0]).get("content", "")[:80]

        # Purge every derived projection before the canonical row is deleted
        _purge_delete_projections(engine, fact_id, profile_id)

        # Delete canonical fact (handles atomic_facts + embedding_metadata)
        engine._db.delete_fact(fact_id, profile_id=profile_id)

    engine._hooks.run_post("delete", context)
    logger.info(
        "DELETE fact_id=%s actor=%s source_agent=%s content=%s",
        fact_id[:16], trusted_actor_id, source_agent_id, content_preview,
    )
    return {
        "ok": True,
        "deleted": fact_id,
        "content_preview": content_preview,
    }


def update_fact_authorized(
    engine: Any,
    fact_id: str,
    content: str,
    *,
    trusted_actor_id: str,
    source_agent_id: str,
    canonical_runtime: Any | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Authorize a fact correction and converge every derived projection."""
    if not content or not content.strip():
        return {"ok": False, "error": "content cannot be empty"}
    content = content.strip()
    profile_id, context = _context(
        engine,
        "update",
        fact_id,
        trusted_actor_id=trusted_actor_id,
        source_agent_id=source_agent_id,
        content_preview=content,
    )
    rows = engine._db.execute(
        "SELECT content FROM atomic_facts "
        "WHERE fact_id = ? AND profile_id = ? LIMIT 1",
        (fact_id, profile_id),
    )
    if not rows:
        return {"ok": False, "error": f"Memory {fact_id} not found"}
    old_content = dict(rows[0]).get("content", "")[:80]
    updates: dict[str, Any] = {"content": content}
    embedding: list[float] | None = None
    if engine._embedder:
        try:
            embedding = engine._embedder.embed(content)
            if embedding:
                updates["embedding"] = embedding
                fisher_mean, fisher_variance = (
                    engine._embedder.compute_fisher_params(embedding)
                )
                updates["fisher_mean"] = fisher_mean
                updates["fisher_variance"] = fisher_variance
        except Exception as exc:
            logger.warning("UPDATE embedding refresh failed: %s", exc)
    if canonical_runtime is not None:
        result = dict(canonical_runtime.update_fact(
            profile_id,
            fact_id,
            updates,
            idempotency_key=idempotency_key,
        ))
        if not result.get("ok"):
            return {"ok": False, "error": f"Memory {fact_id} not found"}
    else:
        engine._db.update_fact(fact_id, updates, profile_id=profile_id)

    # Converge every derived projection to the corrected content
    _converge_update_projections(engine, fact_id, content, profile_id, embedding)

    engine._hooks.run_post("update", context)
    logger.info(
        "UPDATE fact_id=%s actor=%s source_agent=%s old=%s new=%s",
        fact_id[:16], trusted_actor_id, source_agent_id, old_content, content[:80],
    )
    return {"ok": True, "fact_id": fact_id, "content": content}


__all__ = ["delete_fact_authorized", "update_fact_authorized"]
