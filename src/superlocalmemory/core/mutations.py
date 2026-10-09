# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Canonical authorized fact delete and update operations."""

from __future__ import annotations

import logging
import sqlite3
import uuid
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


def _correction_ledger_available(engine: Any) -> bool:
    try:
        rows = engine._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='correction_cases'"
        )
        return bool(rows)
    except Exception:
        return False


def _propose_correction_case(
    engine: Any,
    *,
    profile_id: str,
    scope: str,
    predecessor_fact_id: str,
    successor_fact_id: str,
    trusted_actor_id: str,
    idempotency_key: str,
) -> dict[str, Any]:
    """Record a review-required M042 proposal after successor persistence.

    The ledger deliberately lives on a separate short transaction from the
    canonical writer.  A retry with the same deterministic identifiers is
    idempotent; the predecessor remains untouched if the ledger is unavailable.
    """
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        CorrectionCaseStore,
    )

    actor = CorrectionActor(
        actor_id=trusted_actor_id,
        actor_kind="host_authenticated",
        trust_tier="trusted",
    )
    case_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"slm-correction:{profile_id}:{predecessor_fact_id}:{successor_fact_id}",
    ).hex
    store = CorrectionCaseStore(
        engine._db.db_path,
        is_profile_active=lambda candidate: candidate == profile_id,
        is_actor_trusted=lambda candidate: candidate == actor,
    )
    case = store.propose(
        case_id=case_id,
        profile_id=profile_id,
        scope=scope,
        predecessor_fact_id=predecessor_fact_id,
        successor_fact_id=successor_fact_id,
        reason_code="direct_content_correction",
        actor=actor,
        idempotency_key=idempotency_key,
    )
    return {"case_id": case.case_id, "status": case.status, "version": case.version}


def _purge_delete_projections(
    engine: Any,
    fact_id: str,
    profile_id: str,
    *,
    memory_id: str | None = None,
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

    if memory_id:
        try:
            siblings = db.execute(
                "SELECT 1 FROM atomic_facts "
                "WHERE memory_id = ? AND profile_id = ? LIMIT 1",
                (memory_id, profile_id),
            )
            if not siblings:
                db.execute(
                    "DELETE FROM memories WHERE memory_id = ? AND profile_id = ?",
                    (memory_id, profile_id),
                )
        except Exception:
            logger.debug("memory delete by id skipped for %s", fact_id[:16])
    else:
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


def _canonical_present(engine: Any, fact_id: str, profile_id: str) -> bool:
    return bool(engine._db.execute(
        "SELECT 1 FROM atomic_facts WHERE fact_id = ? AND profile_id = ? LIMIT 1",
        (fact_id, profile_id),
    ))


def _memory_present(engine: Any, memory_id: str, profile_id: str) -> bool:
    return bool(engine._db.execute(
        "SELECT 1 FROM memories WHERE memory_id = ? AND profile_id = ? LIMIT 1",
        (memory_id, profile_id),
    ))


def _memory_has_facts(engine: Any, memory_id: str, profile_id: str) -> bool:
    return bool(engine._db.execute(
        "SELECT 1 FROM atomic_facts WHERE memory_id = ? AND profile_id = ? LIMIT 1",
        (memory_id, profile_id),
    ))


def _finalize_erasure(
    engine: Any,
    service: Any,
    op_ctx: Any,
    fact_id: str,
    profile_id: str,
    memory_id: str | None,
    erasure_id: str,
    requested_at: float,
    *,
    requested_by: str,
    tombstoned: bool,
) -> dict[str, Any]:
    try:
        from superlocalmemory.core.transactions import ErasureProofRecord
        from superlocalmemory.core.transactions.erasure import tombstone_memory_id

        canonical_absent = not _canonical_present(engine, fact_id, profile_id)
        extra = [ErasureProofRecord(
            owner="canonical", erased=canonical_absent, checksum="",
            residue=() if canonical_absent else (fact_id,),
        )]
        if memory_id:
            memory_ok = (
                not _memory_present(engine, memory_id, profile_id)
                or _memory_has_facts(engine, memory_id, profile_id)
            )
            extra.append(ErasureProofRecord(
                owner="memory", erased=memory_ok, checksum="",
                residue=() if memory_ok else (memory_id,),
            ))
        stored_mid = tombstone_memory_id(engine._db, profile_id, fact_id)
        if stored_mid and stored_mid != memory_id:
            stored_mid_ok = (
                not _memory_present(engine, stored_mid, profile_id)
                or _memory_has_facts(engine, stored_mid, profile_id)
            )
            if not stored_mid_ok:
                extra.append(ErasureProofRecord(
                    owner="memory_provenance", erased=False, checksum="",
                    residue=(stored_mid,),
                ))
        receipt = service.finalize(
            engine._db, op_ctx,
            subject_type="fact", subject_id=fact_id,
            requested_by=requested_by, requested_at=requested_at,
            extra_proofs=extra,
        )
        verified = receipt.all_erased and receipt.persisted and tombstoned
        if not verified:
            logger.error(
                "Erasure not verified for %s: state=%s persisted=%s "
                "tombstoned=%s residue=%s",
                fact_id[:16], receipt.state, receipt.persisted, tombstoned,
                [p.owner for p in receipt.proofs if not p.erased],
            )
        return {
            "erasure_id": erasure_id,
            "erasure_verified": verified,
            "erasure_state": receipt.state,
        }
    except Exception as exc:
        logger.error("Erasure finalize failed for %s: %s", fact_id[:16], exc)
        return {
            "erasure_id": erasure_id,
            "erasure_verified": False,
            "erasure_state": "FAILED",
        }


def _refuse_if_correction_protected(db: Any, profile_id: str, fact_id: str) -> None:
    """Refuse before the erasure removes a single projection (see module doc)."""
    from superlocalmemory.core.correction_protection import (
        blocking_cases,
        protection_message,
    )
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    cases = blocking_cases(db, profile_id, fact_id)
    if cases:
        raise CanonicalMutationConflict(protection_message(cases))


def _delete_canonical(
    engine: Any, fact_id: str, profile_id: str, content_preview: str, *,
    trusted_actor_id: str, canonical_runtime: Any | None, idempotency_key: str | None,
) -> str | None:
    """The last protection check and the fact delete, in one transaction.

    Returns the content preview, or None when the writer no longer found the
    fact. Raises ``CanonicalMutationConflict`` when the fact is protected.
    """
    if canonical_runtime is not None:
        result = dict(canonical_runtime.delete_fact(
            profile_id, fact_id, idempotency_key=idempotency_key,
        ))
        if not result.get("ok"):
            return None
        return content_preview or str(result.get("content_preview", ""))
    from superlocalmemory.core.overtaken_cases import cases_naming, overtake

    with engine._db.transaction():  # the delete and the cases it closes: one txn
        _refuse_if_correction_protected(engine._db, profile_id, fact_id)
        overtake(engine._db, cases_naming(engine._db, [fact_id]),
                 user_action="delete", actor_id=trusted_actor_id,
                 operation_id=f"delete:{fact_id}")
        engine._db.delete_fact(fact_id, profile_id=profile_id)
    return content_preview


def delete_fact_authorized(
    engine: Any,
    fact_id: str,
    *,
    trusted_actor_id: str,
    source_agent_id: str,
    canonical_runtime: Any | None = None,
    idempotency_key: str | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Authorize, delete one profile-owned fact, then emit post hooks.

    ``profile_id`` is the profile the fact belongs to; ``None`` = the active one.
    """
    import time as _time
    import uuid

    from superlocalmemory.core.transactions import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import (
        build_erasure_service,
    )
    from superlocalmemory.core.transactions.erasure import (
        is_tombstoned,
        tombstone_memory_id,
    )
    from superlocalmemory.storage.erasure_fence import clear_erasing, mark_erasing

    profile_id, context = _context(
        engine,
        "delete",
        fact_id,
        trusted_actor_id=trusted_actor_id,
        source_agent_id=source_agent_id,
        profile_id=profile_id,
    )

    rows = engine._db.execute(
        "SELECT content, memory_id FROM atomic_facts "
        "WHERE fact_id = ? AND profile_id = ? LIMIT 1",
        (fact_id, profile_id),
    )
    exists = bool(rows)
    content_preview = dict(rows[0]).get("content", "")[:80] if exists else ""
    memory_id = dict(rows[0]).get("memory_id") if exists else None
    from superlocalmemory.core import delete_refusal, erasure_scrub

    entity_ids: list[str] = []
    if exists:  # a refused delete also heals a stale tombstone (delete_refusal)
        delete_refusal.refuse_early(engine, profile_id, fact_id)
        entity_ids = erasure_scrub.entities_of(engine._db, profile_id, fact_id)
        erasure_scrub.prepare(engine._db)

    erasure_id = uuid.uuid4().hex
    requested_at = _time.time()
    op_ctx = OperationContext(
        operation_id=erasure_id, profile_id=profile_id,
        subject_id=fact_id, fact_ids=(fact_id,),
    )
    service = build_erasure_service(engine)

    # A tombstoned fact whose canonical row is already gone is a partially
    # completed prior erasure (e.g. a transient raw-memory delete failure left
    # the source memory behind). Resume cleanup from the stored memory_id
    # instead of reporting the fact "not found".
    resuming = False
    if not exists:
        with engine._db.raw_connection() as _conn:
            resuming = is_tombstoned(_conn, profile_id, fact_id)
        if not resuming:
            return {"ok": False, "error": f"Memory {fact_id} not found"}
        memory_id = tombstone_memory_id(engine._db, profile_id, fact_id)

    remove_result = None
    if exists:
        before_removal = delete_refusal.snapshot(engine, profile_id, fact_id)
        try:
            remove_result = service.remove(engine._db, op_ctx, memory_id=memory_id)
        except Exception as exc:
            logger.error("Erasure remove failed for %s: %s", fact_id[:16], exc)
        if (
            remove_result is None
            or not remove_result.spine_ok
            or not remove_result.tombstoned
        ):
            erasure = _finalize_erasure(
                engine, service, op_ctx, fact_id, profile_id, memory_id,
                erasure_id, requested_at, requested_by=trusted_actor_id,
                tombstoned=bool(remove_result and remove_result.tombstoned),
            )
            logger.error(
                "Erasure blocked (residue or unverified tombstone) for %s — "
                "canonical retained",
                fact_id[:16],
            )
            return {
                "ok": False,
                "error": "erasure incomplete: projection residue",
                "retryable": True,
                **erasure,
            }

    # Fence the fact for the purge window so a concurrent materializer cannot
    # re-write its projections after the durable tombstone is committed. The
    # tombstone remains the cross-process signal; this is the in-process guard.
    mark_erasing(profile_id, fact_id)
    try:
        if exists:
            from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

            try:
                content_preview = _delete_canonical(
                    engine, fact_id, profile_id, content_preview,
                    trusted_actor_id=trusted_actor_id,
                    canonical_runtime=canonical_runtime, idempotency_key=idempotency_key,
                )
            except CanonicalMutationConflict as refused:
                # Refused at the last check (a protecting case appeared after
                # the first): the fact stays, so its search entries come back.
                if delete_refusal.restore(engine, before_removal, erasure_id):
                    raise
                raise CanonicalMutationConflict(
                    delete_refusal.unrestored_message(str(refused))) from refused
            if content_preview is None:
                return {"ok": False, "error": f"Memory {fact_id} not found"}

        # Purge projections for a fresh delete and re-run (idempotently) for a
        # resumed cleanup so an orphaned source memory is reclaimed on retry.
        _purge_delete_projections(engine, fact_id, profile_id, memory_id=memory_id)
        try:  # the erased words outside the projections (core/erasure_scrub.py)
            erasure_scrub.scrub(engine._db, profile_id, fact_id, entity_ids)
        except Exception as exc:
            logger.error("Erasure scrub failed for %s: %s", fact_id[:16], exc)
            return {"ok": False, "retryable": True, "deleted": fact_id,
                    "error": "erasure incomplete: derived copies remain"}

        if exists:
            tombstoned = bool(remove_result and remove_result.tombstoned)
        else:
            with engine._db.raw_connection() as _conn:
                tombstoned = is_tombstoned(_conn, profile_id, fact_id)

        erasure = _finalize_erasure(
            engine, service, op_ctx, fact_id, profile_id, memory_id, erasure_id,
            requested_at, requested_by=trusted_actor_id, tombstoned=tombstoned,
        )

        engine._hooks.run_post("delete", context)
        # Never the erased words themselves: a log line outlives the erasure.
        logger.info(
            "DELETE fact_id=%s actor=%s source_agent=%s content_chars=%d",
            fact_id[:16], trusted_actor_id, source_agent_id, len(content_preview),
        )
        erasure_verified = bool(erasure.get("erasure_verified", False))
        if not erasure_verified:
            logger.error(
                "Erasure unverified for %s — returning retryable failure",
                fact_id[:16],
            )
            return {
                "ok": False,
                "retryable": True,
                "deleted": fact_id,
                "content_preview": content_preview,
                **erasure,
            }
        return {
            "ok": True,
            "deleted": fact_id,
            "content_preview": content_preview,
            **erasure,
        }
    finally:
        clear_erasing(profile_id, fact_id)


def update_fact_authorized(
    engine: Any,
    fact_id: str,
    content: str,
    *,
    trusted_actor_id: str,
    source_agent_id: str,
    canonical_runtime: Any | None = None,
    idempotency_key: str | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Create a review-required correction successor without rewriting history.

    ``profile_id`` is the profile the fact belongs to; ``None`` = the active one.
    """
    if not content or not content.strip():
        return {"ok": False, "error": "content cannot be empty"}
    from superlocalmemory.memory_core import prepare_user_text

    # The new text is prepared like any save, so the stored successor and every
    # event preview built from it hold no personal data when redaction is on.
    content = prepare_user_text(getattr(engine, "_config", None), content.strip()).text
    profile_id, context = _context(
        engine,
        "update",
        fact_id,
        trusted_actor_id=trusted_actor_id,
        source_agent_id=source_agent_id,
        content_preview=content,
        profile_id=profile_id,
    )
    rows = engine._db.execute(
        "SELECT content FROM atomic_facts "
        "WHERE fact_id = ? AND profile_id = ? LIMIT 1",
        (fact_id, profile_id),
    )
    if not rows:
        return {"ok": False, "error": f"Memory {fact_id} not found"}
    source = engine._db.get_fact(fact_id)
    if source is None:
        return {"ok": False, "error": f"Memory {fact_id} not found"}
    if source.content == content:
        return {"ok": True, "fact_id": fact_id, "content": content, "unchanged": True}
    if not _correction_ledger_available(engine):
        return {"ok": False, "error": "review-gated correction ledger is unavailable"}

    request_key = idempotency_key or uuid.uuid4().hex
    successor_fact_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"slm-correction-successor:{profile_id}:{fact_id}:{request_key}",
    ).hex[:16]
    embedding: list[float] | None = None
    fisher_mean: list[float] | None = None
    fisher_variance: list[float] | None = None
    if engine._embedder:
        try:
            embedding = engine._embedder.embed(content)
            if embedding:
                fisher_mean, fisher_variance = (
                    engine._embedder.compute_fisher_params(embedding)
                )
        except Exception as exc:
            logger.warning("Correction successor embedding refresh failed: %s", exc)
    if canonical_runtime is None:
        return {
            "ok": False,
            "retryable": True,
            "error": "canonical correction writer is temporarily unavailable",
        }
    result = dict(canonical_runtime.create_correction_successor(
        profile_id,
        fact_id,
        successor_fact_id,
        content,
        embedding=embedding,
        fisher_mean=fisher_mean,
        fisher_variance=fisher_variance,
        trusted_actor_id=trusted_actor_id,
        idempotency_key=request_key,
    ))
    if not result.get("ok"):
        return {"ok": False, "error": f"Memory {fact_id} not found"}
    successor_fact_id = str(result["successor_fact_id"])

    engine._hooks.run_post("update", context)
    logger.info(
        "CORRECTION_PROPOSED predecessor=%s successor=%s actor=%s source_agent=%s",
        fact_id[:16], successor_fact_id[:16], trusted_actor_id, source_agent_id,
    )
    return {
        "ok": True,
        "fact_id": successor_fact_id,
        "content": content,
        "predecessor_fact_id": fact_id,
        "successor_fact_id": successor_fact_id,
        "correction_case": {
            "case_id": result["case_id"],
            "status": result["status"],
            "version": result["version"],
        },
        "review_required": True,
    }


__all__ = ["delete_fact_authorized", "purge_profile_context_cache", "update_fact_authorized"]
