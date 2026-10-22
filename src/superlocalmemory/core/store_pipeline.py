# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Store pipeline — extracted free functions for MemoryEngine.store().

Direction: engine.py imports this module. This module NEVER imports engine.py.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.hooks import HookRegistry
    from superlocalmemory.storage.database import DatabaseManager

from superlocalmemory.core.receipt_guard import keeps_own_fact
from superlocalmemory.encoding.entity_validity import valid_names
from superlocalmemory.storage.erasure_fence import is_erasing
from superlocalmemory.storage.models import (
    AtomicFact,
    MemoryRecord,
)

logger = logging.getLogger(__name__)

# Langevin initialization radius for new facts (ACTIVE zone < 0.3)
_INIT_LANGEVIN_RADIUS = 0.05


def _reraise_materialization_deferral(exc: Exception) -> None:
    """Keep explicit runtime preemption out of best-effort fallbacks."""
    from superlocalmemory.core.materialization_control import MaterializationDeferred

    if isinstance(exc, MaterializationDeferred):
        raise exc


def _ingestion_effect_id(operation_id: str, *parts: object) -> str:
    """Return a stable ID for a relational effect owned by one ingestion."""
    if not operation_id:
        return uuid.uuid4().hex
    payload = "\0".join((operation_id, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _record_correction_candidate(
    db: DatabaseManager,
    *,
    operation_id: str,
    profile_id: str,
    scope: str,
    predecessor_fact_id: str,
    successor_fact_id: str,
    reason_code: str,
    trusted_actor_id: str,
) -> None:
    """Append a review candidate through the current canonical transaction.

    This intentionally carries identifiers and a controlled reason code only.
    It must not use detector prose because that can contain user memory text.
    If the current path is bound to the canonical coordinator,
    ``raw_connection`` yields its already-open transaction; otherwise the
    database manager owns the short transaction. Either path is atomic.
    """
    if not trusted_actor_id or not predecessor_fact_id or not successor_fact_id:
        return
    if predecessor_fact_id == successor_fact_id:
        return
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        CorrectionCaseError,
        open_case_id,
        propose_on_connection,
    )

    case_id = _ingestion_effect_id(
        operation_id, "correction-case", profile_id, predecessor_fact_id,
        successor_fact_id, reason_code,
    )
    idempotency_key = _ingestion_effect_id(
        operation_id, "correction-proposal", profile_id, predecessor_fact_id,
        successor_fact_id, reason_code,
    )
    from superlocalmemory.core.remember_replaces import ledger_actor_id

    actor = CorrectionActor(
        # A local writer's install-token id (158-165 bytes) is longer than the
        # ledger holds (128): every machine candidate from the Python API, the
        # daemon-less CLI or MCP was refused and only logged. Same digest
        # mapping as replacements use, so one writer is one ledger id.
        actor_id=ledger_actor_id(trusted_actor_id),
        actor_kind="host_attested",
        trust_tier="canonical_writer",
    )
    try:
        with db.raw_connection() as conn:
            # The ledger holds one open case per memory. A candidate for a
            # memory that already has one (another fact of this save, or an
            # earlier save, suspected the same older memory) leaves it as it
            # is: it may be under review or already decided, and cases are
            # history that nothing replaces. Checked inside this transaction,
            # so the check and the insert cannot race.
            open_id = open_case_id(conn, profile_id, predecessor_fact_id)
            if open_id is not None and open_id != case_id:
                logger.debug(
                    "Correction candidate for %s skipped: %s already has open case %s",
                    successor_fact_id, predecessor_fact_id, open_id,
                )
                return
            propose_on_connection(
                conn,
                case_id=case_id,
                profile_id=profile_id,
                scope=scope,
                predecessor_fact_id=predecessor_fact_id,
                successor_fact_id=successor_fact_id,
                reason_code=reason_code,
                actor=actor,
                idempotency_key=idempotency_key,
                # Candidate detection observes a possible correction. It does
                # not claim an event-time boundary from heuristic evidence.
                is_profile_active=lambda candidate_profile: candidate_profile == profile_id,
                is_actor_trusted=lambda candidate_actor: candidate_actor == actor,
            )
    except (CorrectionCaseError, sqlite3.Error, ValueError) as exc:
        # A missing/hot-upgrading M042 ledger must not make memory ingestion
        # unavailable. The candidate is advisory and has no retrieval effect;
        # the warning is the operational signal that an operator must inspect.
        logger.warning("Correction candidate not recorded for %s: %s", successor_fact_id, exc)


#: Entity id pattern for the per-profile placeholder a dated fact with no
#: resolved entity points its temporal event at. Per profile, not shared, so
#: profile deletion cascades it away with everything else.
_UNRESOLVED_ENTITY_PREFIX = "__slm_unresolved__:"


def _ensure_unresolved_entity(db: Any, profile_id: str) -> str | None:
    """Return the placeholder entity id for ``profile_id``, creating it once.

    ``temporal_events.entity_id`` is NOT NULL with a foreign key to
    ``canonical_entities``, so an event for a fact with no recognised entity
    needs something real to reference. This is that something.

    It is deliberately inert as an entity:

      * Its ``canonical_name`` is empty, and the only query that resolves an
        entity by name (``retrieval/temporal_channel.py``, the "events for
        entity X" lookup) is driven by names extracted from the user's
        question, which are never empty.
      * Nothing derives entity links from ``temporal_events``. The entity
        channel builds its map from ``atomic_facts.canonical_entities_json``,
        which this does not touch — so the placeholder cannot become the kind
        of entity that links a quarter of the store and flattens entity
        proximity. That failure mode is real (on the author's store 'State'
        links 1,388 facts) and this is specifically shaped not to add to it.
      * ``fact_count`` stays 0. It is a hook for a foreign key, not a concept.

    Returns None if the row cannot be created, in which case the caller skips
    the temporal event rather than failing the write — a missing temporal row
    is a degraded memory, an aborted remember is a lost one.
    """
    entity_id = f"{_UNRESOLVED_ENTITY_PREFIX}{profile_id}"
    try:
        db.execute(
            "INSERT OR IGNORE INTO canonical_entities "
            "(entity_id, profile_id, canonical_name, entity_type, fact_count) "
            "VALUES (?, ?, '', 'unresolved', 0)",
            (entity_id, profile_id),
        )
        return entity_id
    except Exception as exc:  # noqa: BLE001 — never fail a write over this
        logger.debug(
            "temporal placeholder entity unavailable for %s: %s",
            profile_id, exc,
        )
        return None


def _record_fact_entity_association(
    db: DatabaseManager,
    *,
    operation_id: str,
    profile_id: str,
    fact_id: str,
    entity_id: str,
) -> None:
    """Apply one fact/entity count effect in O(1), exactly once."""
    if not operation_id:
        db.increment_entity_fact_count(entity_id, profile_id)
        return
    with db.transaction():
        claimed = db.execute(
            "INSERT INTO fact_entity_associations "
            "(profile_id,fact_id,entity_id,first_operation_id,count_applied) "
            "SELECT ?,?,?,?,"
            "CASE WHEN fact.rowid > repair.target_fact_rowid THEN 1 ELSE 0 END "
            "FROM canonical_entities AS entity "
            "JOIN atomic_facts AS fact "
            "ON fact.fact_id=? AND fact.profile_id=? "
            "JOIN fact_entity_association_repair_state AS repair "
            "ON repair.repair_key='historical-backfill' "
            "WHERE entity.entity_id=? AND entity.profile_id=? "
            "ON CONFLICT(profile_id,fact_id,entity_id) DO UPDATE SET "
            "count_applied=excluded.count_applied,"
            "first_operation_id=excluded.first_operation_id "
            "WHERE fact_entity_associations.count_applied=0 "
            "AND excluded.count_applied=1 "
            "RETURNING count_applied",
            (
                profile_id,
                fact_id,
                entity_id,
                operation_id,
                fact_id,
                profile_id,
                entity_id,
                profile_id,
            ),
        )
        if claimed and int(claimed[0]["count_applied"]) == 1:
            db.execute(
                "UPDATE canonical_entities SET fact_count=fact_count+1 "
                "WHERE entity_id=? AND profile_id=?",
                (entity_id, profile_id),
            )
        elif not claimed:
            # An empty result has TWO causes: (a) the 'historical-backfill'
            # repair-state row is missing — M028 DDL ran but its INSERT rolled
            # back (partial migration) — so the JOIN returned zero rows and the
            # main INSERT produced no output; or (b) the association already
            # exists and the ON CONFLICT update was correctly skipped (an
            # idempotent re-run). Only (a) needs the fallback insert; (b) is a
            # no-op. Disambiguate by probing the repair-state table directly —
            # a DIFFERENT table, so this stays a constant-query, idempotent
            # effect and never re-fires the insert on an already-applied row.
            repair_ready = db.execute(
                "SELECT 1 FROM fact_entity_association_repair_state "
                "WHERE repair_key='historical-backfill' LIMIT 1",
            )
            if repair_ready:
                # (b) idempotent skip — the association was already applied.
                return
            # (a) partial migration: insert with count_applied=1 (no historical-
            # rowid check needed — all associations in this state are
            # post-migration). A row ``store_fact`` already wrote for the
            # entity index (storage/entity_index.py) is uncounted and is
            # claimed exactly as a fresh insert would be; a counted one is
            # left alone, which keeps this retry-safe.
            fallback = db.execute(
                "INSERT INTO fact_entity_associations "
                "(profile_id,fact_id,entity_id,first_operation_id,count_applied) "
                "SELECT ?,?,?,?,1 "
                "FROM canonical_entities AS entity "
                "JOIN atomic_facts AS fact "
                "ON fact.fact_id=? AND fact.profile_id=? "
                "WHERE entity.entity_id=? AND entity.profile_id=? "
                "ON CONFLICT(profile_id,fact_id,entity_id) DO UPDATE SET "
                "count_applied=1,first_operation_id=excluded.first_operation_id "
                "WHERE fact_entity_associations.count_applied=0 "
                "RETURNING count_applied",
                (
                    profile_id, fact_id, entity_id, operation_id,
                    fact_id, profile_id,
                    entity_id, profile_id,
                ),
            )
            if fallback:
                db.execute(
                    "UPDATE canonical_entities SET fact_count=fact_count+1 "
                    "WHERE entity_id=? AND profile_id=?",
                    (entity_id, profile_id),
                )


def _init_langevin_position(dim: int = 8) -> list[float]:
    """Initialize Langevin position near origin for a new fact.

    Small random perturbation ensures each fact gets a unique position
    while staying deep in the ACTIVE zone (radius < 0.3).
    """
    import numpy as np
    rng = np.random.default_rng()
    direction = rng.standard_normal(dim)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        direction = np.ones(dim)
        norm = float(np.linalg.norm(direction))
    return (direction / norm * _INIT_LANGEVIN_RADIUS).tolist()


# ---------------------------------------------------------------------------
# enrich_fact  (was MemoryEngine._enrich_fact)
# ---------------------------------------------------------------------------

def enrich_fact(
    fact: AtomicFact,
    record: MemoryRecord,
    profile_id: str,
    *,
    embedder: Any,
    entity_resolver: Any,
    temporal_parser: Any,
) -> AtomicFact:
    """Enrich fact fields while preserving access, kind, pin and lifecycle metadata."""
    from superlocalmemory.core.enriched_fact import enrich
    return enrich(fact, record, profile_id, embedder=embedder,
                  entity_resolver=entity_resolver, temporal_parser=temporal_parser,
                  init_position=_init_langevin_position)


# ---------------------------------------------------------------------------
# Vector dual-write helper (P1-2 / embeddings-vector-01)
# ---------------------------------------------------------------------------

def _upsert_fact_vectors(fact, profile_id, ann_index, vector_store, embedder=None) -> bool:
    """Dual-write a fact's embedding to the sqlite-vec projection and the ANN index.

    Write order mirrors _attach_vector: the sqlite-vec projection is attempted
    first because that is what a meaning-based search reads.  The ANN index is
    updated only when the projection accepts the vector.  Returns True when the
    fact is now findable by meaning (the projection accepted the vector).

    Embeds on-demand when the fact has no embedding (e.g. consolidated
    summary facts created without one), so UPDATE/SUPERSEDE and consolidated
    facts remain visible to the semantic channel instead of having a row in
    ``atomic_facts`` but none in the vector store.
    """
    if not getattr(fact, "embedding", None) and embedder is not None and fact.content:
        try:
            from superlocalmemory.core.bounded_embedding import embed_source
            fact.embedding = embed_source(embedder, fact.content)
        except Exception as _emb_exc:  # pragma: no cover - defensive
            _reraise_materialization_deferral(_emb_exc)
            logger.debug("on-demand embed failed for %s: %s", fact.fact_id, _emb_exc)
            return False
    if not getattr(fact, "embedding", None):
        return False

    projected = False
    # Projection first: what a meaning-based search reads.
    if vector_store and getattr(vector_store, "available", False):
        try:
            projected = bool(vector_store.upsert(
                fact_id=fact.fact_id,
                profile_id=profile_id,
                embedding=fact.embedding,
            ))
            if not projected:
                logger.warning(
                    "vector projection refused for %s — stored but not "
                    "findable by meaning", fact.fact_id[:12],
                )
        except Exception as _vs_exc:
            logger.warning(
                "vector projection failed for %s (%s: %s) — stored but not "
                "findable by meaning",
                fact.fact_id[:12], type(_vs_exc).__name__, _vs_exc,
            )

    # ANN index only when the projection accepted the vector so both
    # representations stay consistent.  Writing ANN without a vector-store
    # entry creates a ghost that disappears on restart (ANN is rebuilt from
    # the vector store at startup).
    if projected and ann_index:
        try:
            ann_index.add(fact.fact_id, fact.embedding)
        except Exception:
            logger.warning(
                "in-memory index rejected %s", fact.fact_id[:12], exc_info=True,
            )

    return projected


class _TombstoneReadError(Exception):
    pass


def _fact_is_tombstoned(db: DatabaseManager, profile_id: str, fact_id: str) -> bool:
    try:
        rows = db.execute(
            "SELECT 1 FROM projection_tombstones "
            "WHERE profile_id = ? AND fact_id = ? LIMIT 1",
            (profile_id, fact_id),
        )
        return bool(rows)
    except Exception as exc:
        raise _TombstoneReadError(
            f"tombstone check failed for {fact_id[:16]}: {type(exc).__name__}: {exc}"
        ) from exc


def _residue_table_exists(db: DatabaseManager, name: str) -> bool:
    try:
        return bool(db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
            (name,),
        ))
    except Exception:
        # Cannot determine existence: treat as present so the residue probe runs
        # and fails closed rather than silently reporting "no residue".
        return True


def _read_tombstoned_with_retry(
    db: DatabaseManager, profile_id: str, fact_id: str, attempts: int = 3,
) -> bool:
    last: _TombstoneReadError | None = None
    for _ in range(max(1, attempts)):
        try:
            return _fact_is_tombstoned(db, profile_id, fact_id)
        except _TombstoneReadError as exc:
            last = exc
    raise last if last is not None else _TombstoneReadError("tombstone read failed")


def _vector_residue_present(db: DatabaseManager, fact_id: str) -> bool:
    uncertain = False
    for table in ("embedding_metadata", "vector_row_map"):
        if not _residue_table_exists(db, table):
            continue
        try:
            if db.execute(
                f"SELECT 1 FROM {table} WHERE fact_id = ? LIMIT 1", (fact_id,)
            ):
                return True
        except Exception:
            uncertain = True
    return uncertain


def _drop_resurrected_facts(
    db: DatabaseManager,
    profile_id: str,
    stored_ids: list[str],
    vector_store: Any,
    ann_index: Any,
    retrieval_engine: Any,
) -> list[str]:
    survivors: list[str] = []
    for fid in stored_ids:
        try:
            tombstoned = _read_tombstoned_with_retry(db, profile_id, fid)
        except _TombstoneReadError as exc:
            if is_erasing(profile_id, fid):
                # Tombstone unreadable but the in-process fence confirms an
                # erasure is in flight — clean up rather than leave residue.
                tombstoned = True
            else:
                logger.error(
                    "Tombstone read error in drop_resurrected for %s, deferring: %s",
                    fid[:16], exc,
                )
                continue
        if not tombstoned:
            survivors.append(fid)
            continue
        vec_store_ok = vector_store is not None and getattr(vector_store, "available", False)
        try:
            if vec_store_ok:
                from superlocalmemory.retrieval.vector_store import classify_delete

                if classify_delete(vector_store.delete(fid)).failed:
                    logger.error(
                        "resurrection undo: the vector of %s could not be removed", fid[:16])
            if ann_index is not None and hasattr(ann_index, "remove"):
                ann_index.remove(fid)
            bm25 = getattr(retrieval_engine, "_bm25", None) if retrieval_engine else None
            if bm25 is not None and hasattr(bm25, "remove_fact"):
                bm25.remove_fact(fid)
            db.delete_bm25_tokens_for_fact(fid)
            db.delete_fact(fid, profile_id=profile_id)
        except Exception as exc:
            logger.warning("resurrection undo failed for %s: %s", fid[:16], exc)
        if _vector_residue_present(db, fid):
            logger.error(
                "resurrection undo incomplete for %s: vector residue remains "
                "(vec_store_available=%s); tombstone preserved",
                fid[:16], vec_store_ok,
            )
    return survivors


# ---------------------------------------------------------------------------
# run_store  (was MemoryEngine.store)
# ---------------------------------------------------------------------------

def run_store(
    content: str,
    profile_id: str,
    session_id: str = "",
    session_date: str | None = None,
    speaker: str = "",
    role: str = "user",
    metadata: dict[str, Any] | None = None,
    *,
    scope: str = "personal",
    shared_with: list[str] | None = None,
    config: SLMConfig,
    db: DatabaseManager,
    embedder: Any,
    fact_extractor: Any,
    entity_resolver: Any,
    temporal_parser: Any,
    type_router: Any,
    graph_builder: Any,
    consolidator: Any,
    observation_builder: Any,
    scene_builder: Any,
    entropy_gate: Any,
    ann_index: Any,
    sheaf_checker: Any,
    retrieval_engine: Any,
    provenance: Any,
    hooks: HookRegistry,
    vector_store: Any = None,
    temporal_validator: Any = None,
    auto_linker: Any = None,
    context_generator: Any = None,
    consolidation_engine: Any = None,
    existing_memory_id: str | None = None,
    queryable_fact_ids: tuple[str, ...] = (),
    trusted_actor_id: str = "",
    pre_authorized: bool = False,
    ingestion_source_type: str = "store",
    ingestion_operation_id: str = "",
    derivation_report: dict[str, bool] | None = None,
    precompleted_derivation_stages: frozenset[str] = frozenset(),
    materialization_progress: dict[str, Any] | None = None,
    materialization_checkpoint: Any = None,
    kind_classifier: Any = None,
) -> list[str]:
    """Store content and extract structured facts. Returns fact_ids.

    Multi-scope: ``scope`` sets visibility (personal/shared/global).
    ``shared_with`` is a list of profile_ids for shared scope.
    """
    # Pre-operation hooks (trust gate, ABAC, rate limiter)
    hook_ctx = {
        "operation": "store",
        "agent_id": (
            trusted_actor_id
            or (metadata.get("agent_id", "unknown") if metadata else "unknown")
        ),
        "profile_id": profile_id,
        "content_preview": content[:100],
    }
    # What the audit entry binds: the text's hash and who saved it (L-16, L-15).
    from superlocalmemory.core.agent_attribution import audit_fields
    from superlocalmemory.core.audit_record import content_hash as _content_hash

    hook_ctx["content_hash"] = _content_hash(content)
    hook_ctx.update(audit_fields(metadata))
    if not pre_authorized:
        hooks.run_pre("store", hook_ctx)

    # Admission gates apply to FRESH submissions only. A materialization pass
    # re-runs this pipeline for content whose queryable projection was already
    # committed at submit (``queryable_fact_ids`` is set). Re-applying admission
    # here — the entropy near-duplicate gate especially — would discard that
    # already-committed fact (return []) and wedge the operation in an endless
    # materialize retry loop ("materialization produced no final facts"). The
    # near-duplicate verdict is expected at materialize: the submitted
    # projection itself is in the gate's window. Admission was already decided
    # at submit (store_fast enforces the same gates), so skip it here.
    is_materialization = bool(queryable_fact_ids)

    if entropy_gate and not is_materialization and not entropy_gate.should_pass(content):
        return []

    # v3.5.0: store-side quality gate (H3). Reject prompt-template leakage,
    # empty placeholders, and other non-memory content BEFORE it enters the
    # DB. Uses the shared is_low_quality from core/injection so both store
    # AND injection filter by identical rules. Saves DB IO + recall pollution.
    if not is_materialization:
        try:
            from superlocalmemory.core.injection import is_low_quality
            if is_low_quality(content):
                # Rejected text is saved nowhere; a log line would be its only copy,
                # and a log is not covered by an erasure.
                logger.debug("Store rejected (low-quality content, %d chars)", len(content))
                return []
        except Exception:
            pass  # Best-effort gate; store succeeds if import fails

    from superlocalmemory.encoding.temporal_parser import TemporalParser
    parser = temporal_parser or TemporalParser()
    parsed_date = parser.parse_session_date(session_date) if session_date else None

    queryable_ids = frozenset(queryable_fact_ids)
    queryable_facts: list[AtomicFact] = []
    if existing_memory_id:
        memory_rows = db.execute(
            "SELECT * FROM memories WHERE memory_id=?",
            (existing_memory_id,),
        )
        if not memory_rows:
            raise ValueError("queryable ingestion memory does not exist")
        memory = dict(memory_rows[0])
        if memory["profile_id"] != profile_id:
            raise ValueError("queryable ingestion memory profile mismatch")
        if memory["content"] != content:
            raise ValueError("queryable ingestion memory content mismatch")
        stored_shared = json.loads(memory.get("shared_with") or "null")
        if (memory.get("scope") or "personal") != scope:
            raise ValueError("queryable ingestion memory scope mismatch")
        if (stored_shared or None) != (shared_with or None):
            raise ValueError("queryable ingestion shared scope mismatch")
        queryable_facts = db.get_facts_by_ids(list(queryable_ids), profile_id)
        if len(queryable_facts) != len(queryable_ids):
            raise ValueError("queryable ingestion fact profile mismatch or missing fact")
        if any(fact.memory_id != existing_memory_id for fact in queryable_facts):
            raise ValueError("queryable ingestion fact belongs to another memory")
        record = MemoryRecord(
            memory_id=existing_memory_id,
            profile_id=profile_id,
            content=content,
            session_id=memory.get("session_id") or session_id,
            speaker=memory.get("speaker") or speaker,
            role=memory.get("role") or role,
            session_date=memory.get("session_date") or parsed_date,
            created_at=memory["created_at"],
            metadata=json.loads(memory.get("metadata_json") or "{}"),
            scope=scope,
            shared_with=shared_with,
        )
    else:
        record = MemoryRecord(
            profile_id=profile_id, content=content,
            session_id=session_id, speaker=speaker, role=role,
            session_date=parsed_date, metadata=metadata or {},
            scope=scope, shared_with=shared_with,
        )
        db.store_memory(record)

    extraction_complete = "extraction" in precompleted_derivation_stages
    consolidation_complete = (
        "consolidation" in precompleted_derivation_stages
        or consolidator is not None
    )
    canonicalization_complete = (
        "canonicalization" in precompleted_derivation_stages
        or entity_resolver is not None
    )
    graph_complete = (
        "graph" in precompleted_derivation_stages
        or graph_builder is not None
    )
    temporal_complete = True
    provenance_complete = provenance is not None

    try:
        from superlocalmemory.encoding.speaker_spans import with_actor
        context_source = with_actor(content, record.speaker, record.role)
        facts = fact_extractor.extract_facts(
            turns=[context_source], session_id=session_id,
            session_date=parsed_date, speaker_a=speaker,
        )
        extraction_complete = facts is not None
    except Exception as _extract_exc:
        _reraise_materialization_deferral(_extract_exc)
        # P0-1 (remember-write-04): an extractor EXCEPTION (transient LLM/embed
        # backend error) must NOT orphan the already-committed memory. The None
        # guard below only handled a None *return*, not a raise. Treat a raise
        # as "no facts" so the verbatim/raw fallback persists the content.
        logger.warning(
            "extract_facts() raised — falling back to raw fact: %s", _extract_exc,
        )
        facts = None

    # v3.4.38: Defensive None guard. extract_facts() returns None on transient
    # failures (embedding worker timeout, LLM call fail). Without this guard,
    # line 201's `{f.content for f in facts}` raises 'NoneType' object is not
    # iterable, causing the caller to mark_failed permanently — even though
    # the content is still recoverable. 18 memories were lost to this between
    # April 15-26, 2026.
    if facts is None:
        facts = []

    from superlocalmemory.core.source_fact_set import source_fact_set
    facts, protected_source_ids = source_fact_set(
        facts, content, record, queryable_facts,
        getattr(config.store, "max_verbatim_chars", 24000),
        db=db,
        max_derived=config.encoding.max_facts_per_chunk,
        coverage_limit=config.encoding.coverage_facts_per_memory,
    )

    if not facts:
        return []

    # The extractor already classified every derived fact. Reclassifying here
    # made one extra LLM call (or embedding call) per fact without new evidence.

    # Kinds: the caller's declared kind (confirmed) or a suggestion (label only).
    from superlocalmemory.core.kind_assignment import assign_kinds, kind_update_columns
    facts = assign_kinds(facts, metadata=metadata, source_type=ingestion_source_type,
                         classifier=kind_classifier, db=db)

    stored_ids: list[str] = []
    # Facts this save created (not an older fact a duplicate merged into). A
    # machine correction between two of them hid the save's own newer
    # statement from recall while the case waited for review: 37% of the
    # machine cases on a 22k-fact store paired two facts of one save.
    created_ids: set[str] = set(queryable_ids)
    for fact in facts:
        merged_into_older = False
        try:
            if _fact_is_tombstoned(db, profile_id, fact.fact_id):
                continue
        except _TombstoneReadError as _tse:
            logger.error(
                "Tombstone read error for %s, deferring ingestion: %s",
                fact.fact_id[:16], _tse,
            )
            continue
        if is_erasing(profile_id, fact.fact_id):
            continue
        fact = enrich_fact(
            fact, record, profile_id,
            embedder=embedder,
            entity_resolver=entity_resolver,
            temporal_parser=temporal_parser,
        )

        if (
            materialization_progress is not None
            and not materialization_progress.get("relational_started", False)
        ):
            if materialization_checkpoint is not None:
                materialization_checkpoint(
                    "relational_started",
                    (),
                    {
                        "pipeline_started": True,
                        "relational_started": True,
                        "pipeline": False,
                    },
                )
            materialization_progress["relational_started"] = True

        is_queryable_promotion = fact.fact_id in queryable_ids
        # When a promoted fact is being materialised the embedding column is
        # written in a separate step AFTER the projection attempt so that the
        # two representations stay in the correct order: projection first,
        # canonical column second (mirroring _attach_vector's invariant).
        # Writing the column before the projection and then failing the
        # projection leaves the fact with embedding IS NOT NULL but no entry
        # in fact_embeddings — invisible to meaning-search and invisible to
        # every repair pass that selects WHERE embedding IS NULL.
        _deferred_canonical_embedding = None  # set below if promotion path taken
        if is_queryable_promotion:
            db.update_fact(fact.fact_id, {
                "content": fact.content,
                "fact_type": fact.fact_type,
                "entities_json": fact.entities,
                "canonical_entities_json": fact.canonical_entities,
                "observation_date": fact.observation_date,
                "referenced_date": fact.referenced_date,
                "interval_start": fact.interval_start,
                "interval_end": fact.interval_end,
                "confidence": fact.confidence,
                "importance": fact.importance,
                "evidence_count": fact.evidence_count,
                "access_count": fact.access_count,
                "source_turn_ids_json": fact.source_turn_ids,
                "session_id": fact.session_id,
                # embedding / fisher fields omitted here — written after
                # the projection attempt below so canonical is never ahead
                # of the projection.
                "lifecycle": fact.lifecycle,
                "langevin_position": fact.langevin_position,
                "emotional_valence": fact.emotional_valence,
                "emotional_arousal": fact.emotional_arousal,
                "signal_type": fact.signal_type,
                **kind_update_columns(fact, db),
            })
            if fact.embedding:
                _deferred_canonical_embedding = (
                    fact.fact_id,
                    fact.embedding,
                    fact.fisher_mean,
                    fact.fisher_variance,
                )
        if consolidator:
            try:
                action = consolidator.consolidate(
                    fact,
                    profile_id,
                    exclude_fact_ids=(queryable_ids | created_ids
                                      if fact.fact_id in protected_source_ids else queryable_ids),
                    # Stored by this save, not yet in the vector index.
                    pending_fact_ids=tuple(stored_ids),
                )
            except Exception as _consolidate_exc:
                # P0-1 (remember-write-03): a consolidate failure (e.g. LLM
                # timeout) must NOT orphan the already-committed memory. Fall
                # back to storing the raw enriched fact so the content stays
                # retrievable across all channels.
                logger.warning(
                    "consolidate() failed for fact %s — storing raw fact as "
                    "fallback: %s", fact.fact_id, _consolidate_exc,
                )
                consolidation_complete = False
                action = None

            if action is not None and keeps_own_fact(action, is_queryable_promotion, created_ids):
                action = None  # another memory's duplicate: keep this memory's own fact
            if action is not None:
                if action.action_type.value == "noop":
                    # Same-save duplicate: fold this save's projection into its sibling.
                    target_id = action.existing_fact_id
                    if is_queryable_promotion and target_id:
                        db.delete_fact(fact.fact_id)
                        # Deleted: its deferred canonical embedding write must not happen.
                        _deferred_canonical_embedding = None
                    existing_fact = db.get_fact(target_id) if target_id else None
                    if existing_fact is None:
                        continue
                    fact = existing_fact
                    merged_into_older = target_id not in created_ids

                if action.action_type.value in ("update", "supersede"):
                    # A consolidator UPDATE/SUPERSEDE is a review-required
                    # proposal.  It persists the incoming successor and does
                    # not mutate, delete, archive, or change trust on the
                    # matched predecessor.  Continue materializing the
                    # incoming fact through the common projection pipeline.
                    proposed_fact = db.get_fact(action.new_fact_id)
                    if proposed_fact is None:
                        raise RuntimeError(
                            f"consolidation {action.action_type.value} proposal produced "
                            f"missing fact {action.new_fact_id}"
                        )
                    fact = proposed_fact
                    if action.existing_fact_id and action.existing_fact_id not in created_ids:
                        _record_correction_candidate(
                            db,
                            operation_id=ingestion_operation_id,
                            profile_id=profile_id,
                            scope=scope,
                            predecessor_fact_id=action.existing_fact_id,
                            successor_fact_id=action.new_fact_id,
                            reason_code=f"consolidation_{action.action_type.value}",
                            trusted_actor_id=trusted_actor_id,
                        )
                # ADD case: consolidator already stored the fact (F8 fix)
                # Fall through to post-processing below
            else:
                # Consolidate failed → store the raw fact ourselves so the
                # memory is never left without a retrievable fact, then fall
                # through to post-processing (embeddings, graph, context).
                if not is_queryable_promotion:
                    db.store_fact(fact)
        elif not is_queryable_promotion:
            db.store_fact(fact)

        if fact.fact_id in protected_source_ids and consolidator:
            link_source = getattr(consolidator, "link_source", None)
            if callable(link_source):
                link_source(fact, profile_id, tuple(stored_ids))
        if not merged_into_older:
            created_ids.add(fact.fact_id)
        if fact.fact_id not in stored_ids:
            stored_ids.append(fact.fact_id)
            if materialization_progress is not None:
                materialization_progress["fact_ids"] = tuple(stored_ids)

        # Projection first (vector store then ANN index, on-demand embed for
        # consolidated facts that arrived without one).
        _upsert_fact_vectors(fact, profile_id, ann_index, vector_store, embedder)

        # Canonical embedding write: deferred to here from the queryable-
        # promotion block above so the projection is always attempted first.
        # The column is written regardless of whether the projection accepted
        # the vector (the vector is real data; withholding it would force
        # every repair pass to re-compute a model call that the same
        # installation condition would refuse again anyway).
        if _deferred_canonical_embedding is not None:
            _dc_fid, _dc_emb, _dc_fmean, _dc_fvar = _deferred_canonical_embedding
            try:
                db.update_fact(_dc_fid, {
                    "embedding": _dc_emb,
                    "fisher_mean": _dc_fmean,
                    "fisher_variance": _dc_fvar,
                })
            except Exception as _dc_exc:
                logger.warning(
                    "canonical embedding write failed for %s: %s",
                    _dc_fid[:12], _dc_exc,
                )
            _deferred_canonical_embedding = None

        # Phase 2: Generate contextual description (after consolidator, before graph_builder)
        if context_generator:
            try:
                import json as _json
                ctx_result = context_generator.generate(fact, config.mode.value)
                db.store_fact_context(
                    fact_id=fact.fact_id,
                    profile_id=profile_id,
                    contextual_description=ctx_result.description,
                    keywords=_json.dumps(ctx_result.keywords),
                    generated_by=ctx_result.generated_by,
                )
            except Exception as _ctx_exc:
                logger.debug("Context generation skipped for %s: %s", fact.fact_id, _ctx_exc)

        if graph_builder:
            graph_builder.build_edges(fact, profile_id)

        # Phase 3: AutoLinker creates association_edges (AFTER GraphBuilder)
        if auto_linker is not None:
            try:
                auto_linker.link_new_fact(fact, profile_id)
            except Exception as exc:
                logger.debug("AutoLinker.link_new_fact: %s", exc)

        # Sheaf consistency check (runs after edges exist)
        if (sheaf_checker
                and fact.embedding
                and fact.canonical_entities):
            from superlocalmemory.storage.models import EdgeType, GraphEdge
            try:
                edges_for_fact = db.get_edges_for_node(
                    fact.fact_id, profile_id,
                )
                if len(edges_for_fact) < config.math.sheaf_max_edges_per_check:
                    contradictions = sheaf_checker.check_consistency(
                        fact, profile_id,
                    )
                    for c in contradictions:
                        if c.severity > 0.45:
                            edge = GraphEdge(
                                profile_id=profile_id,
                                source_id=fact.fact_id,
                                target_id=c.fact_id_b,
                                edge_type=EdgeType.SUPERSEDES,
                                weight=c.severity,
                            )
                            db.store_edge(edge)
            except Exception as exc:
                logger.debug("Sheaf check skipped: %s", exc)

        from superlocalmemory.storage.event_time import valid_from as _event_start

        # Event-time start is mandatory even without a contradiction validator.
        # fill_only: a duplicate re-save (merged into an older fact) must
        # never replace a start a correction or extraction already set.
        db.store_temporal_validity(
            fact_id=fact.fact_id, profile_id=profile_id,
            valid_from=_event_start(fact),
            fill_only=True,
        )
        # Phase 4: Temporal validation and contradiction detection
        if temporal_validator:
            try:
                invalidations = temporal_validator.validate_and_invalidate(
                    new_fact=fact,
                    profile_id=profile_id,
                )
                if invalidations:
                    logger.info(
                        "Temporal: %d facts invalidated by new fact %s",
                        len(invalidations), fact.fact_id,
                    )
                    for candidate in invalidations:
                        predecessor_fact_id = str(candidate.get("old_fact_id") or "")
                        if predecessor_fact_id in created_ids:
                            continue  # two facts of this same save
                        _record_correction_candidate(
                            db,
                            operation_id=ingestion_operation_id,
                            profile_id=profile_id,
                            scope=scope,
                            predecessor_fact_id=predecessor_fact_id,
                            successor_fact_id=fact.fact_id,
                            reason_code="temporal_contradiction",
                            trusted_actor_id=trusted_actor_id,
                        )
            except Exception as exc:
                temporal_complete = False
                logger.debug(
                    "Temporal validation skipped for fact %s: %s",
                    fact.fact_id, exc,
                )

        # Phase 4b: fact-augmented key expansion (T3b). Index entity aliases /
        # canonical names as BM25 alt-keys so paraphrased queries match. Mode A
        # (entity graph) is zero-LLM and safe on the hot store path; LLM
        # paraphrase enrichment (Mode B/C) is left to background consolidation.
        try:
            from superlocalmemory.core.key_expander import KeyExpander
            _alt_keys = KeyExpander(db).expand(fact, profile_id, mode="a")
            if _alt_keys:
                db.upsert_fact_expansion(fact.fact_id, _alt_keys)
        except Exception as exc:
            logger.debug("Key expansion skipped for %s: %s", fact.fact_id, exc)

        if observation_builder:
            for eid in fact.canonical_entities:
                observation_builder.update_profile(eid, fact, profile_id)

        # The normalized association key makes this O(1) and exactly-once
        # across retries, crashes, and separate operations that consolidate to
        # the same fact.
        for eid in fact.canonical_entities:
            _record_fact_entity_association(
                db,
                operation_id=ingestion_operation_id,
                profile_id=profile_id,
                fact_id=fact.fact_id,
                entity_id=eid,
            )
        if scene_builder:
            scene_builder.assign_to_scene(fact, profile_id)

        # Populate temporal_events for temporal retrieval.
        #
        # A DATE IS A DATE WHETHER OR NOT AN ENTITY WAS RESOLVED. This used to
        # read `if fact.canonical_entities and has_dates`, so a fact carrying a
        # perfectly good date but no recognised entity got no temporal row at
        # all. Measured on the author's store: 967 of 3,894 genuine facts have
        # no canonical entity (24.8%), and 958 of those have no temporal event
        # either — a quarter of the store that the temporal channel could only
        # reach through its created_at recency fallback, i.e. by being recent
        # rather than by being about the right time.
        #
        # The date-window query does not need the entity: it joins
        # temporal_events to atomic_facts and nothing else
        # (retrieval/temporal_channel.py). Only the "events for entity X"
        # lookup joins canonical_entities, and that lookup is correctly
        # uninterested in a fact with no entity.
        #
        # temporal_events.entity_id is NOT NULL with an FK to
        # canonical_entities, so an entity-less event needs a row to point at.
        # It gets a per-profile sentinel rather than a shared one, and the
        # sentinel is invisible to ranking: the entity channel builds its map
        # from atomic_facts.canonical_entities_json, which is untouched here, so
        # this cannot create the kind of entity that links a quarter of the
        # store and destroys entity proximity.
        has_dates = (fact.observation_date or fact.referenced_date
                     or fact.interval_start)
        if has_dates:
            from superlocalmemory.storage.models import TemporalEvent
            entity_ids = list(fact.canonical_entities)
            if not entity_ids:
                sentinel = _ensure_unresolved_entity(db, profile_id)
                entity_ids = [sentinel] if sentinel else []
            for eid in entity_ids:
                event = TemporalEvent(
                    event_id=_ingestion_effect_id(
                        ingestion_operation_id,
                        "temporal",
                        fact.fact_id,
                        eid,
                        "observed",
                    ),
                    profile_id=profile_id, entity_id=eid,
                    fact_id=fact.fact_id,
                    scope=fact.scope,
                    shared_with=fact.shared_with,
                    observation_date=fact.observation_date,
                    referenced_date=fact.referenced_date,
                    interval_start=fact.interval_start,
                    interval_end=fact.interval_end,
                    description=fact.content[:200],
                )
                db.store_temporal_event(event)

        # Foresight: extract time-bounded predictions
        try:
            from superlocalmemory.encoding.foresight import extract_foresight_signals
            from superlocalmemory.storage.models import TemporalEvent as _TE
            foresight_signals = extract_foresight_signals(fact)
            for signal_index, sig in enumerate(foresight_signals):
                f_event = _TE(
                    event_id=_ingestion_effect_id(
                        ingestion_operation_id,
                        "temporal",
                        fact.fact_id,
                        sig.get("entity_id", ""),
                        "foresight",
                        signal_index,
                    ),
                    profile_id=profile_id,
                    entity_id=sig.get("entity_id", ""),
                    fact_id=fact.fact_id,
                    scope=fact.scope,
                    shared_with=fact.shared_with,
                    interval_start=sig.get("start_time"),
                    interval_end=sig.get("end_time"),
                    description=sig.get("description", ""),
                )
                db.store_temporal_event(f_event)
        except Exception as exc:
            logger.debug("Foresight extraction: %s", exc)

        # Persist BM25 tokens at ingestion
        bm25 = getattr(retrieval_engine, '_bm25', None) if retrieval_engine else None
        if bm25:
            bm25.add(fact.fact_id, fact.content, profile_id)

        # Record provenance for data lineage (EU AI Act Art. 10)
        if provenance:
            try:
                provenance.record(
                    fact_id=fact.fact_id,
                    profile_id=profile_id,
                    source_type=ingestion_source_type,
                    source_id=ingestion_operation_id or session_id,
                    created_by=trusted_actor_id or speaker or "unknown",
                )
            except Exception:
                provenance_complete = False

    stored_ids = _drop_resurrected_facts(
        db, profile_id, stored_ids, vector_store, ann_index, retrieval_engine,
    )

    logger.info("Stored %d facts (session=%s)", len(stored_ids), session_id)

    if derivation_report is not None:
        derivation_report.update({
            "extraction": extraction_complete,
            "canonicalization": canonicalization_complete,
            "consolidation": consolidation_complete,
            "graph": graph_complete,
            "temporal": temporal_complete,
            "provenance": provenance_complete,
        })

    # Stage observations and emitted fact IDs are recorded before optional
    # post-hooks.  A hook failure must not erase the durable retry boundary and
    # cause extraction/consolidation to run again.
    if materialization_progress is not None:
        materialization_progress["fact_ids"] = tuple(stored_ids)
        materialization_progress["relational_complete"] = True
        materialization_progress["post_hooks"] = False
    if materialization_checkpoint is not None and stored_ids:
        materialization_checkpoint(
            "relational_complete",
            tuple(stored_ids),
            {
                "pipeline_started": True,
                "relational_started": True,
                "pipeline": True,
                "post_hooks": False,
                "relational": True,
                **dict(derivation_report or {}),
            },
        )

    # Post-operation hooks (audit, trust signal, event bus)
    hook_ctx["ingestion_operation_id"] = ingestion_operation_id
    hook_ctx["fact_ids"] = stored_ids
    hook_ctx["fact_count"] = len(stored_ids)
    hooks.run_post("store", hook_ctx)
    if materialization_progress is not None:
        materialization_progress["post_hooks"] = True

    # Phase 5: Step-count trigger for lightweight consolidation (L7)
    if consolidation_engine is not None:
        try:
            consolidation_engine.increment_store_count(profile_id)
        except Exception as _cons_exc:
            logger.debug("Consolidation step-count trigger: %s", _cons_exc)

    return stored_ids


# ---------------------------------------------------------------------------
# run_store_fact_direct  (was MemoryEngine.store_fact_direct)
# ---------------------------------------------------------------------------

def run_store_fact_direct(
    fact: AtomicFact,
    profile_id: str,
    *,
    db: DatabaseManager,
    embedder: Any,
    entity_resolver: Any,
    ann_index: Any,
    graph_builder: Any,
    retrieval_engine: Any,
    vector_store: Any = None,
) -> str:
    """Store a pre-built fact with full enrichment.

    Ensures embedding, Fisher params, canonical entities, BM25 tokens,
    and graph edges are all populated — even for auxiliary data.
    Creates a parent memory record to satisfy FK constraint.
    """
    # remember-write-02: gate low-quality content (empty, bare category tags,
    # placeholder/template leakage) at the WRITE boundary, matching run_store's
    # gate. Previously this direct path had no filter, so junk entered the KB
    # and polluted evidence/stats/embeddings (the read-side filter only hid it).
    from superlocalmemory.core.injection import is_low_quality
    if is_low_quality(fact.content):
        logger.debug("run_store_fact_direct: skipping low-quality content")
        return fact.fact_id

    # Create parent memory record (FK: atomic_facts.memory_id → memories.memory_id)
    if not fact.memory_id:
        record = MemoryRecord(
            profile_id=profile_id,
            content=fact.content[:500],
            session_id=fact.session_id,
        )
        db.store_memory(record)
        fact.memory_id = record.memory_id

    if not fact.embedding and embedder:
        from superlocalmemory.core.bounded_embedding import embed_source
        fact.embedding = embed_source(embedder, fact.content)
        if fact.embedding:
            fact.fisher_mean, fact.fisher_variance = (
                embedder.compute_fisher_params(fact.embedding)
            )
    fact.entities = valid_names(list(fact.entities or []), fact.content)
    if entity_resolver and fact.entities:
        canonical = entity_resolver.resolve(
            fact.entities, profile_id,
        )
        fact.canonical_entities = list(canonical.values())
    db.store_fact(fact)
    # Projection first (vector store), ANN only on success — matching
    # _attach_vector's invariant.  The return value (projected) is not used
    # here because run_store_fact_direct is a fire-and-forget path with no
    # receipt to update; the caller is responsible for any enrichment status.
    _upsert_fact_vectors(fact, profile_id, ann_index, vector_store)
    if graph_builder:
        graph_builder.build_edges(fact, profile_id)
    # The graph projection must run after GraphBuilder: syncing immediately
    # after SQLite fact insertion omitted every new fact edge from Cozo.
    _sync_to_graph_backends(fact)
    # BM25 indexing
    bm25 = getattr(retrieval_engine, '_bm25', None) if retrieval_engine else None
    if bm25:
        bm25.add(fact.fact_id, fact.content, profile_id)
    return fact.fact_id


# ---------------------------------------------------------------------------
# run_close_session  (was MemoryEngine.close_session)
# ---------------------------------------------------------------------------

def _session_already_summarised(
    db: DatabaseManager,
    profile_id: str,
    session_id: str,
    entity_ids: list[str],
) -> bool:
    """True if close_session already wrote temporal summaries for this session.

    Its summaries carry the session's entities, so the LIKE runs only over those
    entities' events (idx_tevents_entity), not every event in the profile.
    """
    if not session_id or not entity_ids:
        return False
    marks = ", ".join("?" * len(entity_ids))
    try:
        rows = db.execute(
            "SELECT 1 FROM temporal_events "
            f"WHERE profile_id = ? AND entity_id IN ({marks}) AND description LIKE ? LIMIT 1",
            (profile_id, *entity_ids, f"Session {session_id}:%"),
        )
        return bool(rows)
    except Exception:
        return False


def run_close_session(
    session_id: str,
    profile_id: str,
    *,
    db: DatabaseManager,
) -> int:
    """Create session-level temporal summary for session-level retrieval.

    Aggregates facts from a completed session into temporal_events
    with session scope. Enables temporal queries like "What happened
    in session 3?"

    Idempotent: if temporal summaries for this session already exist,
    returns 0 without writing duplicates.

    Returns number of session summary events created.
    """
    from superlocalmemory.storage.models import TemporalEvent

    if not session_id:
        return 0

    # The session's own facts through idx_facts_session: reading the whole store
    # here held the daemon for seconds on every session end, and its fetch
    # ceiling dropped older sessions entirely.
    session_facts = db.get_session_facts(profile_id, session_id)
    if not session_facts:
        return 0

    # Group by entity for session-level summaries
    entity_facts: dict[str, list[AtomicFact]] = {}
    for f in session_facts:
        for eid in f.canonical_entities:
            entity_facts.setdefault(eid, []).append(f)
    if not entity_facts:
        return 0

    if _session_already_summarised(db, profile_id, session_id, list(entity_facts)):
        logger.debug("Session %s already summarised — skip close", session_id)
        return 0

    count = 0
    session_date = session_facts[0].observation_date or ""
    for eid, efacts in entity_facts.items():
        summary_parts = [f.content[:80] for f in efacts[:5]]
        summary = f"Session {session_id}: " + "; ".join(summary_parts)
        event = TemporalEvent(
            profile_id=profile_id,
            entity_id=eid,
            fact_id=efacts[0].fact_id,
            observation_date=session_date,
            description=summary[:500],
        )
        db.store_temporal_event(event)
        count += 1

    logger.info(
        "Session %s closed: %d summary events for %d facts",
        session_id, count, len(session_facts),
    )
    return count


# ---------------------------------------------------------------------------
# v3.4.5: Incremental sync to CozoDB/LanceDB (F-04)
# ---------------------------------------------------------------------------

def _sync_to_graph_backends(fact: Any) -> None:
    """Sync a newly stored fact to CozoDB/LanceDB.

    Non-blocking, best-effort. Called after SQLite write.
    Failures are logged, not raised — SQLite is already committed.
    """
    try:
        from superlocalmemory.core.backend_orchestrator import get_orchestrator
        orch = get_orchestrator()
        if orch is not None:
            orch.sync_new_fact(fact)
    except Exception:
        pass  # Best-effort — daemon may not have initialized yet
