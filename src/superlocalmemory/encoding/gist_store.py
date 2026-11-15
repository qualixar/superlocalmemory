# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Persist a CCQ replacement atomically, reusing equal source observations."""

from __future__ import annotations

import json
import logging

from superlocalmemory.core.lifecycle_state import set_fact_lifecycle_zone
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.storage.source_identity import store_source_fact

logger = logging.getLogger(__name__)


def _encode_gist(embedder, text) -> list[float] | None:
    """The gist's embedding, kept (it used to be thrown away)."""
    if embedder is None:
        return None
    try:
        vec = embedder.encode(text)
    except Exception as exc:
        logger.warning("Gist embedding generation failed: %s", exc)
        return None
    return [float(x) for x in vec] if vec is not None else None


def _archive_sources(db, fact_ids, profile_id, new_id):
    set_fact_lifecycle_zone(db, fact_ids, "archive", profile_id=profile_id)
    for fact_id in fact_ids:
        db.execute(
            "INSERT INTO fact_access_log "
            "(log_id,fact_id,profile_id,accessed_at,access_type,session_id) "
            "VALUES (?,?,?,datetime('now'),'consolidation','ccq')",
            (new_id(), fact_id, profile_id),
        )


def store_block(db, embedder, cluster, gist, profile_id, new_id, current_confidence):
    """Persist the gist before soft-archiving its sources, in one transaction.

    ``current_confidence(fact_ids)`` runs inside the transaction: it raises when a
    source was retired after selection (the cluster is abandoned)
    and returns the least confident source's confidence, which the gist takes.
    """
    block_id = new_id()
    embedding = _encode_gist(embedder, gist.gist_text)  # model work stays outside the transaction
    with db.transaction():
        confidence = current_confidence(cluster.fact_ids)
        record = MemoryRecord(
            memory_id=block_id, profile_id=profile_id, content=gist.gist_text, session_id="ccq"
        )
        fact = AtomicFact(
            memory_id=block_id,
            profile_id=profile_id,
            content=gist.gist_text,
            canonical_entities=list(gist.key_entities),
            importance=0.8,
            confidence=confidence,
            session_id="ccq",
            embedding=embedding,
        )
        from superlocalmemory.storage import fact_origin

        with fact_origin.declare(fact_origin.Origin(
                "derived", "model_rewritten", source_type="ccq-gist")):
            store_source_fact(db, record, fact)
        db.store_ccq_block(
            block_id=block_id,
            profile_id=profile_id,
            content=gist.gist_text,
            source_fact_ids=json.dumps(list(cluster.fact_ids)),
            gist_embedding_rowid=None,
            char_count=len(gist.gist_text),
            cluster_id=cluster.cluster_id,
        )
        _archive_sources(db, cluster.fact_ids, profile_id, new_id)
    return block_id
