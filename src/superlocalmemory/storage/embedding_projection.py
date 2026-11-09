# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Embedding-only reads for the retrieval channels.

WHY THIS EXISTS. Three channels (spreading activation, semantic, Hopfield)
score candidates against the query using nothing but ``fact_id`` and the
embedding.  They used to read those candidates as fully hydrated
``AtomicFact`` objects: ``SELECT *`` plus ``_row_to_fact``, which decodes the
embedding AND the two 768-float Fisher vectors through ``.tolist()`` and four
JSON fields, per row, per recall.  At about 1,800 cross-scope facts that was
1.3 s of every cross-scope recall (GitHub #147), for data nobody read.

The functions here run the SAME predicate as the hydrating readers they stand
in for -- built by the same ``_scope_where`` and ``visible_fact_clause`` calls,
in the same order, with the same ``ORDER BY`` -- and project only
``fact_id, embedding``.  The predicate is the authorization rule, so it must
not drift; tests/test_storage/test_embedding_projection.py asserts both readers
return the same ids in the same order.

Decoding is bit-identical to the hydrating path.  A full-width float32 BLOB is
read with ``np.frombuffer``: the old path went float32 -> Python float ->
float32, which is exact, so the values are the same bits.  Everything else
(legacy JSON text, short test vectors) goes through ``decode_embedding``, the
one codec every reader uses, and then ``np.array(..., float32)`` exactly as the
channels did.

One deliberate behaviour change: a row whose embedding cannot be decoded is
skipped with a warning.  Before, one corrupt cross-scope row raised inside
``_row_to_fact``; spreading activation swallowed that and silently lost EVERY
cross-scope seed, while semantic and Hopfield failed the channel.  Losing one
unreadable row is strictly better than losing the channel.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

from superlocalmemory.storage.database import _scope_where
from superlocalmemory.storage.embedding_codec import (
    decode_embedding,
    expected_bytes,
)

logger = logging.getLogger(__name__)

__all__ = [
    "decode_embedding_array",
    "fetch_external_visible_embeddings",
    "fetch_fact_embeddings_by_ids",
]

EmbeddingRow = tuple[str, np.ndarray]


def decode_embedding_array(raw: Any, *, fact_id: str) -> np.ndarray | None:
    """Decode a stored embedding to a float32 vector, or ``None`` if absent.

    Raises ``ValueError``/``TypeError`` for a value that is not an embedding,
    exactly like ``decode_embedding`` -- callers decide what to do with it.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, (bytes, bytearray)) and len(raw) == expected_bytes():
        return np.frombuffer(raw, dtype=np.float32)
    values = decode_embedding(raw, fact_id=fact_id)
    if values is None:
        return None
    return np.array(values, dtype=np.float32)


def _project_rows(rows: Iterable[Any]) -> list[EmbeddingRow]:
    """Decode ``(fact_id, embedding)`` rows, keeping row order.

    Rows without an embedding are omitted: every caller skipped them already.
    A row that cannot be decoded is skipped with a warning (see module doc).
    """
    out: list[EmbeddingRow] = []
    for row in rows:
        fact_id = str(row["fact_id"])
        try:
            vector = decode_embedding_array(row["embedding"], fact_id=fact_id)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "skipping fact %s in retrieval: unreadable embedding (%s)",
                fact_id, exc,
            )
            continue
        if vector is None:
            continue
        if vector.ndim != 1:
            logger.warning(
                "skipping fact %s in retrieval: embedding has shape %s",
                fact_id, vector.shape,
            )
            continue
        out.append((fact_id, vector))
    return out


def fetch_external_visible_embeddings(
    db: Any,
    profile_id: str,
    *,
    include_global: bool = False,
    include_shared: bool = False,
) -> list[EmbeddingRow]:
    """``get_external_visible_facts``, projected to ``(fact_id, embedding)``.

    Same predicate, same exclusion of the requester's own partition, same
    order (newest first).
    """
    if not include_global and not include_shared:
        return []
    where, params = _scope_where(
        profile_id,
        include_global=include_global,
        include_shared=include_shared,
    )
    rows = db.execute(
        f"SELECT fact_id, embedding FROM atomic_facts WHERE {where} "
        f"AND profile_id != ?{db.visible_fact_clause()} ORDER BY created_at DESC",
        (*params, profile_id),
    )
    return _project_rows(rows)


def fetch_fact_embeddings_by_ids(
    db: Any,
    fact_ids: list[str],
    profile_id: str,
    *,
    include_global: bool = False,
    include_shared: bool = False,
) -> list[EmbeddingRow]:
    """``get_facts_by_ids``, projected to ``(fact_id, embedding)``.

    Same authorization predicate (scope plus withheld/archived exclusion) and
    the same single query with the same ``ORDER BY``, so the rows come back in
    the order the hydrating reader produced them.  Hopfield sums over its
    sub-matrix in row order, so the order is part of the score.
    """
    if not fact_ids:
        return []
    where, params = _scope_where(
        profile_id,
        include_global=include_global,
        include_shared=include_shared,
    )
    placeholders = ",".join("?" for _ in fact_ids)
    rows = db.execute(
        f"SELECT fact_id, embedding FROM atomic_facts "
        f"WHERE fact_id IN ({placeholders}) "
        f"AND {where}{db.visible_fact_clause()} "
        "ORDER BY created_at DESC",
        (*fact_ids, *params),
    )
    return _project_rows(rows)
