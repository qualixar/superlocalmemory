# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""P1-2 (embeddings-vector-01): UPDATE/SUPERSEDE facts must reach the vector
store, and consolidated facts that lack an embedding must be embedded
on-demand — otherwise they are invisible to the semantic channel.

Unit tests for the extracted dual-write helper ``_upsert_fact_vectors``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from superlocalmemory.core.store_pipeline import _upsert_fact_vectors
from superlocalmemory.storage.models import AtomicFact, FactType


def _fact(fid: str = "f1", content: str = "merged superseded content", embedding=None):
    return AtomicFact(fact_id=fid, content=content, fact_type=FactType.SEMANTIC,
                      embedding=embedding)


def test_dualwrites_existing_embedding():
    fact = _fact(embedding=[0.1] * 8)
    ann = MagicMock()
    vs = MagicMock()
    vs.available = True
    _upsert_fact_vectors(fact, "default", ann, vs, embedder=None)
    ann.add.assert_called_once_with("f1", [0.1] * 8)
    vs.upsert.assert_called_once()
    assert vs.upsert.call_args.kwargs["fact_id"] == "f1"


def test_embeds_on_demand_when_missing():
    fact = _fact(embedding=None)
    assert fact.embedding is None
    embedder = MagicMock()
    embedder.embed.return_value = [0.2] * 8
    ann = MagicMock()
    vs = MagicMock()
    vs.available = True
    _upsert_fact_vectors(fact, "default", ann, vs, embedder=embedder)
    embedder.embed.assert_called_once_with("merged superseded content")
    ann.add.assert_called_once_with("f1", [0.2] * 8)
    vs.upsert.assert_called_once()


def test_skips_when_no_embedding_and_no_embedder():
    fact = _fact(embedding=None)
    ann = MagicMock()
    vs = MagicMock()
    vs.available = True
    _upsert_fact_vectors(fact, "default", ann, vs, embedder=None)
    ann.add.assert_not_called()
    vs.upsert.assert_not_called()


def test_respects_unavailable_vector_store():
    # When the vector store is unavailable there is no projection, so the ANN
    # index must not receive the vector either. Writing to ANN without a
    # matching vector-store entry creates a ghost: findable in the in-memory
    # channel this session, invisible after a restart that rebuilds ANN from
    # the vector store.
    fact = _fact(embedding=[0.3] * 8)
    ann = MagicMock()
    vs = MagicMock()
    vs.available = False
    result = _upsert_fact_vectors(fact, "default", ann, vs, embedder=None)
    assert result is False             # no projection → not findable by meaning
    ann.add.assert_not_called()        # ANN must not get the vector without a projection
    vs.upsert.assert_not_called()      # vec store skipped when unavailable


# ---------------------------------------------------------------------------
# New tests for write-order invariant (projection before ANN, honest return value)
# ---------------------------------------------------------------------------


def test_ann_not_written_when_projection_refuses():
    """ANN must not receive a vector when the projection store refuses it.

    The correct write order mirrors _attach_vector: projection first, ANN only
    when projected is True. Writing to ANN unconditionally creates a ghost
    entry: the fact appears in the in-memory ANN channel for the current
    session but disappears after a restart that rebuilds ANN from the vector
    store (which has no entry for it).
    """
    fact = _fact(embedding=[0.1] * 8)
    ann = MagicMock()
    vs = MagicMock()
    vs.available = True
    vs.upsert.return_value = False  # projection refused

    result = _upsert_fact_vectors(fact, "default", ann, vs, embedder=None)

    assert result is False, "returned True for a refused projection"
    ann.add.assert_not_called()
    vs.upsert.assert_called_once()


def test_ann_not_written_when_projection_raises():
    """ANN must not receive a vector when the projection store raises.

    A raised exception and a returned False are indistinguishable to the
    caller; both mean the projection did not accept the vector.
    """
    fact = _fact(embedding=[0.1] * 8)
    ann = MagicMock()
    vs = MagicMock()
    vs.available = True
    vs.upsert.side_effect = RuntimeError("vec0 extension not loaded")

    result = _upsert_fact_vectors(fact, "default", ann, vs, embedder=None)

    assert result is False, "returned True when projection raised"
    ann.add.assert_not_called()


def test_returns_true_and_writes_ann_when_projection_succeeds():
    """Happy path: projection succeeds, ANN is written, return value is True."""
    fact = _fact(embedding=[0.1] * 8)
    ann = MagicMock()
    vs = MagicMock()
    vs.available = True
    vs.upsert.return_value = True

    result = _upsert_fact_vectors(fact, "default", ann, vs, embedder=None)

    assert result is True, "returned False for a successful projection"
    ann.add.assert_called_once_with("f1", [0.1] * 8)


def test_returns_false_when_vector_store_is_none():
    """No vector store means no projection — return False and do not touch ANN."""
    fact = _fact(embedding=[0.1] * 8)
    ann = MagicMock()

    result = _upsert_fact_vectors(fact, "default", ann, None, embedder=None)

    assert result is False
    ann.add.assert_not_called()
