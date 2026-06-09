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
    vs = MagicMock(); vs.available = True
    _upsert_fact_vectors(fact, "default", ann, vs, embedder=None)
    ann.add.assert_called_once_with("f1", [0.1] * 8)
    vs.upsert.assert_called_once()
    assert vs.upsert.call_args.kwargs["fact_id"] == "f1"


def test_embeds_on_demand_when_missing():
    fact = _fact(embedding=None)
    assert fact.embedding is None
    embedder = MagicMock(); embedder.embed.return_value = [0.2] * 8
    ann = MagicMock()
    vs = MagicMock(); vs.available = True
    _upsert_fact_vectors(fact, "default", ann, vs, embedder=embedder)
    embedder.embed.assert_called_once_with("merged superseded content")
    ann.add.assert_called_once_with("f1", [0.2] * 8)
    vs.upsert.assert_called_once()


def test_skips_when_no_embedding_and_no_embedder():
    fact = _fact(embedding=None)
    ann = MagicMock()
    vs = MagicMock(); vs.available = True
    _upsert_fact_vectors(fact, "default", ann, vs, embedder=None)
    ann.add.assert_not_called()
    vs.upsert.assert_not_called()


def test_respects_unavailable_vector_store():
    fact = _fact(embedding=[0.3] * 8)
    ann = MagicMock()
    vs = MagicMock(); vs.available = False
    _upsert_fact_vectors(fact, "default", ann, vs, embedder=None)
    ann.add.assert_called_once()       # ANN still gets it
    vs.upsert.assert_not_called()      # vec store skipped when unavailable
