# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""A fact correction must replace the fact's semantic representation.

These tests exercise the REAL ``VectorStore`` and ``ANNIndex`` — no mocks —
so a regression that calls a method the backend does not expose is caught
here. The sync helper is deliberately fail-open (it logs and swallows), which
means a wrong method name would otherwise pass silently and leave stale
embeddings queryable forever.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core.mutations import _sync_vector_ann
from superlocalmemory.retrieval.ann_index import ANNIndex
from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig


class _Retrieval:
    """Real attribute holder — intentionally NOT a mock, so a call to a
    non-existent backend method raises AttributeError instead of being
    fabricated and hidden."""

    def __init__(self, vector_store: object | None, ann_index: object | None) -> None:
        self._vector_store = vector_store
        self._ann_index = ann_index


def test_correction_replaces_ann_embedding() -> None:
    ann = ANNIndex(dimension=4)
    ann.add("f1", [1.0, 0.0, 0.0, 0.0])
    retrieval = _Retrieval(vector_store=None, ann_index=ann)

    _sync_vector_ann(
        retrieval, "f1", "default", [0.0, 1.0, 0.0, 0.0], operation="update"
    )

    idx = ann._id_to_idx["f1"]
    stored = ann._vectors[idx]
    # The normalized vector now points along the corrected axis, not the old one.
    assert stored[1] == pytest.approx(1.0, abs=1e-6)
    assert stored[0] == pytest.approx(0.0, abs=1e-6)


def test_correction_replaces_vector_store_embedding(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "vec.db", VectorStoreConfig(dimension=4))
    if not store.available:
        pytest.skip("vector extension unavailable in this environment")

    store.upsert("f1", "default", [1.0, 0.0, 0.0, 0.0])
    retrieval = _Retrieval(vector_store=store, ann_index=None)

    _sync_vector_ann(
        retrieval, "f1", "default", [0.0, 1.0, 0.0, 0.0], operation="update"
    )

    # A query along the corrected direction now matches strongly; if the old
    # embedding had survived, this score would be ~0.
    hits = store.search([0.0, 1.0, 0.0, 0.0], top_k=1, profile_id="default")
    assert hits and hits[0][0] == "f1"
    assert hits[0][1] > 0.99


def test_delete_removes_from_both_backends(tmp_path: Path) -> None:
    ann = ANNIndex(dimension=4)
    ann.add("f1", [1.0, 0.0, 0.0, 0.0])
    store = VectorStore(tmp_path / "vec.db", VectorStoreConfig(dimension=4))
    store.upsert("f1", "default", [1.0, 0.0, 0.0, 0.0])
    retrieval = _Retrieval(vector_store=store, ann_index=ann)

    _sync_vector_ann(retrieval, "f1", "default", None, operation="delete")

    assert "f1" not in ann._id_to_idx
    if store.available:
        assert store.search([1.0, 0.0, 0.0, 0.0], top_k=1, profile_id="default") == []
