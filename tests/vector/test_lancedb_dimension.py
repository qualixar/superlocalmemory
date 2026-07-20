# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""v3.7.6 (#72): the LanceDB vector width must follow the configured embedding
dimension instead of a hardcoded 768, so custom OpenAI-compatible endpoints
(e.g. 1024d Qwen3-Embedding) no longer collide with a fixed schema/decode."""

from __future__ import annotations

import importlib.util
import shutil
import struct

import numpy as np
import pytest

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        importlib.util.find_spec("lancedb") is None,
        reason="LanceDB optional dependency not installed",
    ),
]


def _backend(path: str, dimension=None):
    from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend

    return LanceDBVectorBackend(path, dimension=dimension)


def test_default_dimension_is_768(tmp_path):
    be = _backend(str(tmp_path / "d_default"))
    try:
        assert be.dimension == 768
    finally:
        be.close()


def test_custom_dimension_schema_and_search(tmp_path):
    be = _backend(str(tmp_path / "d_1024"), dimension=1024)
    try:
        assert be.dimension == 1024
        vecs = [np.random.rand(1024).tolist() for _ in range(3)]
        added = be.add_vectors(["a", "b", "c"], vecs, ["active"] * 3)
        assert added == 3
        hits = be.similarity_search(vecs[0], top_k=3)
        assert any(fid == "a" for fid, _ in hits)
    finally:
        be.close()
        shutil.rmtree(str(tmp_path / "d_1024"), ignore_errors=True)


def test_decode_blob_uses_configured_width(tmp_path):
    be = _backend(str(tmp_path / "d_decode"), dimension=1024)
    try:
        blob = struct.pack("1024f", *([0.1] * 1024))
        vec = be._decode_vector_blob(blob)
        assert len(vec) == 1024
        # A 768-wide blob must now be rejected against a 1024 backend.
        with pytest.raises(ValueError):
            be._decode_vector_blob(struct.pack("768f", *([0.1] * 768)))
    finally:
        be.close()


def test_existing_table_width_is_adopted_over_request(tmp_path):
    """A persisted 1024d store stays readable even if a later config asks 768."""
    path = str(tmp_path / "d_persist")
    be = _backend(path, dimension=1024)
    be.add_vectors(["x"], [np.random.rand(1024).tolist()], ["active"])
    be.close()
    reopened = _backend(path, dimension=768)
    try:
        assert reopened.dimension == 1024  # on-disk width wins
    finally:
        reopened.close()
        shutil.rmtree(path, ignore_errors=True)
