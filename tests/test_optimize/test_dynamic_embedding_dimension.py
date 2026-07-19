"""Optimize semantic cache must honor the configured embedding width."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from superlocalmemory.optimize.cache.centroid_store import CentroidStore
from superlocalmemory.optimize.cache.semantic import VCacheSemantic


def _semantic_config() -> SimpleNamespace:
    return SimpleNamespace(
        semantic_enabled=True,
        semantic_boundary_init=0.95,
        semantic_boundary_floor=0.85,
        semantic_boundary_ceiling=0.995,
        semantic_boundary_step=0.01,
        semantic_error_target=0.02,
        semantic_context_window_turns=3,
        semantic_max_index_entries=100,
    )


def test_semantic_cache_indexes_configured_1024d_vectors() -> None:
    db = MagicMock()
    db.boundary_get.return_value = None
    cache = VCacheSemantic(
        db=db,
        config=_semantic_config(),
        embedding_dimension=1024,
    )
    vector = np.ones(1024, dtype=np.float32)

    cache._set_inner("tenant", "entry", vector, "context")

    assert cache._index["tenant"][0][2].shape == (1024,)
    assert db.vec_add.call_args.kwargs["meta"]["dim"] == 1024


def test_centroid_rebuild_accepts_configured_1024d_vectors() -> None:
    vector = np.ones(1024, dtype=np.float32)
    db = MagicMock()
    db.get_all_vectors.return_value = [
        (f"entry-{index}", vector.tobytes(), "context")
        for index in range(5)
    ]
    centroids = CentroidStore(embedding_dimension=1024)

    centroids.rebuild_from_db(db, "tenant")

    assert centroids.count("tenant") == 5
    assert centroids.get_centroid("tenant").shape == (1024,)


def test_configured_dimension_rejects_incompatible_stale_vectors() -> None:
    db = MagicMock()
    db.boundary_get.return_value = None
    cache = VCacheSemantic(
        db=db,
        config=_semantic_config(),
        embedding_dimension=1024,
    )

    cache._set_inner("tenant", "entry", np.ones(768, dtype=np.float32), "context")

    db.vec_add.assert_not_called()
    assert "tenant" not in cache._index
