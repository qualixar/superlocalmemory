# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-08 invariant: aborted embedding migration is not a silent zero."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
from superlocalmemory.storage.embedding_migrator import (
    EmbeddingMigrationAborted,
    _read_stored_signature,
    _write_stored_signature,
    run_embedding_migration,
)
from superlocalmemory.storage.models import Mode


def _make_config(tmp_path: Path) -> SLMConfig:
    cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    cfg.embedding = EmbeddingConfig(
        provider="sentence-transformers",
        model_name="test-model",
        dimension=768,
    )
    return cfg


def _make_mock_db(facts=None):
    if facts is None:
        facts = [("f1", "content one")]
    norm = []
    for f in facts:
        if len(f) == 2:
            norm.append({"fact_id": f[0], "profile_id": "default", "content": f[1]})
        else:
            norm.append({"fact_id": f[0], "profile_id": f[1], "content": f[2]})
    db = MagicMock()
    db.execute.return_value = norm
    return db


def _make_mock_embedder(dim: int = 768):
    emb = MagicMock()
    emb.embed_batch.return_value = [[0.1] * dim]
    return emb


def test_noop_no_embedder_returns_zero(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    assert run_embedding_migration(cfg, _make_mock_db(), None) == 0


def test_abort_raises_not_zero(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    _write_stored_signature(tmp_path, "old-model::768")
    db = _make_mock_db(facts=[("f1", "content 1")])
    emb = _make_mock_embedder()
    emb.embed_batch.side_effect = RuntimeError("GPU exploded")

    with pytest.raises(EmbeddingMigrationAborted):
        run_embedding_migration(cfg, db, emb)

    # Distinct from no-op: no-op returns 0, abort raises.
    assert run_embedding_migration(cfg, db, None) == 0


def test_abort_leaves_stored_signature_unchanged(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    _write_stored_signature(tmp_path, "old-model::768")
    db = _make_mock_db(facts=[("f1", "content 1")])
    emb = _make_mock_embedder()
    emb.embed_batch.side_effect = RuntimeError("GPU exploded")

    with pytest.raises(EmbeddingMigrationAborted):
        run_embedding_migration(cfg, db, emb)

    assert _read_stored_signature(tmp_path) == "old-model::768"
