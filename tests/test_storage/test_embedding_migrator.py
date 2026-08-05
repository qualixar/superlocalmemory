# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for storage.embedding_migrator — model switch detection + re-indexing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from superlocalmemory.storage.embedding_migrator import (
    _NO_MODEL,
    _REINDEX_BATCH_SIZE,
    _model_signature,
    _read_stored_signature,
    _write_stored_signature,
    check_embedding_migration,
    run_embedding_migration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    tmp_path: Path,
    provider: str = "sentence-transformers",
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    dimension: int = 768,
    profile: str = "default",
):
    """Create a minimal SLMConfig with specific embedding settings."""
    from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
    from superlocalmemory.storage.models import Mode

    config = SLMConfig(
        mode=Mode.A,
        base_dir=tmp_path,
        embedding=EmbeddingConfig(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
        ),
        active_profile=profile,
    )
    return config


def _make_mock_db(facts: list[tuple[str, str]] | None = None):
    """Create a mock DB that returns the given (fact_id, content) pairs."""
    db = MagicMock()
    if facts is None:
        facts = []
    rows = []
    for fid, content in facts:
        row = MagicMock()
        row.__iter__ = MagicMock(return_value=iter([fid, content]))
        row.keys = MagicMock(return_value=["fact_id", "content"])
        # Support dict(r) via sqlite3.Row-like interface
        row_dict = {"fact_id": fid, "content": content}
        row.__getitem__ = lambda self, k, d=row_dict: d[k]

        class _FakeRow:
            def __init__(self, d):
                self._d = d
            def __getitem__(self, k):
                return self._d[k]
            def keys(self):
                return self._d.keys()

        rows.append(_FakeRow(row_dict))

    db.execute.return_value = rows
    return db


def _make_mock_embedder(dimension: int = 768):
    """Create a mock embedder that returns deterministic vectors."""
    emb = MagicMock()

    def _embed_batch(texts):
        result = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % 2**31)
            vec = rng.randn(dimension).astype(np.float32).tolist()
            result.append(vec)
        return result

    emb.embed_batch.side_effect = _embed_batch
    return emb


# ---------------------------------------------------------------------------
# _model_signature
# ---------------------------------------------------------------------------

class TestModelSignature:
    def test_deterministic(self, tmp_path):
        cfg = _make_config(tmp_path)
        sig1 = _model_signature(cfg)
        sig2 = _model_signature(cfg)
        assert sig1 == sig2

    def test_different_provider_same_signature(self, tmp_path):
        """V3.3.4+: provider doesn't affect signature — same model = same space."""
        cfg_a = _make_config(tmp_path, provider="sentence-transformers")
        cfg_b = _make_config(tmp_path, provider="ollama")
        assert _model_signature(cfg_a) == _model_signature(cfg_b)

    def test_different_model(self, tmp_path):
        cfg_a = _make_config(tmp_path, model_name="model-a")
        cfg_b = _make_config(tmp_path, model_name="model-b")
        assert _model_signature(cfg_a) != _model_signature(cfg_b)

    def test_different_dimension(self, tmp_path):
        cfg_a = _make_config(tmp_path, dimension=768)
        cfg_b = _make_config(tmp_path, dimension=3072)
        assert _model_signature(cfg_a) != _model_signature(cfg_b)

    def test_format(self, tmp_path):
        """V3.3.4+: signature is model_name::dimension (provider excluded)."""
        cfg = _make_config(
            tmp_path, provider="ollama",
            model_name="nomic", dimension=768,
        )
        sig = _model_signature(cfg)
        assert sig == "nomic::768"


# ---------------------------------------------------------------------------
# _read_stored_signature / _write_stored_signature
# ---------------------------------------------------------------------------

class TestStoredSignature:
    def test_no_config_file(self, tmp_path):
        assert _read_stored_signature(tmp_path) == _NO_MODEL

    def test_write_then_read(self, tmp_path):
        _write_stored_signature(tmp_path, "test::sig::768")
        assert _read_stored_signature(tmp_path) == "test::sig::768"

    def test_write_preserves_other_keys(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"mode": "a", "active_profile": "default"}))
        _write_stored_signature(tmp_path, "my::sig::768")
        data = json.loads(config_path.read_text())
        assert data["mode"] == "a"
        assert data["active_profile"] == "default"
        assert data["embedding_signature"] == "my::sig::768"

    def test_corrupt_json_returns_no_model(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{invalid json!!!")
        assert _read_stored_signature(tmp_path) == _NO_MODEL

    def test_missing_key_returns_no_model(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"mode": "a"}))
        assert _read_stored_signature(tmp_path) == _NO_MODEL

    def test_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        _write_stored_signature(deep, "sig")
        assert _read_stored_signature(deep) == "sig"


# ---------------------------------------------------------------------------
# check_embedding_migration
# ---------------------------------------------------------------------------

class TestCheckEmbeddingMigration:
    def test_first_run_no_migration(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert check_embedding_migration(cfg) is False
        # Signature should now be stored
        assert _read_stored_signature(tmp_path) != _NO_MODEL

    def test_same_model_no_migration(self, tmp_path):
        cfg = _make_config(tmp_path)
        # First run stores signature
        check_embedding_migration(cfg)
        # Second run with same config
        assert check_embedding_migration(cfg) is False

    def test_different_model_triggers_migration(self, tmp_path):
        cfg_a = _make_config(tmp_path, provider="ollama", model_name="nomic")
        check_embedding_migration(cfg_a)
        cfg_b = _make_config(
            tmp_path, provider="sentence-transformers",
            model_name="nomic-ai/nomic-embed-text-v1.5",
        )
        assert check_embedding_migration(cfg_b) is True

    def test_different_dimension_triggers_migration(self, tmp_path):
        cfg_a = _make_config(tmp_path, dimension=768)
        check_embedding_migration(cfg_a)
        cfg_b = _make_config(tmp_path, dimension=3072)
        assert check_embedding_migration(cfg_b) is True

    def test_provider_change_no_migration(self, tmp_path):
        """V3.3.4+: provider change alone doesn't trigger migration."""
        cfg_a = _make_config(tmp_path, provider="ollama")
        check_embedding_migration(cfg_a)
        cfg_b = _make_config(tmp_path, provider="sentence-transformers")
        assert check_embedding_migration(cfg_b) is False


# ---------------------------------------------------------------------------
# run_embedding_migration
# ---------------------------------------------------------------------------

class TestRunEmbeddingMigration:
    def test_no_embedder_returns_zero(self, tmp_path):
        cfg = _make_config(tmp_path)
        db = _make_mock_db()
        assert run_embedding_migration(cfg, db, None) == 0

    def test_no_facts_returns_zero(self, tmp_path):
        cfg = _make_config(tmp_path)
        db = _make_mock_db(facts=[])
        emb = _make_mock_embedder()
        result = run_embedding_migration(cfg, db, emb)
        assert result == 0

    def test_reindexes_all_facts(self, tmp_path):
        facts = [
            ("fact-1", "Alice went to Paris"),
            ("fact-2", "Bob likes pizza"),
            ("fact-3", "Weather is sunny"),
        ]
        cfg = _make_config(tmp_path)
        db = _make_mock_db(facts=facts)

        # Make execute return facts on first call, then succeed on updates
        call_count = [0]
        original_return = db.execute.return_value

        def _side_effect(sql, params=()):
            call_count[0] += 1
            if call_count[0] == 1:
                return original_return
            return []

        db.execute.side_effect = _side_effect

        emb = _make_mock_embedder()
        result = run_embedding_migration(cfg, db, emb)

        assert result == 3
        # Verify embed_batch was called
        emb.embed_batch.assert_called_once()

    def test_updates_signature_after_migration(self, tmp_path):
        """V3.3.4+: signature is model_name::dimension (no provider)."""
        cfg = _make_config(tmp_path, provider="new-provider", model_name="test-model", dimension=512)
        db = _make_mock_db(facts=[])
        emb = _make_mock_embedder()
        run_embedding_migration(cfg, db, emb)
        stored = _read_stored_signature(tmp_path)
        assert "test-model::512" in stored

    def test_embed_batch_failure_stops_gracefully(self, tmp_path):
        facts = [("f1", "content 1")]
        cfg = _make_config(tmp_path)
        _write_stored_signature(tmp_path, "old-model::768")
        db = _make_mock_db(facts=facts)
        emb = _make_mock_embedder()
        emb.embed_batch.side_effect = RuntimeError("GPU exploded")
        result = run_embedding_migration(cfg, db, emb)
        assert result == 0
        assert _read_stored_signature(tmp_path) == "old-model::768"

    def test_individual_update_failure_aborts_without_activation(self, tmp_path):
        facts = [
            ("f1", "content 1"),
            ("f2", "content 2"),
        ]
        cfg = _make_config(tmp_path)
        _write_stored_signature(tmp_path, "old-model::768")
        db = _make_mock_db(facts=facts)
        db.transaction.return_value.__enter__.return_value = None
        db.transaction.return_value.__exit__.return_value = False

        call_count = [0]
        original_return = db.execute.return_value

        def _side_effect(sql, params=()):
            call_count[0] += 1
            if call_count[0] == 1:
                return original_return
            # Fail on first UPDATE (f1), succeed on rest
            if call_count[0] == 2 and "UPDATE atomic_facts" in sql:
                raise RuntimeError("disk full")
            return []

        db.execute.side_effect = _side_effect

        emb = _make_mock_embedder()
        result = run_embedding_migration(cfg, db, emb)
        assert result == 0
        assert _read_stored_signature(tmp_path) == "old-model::768"

    def test_vector_count_mismatch_aborts_without_activation(self, tmp_path):
        facts = [("f1", "content 1"), ("f2", "content 2")]
        cfg = _make_config(tmp_path)
        _write_stored_signature(tmp_path, "old-model::768")
        db = _make_mock_db(facts=facts)
        emb = MagicMock()
        emb.embed_batch.return_value = [[0.1] * 768]

        result = run_embedding_migration(cfg, db, emb)

        assert result == 0
        assert _read_stored_signature(tmp_path) == "old-model::768"


# ---------------------------------------------------------------------------
# Mode config integration
# ---------------------------------------------------------------------------

class TestModeConfigDefaults:
    """Verify that Mode A/B/C configs have correct embedding/retrieval settings."""

    def test_mode_a_uses_sentence_transformers(self, tmp_path):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode

        cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
        assert cfg.embedding.provider == "sentence-transformers"
        # V3.3.18: PyTorch backend (empty string) — ONNX leaked 28GB on ARM64
        assert cfg.retrieval.use_cross_encoder is True
        assert cfg.retrieval.cross_encoder_backend == ""

    def test_mode_b_uses_ollama(self, tmp_path):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode

        cfg = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
        assert cfg.embedding.provider == "ollama"
        # V3.3.18: PyTorch backend (empty string) — ONNX leaked 28GB on ARM64
        assert cfg.retrieval.use_cross_encoder is True
        assert cfg.retrieval.cross_encoder_backend == ""

    def test_mode_c_keeps_cross_encoder(self, tmp_path):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode

        cfg = SLMConfig.for_mode(Mode.C, base_dir=tmp_path)
        assert cfg.retrieval.use_cross_encoder is True

    def test_mode_a_explicit_override(self, tmp_path):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode

        cfg = SLMConfig.for_mode(
            Mode.A, base_dir=tmp_path,
            embedding_provider="ollama",
        )
        assert cfg.embedding.provider == "ollama"

    def test_mode_b_explicit_override(self, tmp_path):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode

        cfg = SLMConfig.for_mode(
            Mode.B, base_dir=tmp_path,
            embedding_provider="sentence-transformers",
        )
        assert cfg.embedding.provider == "sentence-transformers"
