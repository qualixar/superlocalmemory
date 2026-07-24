# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for v3.8.2 embedding-signature normalization (self-heal).

The SAME embedding model has been recorded under different name strings across
releases — notably the HuggingFace org prefix drifting
(``nomic-ai/nomic-embed-text-v1.5`` vs the bare ``nomic-embed-text-v1.5``).
A prefix-only difference is the same vector space and MUST NOT trigger a full
multi-hour re-embed on upgrade. These tests pin that behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.storage.embedding_migrator import (
    _normalize_signature,
    check_embedding_migration,
)


class TestNormalizeSignature:
    def test_org_prefix_collapses(self) -> None:
        assert _normalize_signature("nomic-ai/nomic-embed-text-v1.5::768") == \
            _normalize_signature("nomic-embed-text-v1.5::768")

    def test_identical_unchanged(self) -> None:
        assert _normalize_signature("nomic-embed-text-v1.5::768") == \
            "nomic-embed-text-v1.5::768"

    def test_different_model_differs(self) -> None:
        assert _normalize_signature("nomic-embed-text-v1.5::768") != \
            _normalize_signature("nomic-embed-text-v2::768")

    def test_different_dimension_differs(self) -> None:
        assert _normalize_signature("nomic-embed-text-v1.5::768") != \
            _normalize_signature("nomic-embed-text-v1.5::1024")

    def test_no_dimension_suffix(self) -> None:
        # graceful on a bare model name without ::dim
        assert _normalize_signature("org/model") == "model"


def _cfg(tmp_path: Path, model_name: str, dimension: int = 768):
    """Minimal config duck-type: .embedding + .base_dir (what the migrator reads)."""
    return SimpleNamespace(
        embedding=SimpleNamespace(model_name=model_name, dimension=dimension),
        base_dir=tmp_path,
    )


class TestCheckEmbeddingMigration:
    def test_prefix_drift_does_not_trigger_reembed(self, tmp_path: Path) -> None:
        """Stored 'nomic-ai/...' + current bare '...' => NO migration, and the
        stored signature is refreshed to the current form (converges)."""
        # seed config.json with the OLD prefixed signature (as a real M5 DB has)
        (tmp_path / "config.json").write_text(json.dumps(
            {"embedding_signature": "nomic-ai/nomic-embed-text-v1.5::768"}))
        cfg = _cfg(tmp_path, "nomic-embed-text-v1.5", 768)

        assert check_embedding_migration(cfg) is False  # no re-embed

        # stored signature refreshed to current form
        stored = json.loads((tmp_path / "config.json").read_text())["embedding_signature"]
        assert stored == "nomic-embed-text-v1.5::768"
        # idempotent second call
        assert check_embedding_migration(cfg) is False

    def test_first_run_initializes_no_migration(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path, "nomic-embed-text-v1.5", 768)
        assert check_embedding_migration(cfg) is False
        stored = json.loads((tmp_path / "config.json").read_text())["embedding_signature"]
        assert stored == "nomic-embed-text-v1.5::768"

    def test_genuine_model_change_triggers_migration(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps(
            {"embedding_signature": "nomic-embed-text-v1.5::768"}))
        cfg = _cfg(tmp_path, "bge-large-en-v1.5", 1024)
        assert check_embedding_migration(cfg) is True

    def test_dimension_change_triggers_migration(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps(
            {"embedding_signature": "nomic-embed-text-v1.5::768"}))
        cfg = _cfg(tmp_path, "nomic-embed-text-v1.5", 1024)
        assert check_embedding_migration(cfg) is True
