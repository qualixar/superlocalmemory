# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""Root conftest — shared fixtures for Phase 0 Safety Net.

Provides in-memory DB, mock embedder, Mode A config, and
engine-with-mock-deps fixtures used across all test modules.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database with full SLM schema.

    Returns a real sqlite3 Connection backed by :memory:.
    Gives real SQL execution without touching disk.
    """
    from superlocalmemory.storage import schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    schema.create_all_tables(conn)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic 768-dim vectors.

    Uses seeded RNG keyed on input string for reproducibility.
    Implements: embed(), is_available, compute_fisher_params().
    """
    emb = MagicMock()

    def _embed(text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    emb.embed.side_effect = _embed
    emb.is_available = True
    emb.compute_fisher_params.return_value = ([0.0] * 768, [1.0] * 768)
    return emb


@pytest.fixture
def mode_a_config(tmp_path):
    """SLMConfig for Mode A using tmp_path as base_dir."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    return config


@pytest.fixture
def engine_with_mock_deps(mode_a_config, mock_embedder, tmp_path):
    """A MemoryEngine with mocked LLM and embedder for fast unit tests.

    Initializes with real DB (on disk in tmp_path) and real schema,
    but mocked embeddings and no LLM. Suitable for testing store/recall
    flow without heavy ML dependencies.
    """
    from superlocalmemory.core.engine import MemoryEngine

    engine = MemoryEngine(mode_a_config)

    # Patch embedder initialization to use our mock
    with patch('superlocalmemory.core.engine_wiring.init_embedder', return_value=mock_embedder):
        engine.initialize()
        engine._embedder = mock_embedder

    yield engine
    engine.close()
