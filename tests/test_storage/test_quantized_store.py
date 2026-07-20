# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for QuantizedEmbeddingStore.

TDD sequence (LLD Section 6):
  11. test_store_and_load
  12. test_search_returns_results
  13. test_compress_fact_reduces_size
  14. test_batch_compress
  15. test_migration_float32_to_int8 (B-HIGH-01 audit fix)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from superlocalmemory.core.config import (
    PolarQuantConfig,
    QJLConfig,
    QuantizationConfig,
)
from superlocalmemory.math.polar_quant import PolarQuantEncoder, QuantizedEmbedding
from superlocalmemory.math.qjl import QJLEncoder
from superlocalmemory.storage.quantized_store import QuantizedEmbeddingStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_test_db(tmp_path: Path) -> MagicMock:
    """Create a mock DatabaseManager backed by real SQLite."""
    db = MagicMock()
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")  # No FK deps for unit test

    # Create required tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS polar_embeddings (
            fact_id     TEXT PRIMARY KEY,
            profile_id  TEXT NOT NULL,
            radius      REAL NOT NULL,
            angle_indices BLOB NOT NULL,
            qjl_bits    BLOB,
            bit_width   INTEGER NOT NULL DEFAULT 4,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_polar_profile
            ON polar_embeddings (profile_id);

        CREATE TABLE IF NOT EXISTS embedding_quantization_metadata (
            fact_id               TEXT PRIMARY KEY,
            profile_id            TEXT NOT NULL,
            quantization_level    TEXT NOT NULL DEFAULT 'float32',
            bit_width             INTEGER NOT NULL DEFAULT 32,
            compressed_size_bytes INTEGER,
            created_at            TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_eqm_profile_level
            ON embedding_quantization_metadata (profile_id, quantization_level);

        CREATE TABLE IF NOT EXISTS fact_retention (
            fact_id         TEXT PRIMARY KEY,
            profile_id      TEXT NOT NULL,
            retention_score REAL NOT NULL DEFAULT 1.0,
            lifecycle_zone  TEXT NOT NULL DEFAULT 'active'
        );
    """)
    conn.commit()

    # Wire execute to the real connection
    def _execute(sql, params=()):
        rows = conn.execute(sql, params).fetchall()
        conn.commit()
        return rows

    db.execute.side_effect = _execute
    db._test_conn = conn  # Keep ref for cleanup
    return db


@pytest.fixture
def test_db(tmp_path: Path) -> MagicMock:
    db = _create_test_db(tmp_path)
    yield db
    db._test_conn.close()


@pytest.fixture
def polar_encoder(tmp_path: Path) -> PolarQuantEncoder:
    config = PolarQuantConfig(
        dimension=768,
        rotation_matrix_path=str(tmp_path / "polar_rot.npy"),
        seed=42,
    )
    return PolarQuantEncoder(config)


@pytest.fixture
def qjl_encoder() -> QJLEncoder:
    return QJLEncoder(QJLConfig(projection_dim=128, seed=43))


@pytest.fixture
def quant_config() -> QuantizationConfig:
    return QuantizationConfig()


@pytest.fixture
def store(
    test_db: MagicMock,
    polar_encoder: PolarQuantEncoder,
    qjl_encoder: QJLEncoder,
    quant_config: QuantizationConfig,
) -> QuantizedEmbeddingStore:
    return QuantizedEmbeddingStore(test_db, polar_encoder, qjl_encoder, quant_config)


def _random_vec(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# 11. Store and load roundtrip
# ---------------------------------------------------------------------------


def test_store_and_load(store: QuantizedEmbeddingStore) -> None:
    """Store a quantized embedding, load it back, verify fields match."""
    qe = QuantizedEmbedding(
        fact_id="fact-001",
        radius=1.5,
        angle_indices=b"\x01\x02\x03",
        bit_width=4,
        qjl_bits=b"\xff",
    )

    result = store.store("fact-001", "profile-1", qe)
    assert result is True

    loaded = store.load("fact-001", "profile-1")
    assert loaded is not None
    assert loaded.fact_id == "fact-001"
    assert loaded.bit_width == 4
    assert loaded.angle_indices == b"\x01\x02\x03"
    assert loaded.qjl_bits == b"\xff"
    assert abs(loaded.radius - 1.5) < 0.01


def test_load_nonexistent(store: QuantizedEmbeddingStore) -> None:
    """Load non-existent fact returns None."""
    loaded = store.load("nonexistent", "profile-1")
    assert loaded is None


# ---------------------------------------------------------------------------
# 12. Search returns results
# ---------------------------------------------------------------------------


def test_search_returns_results(
    store: QuantizedEmbeddingStore,
    polar_encoder: PolarQuantEncoder,
    test_db: MagicMock,
) -> None:
    """Search returns ranked results for profile's polar embeddings."""
    # Insert fact_retention rows so JOIN works
    test_db._test_conn.execute(
        "INSERT INTO fact_retention (fact_id, profile_id, retention_score, lifecycle_zone) "
        "VALUES ('f1', 'p1', 0.9, 'active')"
    )
    test_db._test_conn.execute(
        "INSERT INTO fact_retention (fact_id, profile_id, retention_score, lifecycle_zone) "
        "VALUES ('f2', 'p1', 0.6, 'warm')"
    )
    test_db._test_conn.commit()

    # Compress two facts
    v1 = _random_vec(768, seed=10)
    v2 = _random_vec(768, seed=11)
    store.compress_fact("f1", "p1", v1, 4)
    store.compress_fact("f2", "p1", v2, 4)

    # Search with v1 as query
    results = store.search(v1, "p1", top_k=10)
    assert len(results) >= 1
    # First result should be f1 (self-match)
    fact_ids = [r[0] for r in results]
    assert "f1" in fact_ids


def test_search_filters_persisted_precision_tiers(
    store: QuantizedEmbeddingStore,
    test_db: MagicMock,
) -> None:
    """Int8 and polar retrieval must query distinct persisted rows."""
    for fact_id in ("f-int8", "f-polar"):
        test_db._test_conn.execute(
            "INSERT INTO fact_retention "
            "(fact_id, profile_id, retention_score, lifecycle_zone) "
            "VALUES (?, 'p1', 1.0, 'active')",
            (fact_id,),
        )
    test_db._test_conn.commit()

    query = _random_vec(768, seed=12)
    assert store.compress_fact("f-int8", "p1", query, 8)
    assert store.compress_fact("f-polar", "p1", query, 4)

    int8 = store.search(query, "p1", top_k=10, bit_widths=(8,))
    polar = store.search(query, "p1", top_k=10, bit_widths=(2, 4))

    assert [fact_id for fact_id, _ in int8] == ["f-int8"]
    assert [fact_id for fact_id, _ in polar] == ["f-polar"]


# ---------------------------------------------------------------------------
# 13. compress_fact reduces size
# ---------------------------------------------------------------------------


def test_compress_fact_reduces_size(
    store: QuantizedEmbeddingStore,
) -> None:
    """compress_fact produces a stored polar embedding with < float32 size."""
    v = _random_vec(768, seed=20)
    float32_size = 768 * 4  # 3072 bytes

    result = store.compress_fact("f-compress", "p1", v, target_bit_width=4)
    assert result is True

    loaded = store.load("f-compress", "p1")
    assert loaded is not None
    assert len(loaded.angle_indices) < float32_size


# ---------------------------------------------------------------------------
# 14. Batch compress
# ---------------------------------------------------------------------------


def test_batch_compress(store: QuantizedEmbeddingStore) -> None:
    """batch_compress processes all facts and returns count."""
    embeddings = {
        f"bf-{i}": _random_vec(768, seed=30 + i) for i in range(5)
    }
    fact_ids = list(embeddings.keys())

    count = store.batch_compress(fact_ids, "p1", embeddings, target_bit_width=4)
    assert count == 5

    # Verify all are loadable
    for fid in fact_ids:
        loaded = store.load(fid, "p1")
        assert loaded is not None
        assert loaded.bit_width == 4


# ---------------------------------------------------------------------------
# 15. Store without QJL (HR-07: QJL is optional)
# ---------------------------------------------------------------------------


def test_store_without_qjl(
    test_db: MagicMock,
    polar_encoder: PolarQuantEncoder,
    quant_config: QuantizationConfig,
) -> None:
    """System works without QJL encoder (HR-07)."""
    store_no_qjl = QuantizedEmbeddingStore(
        test_db, polar_encoder, None, quant_config,
    )
    v = _random_vec(768, seed=40)
    result = store_no_qjl.compress_fact("f-noqjl", "p1", v, target_bit_width=4)
    assert result is True

    loaded = store_no_qjl.load("f-noqjl", "p1")
    assert loaded is not None
    assert loaded.qjl_bits is None


# ---------------------------------------------------------------------------
# Error path coverage
# ---------------------------------------------------------------------------


def test_store_error_returns_false(
    polar_encoder: PolarQuantEncoder,
    qjl_encoder: QJLEncoder,
    quant_config: QuantizationConfig,
) -> None:
    """Store returns False when DB execute raises."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("db broke")
    s = QuantizedEmbeddingStore(db, polar_encoder, qjl_encoder, quant_config)

    qe = QuantizedEmbedding(
        fact_id="bad", radius=1.0, angle_indices=b"\x00",
        bit_width=4, qjl_bits=None,
    )
    assert s.store("bad", "p1", qe) is False


def test_load_error_returns_none(
    polar_encoder: PolarQuantEncoder,
    qjl_encoder: QJLEncoder,
    quant_config: QuantizationConfig,
) -> None:
    """Load returns None when DB execute raises."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("db broke")
    s = QuantizedEmbeddingStore(db, polar_encoder, qjl_encoder, quant_config)
    assert s.load("bad", "p1") is None


def test_search_error_returns_empty(
    polar_encoder: PolarQuantEncoder,
    qjl_encoder: QJLEncoder,
    quant_config: QuantizationConfig,
) -> None:
    """Search returns [] when DB execute raises."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("db broke")
    s = QuantizedEmbeddingStore(db, polar_encoder, qjl_encoder, quant_config)
    results = s.search(_random_vec(768, seed=99), "p1")
    assert results == []


def test_compress_fact_error_returns_false(
    polar_encoder: PolarQuantEncoder,
    qjl_encoder: QJLEncoder,
    quant_config: QuantizationConfig,
) -> None:
    """compress_fact returns False on encode error (wrong dimension)."""
    db = MagicMock()
    s = QuantizedEmbeddingStore(db, polar_encoder, qjl_encoder, quant_config)
    # Wrong dimension -- encode will raise
    bad_vec = np.zeros(10)
    assert s.compress_fact("bad", "p1", bad_vec, 4) is False


def test_search_no_rows(
    store: QuantizedEmbeddingStore,
) -> None:
    """Search with empty polar_embeddings returns []."""
    results = store.search(_random_vec(768, seed=88), "p-empty")
    assert results == []


def test_compress_at_8bit_no_qjl(
    store: QuantizedEmbeddingStore,
) -> None:
    """8-bit compression skips QJL (only applied at <= 4bit)."""
    v = _random_vec(768, seed=50)
    result = store.compress_fact("f-8bit", "p1", v, target_bit_width=8)
    assert result is True
    loaded = store.load("f-8bit", "p1")
    assert loaded is not None
    assert loaded.bit_width == 8
    assert loaded.qjl_bits is None


def test_batch_compress_missing_embedding(
    store: QuantizedEmbeddingStore,
) -> None:
    """batch_compress skips fact_ids not in embeddings dict."""
    embeddings = {"exists": _random_vec(768, seed=60)}
    count = store.batch_compress(
        ["exists", "missing"], "p1", embeddings, target_bit_width=4,
    )
    assert count == 1
