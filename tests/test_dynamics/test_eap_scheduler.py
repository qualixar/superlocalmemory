# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for EAP Scheduler (Embedding-Aware Precision).

TDD sequence (LLD Section 6):
  18. test_retention_to_bitwidth_mapping
  19. test_eap_cycle_compresses_cold_facts
  22. test_storage_reduction_at_scale (B-HIGH-01 audit fix)
  23. test_latency_degradation (B-HIGH-01 audit fix)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from superlocalmemory.core.config import (
    ForgettingConfig,
    PolarQuantConfig,
    QJLConfig,
    QuantizationConfig,
)
from superlocalmemory.dynamics.eap_scheduler import EAPScheduler, retention_to_bit_width
from superlocalmemory.math.ebbinghaus import EbbinghausCurve
from superlocalmemory.math.polar_quant import PolarQuantEncoder
from superlocalmemory.math.qjl import QJLEncoder
from superlocalmemory.storage.quantized_store import QuantizedEmbeddingStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _random_vec(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _create_test_db(tmp_path: Path) -> MagicMock:
    """Create mock DatabaseManager backed by real SQLite."""
    db = MagicMock()
    conn = sqlite3.connect(str(tmp_path / "eap_test.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")

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
        CREATE TABLE IF NOT EXISTS embedding_quantization_metadata (
            fact_id               TEXT PRIMARY KEY,
            profile_id            TEXT NOT NULL,
            quantization_level    TEXT NOT NULL DEFAULT 'float32',
            bit_width             INTEGER NOT NULL DEFAULT 32,
            compressed_size_bytes INTEGER,
            created_at            TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS fact_retention (
            fact_id         TEXT PRIMARY KEY,
            profile_id      TEXT NOT NULL,
            retention_score REAL NOT NULL DEFAULT 1.0,
            memory_strength REAL NOT NULL DEFAULT 1.0,
            access_count    INTEGER NOT NULL DEFAULT 0,
            last_accessed_at TEXT,
            last_computed_at TEXT NOT NULL DEFAULT (datetime('now')),
            lifecycle_zone  TEXT NOT NULL DEFAULT 'active'
        );
    """)
    conn.commit()

    def _execute(sql, params=()):
        rows = conn.execute(sql, params).fetchall()
        conn.commit()
        return rows

    db.execute.side_effect = _execute
    db._test_conn = conn
    return db


@pytest.fixture
def test_db(tmp_path: Path) -> MagicMock:
    db = _create_test_db(tmp_path)
    yield db
    db._test_conn.close()


@pytest.fixture
def polar_encoder(tmp_path: Path) -> PolarQuantEncoder:
    return PolarQuantEncoder(PolarQuantConfig(
        dimension=768,
        rotation_matrix_path=str(tmp_path / "polar_rot.npy"),
        seed=42,
    ))


@pytest.fixture
def qjl_encoder() -> QJLEncoder:
    return QJLEncoder(QJLConfig(projection_dim=128, seed=43))


@pytest.fixture
def quant_config() -> QuantizationConfig:
    return QuantizationConfig(keep_float32_backup=True)


@pytest.fixture
def quantized_store(
    test_db: MagicMock,
    polar_encoder: PolarQuantEncoder,
    qjl_encoder: QJLEncoder,
    quant_config: QuantizationConfig,
) -> QuantizedEmbeddingStore:
    return QuantizedEmbeddingStore(test_db, polar_encoder, qjl_encoder, quant_config)


@pytest.fixture
def ebbinghaus() -> EbbinghausCurve:
    return EbbinghausCurve(ForgettingConfig())


@pytest.fixture
def scheduler(
    test_db: MagicMock,
    ebbinghaus: EbbinghausCurve,
    quantized_store: QuantizedEmbeddingStore,
    quant_config: QuantizationConfig,
) -> EAPScheduler:
    return EAPScheduler(test_db, ebbinghaus, quantized_store, quant_config)


# ---------------------------------------------------------------------------
# 18. Retention-to-bitwidth mapping
# ---------------------------------------------------------------------------


def test_retention_to_bitwidth_mapping() -> None:
    """Verify the retention -> bit_width mapping from LLD."""
    assert retention_to_bit_width(0.9) == 32   # R > 0.8
    assert retention_to_bit_width(0.81) == 32
    assert retention_to_bit_width(0.8) == 8    # R > 0.5 (boundary: 0.8 is NOT > 0.8)
    assert retention_to_bit_width(0.6) == 8
    assert retention_to_bit_width(0.51) == 8
    assert retention_to_bit_width(0.5) == 4    # R > 0.2 (boundary)
    assert retention_to_bit_width(0.3) == 4
    assert retention_to_bit_width(0.21) == 4
    assert retention_to_bit_width(0.2) == 2    # R > 0.05 (boundary)
    assert retention_to_bit_width(0.1) == 2
    assert retention_to_bit_width(0.06) == 2
    assert retention_to_bit_width(0.05) == 0   # R <= 0.05 (forgotten)
    assert retention_to_bit_width(0.01) == 0
    assert retention_to_bit_width(0.0) == 0


def test_eap_dry_run_computes_without_precision_mutation(
    scheduler: EAPScheduler,
    test_db: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_db.execute(
        "INSERT INTO fact_retention "
        "(fact_id, profile_id, retention_score, lifecycle_zone) "
        "VALUES (?, ?, ?, ?)",
        ("dry-fact", "test_profile", 0.3, "cold"),
    )
    downgrade = MagicMock(return_value=True)
    monkeypatch.setattr(scheduler, "_handle_downgrade", downgrade)

    stats = scheduler.run_eap_cycle("test_profile", dry_run=True)

    assert stats["total"] == 1
    assert stats["downgrades"] == 1
    downgrade.assert_not_called()
    rows = test_db.execute(
        "SELECT COUNT(*) AS n FROM embedding_quantization_metadata "
        "WHERE fact_id = ?",
        ("dry-fact",),
    )
    assert rows[0]["n"] == 0


# ---------------------------------------------------------------------------
# 19. EAP cycle compresses cold facts
# ---------------------------------------------------------------------------


def test_eap_cycle_compresses_cold_facts(
    scheduler: EAPScheduler,
    test_db: MagicMock,
    quantized_store: QuantizedEmbeddingStore,
) -> None:
    """Run EAP cycle: cold facts (R=0.3) should get compressed to 4-bit."""
    # Insert a fact with cold retention
    test_db._test_conn.execute(
        "INSERT INTO fact_retention "
        "(fact_id, profile_id, retention_score, memory_strength, access_count, lifecycle_zone) "
        "VALUES ('cold-1', 'p1', 0.3, 5.0, 3, 'cold')"
    )
    test_db._test_conn.commit()

    # Provide embedding via mock get_fact_embedding
    embedding = _random_vec(768, seed=50)

    with patch.object(scheduler, '_get_fact_embedding', return_value=embedding):
        stats = scheduler.run_eap_cycle("p1")

    assert stats["total"] >= 1
    assert stats["downgrades"] >= 1
    assert stats["errors"] == 0

    # Verify the fact was compressed
    loaded = quantized_store.load("cold-1", "p1")
    assert loaded is not None
    assert loaded.bit_width == 4


# ---------------------------------------------------------------------------
# Edge: skip facts already at target bit_width
# ---------------------------------------------------------------------------


def test_eap_cycle_skips_already_compressed(
    scheduler: EAPScheduler,
    test_db: MagicMock,
) -> None:
    """Facts already at correct bit_width are skipped."""
    test_db._test_conn.execute(
        "INSERT INTO fact_retention "
        "(fact_id, profile_id, retention_score, memory_strength, access_count, lifecycle_zone) "
        "VALUES ('ok-1', 'p1', 0.9, 10.0, 5, 'active')"
    )
    test_db._test_conn.commit()

    # No embedding_quantization_metadata => current_bw defaults to 32
    # retention 0.9 => target_bw 32 => skip (already at target)

    stats = scheduler.run_eap_cycle("p1")
    assert stats["skipped"] >= 1
    assert stats["downgrades"] == 0


# ---------------------------------------------------------------------------
# Edge: forgotten facts get bit_width 0
# ---------------------------------------------------------------------------


def test_eap_cycle_handles_forgotten(
    scheduler: EAPScheduler,
    test_db: MagicMock,
) -> None:
    """Forgotten facts (R <= 0.05) get target bit_width 0."""
    test_db._test_conn.execute(
        "INSERT INTO fact_retention "
        "(fact_id, profile_id, retention_score, memory_strength, access_count, lifecycle_zone) "
        "VALUES ('dead-1', 'p1', 0.02, 0.1, 0, 'forgotten')"
    )
    test_db._test_conn.commit()

    stats = scheduler.run_eap_cycle("p1")
    # Forgotten facts are handled (deleted or marked)
    assert stats["total"] >= 1
    assert stats["deleted"] >= 1


# ---------------------------------------------------------------------------
# Error path coverage
# ---------------------------------------------------------------------------


def test_eap_cycle_query_error() -> None:
    """EAP cycle handles DB query failure gracefully."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("db exploded")

    scheduler = EAPScheduler(
        db, EbbinghausCurve(ForgettingConfig()),
        MagicMock(), QuantizationConfig(),
    )
    stats = scheduler.run_eap_cycle("p1")
    assert stats["errors"] == 1
    assert stats["total"] == 0


def test_eap_cycle_empty_profile(
    scheduler: EAPScheduler,
) -> None:
    """Empty profile returns zero stats."""
    stats = scheduler.run_eap_cycle("nonexistent-profile")
    assert stats["total"] == 0
    assert stats["downgrades"] == 0


def test_eap_downgrade_no_embedding(
    scheduler: EAPScheduler,
    test_db: MagicMock,
) -> None:
    """Downgrade fails (error) when no float32 embedding available."""
    # Insert a warm fact (R=0.3 -> target_bw=4, but current_bw defaults to 32)
    test_db._test_conn.execute(
        "INSERT INTO fact_retention "
        "(fact_id, profile_id, retention_score, memory_strength, access_count, lifecycle_zone) "
        "VALUES ('no-emb', 'p1', 0.3, 5.0, 1, 'cold')"
    )
    test_db._test_conn.commit()

    # _get_fact_embedding will try atomic_facts but find nothing -> returns None
    stats = scheduler.run_eap_cycle("p1")
    assert stats["errors"] >= 1


def test_eap_upgrade_to_float32(
    scheduler: EAPScheduler,
    test_db: MagicMock,
) -> None:
    """Upgrade from lower bw back to float32 updates metadata."""
    # Insert a fact that was previously downgraded to 4-bit but now has high retention
    test_db._test_conn.execute(
        "INSERT INTO fact_retention "
        "(fact_id, profile_id, retention_score, memory_strength, access_count, lifecycle_zone) "
        "VALUES ('upgrade-1', 'p1', 0.9, 10.0, 5, 'active')"
    )
    test_db._test_conn.execute(
        "INSERT INTO embedding_quantization_metadata "
        "(fact_id, profile_id, quantization_level, bit_width) "
        "VALUES ('upgrade-1', 'p1', 'polar4', 4)"
    )
    test_db._test_conn.commit()

    stats = scheduler.run_eap_cycle("p1")
    # R=0.9 -> target_bw=32, current_bw=4 -> upgrade
    assert stats["upgrades"] >= 1


def test_eap_upgrade_to_int8_no_embedding(
    scheduler: EAPScheduler,
    test_db: MagicMock,
) -> None:
    """Upgrade from 2-bit to int8 is skipped when no embedding available."""
    test_db._test_conn.execute(
        "INSERT INTO fact_retention "
        "(fact_id, profile_id, retention_score, memory_strength, access_count, lifecycle_zone) "
        "VALUES ('up-int8', 'p1', 0.6, 5.0, 2, 'warm')"
    )
    test_db._test_conn.execute(
        "INSERT INTO embedding_quantization_metadata "
        "(fact_id, profile_id, quantization_level, bit_width) "
        "VALUES ('up-int8', 'p1', 'polar2', 2)"
    )
    test_db._test_conn.commit()

    # R=0.6 -> target_bw=8, current_bw=2 -> upgrade
    # But no embedding available -> skipped
    stats = scheduler.run_eap_cycle("p1")
    assert stats["skipped"] >= 1


def test_handle_deletion_error(
    ebbinghaus: EbbinghausCurve,
    quant_config: QuantizationConfig,
) -> None:
    """_handle_deletion logs error on DB failure (does not raise)."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("deletion insert failed")

    scheduler = EAPScheduler(db, ebbinghaus, MagicMock(), quant_config)
    # Should NOT raise -- just logs the error
    scheduler._handle_deletion("some-fact", "p1")


def test_get_fact_embedding_from_atomic_facts(
    scheduler: EAPScheduler,
    test_db: MagicMock,
) -> None:
    """_get_fact_embedding retrieves embedding from atomic_facts JSON column."""
    import json
    # Create atomic_facts table
    test_db._test_conn.execute(
        "CREATE TABLE IF NOT EXISTS atomic_facts ("
        "  fact_id TEXT PRIMARY KEY, "
        "  embedding TEXT"
        ")"
    )
    embedding_data = [1.0, 2.0, 3.0]
    test_db._test_conn.execute(
        "INSERT INTO atomic_facts (fact_id, embedding) VALUES (?, ?)",
        ("with-emb", json.dumps(embedding_data)),
    )
    test_db._test_conn.commit()

    result = scheduler._get_fact_embedding("with-emb")
    assert result is not None
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])


def test_get_fact_embedding_null(
    scheduler: EAPScheduler,
    test_db: MagicMock,
) -> None:
    """_get_fact_embedding returns None for null/empty embedding."""
    test_db._test_conn.execute(
        "CREATE TABLE IF NOT EXISTS atomic_facts ("
        "  fact_id TEXT PRIMARY KEY, "
        "  embedding TEXT"
        ")"
    )
    test_db._test_conn.execute(
        "INSERT INTO atomic_facts (fact_id, embedding) VALUES (?, ?)",
        ("null-emb", "null"),
    )
    test_db._test_conn.commit()

    result = scheduler._get_fact_embedding("null-emb")
    assert result is None


def test_get_fact_embedding_not_found(
    scheduler: EAPScheduler,
    test_db: MagicMock,
) -> None:
    """_get_fact_embedding returns None when fact doesn't exist."""
    test_db._test_conn.execute(
        "CREATE TABLE IF NOT EXISTS atomic_facts ("
        "  fact_id TEXT PRIMARY KEY, "
        "  embedding TEXT"
        ")"
    )
    test_db._test_conn.commit()

    result = scheduler._get_fact_embedding("nonexistent")
    assert result is None


def test_get_fact_embedding_db_error(
    ebbinghaus: EbbinghausCurve,
    quant_config: QuantizationConfig,
) -> None:
    """_get_fact_embedding returns None on DB error."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("db error")

    scheduler = EAPScheduler(db, ebbinghaus, MagicMock(), quant_config)
    result = scheduler._get_fact_embedding("any-fact")
    assert result is None


def test_eap_upgrade_to_float32_db_error(
    ebbinghaus: EbbinghausCurve,
    quant_config: QuantizationConfig,
) -> None:
    """_handle_upgrade to float32 returns False on DB error."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("upgrade insert failed")

    scheduler = EAPScheduler(db, ebbinghaus, MagicMock(), quant_config)
    # Directly test _handle_upgrade since the error path is internal
    result = scheduler._handle_upgrade("fact-err", "p1", 32)
    assert result is False


def test_eap_upgrade_to_int8_with_embedding(
    test_db: MagicMock,
    ebbinghaus: EbbinghausCurve,
    polar_encoder: PolarQuantEncoder,
    qjl_encoder: QJLEncoder,
    quant_config: QuantizationConfig,
) -> None:
    """Upgrade from 2-bit to int8 works when embedding is available."""
    import json

    # Create atomic_facts table with embedding
    test_db._test_conn.execute(
        "CREATE TABLE IF NOT EXISTS atomic_facts ("
        "  fact_id TEXT PRIMARY KEY, "
        "  embedding TEXT"
        ")"
    )
    embedding = _random_vec(768, seed=77).tolist()
    test_db._test_conn.execute(
        "INSERT INTO atomic_facts (fact_id, embedding) VALUES (?, ?)",
        ("up-ok", json.dumps(embedding)),
    )
    test_db._test_conn.execute(
        "INSERT INTO fact_retention "
        "(fact_id, profile_id, retention_score, memory_strength, access_count, lifecycle_zone) "
        "VALUES ('up-ok', 'p1', 0.6, 5.0, 3, 'warm')"
    )
    test_db._test_conn.execute(
        "INSERT INTO embedding_quantization_metadata "
        "(fact_id, profile_id, quantization_level, bit_width) "
        "VALUES ('up-ok', 'p1', 'polar2', 2)"
    )
    test_db._test_conn.commit()

    qs = QuantizedEmbeddingStore(test_db, polar_encoder, qjl_encoder, quant_config)
    scheduler = EAPScheduler(test_db, ebbinghaus, qs, quant_config)

    # R=0.6 -> target_bw=8, current_bw=2 -> upgrade
    stats = scheduler.run_eap_cycle("p1")
    assert stats["upgrades"] >= 1
