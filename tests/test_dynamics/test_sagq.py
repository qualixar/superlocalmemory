# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for Spreading Activation-Guided Quantization (SAGQ).

TDD sequence (Phase D LLD Section 6):
  1. test_centrality_computation_weighted_sum
  2. test_centrality_normalization_bounds
  3. test_sagq_precision_mapping_endpoints
  4. test_sagq_precision_mapping_midpoints
  5. test_combined_eap_sagq_takes_max
  6. test_high_centrality_overrides_low_retention
  7. test_isolated_nodes_get_minimum_precision
  8. test_centrality_empty_graph_returns_empty
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import SAGQConfig
from superlocalmemory.dynamics.activation_guided_quantization import (
    ActivationGuidedQuantizer,
    CentralityScore,
    SAGQPrecision,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_sagq_test_db(tmp_path: Path) -> MagicMock:
    """Create mock DatabaseManager backed by real SQLite for SAGQ tests."""
    db = MagicMock()
    conn = sqlite3.connect(str(tmp_path / "sagq_test.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fact_importance (
            fact_id            TEXT NOT NULL,
            profile_id         TEXT NOT NULL,
            pagerank_score     REAL NOT NULL DEFAULT 0.0,
            community_id       INTEGER,
            degree_centrality  REAL NOT NULL DEFAULT 0.0,
            computed_at        TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (fact_id, profile_id)
        );

        CREATE TABLE IF NOT EXISTS activation_cache (
            cache_id         TEXT PRIMARY KEY,
            profile_id       TEXT NOT NULL,
            query_hash       TEXT NOT NULL,
            node_id          TEXT NOT NULL,
            activation_value REAL NOT NULL,
            iteration        INTEGER NOT NULL DEFAULT 0,
            created_at       TEXT NOT NULL DEFAULT (datetime('now')),
            expires_at       TEXT NOT NULL DEFAULT (datetime('now', '+1 hour'))
        );

        CREATE TABLE IF NOT EXISTS embedding_metadata (
            vec_rowid    INTEGER,
            fact_id      TEXT NOT NULL,
            profile_id   TEXT NOT NULL,
            model_name   TEXT,
            dimension    INTEGER,
            bit_width    INTEGER NOT NULL DEFAULT 32,
            quantization_level TEXT NOT NULL DEFAULT 'float32',
            compressed_size_bytes INTEGER,
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (fact_id, profile_id)
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
def sagq_config() -> SAGQConfig:
    """Default SAGQ configuration."""
    return SAGQConfig()


@pytest.fixture
def disabled_config() -> SAGQConfig:
    """SAGQ config with enabled=False."""
    return SAGQConfig(enabled=False)


@pytest.fixture
def test_db(tmp_path: Path) -> MagicMock:
    db = _create_sagq_test_db(tmp_path)
    yield db
    db._test_conn.close()


@pytest.fixture
def quantizer(test_db: MagicMock, sagq_config: SAGQConfig) -> ActivationGuidedQuantizer:
    return ActivationGuidedQuantizer(test_db, sagq_config)


# ---------------------------------------------------------------------------
# 1. Centrality computation with weighted sum
# ---------------------------------------------------------------------------


def test_centrality_computation_weighted_sum(
    quantizer: ActivationGuidedQuantizer,
    test_db: MagicMock,
) -> None:
    """centrality(pr=max, deg=max, sa=max) == 1.0 and midpoint == 0.5."""
    conn = test_db._test_conn
    # Insert one fact at max importance (will self-normalize to 1.0)
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('f1', 'p1', 1.0, 1.0)"
    )
    # Insert one SA cache entry for this fact
    conn.execute(
        "INSERT INTO activation_cache (cache_id, profile_id, query_hash, node_id, "
        "activation_value, iteration, created_at) "
        "VALUES ('c1', 'p1', 'q1', 'f1', 0.9, 3, datetime('now'))"
    )
    conn.commit()

    result = quantizer.compute_centrality_batch("p1")
    assert len(result) == 1

    cs = result[0]
    assert cs.fact_id == "f1"
    # With only one fact, max_pr=1.0, max_deg=1.0, max_sa=1
    # pr_norm=1.0, deg_norm=1.0, sa_freq_norm=1.0
    # combined = 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0
    assert cs.combined_centrality == pytest.approx(1.0, abs=1e-6)
    assert cs.pagerank_norm == pytest.approx(1.0)
    assert cs.degree_norm == pytest.approx(1.0)
    assert cs.sa_freq_norm == pytest.approx(1.0)


def test_centrality_midpoint(
    quantizer: ActivationGuidedQuantizer,
    test_db: MagicMock,
) -> None:
    """Two facts: one at max, one at half should give centrality ~0.5."""
    conn = test_db._test_conn
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('hub', 'p1', 1.0, 1.0)"
    )
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('mid', 'p1', 0.5, 0.5)"
    )
    # 2 SA entries for hub, 1 for mid
    conn.execute(
        "INSERT INTO activation_cache (cache_id, profile_id, query_hash, node_id, "
        "activation_value, iteration, created_at) "
        "VALUES ('c1', 'p1', 'q1', 'hub', 0.9, 3, datetime('now'))"
    )
    conn.execute(
        "INSERT INTO activation_cache (cache_id, profile_id, query_hash, node_id, "
        "activation_value, iteration, created_at) "
        "VALUES ('c2', 'p1', 'q2', 'hub', 0.8, 3, datetime('now'))"
    )
    conn.execute(
        "INSERT INTO activation_cache (cache_id, profile_id, query_hash, node_id, "
        "activation_value, iteration, created_at) "
        "VALUES ('c3', 'p1', 'q3', 'mid', 0.5, 3, datetime('now'))"
    )
    conn.commit()

    result = quantizer.compute_centrality_batch("p1")
    by_id = {cs.fact_id: cs for cs in result}

    hub = by_id["hub"]
    assert hub.combined_centrality == pytest.approx(1.0, abs=1e-6)

    mid = by_id["mid"]
    # pr_norm = 0.5/1.0 = 0.5, deg_norm = 0.5/1.0 = 0.5, sa_norm = 1/2 = 0.5
    # combined = 0.5*0.5 + 0.3*0.5 + 0.2*0.5 = 0.25 + 0.15 + 0.10 = 0.50
    assert mid.combined_centrality == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Centrality normalization bounds
# ---------------------------------------------------------------------------


def test_centrality_normalization_bounds(
    quantizer: ActivationGuidedQuantizer,
    test_db: MagicMock,
) -> None:
    """All centrality scores must be in [0.0, 1.0]."""
    conn = test_db._test_conn
    # Insert diverse facts
    for i in range(20):
        pr = i * 0.05
        deg = (20 - i) * 0.05
        conn.execute(
            "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
            "VALUES (?, 'p1', ?, ?)",
            (f"f{i}", pr, deg),
        )
    conn.commit()

    result = quantizer.compute_centrality_batch("p1")
    assert len(result) == 20
    for cs in result:
        assert 0.0 <= cs.combined_centrality <= 1.0
        assert 0.0 <= cs.pagerank_norm <= 1.0
        assert 0.0 <= cs.degree_norm <= 1.0
        assert 0.0 <= cs.sa_freq_norm <= 1.0


# ---------------------------------------------------------------------------
# 3. SAGQ precision mapping: endpoints
# ---------------------------------------------------------------------------


def test_sagq_precision_mapping_endpoints(sagq_config: SAGQConfig) -> None:
    """centrality=0.0 -> b_min=2, centrality=1.0 -> b_max=32."""
    db = MagicMock()
    q = ActivationGuidedQuantizer(db, sagq_config)

    assert q.centrality_to_bit_width(0.0) == 2
    assert q.centrality_to_bit_width(1.0) == 32


# ---------------------------------------------------------------------------
# 4. SAGQ precision mapping: midpoints with ceiling snap
# ---------------------------------------------------------------------------


def test_sagq_precision_mapping_midpoints(sagq_config: SAGQConfig) -> None:
    """Verify ceiling-snap to valid_bit_widths [2, 4, 8, 32].

    centrality=0.5 -> raw_bw = 2 + 30*0.5 = 17.0 -> snapped to 32
    centrality=0.2 -> raw_bw = 2 + 30*0.2 = 8.0 -> snapped to 8
    centrality=0.05 -> raw_bw = 2 + 30*0.05 = 3.5 -> snapped to 4
    """
    db = MagicMock()
    q = ActivationGuidedQuantizer(db, sagq_config)

    assert q.centrality_to_bit_width(0.5) == 32
    assert q.centrality_to_bit_width(0.2) == 8
    assert q.centrality_to_bit_width(0.05) == 4


def test_sagq_precision_clamps_input() -> None:
    """centrality outside [0,1] is clamped before computation."""
    db = MagicMock()
    config = SAGQConfig()
    q = ActivationGuidedQuantizer(db, config)

    # Negative clamped to 0.0 -> b_min=2
    assert q.centrality_to_bit_width(-0.5) == 2
    # >1.0 clamped to 1.0 -> b_max=32
    assert q.centrality_to_bit_width(1.5) == 32


# ---------------------------------------------------------------------------
# 5. Combined EAP + SAGQ takes max
# ---------------------------------------------------------------------------


def test_combined_eap_sagq_takes_max(
    quantizer: ActivationGuidedQuantizer,
    test_db: MagicMock,
) -> None:
    """max(eap, sagq): EAP=4 + SAGQ=32 -> 32; EAP=32 + SAGQ=2 -> 32."""
    conn = test_db._test_conn

    # High centrality fact -> SAGQ wants 32-bit
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('high_c', 'p1', 1.0, 1.0)"
    )
    # Low centrality fact -> SAGQ wants 2-bit
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('low_c', 'p1', 0.0, 0.0)"
    )
    # Current bit-widths
    conn.execute(
        "INSERT INTO embedding_metadata (fact_id, profile_id, bit_width) "
        "VALUES ('high_c', 'p1', 32)"
    )
    conn.execute(
        "INSERT INTO embedding_metadata (fact_id, profile_id, bit_width) "
        "VALUES ('low_c', 'p1', 32)"
    )
    conn.commit()

    def eap_mapper(fact_id: str) -> int:
        # EAP says 4-bit for high_c, 32-bit for low_c
        return 4 if fact_id == "high_c" else 32

    result = quantizer.compute_sagq_precision_batch("p1", eap_mapper)
    by_id = {p.fact_id: p for p in result}

    # high_c: sagq=32, eap=4 -> final = max(32, 4) = 32
    assert by_id["high_c"].final_bit_width == 32
    assert by_id["high_c"].sagq_bit_width == 32
    assert by_id["high_c"].eap_bit_width == 4

    # low_c: sagq=2, eap=32 -> final = max(2, 32) = 32
    assert by_id["low_c"].final_bit_width == 32
    assert by_id["low_c"].sagq_bit_width == 2
    assert by_id["low_c"].eap_bit_width == 32


# ---------------------------------------------------------------------------
# 6. High centrality overrides low retention
# ---------------------------------------------------------------------------


def test_high_centrality_overrides_low_retention(
    quantizer: ActivationGuidedQuantizer,
    test_db: MagicMock,
) -> None:
    """Fact with retention=0.3 (EAP->4-bit) but centrality=0.9 (SAGQ->32-bit) -> final=32."""
    conn = test_db._test_conn

    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('rescue', 'p1', 1.0, 1.0)"
    )
    conn.execute(
        "INSERT INTO activation_cache (cache_id, profile_id, query_hash, node_id, "
        "activation_value, iteration, created_at) "
        "VALUES ('c1', 'p1', 'q1', 'rescue', 0.9, 3, datetime('now'))"
    )
    conn.execute(
        "INSERT INTO embedding_metadata (fact_id, profile_id, bit_width) "
        "VALUES ('rescue', 'p1', 4)"
    )
    conn.commit()

    # EAP mapper: retention=0.3 -> 4-bit
    def eap_mapper(fact_id: str) -> int:
        return 4

    result = quantizer.compute_sagq_precision_batch("p1", eap_mapper)
    assert len(result) == 1

    prec = result[0]
    # centrality = 1.0 (single node, all normalized to 1.0)
    # SAGQ -> 32-bit, EAP -> 4-bit, final = max(32, 4) = 32
    assert prec.sagq_bit_width == 32
    assert prec.final_bit_width == 32
    # Current is 4, final is 32 -> upgrade
    assert prec.action == "upgrade"


# ---------------------------------------------------------------------------
# 7. Isolated nodes get minimum precision
# ---------------------------------------------------------------------------


def test_isolated_nodes_get_minimum_precision(
    test_db: MagicMock,
    sagq_config: SAGQConfig,
) -> None:
    """Fact with no edges (pr=0, deg=0, sa=0) -> centrality=0.0 -> sagq_bw=2."""
    conn = test_db._test_conn

    # Two facts: one with values, one with zeros
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('hub', 'p1', 0.5, 0.8)"
    )
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('isolated', 'p1', 0.0, 0.0)"
    )
    conn.execute(
        "INSERT INTO embedding_metadata (fact_id, profile_id, bit_width) "
        "VALUES ('hub', 'p1', 32)"
    )
    conn.execute(
        "INSERT INTO embedding_metadata (fact_id, profile_id, bit_width) "
        "VALUES ('isolated', 'p1', 32)"
    )
    conn.commit()

    q = ActivationGuidedQuantizer(test_db, sagq_config)

    # EAP also says minimum for the isolated node
    def eap_mapper(fact_id: str) -> int:
        return 2

    result = q.compute_sagq_precision_batch("p1", eap_mapper)
    by_id = {p.fact_id: p for p in result}

    assert by_id["isolated"].sagq_bit_width == 2
    assert by_id["isolated"].centrality == pytest.approx(0.0)
    # final = max(sagq=2, eap=2) = 2
    assert by_id["isolated"].final_bit_width == 2
    # current=32, final=2 -> downgrade
    assert by_id["isolated"].action == "downgrade"


# ---------------------------------------------------------------------------
# 8. Empty graph returns empty list
# ---------------------------------------------------------------------------


def test_centrality_empty_graph_returns_empty(
    quantizer: ActivationGuidedQuantizer,
) -> None:
    """Profile with no fact_importance rows -> empty list."""
    result = quantizer.compute_centrality_batch("nonexistent-profile")
    assert result == []


def test_sagq_precision_empty_returns_empty(
    quantizer: ActivationGuidedQuantizer,
) -> None:
    """compute_sagq_precision_batch with no data -> empty list."""
    result = quantizer.compute_sagq_precision_batch("empty-p", lambda _: 32)
    assert result == []


# ---------------------------------------------------------------------------
# HR-07: Disabled config returns empty results
# ---------------------------------------------------------------------------


def test_disabled_returns_empty(disabled_config: SAGQConfig) -> None:
    """SAGQ is a NO-OP when config.enabled=False (HR-07)."""
    db = MagicMock()
    q = ActivationGuidedQuantizer(db, disabled_config)

    assert q.compute_centrality_batch("p1") == []
    assert q.compute_sagq_precision_batch("p1", lambda _: 32) == []
    assert q.get_centrality_for_fact("f1", "p1") == 0.0
    # DB should never be called
    db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# get_centrality_for_fact single-fact lookup
# ---------------------------------------------------------------------------


def test_get_centrality_for_fact_found(
    quantizer: ActivationGuidedQuantizer,
    test_db: MagicMock,
) -> None:
    """Single-fact centrality lookup returns correct value."""
    conn = test_db._test_conn
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('f1', 'p1', 0.5, 0.8)"
    )
    conn.commit()

    c = quantizer.get_centrality_for_fact("f1", "p1")
    # max_pr=0.5, max_deg=0.8 -> pr_norm=1.0, deg_norm=1.0
    # sa_norm=0 (no activation_cache entries)
    # combined = 0.5*1.0 + 0.3*1.0 + 0.2*0 = 0.8
    assert c == pytest.approx(0.8, abs=1e-6)


def test_get_centrality_for_fact_not_found(
    quantizer: ActivationGuidedQuantizer,
) -> None:
    """Unknown fact returns 0.0."""
    c = quantizer.get_centrality_for_fact("unknown", "p1")
    assert c == 0.0


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_sagq_config_weight_validation() -> None:
    """Weights that don't sum to 1.0 raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SAGQConfig(w_pagerank=0.5, w_degree=0.5, w_sa_freq=0.5)


def test_sagq_config_empty_bit_widths() -> None:
    """Empty valid_bit_widths raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        SAGQConfig(valid_bit_widths=())


def test_sagq_config_invalid_b_min() -> None:
    """b_min < 1 raises ValueError."""
    with pytest.raises(ValueError, match="b_min must be >= 1"):
        SAGQConfig(b_min=0)


def test_sagq_config_bmax_lt_bmin() -> None:
    """b_max < b_min raises ValueError."""
    with pytest.raises(ValueError, match="b_max"):
        SAGQConfig(b_min=32, b_max=2)


# ---------------------------------------------------------------------------
# NaN safety
# ---------------------------------------------------------------------------


def test_nan_centrality_defaults_to_zero(
    test_db: MagicMock,
) -> None:
    """If pagerank and degree are both zero, centrality is 0.0, not NaN."""
    conn = test_db._test_conn
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('z', 'p1', 0.0, 0.0)"
    )
    conn.commit()

    q = ActivationGuidedQuantizer(test_db, SAGQConfig())
    result = q.compute_centrality_batch("p1")
    assert len(result) == 1
    assert result[0].combined_centrality == pytest.approx(0.0)
    assert result[0].pagerank_norm == pytest.approx(0.0)
    assert result[0].degree_norm == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Action determination (skip)
# ---------------------------------------------------------------------------


def test_skip_action_when_no_change(
    quantizer: ActivationGuidedQuantizer,
    test_db: MagicMock,
) -> None:
    """When final_bw == current_bw, action is 'skip'."""
    conn = test_db._test_conn

    # High centrality -> SAGQ recommends 32; current is 32
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('same', 'p1', 1.0, 1.0)"
    )
    conn.execute(
        "INSERT INTO embedding_metadata (fact_id, profile_id, bit_width) "
        "VALUES ('same', 'p1', 32)"
    )
    conn.commit()

    result = quantizer.compute_sagq_precision_batch("p1", lambda _: 32)
    assert len(result) == 1
    assert result[0].action == "skip"


# ---------------------------------------------------------------------------
# Frozen dataclass enforcement
# ---------------------------------------------------------------------------


def test_centrality_score_frozen() -> None:
    """CentralityScore is immutable."""
    cs = CentralityScore(
        fact_id="f1",
        pagerank_norm=0.5,
        degree_norm=0.3,
        sa_freq_norm=0.2,
        combined_centrality=0.4,
    )
    with pytest.raises(AttributeError):
        cs.combined_centrality = 0.9  # type: ignore[misc]


def test_sagq_precision_frozen() -> None:
    """SAGQPrecision is immutable."""
    sp = SAGQPrecision(
        fact_id="f1",
        centrality=0.9,
        sagq_bit_width=32,
        eap_bit_width=4,
        final_bit_width=32,
        current_bit_width=4,
        action="upgrade",
    )
    with pytest.raises(AttributeError):
        sp.action = "skip"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Error path coverage
# ---------------------------------------------------------------------------


def test_fact_importance_query_error() -> None:
    """compute_centrality_batch handles DB query failure gracefully."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("db exploded")
    q = ActivationGuidedQuantizer(db, SAGQConfig())
    assert q.compute_centrality_batch("p1") == []


def test_activation_cache_query_error(
    test_db: MagicMock,
    sagq_config: SAGQConfig,
) -> None:
    """When activation_cache query fails, SA freq defaults to 0."""
    conn = test_db._test_conn
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('f1', 'p1', 0.5, 0.5)"
    )
    conn.commit()

    # Drop activation_cache to force error on SA query
    conn.execute("DROP TABLE activation_cache")
    conn.commit()

    q = ActivationGuidedQuantizer(test_db, sagq_config)
    result = q.compute_centrality_batch("p1")
    assert len(result) == 1
    # SA freq should be 0 due to error fallback
    assert result[0].sa_freq_norm == 0.0


def test_embedding_metadata_query_error() -> None:
    """compute_sagq_precision_batch handles embedding_metadata query failure."""
    db = MagicMock()
    config = SAGQConfig()

    # First call (fact_importance) succeeds, second (activation_cache) returns empty,
    # third (embedding_metadata) fails
    call_count = [0]
    def _mock_execute(sql, params=()):
        call_count[0] += 1
        if "fact_importance" in sql and "MAX" not in sql:
            return [{"fact_id": "f1", "pagerank_score": 0.5, "degree_centrality": 0.5}]
        if "activation_cache" in sql:
            return []
        if "embedding_metadata" in sql:
            raise RuntimeError("metadata table missing")
        return []

    db.execute.side_effect = _mock_execute
    q = ActivationGuidedQuantizer(db, config)
    result = q.compute_sagq_precision_batch("p1", lambda _: 32)
    # Should still return results, just with default bit-width of 32
    assert len(result) == 1
    assert result[0].current_bit_width == 32


def test_get_centrality_single_fact_db_error() -> None:
    """get_centrality_for_fact returns 0.0 on DB error."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("db offline")
    q = ActivationGuidedQuantizer(db, SAGQConfig())
    assert q.get_centrality_for_fact("f1", "p1") == 0.0


def test_get_centrality_max_query_error(test_db: MagicMock) -> None:
    """get_centrality_for_fact handles max-query failure gracefully."""
    conn = test_db._test_conn
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('f1', 'p1', 0.5, 0.5)"
    )
    conn.commit()

    config = SAGQConfig()
    call_count = [0]
    original_execute = test_db.execute.side_effect

    def _failing_max_execute(sql, params=()):
        call_count[0] += 1
        if "MAX(pagerank_score)" in sql:
            raise RuntimeError("max query broken")
        return original_execute(sql, params)

    test_db.execute.side_effect = _failing_max_execute
    q = ActivationGuidedQuantizer(test_db, config)
    result = q.get_centrality_for_fact("f1", "p1")
    assert result == 0.0


def test_get_centrality_sa_count_error(test_db: MagicMock) -> None:
    """get_centrality_for_fact handles SA count query failure."""
    conn = test_db._test_conn
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('f1', 'p1', 0.5, 0.5)"
    )
    conn.commit()

    # Drop activation_cache to force SA query errors
    conn.execute("DROP TABLE activation_cache")
    conn.commit()

    q = ActivationGuidedQuantizer(test_db, SAGQConfig())
    result = q.get_centrality_for_fact("f1", "p1")
    # Should still return a value based on PR + degree only, SA=0
    assert result == pytest.approx(0.8, abs=1e-6)


def test_bit_width_cap_at_maximum() -> None:
    """When raw_bw exceeds all valid bit-widths, cap at maximum."""
    # Create config where b_max exceeds largest valid_bit_width
    config = SAGQConfig(
        b_min=2, b_max=64,
        valid_bit_widths=(2, 4, 8, 32),
    )
    db = MagicMock()
    q = ActivationGuidedQuantizer(db, config)
    # centrality=1.0 -> raw_bw = 2 + (64-2)*1.0 = 64, which exceeds 32
    assert q.centrality_to_bit_width(1.0) == 32


def test_nan_centrality_in_batch() -> None:
    """NaN from centrality computation defaults to 0.0 (line 167-168)."""
    db = MagicMock()
    config = SAGQConfig()

    # Inject NaN via a custom config with NaN weight
    # We need to bypass the frozen config validation, so we use mock
    mock_config = MagicMock()
    mock_config.enabled = True
    mock_config.sa_frequency_window_days = 7
    mock_config.w_pagerank = float("nan")  # This will produce NaN combined
    mock_config.w_degree = 0.3
    mock_config.w_sa_freq = 0.2

    db.execute.side_effect = [
        # fact_importance query
        [{"fact_id": "f1", "pagerank_score": 0.5, "degree_centrality": 0.5}],
        # activation_cache query
        [],
    ]

    q = ActivationGuidedQuantizer(db, mock_config)
    result = q.compute_centrality_batch("p1")
    assert len(result) == 1
    assert result[0].combined_centrality == 0.0  # NaN replaced with 0.0


def test_get_centrality_for_fact_nan() -> None:
    """get_centrality_for_fact returns 0.0 when NaN is produced (line 370-371)."""
    db = MagicMock()

    # Use mock config with NaN weight to force NaN result
    mock_config = MagicMock()
    mock_config.enabled = True
    mock_config.sa_frequency_window_days = 7
    mock_config.w_pagerank = float("nan")
    mock_config.w_degree = 0.3
    mock_config.w_sa_freq = 0.2

    call_count = [0]
    def _mock_execute(sql, params=()):
        call_count[0] += 1
        if "fact_importance" in sql and "MAX" not in sql:
            return [{"pagerank_score": 0.5, "degree_centrality": 0.5}]
        if "MAX(pagerank_score)" in sql:
            return [{"max_pr": 0.5, "max_deg": 0.5}]
        if "activation_cache" in sql and "MAX" not in sql:
            return [{"cnt": 0}]
        if "MAX(cnt)" in sql:
            return [{"max_cnt": 1}]
        return []

    db.execute.side_effect = _mock_execute
    q = ActivationGuidedQuantizer(db, mock_config)
    result = q.get_centrality_for_fact("f1", "p1")
    assert result == 0.0  # NaN replaced with 0.0


def test_get_centrality_sa_max_subquery_error(test_db: MagicMock) -> None:
    """get_centrality_for_fact handles SA max subquery failure (line 370-371)."""
    conn = test_db._test_conn
    conn.execute(
        "INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, degree_centrality) "
        "VALUES ('f1', 'p1', 0.5, 0.5)"
    )
    conn.execute(
        "INSERT INTO activation_cache (cache_id, profile_id, query_hash, node_id, "
        "activation_value, iteration, created_at) "
        "VALUES ('c1', 'p1', 'q1', 'f1', 0.5, 3, datetime('now'))"
    )
    conn.commit()

    config = SAGQConfig()
    original_execute = test_db.execute.side_effect
    call_count = [0]

    def _failing_sa_max(sql, params=()):
        call_count[0] += 1
        # Let the SA max subquery fail (the one with nested SELECT)
        if "MAX(cnt)" in sql:
            raise RuntimeError("sa max query broken")
        return original_execute(sql, params)

    test_db.execute.side_effect = _failing_sa_max
    q = ActivationGuidedQuantizer(test_db, config)
    # Should still return a result, with sa_norm based on max_sa=1 fallback
    result = q.get_centrality_for_fact("f1", "p1")
    assert 0.0 <= result <= 1.0
