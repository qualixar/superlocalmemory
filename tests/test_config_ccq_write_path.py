# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Write-path regression tests.

Two independent behavioural defects are covered:

1. SLMConfig.save() must serialize in-memory mutations to the forgetting,
   quantization, sagq, and auto_invoke sections so a mutate→save→load cycle
   preserves the mutated values.  Previously save() copied the on-disk value
   back, silently discarding any in-memory change.

2. CognitiveConsolidator._step5_store_block must wrap its four writes
   (store_memory, store_fact, store_ccq_block, set_fact_lifecycle_zone) in a
   single transaction so a failure after store_fact cannot leave an orphan
   gist AtomicFact with no ccq_consolidated_blocks row.

Tests use real files and the real DatabaseManager — no MagicMock on the paths
under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.core.config import (
    AutoInvokeConfig,
    ForgettingConfig,
    PolarQuantConfig,
    QJLConfig,
    QuantizationConfig,
    SAGQConfig,
    SLMConfig,
)
from superlocalmemory.storage.models import Mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_config(tmp_path: Path) -> tuple[Path, SLMConfig]:
    """Write a default Mode-A config file and return (path, config)."""
    config_path = tmp_path / "config.json"
    cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    cfg.save(config_path)
    return config_path, cfg


# ---------------------------------------------------------------------------
# Defect 1 — SLMConfig.save() mutation round-trip
# ---------------------------------------------------------------------------

class TestForgettingRoundTrip:
    """save() must persist in-memory mutations to config.forgetting."""

    def test_alpha_mutation_survives_save_load(self, tmp_path: Path) -> None:
        config_path, _ = _fresh_config(tmp_path)

        cfg = SLMConfig.load(config_path)
        cfg.forgetting = ForgettingConfig(alpha=3.0, beta=2.5)
        cfg.save(config_path)

        reloaded = SLMConfig.load(config_path)
        assert reloaded.forgetting.alpha == 3.0
        assert reloaded.forgetting.beta == 2.5

    def test_enabled_flag_mutation_survives_save_load(self, tmp_path: Path) -> None:
        config_path, _ = _fresh_config(tmp_path)

        cfg = SLMConfig.load(config_path)
        cfg.forgetting = ForgettingConfig(enabled=False, alpha=1.0)
        cfg.save(config_path)

        reloaded = SLMConfig.load(config_path)
        assert reloaded.forgetting.enabled is False
        assert reloaded.forgetting.alpha == 1.0

    def test_unmodified_round_trip_preserves_defaults(self, tmp_path: Path) -> None:
        """A load→save→load cycle without mutation must preserve field defaults."""
        config_path, _ = _fresh_config(tmp_path)

        cfg = SLMConfig.load(config_path)
        cfg.save(config_path)

        reloaded = SLMConfig.load(config_path)
        default_alpha = ForgettingConfig.__dataclass_fields__["alpha"].default
        assert reloaded.forgetting.alpha == default_alpha
        assert reloaded.forgetting.enabled is True

    def test_unknown_top_level_key_survives_alongside_mutation(
        self, tmp_path: Path
    ) -> None:
        """Forward-compat unknown top-level keys must not be dropped when
        config.forgetting is mutated and saved."""
        config_path = tmp_path / "config.json"
        raw = {
            "mode": "a",
            "base_dir": str(tmp_path),
            "future_vendor_extension": {"enabled": True, "count": 42},
        }
        config_path.write_text(json.dumps(raw))

        cfg = SLMConfig.load(config_path)
        cfg.forgetting = ForgettingConfig(alpha=5.0)
        cfg.save(config_path)

        saved = json.loads(config_path.read_text())
        assert saved.get("future_vendor_extension") == {"enabled": True, "count": 42}

        reloaded = SLMConfig.load(config_path)
        assert reloaded.forgetting.alpha == 5.0


class TestQuantizationRoundTrip:
    """save() must persist in-memory mutations to quantization."""

    def test_enabled_flag_survives_save_load(self, tmp_path: Path) -> None:
        config_path, _ = _fresh_config(tmp_path)

        cfg = SLMConfig.load(config_path)
        cfg.quantization = QuantizationConfig(enabled=False)
        cfg.save(config_path)

        reloaded = SLMConfig.load(config_path)
        assert reloaded.quantization.enabled is False

    def test_nested_polar_fields_survive_save_load(self, tmp_path: Path) -> None:
        config_path, _ = _fresh_config(tmp_path)

        cfg = SLMConfig.load(config_path)
        cfg.quantization = QuantizationConfig(
            polar=PolarQuantConfig(seed=99, codebook_method="polar_legacy"),
            qjl=QJLConfig(projection_dim=64),
        )
        cfg.save(config_path)

        reloaded = SLMConfig.load(config_path)
        assert reloaded.quantization.polar.seed == 99
        assert reloaded.quantization.polar.codebook_method == "polar_legacy"
        assert reloaded.quantization.qjl.projection_dim == 64


class TestSAGQRoundTrip:
    """save() must persist in-memory mutations to sagq, including the tuple field."""

    def test_b_min_b_max_survive_save_load(self, tmp_path: Path) -> None:
        config_path, _ = _fresh_config(tmp_path)

        cfg = SLMConfig.load(config_path)
        cfg.sagq = SAGQConfig(b_min=4, b_max=8, valid_bit_widths=(4, 8))
        cfg.save(config_path)

        reloaded = SLMConfig.load(config_path)
        assert reloaded.sagq.b_min == 4
        assert reloaded.sagq.b_max == 8

    def test_valid_bit_widths_is_tuple_after_round_trip(self, tmp_path: Path) -> None:
        """valid_bit_widths must come back as tuple, not list, after JSON round-trip."""
        config_path, _ = _fresh_config(tmp_path)

        cfg = SLMConfig.load(config_path)
        cfg.sagq = SAGQConfig(b_min=4, b_max=8, valid_bit_widths=(4, 8))
        cfg.save(config_path)

        reloaded = SLMConfig.load(config_path)
        assert isinstance(reloaded.sagq.valid_bit_widths, tuple)
        assert reloaded.sagq.valid_bit_widths == (4, 8)


class TestAutoInvokeRoundTrip:
    """save() must persist in-memory mutations to auto_invoke."""

    def test_fok_threshold_survives_save_load(self, tmp_path: Path) -> None:
        config_path, _ = _fresh_config(tmp_path)

        cfg = SLMConfig.load(config_path)
        cfg.auto_invoke = AutoInvokeConfig(fok_threshold=0.25, max_memories_injected=5)
        cfg.save(config_path)

        reloaded = SLMConfig.load(config_path)
        assert reloaded.auto_invoke.fok_threshold == 0.25
        assert reloaded.auto_invoke.max_memories_injected == 5

    def test_weights_dict_survives_save_load(self, tmp_path: Path) -> None:
        custom_weights = {
            "similarity": 0.50,
            "recency": 0.20,
            "frequency": 0.15,
            "trust": 0.15,
        }
        config_path, _ = _fresh_config(tmp_path)

        cfg = SLMConfig.load(config_path)
        cfg.auto_invoke = AutoInvokeConfig(weights=custom_weights)
        cfg.save(config_path)

        reloaded = SLMConfig.load(config_path)
        assert reloaded.auto_invoke.weights == custom_weights


# ---------------------------------------------------------------------------
# Defect 2 — CognitiveConsolidator._step5_store_block transaction atomicity
# ---------------------------------------------------------------------------

def _make_real_db(tmp_path: Path):
    """DatabaseManager backed by a temp sqlite file with full schema."""
    from superlocalmemory.storage import schema as real_schema
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(tmp_path / "test_ccq.db")
    db.initialize(real_schema)
    return db


def _make_cluster_and_gist(db=None):
    """Minimal cluster + gist for _step5_store_block testing.

    When ``db`` is provided, three source facts are pre-inserted so that the
    fact_access_log FK constraint and set_fact_lifecycle_zone operate on real
    rows.  Pass ``db=None`` only for rollback tests where the transaction aborts
    before the FK-constrained writes are reached.
    """
    from superlocalmemory.encoding.cognitive_consolidator import (
        ConsolidationCluster,
        GistResult,
    )
    from superlocalmemory.storage.models import AtomicFact, MemoryRecord

    src_ids = ("src-fact-a", "src-fact-b", "src-fact-c")

    if db is not None:
        # Pre-insert the source facts so FK constraints on fact_access_log
        # and fact_retention are satisfied during the happy-path writes.
        for fid in src_ids:
            mem_id = fid + "-mem"
            db.store_memory(MemoryRecord(memory_id=mem_id, profile_id="default", content=f"source content {fid}"))
            db.execute(
                "INSERT OR IGNORE INTO atomic_facts "
                "(fact_id, memory_id, profile_id, content, fact_type, lifecycle) "
                "VALUES (?, ?, 'default', ?, 'semantic', 'warm')",
                (fid, mem_id, f"source content {fid}"),
            )

    cluster = ConsolidationCluster(
        cluster_id="test-cluster-001",
        fact_ids=src_ids,
        shared_entities=("entity_x",),
        temporal_centroid="2026-01-01T00:00:00",
        avg_retention=0.3,
        fact_count=3,
    )
    gist = GistResult(
        gist_text="Consolidated knowledge about entity_x from test cluster",
        key_entities=("entity_x",),
        extraction_mode="rules",
        representative_fact_id="src-fact-a",
    )
    return cluster, gist


class TestStep5StoreBlockAtomicity:
    """_step5_store_block writes must be atomic."""

    def test_rollback_on_store_ccq_block_failure_leaves_no_orphan(
        self, tmp_path: Path
    ) -> None:
        """When store_ccq_block raises after store_fact succeeds, the transaction
        must roll back so no orphan gist fact or memory row remains."""
        db = _make_real_db(tmp_path)
        # No source facts pre-created: the injected failure aborts before the
        # FK-constrained fact_access_log writes are reached, so this is safe.
        cluster, gist = _make_cluster_and_gist(db=None)

        from superlocalmemory.encoding.cognitive_consolidator import CognitiveConsolidator

        consolidator = CognitiveConsolidator(db=db)

        # Replace store_ccq_block on the instance so it raises after the
        # preceding store_fact has succeeded inside the transaction.
        real_store_ccq_block = db.store_ccq_block

        def _failing_ccq_block(**kwargs):
            raise RuntimeError("injected failure for atomicity test")

        db.store_ccq_block = _failing_ccq_block

        with pytest.raises(RuntimeError, match="injected failure"):
            consolidator._step5_store_block(cluster, gist, "default")

        db.store_ccq_block = real_store_ccq_block

        # Transaction must have rolled back — no orphan rows in any table.
        facts = db.execute(
            "SELECT fact_id FROM atomic_facts WHERE content = ?",
            (gist.gist_text,),
        )
        assert len(facts) == 0, (
            f"Orphan gist AtomicFact found after rollback: {facts}"
        )

        memories = db.execute(
            "SELECT memory_id FROM memories WHERE content = ?",
            (gist.gist_text,),
        )
        assert len(memories) == 0, (
            f"Orphan gist memory found after rollback: {memories}"
        )

        blocks = db.execute("SELECT block_id FROM ccq_consolidated_blocks", ())
        assert len(blocks) == 0, (
            f"Unexpected ccq_consolidated_blocks row after rollback: {blocks}"
        )

    def test_happy_path_commits_all_rows(self, tmp_path: Path) -> None:
        """On success, gist fact, gist memory, and ccq block are all committed."""
        db = _make_real_db(tmp_path)
        cluster, gist = _make_cluster_and_gist(db=db)

        from superlocalmemory.encoding.cognitive_consolidator import CognitiveConsolidator

        consolidator = CognitiveConsolidator(db=db)
        block_id = consolidator._step5_store_block(cluster, gist, "default")

        facts = db.execute(
            "SELECT fact_id FROM atomic_facts WHERE content = ?",
            (gist.gist_text,),
        )
        assert len(facts) == 1, f"Expected 1 gist fact, got {len(facts)}"

        memories = db.execute(
            "SELECT memory_id FROM memories WHERE content = ?",
            (gist.gist_text,),
        )
        assert len(memories) == 1, f"Expected 1 gist memory, got {len(memories)}"

        blocks = db.execute(
            "SELECT block_id, cluster_id, profile_id "
            "FROM ccq_consolidated_blocks WHERE block_id = ?",
            (block_id,),
        )
        assert len(blocks) == 1, f"Expected 1 ccq block, got {len(blocks)}"
        block_row = dict(blocks[0])
        assert block_row["cluster_id"] == cluster.cluster_id
        assert block_row["profile_id"] == "default"

    def test_second_call_with_same_gist_does_not_duplicate(
        self, tmp_path: Path
    ) -> None:
        """Calling _step5_store_block twice with identical content is idempotent
        in the fact table (store_fact deduplicates on content)."""
        db = _make_real_db(tmp_path)
        cluster, gist = _make_cluster_and_gist(db=db)

        from superlocalmemory.encoding.cognitive_consolidator import CognitiveConsolidator

        consolidator = CognitiveConsolidator(db=db)
        consolidator._step5_store_block(cluster, gist, "default")
        consolidator._step5_store_block(cluster, gist, "default")

        facts = db.execute(
            "SELECT fact_id FROM atomic_facts WHERE content = ?",
            (gist.gist_text,),
        )
        # store_fact deduplicates on content — exactly one canonical fact.
        assert len(facts) == 1, (
            f"Expected exactly 1 canonical fact after two identical stores, "
            f"got {len(facts)}"
        )
