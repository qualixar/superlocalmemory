# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for backfill_missing_embeddings in embedding_migrator.py.

Covers:
- Basic: NULL embeddings are written, embedding_metadata upserted
- Idempotent: re-running only touches remaining NULLs
- Limit: only N facts are embedded per call (bounded)
- all_profiles: spans facts across all profile_ids
- Fail-open: a bad fact is skipped, rest continue
- No-op: 0 NULLs returns early
- Return dict: scanned/embedded/remaining_null counts are correct
- Byte-compat: embedding JSON is a float list (~16-17KB for dim=768)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    """DatabaseManager with full SLM schema."""
    manager = DatabaseManager(tmp_path / "memory.db")
    manager.initialize(schema)
    return manager


@pytest.fixture
def config(tmp_path: Path) -> SLMConfig:
    """Mode A SLMConfig pointing at tmp_path."""
    from superlocalmemory.storage.models import Mode
    cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    cfg.active_profile = "default"
    return cfg


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock embedder returning deterministic 768-dim unit vectors."""
    emb = MagicMock()

    def _embed(text: str) -> list[float]:
        rng = np.random.RandomState(abs(hash(text)) % 2**31)
        vec = rng.randn(768).astype(np.float32)
        return (vec / np.linalg.norm(vec)).tolist()

    def _embed_batch(texts: list[str]) -> list[list[float] | None]:
        return [_embed(t) for t in texts]

    emb.embed.side_effect = _embed
    emb.embed_batch.side_effect = _embed_batch
    return emb


def _seed_facts(
    db: DatabaseManager,
    profile_id: str,
    contents: list[str],
    *,
    with_embedding: bool = False,
) -> list[str]:
    """Insert atomic_facts rows with optional NULL/present embeddings.
    Returns list of fact_ids inserted.

    Uses PRAGMA foreign_keys=OFF so we can insert without seeding memories.
    """
    fact_ids: list[str] = []
    # Synthetic memory_id (one per profile; FK off so no actual memories row needed)
    mem_id = f"mem-{profile_id[:8]}-seed"
    with db.raw_connection() as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT OR IGNORE INTO profiles(profile_id, name) VALUES (?, ?)",
            (profile_id, profile_id),
        )
        for i, content in enumerate(contents):
            fid = f"fact-{profile_id[:4]}-{i:04d}"
            fact_ids.append(fid)
            embedding = (
                json.dumps([0.01] * 768) if with_embedding else None
            )
            conn.execute(
                "INSERT OR IGNORE INTO atomic_facts"
                " (fact_id, memory_id, profile_id, content, embedding)"
                " VALUES (?, ?, ?, ?, ?)",
                (fid, mem_id, profile_id, content, embedding),
            )
            if with_embedding:
                conn.execute(
                    "INSERT OR IGNORE INTO embedding_metadata"
                    " (fact_id, profile_id, model_name, dimension)"
                    " VALUES (?, ?, ?, ?)",
                    (fid, profile_id, "nomic-ai/nomic-embed-text-v1.5", 768),
                )
    return fact_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBackfillMissingEmbeddings:

    def test_null_embeddings_get_written(
        self, db: DatabaseManager, config: SLMConfig, mock_embedder: MagicMock
    ) -> None:
        """After backfill, all previously-NULL facts have a non-NULL embedding."""
        from superlocalmemory.storage.embedding_migrator import backfill_missing_embeddings

        _seed_facts(db, "default", ["fact A", "fact B", "fact C"])

        # PRE: 3 NULLs
        before = db.execute(
            "SELECT count(*) AS c FROM atomic_facts WHERE embedding IS NULL"
        )
        assert before[0]["c"] == 3

        result = backfill_missing_embeddings(config, db, mock_embedder)

        # POST: 0 NULLs
        after = db.execute(
            "SELECT count(*) AS c FROM atomic_facts WHERE embedding IS NULL"
        )
        assert after[0]["c"] == 0
        assert result["embedded"] == 3
        assert result["remaining_null"] == 0

    def test_embedding_metadata_upserted(
        self, db: DatabaseManager, config: SLMConfig, mock_embedder: MagicMock
    ) -> None:
        """embedding_metadata rows are created for each backfilled fact."""
        from superlocalmemory.storage.embedding_migrator import backfill_missing_embeddings

        fact_ids = _seed_facts(db, "default", ["alpha", "beta"])

        # PRE: no metadata rows for these facts
        before = db.execute(
            "SELECT count(*) AS c FROM embedding_metadata WHERE fact_id IN (?, ?)",
            tuple(fact_ids),
        )
        assert before[0]["c"] == 0

        backfill_missing_embeddings(config, db, mock_embedder)

        after = db.execute(
            "SELECT count(*) AS c FROM embedding_metadata WHERE fact_id IN (?, ?)",
            tuple(fact_ids),
        )
        assert after[0]["c"] == 2

    def test_byte_compatibility_embedding_format(
        self, db: DatabaseManager, config: SLMConfig, mock_embedder: MagicMock
    ) -> None:
        """Backfilled embedding is a JSON float list compatible with get_all_facts.

        Dimension: 768, size: ~12-20KB (matching existing embeddings).
        Can be deserialized to a list of 768 floats.
        """
        from superlocalmemory.storage.embedding_migrator import backfill_missing_embeddings

        _seed_facts(db, "default", ["content to embed"])
        backfill_missing_embeddings(config, db, mock_embedder)

        rows = db.execute(
            "SELECT embedding FROM atomic_facts WHERE profile_id = 'default'"
        )
        assert len(rows) == 1
        emb_json = rows[0]["embedding"]
        assert emb_json is not None

        parsed = json.loads(emb_json)
        assert isinstance(parsed, list)
        assert len(parsed) == 768
        assert isinstance(parsed[0], float)

        # Size should be in the range of existing embeddings (~12KB-20KB)
        assert 10_000 < len(emb_json) < 25_000, (
            f"Unexpected embedding JSON size: {len(emb_json)} bytes"
        )

    def test_idempotent_resumable(
        self, db: DatabaseManager, config: SLMConfig, mock_embedder: MagicMock
    ) -> None:
        """Re-running after partial completion only embeds remaining NULLs."""
        from superlocalmemory.storage.embedding_migrator import backfill_missing_embeddings

        _seed_facts(db, "default", ["a", "b", "c", "d", "e"])

        # First pass: limit=2
        result1 = backfill_missing_embeddings(config, db, mock_embedder, limit=2)
        assert result1["embedded"] == 2
        assert result1["remaining_null"] == 3

        # Second pass: limit=2 (only 3 remain, so max 2 more)
        result2 = backfill_missing_embeddings(config, db, mock_embedder, limit=2)
        assert result2["embedded"] == 2
        assert result2["remaining_null"] == 1

        # Third pass: no limit (last 1)
        result3 = backfill_missing_embeddings(config, db, mock_embedder)
        assert result3["embedded"] == 1
        assert result3["remaining_null"] == 0

    def test_limit_bounds_per_call(
        self, db: DatabaseManager, config: SLMConfig, mock_embedder: MagicMock
    ) -> None:
        """With limit=N, at most N facts are embedded per call."""
        from superlocalmemory.storage.embedding_migrator import backfill_missing_embeddings

        _seed_facts(db, "default", ["x"] * 10)

        result = backfill_missing_embeddings(config, db, mock_embedder, limit=3)
        assert result["embedded"] == 3
        assert result["scanned"] == 10  # total scanned = all NULLs
        assert result["remaining_null"] == 7

    def test_all_profiles_flag(
        self, db: DatabaseManager, config: SLMConfig, mock_embedder: MagicMock
    ) -> None:
        """With all_profiles=True, facts from all profiles are backfilled."""
        from superlocalmemory.storage.embedding_migrator import backfill_missing_embeddings

        _seed_facts(db, "default", ["p1 fact 1", "p1 fact 2"])
        _seed_facts(db, "work", ["p2 fact 1"])

        # Scope to active_profile only (default): only 2 facts
        result_single = backfill_missing_embeddings(
            config, db, mock_embedder, all_profiles=False
        )
        assert result_single["embedded"] == 2

        # Re-add NULLs for work profile by seeding again (work facts still NULL)
        result_all = backfill_missing_embeddings(
            config, db, mock_embedder, all_profiles=True
        )
        assert result_all["embedded"] == 1  # 1 remaining work fact

    def test_fail_open_bad_fact_skipped(
        self, db: DatabaseManager, config: SLMConfig
    ) -> None:
        """A fact that fails to embed is skipped; remaining facts continue."""
        from superlocalmemory.storage.embedding_migrator import backfill_missing_embeddings

        _seed_facts(db, "default", ["good fact 1", "bad fact", "good fact 2"])

        bad_embedder = MagicMock()
        call_count = [0]

        def _selective_embed_batch(texts: list[str]) -> list[list[float] | None]:
            results = []
            for t in texts:
                if "bad" in t:
                    results.append(None)  # simulate NULL return for bad fact
                else:
                    rng = np.random.RandomState(abs(hash(t)) % 2**31)
                    vec = rng.randn(768).astype(np.float32)
                    results.append((vec / np.linalg.norm(vec)).tolist())
            return results

        bad_embedder.embed_batch.side_effect = _selective_embed_batch

        result = backfill_missing_embeddings(config, db, bad_embedder)

        # 2 good facts succeed, bad fact has None vector — should be skipped
        assert result["embedded"] == 2
        assert result["remaining_null"] == 1  # bad fact still NULL

    def test_no_op_when_no_nulls(
        self, db: DatabaseManager, config: SLMConfig, mock_embedder: MagicMock
    ) -> None:
        """Returns early with zeros when no NULL embeddings exist."""
        from superlocalmemory.storage.embedding_migrator import backfill_missing_embeddings

        _seed_facts(db, "default", ["already embedded"], with_embedding=True)

        result = backfill_missing_embeddings(config, db, mock_embedder)
        assert result["scanned"] == 0
        assert result["embedded"] == 0
        assert result["remaining_null"] == 0
        # embed_batch must NOT be called at all
        mock_embedder.embed_batch.assert_not_called()

    def test_no_embedder_returns_zero_counts(
        self, db: DatabaseManager, config: SLMConfig
    ) -> None:
        """When embedder=None, returns dict with scanned=0, embedded=0."""
        from superlocalmemory.storage.embedding_migrator import backfill_missing_embeddings

        _seed_facts(db, "default", ["unembed me"])

        result = backfill_missing_embeddings(config, db, embedder=None)
        assert result["scanned"] == 0
        assert result["embedded"] == 0


class TestBackfillInMaintenance:
    """run_maintenance calls backfill when embedder is provided."""

    def test_maintenance_calls_backfill_when_embedder_given(
        self, db: DatabaseManager, config: SLMConfig, mock_embedder: MagicMock, tmp_path: Path
    ) -> None:
        """run_maintenance with embedder= fills NULLs up to the bounded limit."""
        from superlocalmemory.core.maintenance import run_maintenance

        # Seed facts that run_maintenance can see
        _seed_facts(db, "default", ["maint fact 1", "maint fact 2"])

        counts = run_maintenance(db, config, profile_id="default", embedder=mock_embedder)

        assert "embeddings_backfilled" in counts
        # At least 1 fact embedded (bounded to 100 per pass)
        assert counts["embeddings_backfilled"] >= 1

    def test_maintenance_no_op_when_no_nulls(
        self, db: DatabaseManager, config: SLMConfig, mock_embedder: MagicMock
    ) -> None:
        """run_maintenance does NOT call embedder when 0 NULLs exist."""
        from superlocalmemory.core.maintenance import run_maintenance

        _seed_facts(db, "default", ["embedded fact"], with_embedding=True)

        counts = run_maintenance(db, config, profile_id="default", embedder=mock_embedder)

        assert counts.get("embeddings_backfilled", 0) == 0
        mock_embedder.embed_batch.assert_not_called()

    def test_maintenance_skips_backfill_when_no_embedder(
        self, db: DatabaseManager, config: SLMConfig
    ) -> None:
        """run_maintenance without embedder does not raise; backfill is skipped."""
        from superlocalmemory.core.maintenance import run_maintenance

        _seed_facts(db, "default", ["no embedder fact"])

        # Must not raise
        counts = run_maintenance(db, config, profile_id="default")
        # No embeddings_backfilled key or 0
        assert counts.get("embeddings_backfilled", 0) == 0
