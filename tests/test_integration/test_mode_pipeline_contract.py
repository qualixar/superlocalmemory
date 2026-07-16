# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Sandbox contracts for the Mode A/B/C canonical memory pipeline.

These tests deliberately exercise the production ``MemoryEngine`` facade,
the M018 canonical-ingestion state machine, and the real retrieval wiring.
They use deterministic local doubles only for embedding/LLM inference; a
network provider's availability is *not* evidence that a mode's durable local
pipeline works.

Mode C's configured cloud provider and Mode B's configured Ollama provider
therefore remain separately covered by their endpoint/provider tests.  This
file proves the invariant that all modes share: remember -> canonical durable
derivation -> indexed recall, with dates and provenance preserved.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.core.ingestion_command import (
    IngestionOperationRepository,
    IngestionState,
)
from superlocalmemory.storage.models import Mode


class _DeterministicEmbedder:
    """Network-free embedding double with the configured dimensionality."""

    is_available = True

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vector = rng.standard_normal(self.dimension).astype(np.float32)
        vector /= np.linalg.norm(vector)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def compute_fisher_params(
        self, embedding: list[float],
    ) -> tuple[list[float], list[float]]:
        vector = np.asarray(embedding, dtype=np.float64)
        return vector.tolist(), np.ones(len(vector), dtype=np.float64).tolist()


class _LocalLLM:
    """Minimal deterministic local LLM contract for B/C pipeline wiring."""

    def __init__(self, _config) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def generate(self, *_args, **_kwargs) -> str:
        # FactExtractor accepts this exact JSON contract.  Keeping it local
        # proves B/C plumbing without representing a cloud-provider test.
        return (
            '[{"text":"Ada maintains Atlas retrieval graph operations on '
            '2026-07-15.","fact_type":"semantic","entities":'
            '["Ada","Atlas"],"referenced_date":"2026-07-15",'
            '"importance":8,"confidence":0.95}]'
        )


def _make_engine(mode: Mode, base_dir: Path) -> MemoryEngine:
    config = SLMConfig.for_mode(mode, base_dir=base_dir)
    config.retrieval.use_cross_encoder = False
    # The test is about the canonical engine path, never an external provider.
    embedder = _DeterministicEmbedder(config.embedding.dimension)
    engine = MemoryEngine(config)
    with ExitStack() as stack:
        stack.enter_context(patch(
            "superlocalmemory.core.engine_wiring.init_embedder",
            return_value=embedder,
        ))
        stack.enter_context(patch(
            "superlocalmemory.llm.backbone.LLMBackbone", _LocalLLM,
        ))
        engine.initialize()
    return engine


def _assert_sqlite_integrity(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


@pytest.mark.parametrize("mode", [Mode.A, Mode.B, Mode.C])
def test_every_mode_completes_canonical_ingestion_and_graph_aware_recall(
    mode: Mode, tmp_path: Path,
) -> None:
    """Every mode retains dated evidence and runs its configured retrieval layers."""
    engine = _make_engine(mode, tmp_path)
    try:
        fact_ids = engine.store(
            "Ada maintains Atlas retrieval graph operations for the platform "
            "team. The production review happened on July 15, 2026.",
            session_id=f"mode-{mode.value}-session",
            session_date="2026-07-15",
            speaker="Ada",
            metadata={"source": "mode-contract"},
        )
        assert fact_ids

        operation = IngestionOperationRepository(engine.db).list_operations()
        assert len(operation) == 1
        operation = operation[0]
        assert operation.state is IngestionState.COMPLETE
        assert operation.session_date == "2026-07-15"
        assert set(fact_ids) == set(operation.final_fact_ids)
        assert all(operation.derivation_state.values()), operation.derivation_state
        assert {
            "relational", "fts", "extraction", "canonicalization",
            "consolidation", "graph", "temporal", "provenance",
            "trust_policy", "embeddings", "ann", "vector", "bm25",
        }.issubset(operation.derivation_state)

        facts = engine.db.get_facts_by_ids(fact_ids, engine.profile_id)
        assert facts
        # Some facts retain the parsed UTC midnight timestamp while an LLM
        # extraction can retain the source date string.  Both forms preserve
        # the same explicit calendar tag and are intentionally accepted.
        assert all(
            (fact.observation_date or "").startswith("2026-07-15")
            for fact in facts
        )
        assert all(fact.embedding is not None for fact in facts)
        # A source turn can yield a contextual fact with no named entity.  At
        # least one fact must still reach canonical entities so graph and
        # temporal projections have a real durable seed.
        assert any(fact.canonical_entities for fact in facts)

        retrieval = engine._retrieval_engine
        channel_spies = {}
        for name, channel in {
            "semantic": retrieval._semantic,
            "bm25": retrieval._bm25,
            "temporal": retrieval._temporal,
            "hopfield": retrieval._hopfield,
            "spreading_activation": retrieval._spreading_activation,
        }.items():
            assert channel is not None, f"{name} was not wired in Mode {mode.value}"
            spy = MagicMock(wraps=channel.search)
            channel.search = spy
            channel_spies[name] = spy
        graph_score = MagicMock(wraps=retrieval._entity.score_candidates)
        retrieval._entity.score_candidates = graph_score
        started = time.perf_counter()
        response = engine.recall("What does Ada maintain in Atlas?", mode=mode)
        wall_clock_ms = (time.perf_counter() - started) * 1000.0

        assert response.results
        assert any("Ada" in result.fact.content for result in response.results)
        # The response metric is part of the public latency observability
        # contract.  This test intentionally does not impose a machine- or
        # corpus-independent one-second SLA on a mocked sandbox.
        assert 0.0 <= response.retrieval_time_ms <= wall_clock_ms + 50.0
        for name, spy in channel_spies.items():
            assert spy.call_count >= 1, f"{name} did not participate"
        # Entity graph is intentionally a post-fusion enhancer: it scores the
        # candidates earned by semantic/BM25/temporal evidence rather than
        # injecting ungrounded graph-only candidates into the answer.
        assert graph_score.call_count >= 1
        assert "semantic" in response.channel_weights
        assert "bm25" in response.channel_weights
        assert "entity_graph" in response.channel_weights
        assert "temporal" in response.channel_weights
        assert "spreading_activation" in response.channel_weights
        assert "hopfield" in response.channel_weights

        # The product's durable stores are memory.db, learning.db, and the
        # tamper-evident audit chain.  They must exist and remain SQLite-valid.
        for database in (
            engine.db.db_path,
            engine._config.base_dir / "learning.db",
            engine._config.base_dir / "audit_chain.db",
        ):
            assert database.exists(), database
            _assert_sqlite_integrity(database)
    finally:
        engine.close()


def test_spreading_activation_remains_wired_without_sqlite_vec(
    tmp_path: Path,
) -> None:
    """The graph layer needs a SQLite seed fallback when vec0 is unavailable."""
    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    config.retrieval.use_cross_encoder = False
    embedder = _DeterministicEmbedder(config.embedding.dimension)
    engine = MemoryEngine(config)
    with ExitStack() as stack:
        stack.enter_context(patch(
            "superlocalmemory.core.engine_wiring.init_embedder",
            return_value=embedder,
        ))
        stack.enter_context(patch(
            "superlocalmemory.core.engine_wiring._init_vector_store",
            return_value=None,
        ))
        stack.enter_context(patch(
            "superlocalmemory.llm.backbone.LLMBackbone", _LocalLLM,
        ))
        engine.initialize()
    try:
        engine.store(
            "Ada maintains the Atlas graph recovery procedure on July 15, 2026.",
            session_id="sqlite-fallback-session",
            session_date="2026-07-15",
        )
        retrieval = engine._retrieval_engine
        assert retrieval._spreading_activation is not None
        spreading_search = MagicMock(
            wraps=retrieval._spreading_activation.search,
        )
        retrieval._spreading_activation.search = spreading_search

        response = engine.recall("What does Ada maintain in Atlas?", mode=Mode.A)

        assert response.results
        assert spreading_search.call_count >= 1
        assert "spreading_activation" in response.channel_weights
    finally:
        engine.close()
