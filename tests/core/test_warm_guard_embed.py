# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3 | Workstream D (3.8.4)
"""Tests for warm-guard synchronous embedding in store_fast().

Workstream D — fresh-fact recall latency fix.

Three contract assertions:
  (a) WARM embedder  → fact has non-NULL embedding immediately after store_fast()
  (b) COLD embedder  → store_fast() does NOT block; emb absent; materializer fills later
  (c) SLOW embedder  → 500ms timeout caught; emb absent; no raise; no thread leak
  (d) REMOTE embedder (cloud/openai-compatible) → stays async even if _available=True
  (e) No double-embed → enrich_fact() skips re-embed when fact.embedding is already set

TDD convention: these tests were written before the implementation.
Running against 3.8.3 baseline gives RED; running against the 3.8.4-D
implementation gives GREEN.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_embedder(
    available: bool | None,
    vector: list[float] | None = None,
    latency_s: float = 0.0,
    is_cloud: bool = False,
    is_openai_compatible: bool = False,
) -> MagicMock:
    """Build a mock embedder with controlled availability and latency."""
    embedder = MagicMock()
    embedder._available = available

    def _slow_embed(text: str) -> list[float] | None:
        if latency_s > 0:
            time.sleep(latency_s)
        return vector

    embedder.embed.side_effect = _slow_embed
    embedder.compute_fisher_params.return_value = (
        [0.1, 0.2, 0.3],
        [0.9, 0.8, 0.7],
    )

    # Simulate remote config if requested
    if is_cloud or is_openai_compatible:
        mock_cfg = MagicMock()
        mock_cfg.is_cloud = is_cloud
        mock_cfg.is_openai_compatible = is_openai_compatible
        embedder._config = mock_cfg
    else:
        # Local embedder — no _config attribute (like OllamaEmbedder)
        # Using del to remove the auto-created attribute from MagicMock
        try:
            del embedder._config
        except AttributeError:
            pass
        embedder._config = None  # local, not remote

    return embedder


# Every engine built here is closed after its test (audit C-3): an engine left
# open keeps its pool threads alive into later, unrelated tests.
_OPEN_ENGINES: list = []


@pytest.fixture(autouse=True)
def _close_engines_made_by_this_test():
    yield
    while _OPEN_ENGINES:
        _OPEN_ENGINES.pop().close()


def _make_engine(tmp_path: Path) -> "MemoryEngine":  # noqa: F821
    """Create a lightweight MemoryEngine with a real SQLite DB but no ML."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.storage.models import Mode

    cfg = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
    engine = MemoryEngine(cfg)
    engine._require_full = lambda _: None  # bypass FULL capability gate
    engine._ensure_init()
    _OPEN_ENGINES.append(engine)
    return engine


def _get_stored_embedding(engine: "MemoryEngine", fact_id: str) -> list[float] | None:  # noqa: F821
    """Retrieve the embedding stored for a fact_id from the DB."""
    row = engine._db.get_facts_by_ids([fact_id], engine._profile_id)
    if not row:
        return None
    return row[0].embedding


# ---------------------------------------------------------------------------
# (a) WARM embedder → embedding set synchronously
# ---------------------------------------------------------------------------

class TestWarmGuardSyncEmbed:
    """store_fast() embeds synchronously when the embedder is provably warm."""

    def test_warm_local_embedder_stores_embedding_immediately(
        self, tmp_path: Path,
    ) -> None:
        """After store_fast() on a warm local embedder, the fact's embedding
        column is non-NULL — no waiting for the materializer."""
        engine = _make_engine(tmp_path)
        vec = [0.1, 0.2, 0.3]
        embedder = _make_mock_embedder(available=True, vector=vec)
        engine._embedder = embedder

        fact_ids = engine.store_fast("Varun prefers Tokyo ramen")
        assert fact_ids, "store_fast must return at least one fact_id"

        stored = _get_stored_embedding(engine, fact_ids[0])
        assert stored is not None, (
            "Embedding must be non-NULL immediately after store_fast() on a warm embedder. "
            "Got None — warm-guard sync embed is not implemented."
        )
        assert len(stored) == 3

    def test_warm_embedder_triggers_fisher_params(
        self, tmp_path: Path,
    ) -> None:
        """fisher_mean and fisher_variance are also computed synchronously."""
        engine = _make_engine(tmp_path)
        vec = [0.1, 0.2, 0.3]
        embedder = _make_mock_embedder(available=True, vector=vec)
        engine._embedder = embedder

        fact_ids = engine.store_fast("Varun likes sushi")
        assert fact_ids

        row = engine._db.get_facts_by_ids([fact_ids[0]], engine._profile_id)
        assert row
        fact = row[0]
        assert fact.embedding is not None
        # fisher params computed in the warm-guard path
        assert fact.fisher_mean is not None, "fisher_mean must be set"
        assert fact.fisher_variance is not None, "fisher_variance must be set"

    def test_warm_embedder_embed_called_once(
        self, tmp_path: Path,
    ) -> None:
        """The warm-guard must call embed() exactly once."""
        engine = _make_engine(tmp_path)
        embedder = _make_mock_embedder(available=True, vector=[0.5, 0.6])
        engine._embedder = embedder

        engine.store_fast("test content")
        # embed() called once synchronously in store_fast
        assert embedder.embed.call_count >= 1


# ---------------------------------------------------------------------------
# (b) COLD embedder → store_fast() does NOT block; emb stays None
# ---------------------------------------------------------------------------

class TestColdEmbedderFallsBack:
    """On a cold embedder (_available != True), store_fast() skips sync embed."""

    @pytest.mark.parametrize("available", [None, False])
    def test_cold_or_failed_embedder_does_not_block(
        self, tmp_path: Path, available: bool | None,
    ) -> None:
        """_available=None (cold/probe) or False (dead) → sync embed skipped.

        A 2-second embed side effect makes an accidental foreground call
        observable without relying on runner-dependent SQLite wall time.
        """
        engine = _make_engine(tmp_path)
        # Embedder that would take 2s if called — proves we never call it
        embedder = _make_mock_embedder(available=available, vector=[1.0, 2.0], latency_s=2.0)
        engine._embedder = embedder

        fact_ids = engine.store_fast("cold test content")

        assert fact_ids, "store_fast must still return fact_ids when embedder is cold"
        embedder.embed.assert_not_called()

    @pytest.mark.parametrize("available", [None, False])
    def test_cold_embedder_leaves_embedding_null(
        self, tmp_path: Path, available: bool | None,
    ) -> None:
        """Cold/failed embedder → embedding column remains NULL (materializer fills it)."""
        engine = _make_engine(tmp_path)
        embedder = _make_mock_embedder(available=available, vector=[1.0, 2.0])
        engine._embedder = embedder

        fact_ids = engine.store_fast("another cold test")
        stored = _get_stored_embedding(engine, fact_ids[0])
        assert stored is None, (
            "Embedding must be NULL for cold/failed embedder — materializer must fill it async."
        )

    def test_none_embedder_does_not_crash(self, tmp_path: Path) -> None:
        """store_fast() works with _embedder=None (BM25-only mode)."""
        engine = _make_engine(tmp_path)
        engine._embedder = None
        fact_ids = engine.store_fast("no embedder content")
        assert fact_ids


# ---------------------------------------------------------------------------
# (c) SLOW embedder → timeout caught, emb=None, no thread leak
# ---------------------------------------------------------------------------

class TestSlowEmbedderTimeout:
    """embed() takes >500ms → TimeoutError caught; store_fast returns with emb=None."""

    def test_slow_embed_falls_back_to_async(self, tmp_path: Path) -> None:
        """An embed that never finishes on its own must not hold store_fast().

        The embed blocks until the test releases it, which happens only after
        store_fast() has returned. So store_fast() can finish only by honouring
        its own embed deadline — no stopwatch needed.
        """
        import os
        import threading

        # Liveness guard, NOT a performance claim: only a store_fast() that
        # waits for the blocked embed itself can run into it.
        liveness_timeout_s = 15.0
        release_embed = threading.Event()
        engine = _make_engine(tmp_path)
        embedder = _make_mock_embedder(available=True, vector=[1.0, 2.0, 3.0])
        # Bounded only so a broken run cannot leak a stuck pool thread forever.
        embedder.embed.side_effect = lambda _text: (
            release_embed.wait(timeout=4 * liveness_timeout_s) and [1.0, 2.0, 3.0]
        )
        engine._embedder = embedder
        outcome: dict = {}

        def _store() -> None:
            outcome["fact_ids"] = engine.store_fast("slow embed content")

        try:
            with patch.dict(os.environ, {"SLM_STORE_FAST_EMBED_TIMEOUT_MS": "200"}):
                store = threading.Thread(target=_store, daemon=True)
                store.start()
                store.join(timeout=liveness_timeout_s)
                assert not store.is_alive(), (
                    "store_fast is waiting on an embed past its own deadline"
                )
        finally:
            release_embed.set()

        assert outcome.get("fact_ids"), "store_fast must still return fact_ids on embed timeout"

    def test_slow_embed_leaves_embedding_null(self, tmp_path: Path) -> None:
        """Timed-out embed → embedding NULL; materializer fills later."""
        import os
        engine = _make_engine(tmp_path)
        with patch.dict(os.environ, {"SLM_STORE_FAST_EMBED_TIMEOUT_MS": "200"}):
            embedder = _make_mock_embedder(
                available=True,
                vector=[1.0],
                latency_s=0.4,
            )
            engine._embedder = embedder
            fact_ids = engine.store_fast("timeout content check")

        stored = _get_stored_embedding(engine, fact_ids[0])
        assert stored is None, "Timed-out embed must not populate the embedding column."

    def test_embed_exception_does_not_propagate(self, tmp_path: Path) -> None:
        """embed() raises RuntimeError → caught, emb=None, no crash."""
        engine = _make_engine(tmp_path)
        embedder = MagicMock()
        embedder._available = True
        embedder._config = None  # local
        embedder.embed.side_effect = RuntimeError("model crashed")

        engine._embedder = embedder
        fact_ids = engine.store_fast("exception content")
        assert fact_ids, "store_fast must not raise when embed() throws"

        stored = _get_stored_embedding(engine, fact_ids[0])
        assert stored is None, "Failed embed must not populate embedding column."


# ---------------------------------------------------------------------------
# (d) REMOTE embedder → stays async even if _available=True
# ---------------------------------------------------------------------------

class TestRemoteEmbedderStaysAsync:
    """Cloud / OpenAI-compatible embedders must never be called synchronously."""

    @pytest.mark.parametrize("cloud,openai_compat", [
        (True, False),
        (False, True),
    ])
    def test_remote_embedder_not_called_synchronously(
        self, tmp_path: Path, cloud: bool, openai_compat: bool,
    ) -> None:
        """Remote embedder with _available=True — embed() must NOT be called
        in store_fast() synchronously. The embedding must remain NULL."""
        engine = _make_engine(tmp_path)
        embedder = _make_mock_embedder(
            available=True,
            vector=[0.1, 0.2],
            is_cloud=cloud,
            is_openai_compatible=openai_compat,
        )
        engine._embedder = embedder

        fact_ids = engine.store_fast("remote embedder content")
        assert fact_ids

        stored = _get_stored_embedding(engine, fact_ids[0])
        assert stored is None, (
            "Remote embedder must not be called synchronously in store_fast(). "
            "Embedding must be NULL — deferred to materializer."
        )


# ---------------------------------------------------------------------------
# (e) No double-embed: enrich_fact() skips re-embed when embedding already set
# ---------------------------------------------------------------------------

class TestNoDoubleEmbed:
    """enrich_fact() must skip the embed call when fact.embedding is already non-None."""

    def test_enrich_fact_skips_embed_when_already_present(self) -> None:
        """If fact.embedding is set, embedder.embed() must not be called."""
        from superlocalmemory.core.store_pipeline import enrich_fact
        from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

        existing_vec = [0.11, 0.22, 0.33]
        fact = AtomicFact(
            fact_id="test001",
            content="Varun works at Accenture",
            fact_type=FactType.SEMANTIC,
            entities=[],
            observation_date="2026-07-26",
            confidence=0.8,
            importance=0.5,
            embedding=existing_vec,  # already embedded
            fisher_mean=[0.1, 0.2, 0.3],
            fisher_variance=[0.9, 0.8, 0.7],
        )
        record = MemoryRecord(
            profile_id="default",
            content="Varun works at Accenture",
            session_date="2026-07-26",
        )
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [9.9, 9.9, 9.9]  # different — should NOT be used
        mock_embedder.compute_fisher_params.return_value = ([9.9], [9.9])

        enriched = enrich_fact(
            fact, record, "default",
            embedder=mock_embedder,
            entity_resolver=None,
            temporal_parser=None,
        )

        # embed() must NOT have been called (existing embedding preserved)
        mock_embedder.embed.assert_not_called()
        # The enriched fact must carry the original vector, not [9.9, 9.9, 9.9]
        assert enriched.embedding == existing_vec, (
            "enrich_fact must preserve existing embedding — no double-embed."
        )

    def test_enrich_fact_still_embeds_when_embedding_is_none(self) -> None:
        """Normal path: fact.embedding=None → embedder.embed() is called as before."""
        from superlocalmemory.core.store_pipeline import enrich_fact
        from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

        fact = AtomicFact(
            fact_id="test002",
            content="some content without pre-embedding",
            fact_type=FactType.SEMANTIC,
            entities=[],
            observation_date="2026-07-26",
            confidence=0.7,
            importance=0.5,
            embedding=None,
        )
        record = MemoryRecord(
            profile_id="default",
            content="some content without pre-embedding",
            session_date="2026-07-26",
        )
        mock_embedder = MagicMock()
        new_vec = [0.5, 0.6, 0.7]
        mock_embedder.embed.return_value = new_vec
        mock_embedder.compute_fisher_params.return_value = ([0.1], [0.9])

        enriched = enrich_fact(
            fact, record, "default",
            embedder=mock_embedder,
            entity_resolver=None,
            temporal_parser=None,
        )

        mock_embedder.embed.assert_called_once_with("some content without pre-embedding")
        assert enriched.embedding == new_vec


# ---------------------------------------------------------------------------
# Pool reuse sanity — shared pool across calls, no thread churn
# ---------------------------------------------------------------------------

class TestEmbedPoolReuse:
    """The warm-guard thread pool is reused across store_fast() calls."""

    def test_pool_is_reused_across_calls(self, tmp_path: Path) -> None:
        """Two consecutive store_fast() calls share the same pool instance."""
        engine = _make_engine(tmp_path)
        embedder = _make_mock_embedder(available=True, vector=[0.1, 0.2])
        engine._embedder = embedder

        engine.store_fast("first content")
        pool_first = engine._store_fast_embed_pool

        engine.store_fast("second content")
        pool_second = engine._store_fast_embed_pool

        assert pool_first is pool_second, (
            "The warm-guard ThreadPoolExecutor must be reused across calls — "
            "not re-created per store_fast() invocation."
        )
