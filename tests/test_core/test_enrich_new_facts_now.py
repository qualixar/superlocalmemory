# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A memory should be findable by meaning as soon as it is stored.

Storing a memory and being able to find it again are different things. A fact
with no vector can only be matched on its own wording, which is not how anyone
asks a question — "what am I working on" shares no distinctive word with any
particular memory. Until the vector exists, the memory describing what is
happening right now is the hardest one in the store to find.

Every entry point — the command line, the tool interface and the dashboard —
commits through one durable receipt and then relies on a background pass to
attach the vector. These tests cover the bounded attempt made before that
receipt is returned, so the window is closed for all three at once rather than
for whichever door happened to be used.

The durable write has already happened by then. Nothing here can fail it: on any
error, any timeout, or a cold embedder, the fact simply keeps its place in the
background queue.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.storage.models import Mode
from tests.helpers.env_capabilities import (
    NO_VECTOR_SEARCH_REASON,
    vector_search_available,
)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[MemoryEngine]:
    eng = MemoryEngine(
        SLMConfig(
            mode=Mode.A,
            base_dir=tmp_path,
            db_path=tmp_path / "memory.db",
            active_profile="default",
            embedding=EmbeddingConfig(model_name="nomic-embed-text", dimension=768),
        )
    )
    eng._ensure_init()
    yield eng
    eng.close()


def _local_embedder_mock() -> MagicMock:
    """A mock that the engine treats as a LOCAL embedder.

    `_is_remote_embedder` reads `embedder._config` and a bare `MagicMock`
    auto-creates it, so `is_cloud` comes back as a truthy mock and the engine
    classifies the mock as remote — returning before it touches the code below.
    Three tests in this file passed that way without ever executing the path
    they claimed to cover. `_config = None` is what a local embedder looks like.
    """
    m = MagicMock()
    m._config = None
    return m


def _fact_without_a_vector(engine: MemoryEngine, content: str) -> str:
    """Write a fact and strip every trace of its enrichment.

    This is the state a fact is in between "stored durably" and "the background
    pass has caught up", which is the window under test.
    """
    fact_id = engine.store_fast(content)[0]
    conn = sqlite3.connect(str(engine._config.db_path))
    try:
        conn.execute("DELETE FROM embedding_metadata WHERE fact_id = ?", (fact_id,))
        conn.execute("UPDATE atomic_facts SET embedding = NULL WHERE fact_id = ?", (fact_id,))
        conn.commit()
    finally:
        conn.close()
    return fact_id


def _state(engine: MemoryEngine, fact_id: str) -> tuple[str, int]:
    conn = sqlite3.connect(f"file:{engine._config.db_path}?mode=ro", uri=True)
    try:
        kind = conn.execute(
            "SELECT typeof(embedding) FROM atomic_facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()[0]
        projected = conn.execute(
            "SELECT COUNT(*) FROM embedding_metadata WHERE fact_id = ?", (fact_id,)
        ).fetchone()[0]
        return kind, projected
    finally:
        conn.close()


@pytest.mark.skipif(
    not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON,
)
class TestTheVectorIsAttached:
    """Requires a working vector store, which requires SQLite extensions.

    Without them ``VectorStore.available`` is False, the engine's vector store
    is None, and "is this findable by meaning" answers no for everything — so
    these assertions fail on a count of 0 with nothing pointing at the cause.
    That is an interpreter capability, not a product defect, and skipping with
    the remedy named beats failing four frames from the reason.
    """

    def test_both_representations_are_written(self, engine: MemoryEngine) -> None:
        """The column alone is not enough, and this is the trap worth pinning.

        Searching by meaning reads the vector projection, not the canonical
        column. Writing only the column looks like success at the database level
        and leaves the memory exactly as unfindable as before.

        A deterministic local embedder is injected through the engine's own
        seam (``engine._embedder`` — see ``_local_embedder_mock`` above and
        ``tests/core/test_warm_guard_embed.py``) rather than relying on
        whatever real embedder the engine fixture would otherwise build: CI has
        no local model and no Ollama, so the real embedder is never warm and
        this test would otherwise fail on "is it embedded" instead of on the
        thing it is actually pinning (the column/projection pair).
        """
        fake = _local_embedder_mock()
        fake._available = True
        fake.embed.return_value = [0.01] * 768
        fake.compute_fisher_params.return_value = ([0.0] * 768, [1.0] * 768)
        engine._embedder = fake

        fact_id = _fact_without_a_vector(engine, "The Helsinki ledger reconciliation is deferred.")
        assert _state(engine, fact_id) == ("null", 0), "test setup did not strip the vector"

        assert engine.enrich_new_facts_now([fact_id]) == 1

        kind, projected = _state(engine, fact_id)
        assert kind == "blob", (
            f"the embedding was stored as {kind!r}; a text value here means a writer "
            f"bypassed the codec and reintroduced the previous format"
        )
        assert projected == 1, (
            "the canonical column was written but the vector projection was not, so "
            "searching by meaning still cannot reach this memory"
        )

    def test_a_fact_that_already_has_one_counts_without_being_re_embedded(
        self, engine: MemoryEngine,
    ) -> None:
        """It is already searchable, so it counts — and costs nothing.

        The returned number answers "how many of these can be found by meaning",
        which is what the caller reports to the user. Returning 0 for a fact that
        is already fully searchable would make the receipt say "wording only"
        about a memory that is in fact findable.

        Three halves are asserted, and the third is the one that matters. The
        count alone would pass on an implementation that re-embeds every time;
        the embedder alone would pass on one that always returns 0; and both
        together would still pass for a fact that has a vector in its canonical
        column and no projection behind it — which is unreachable by a search on
        meaning, and is exactly the state this number must not claim success for.
        So the end state is checked too.
        """
        fake = _local_embedder_mock()
        fake._available = True
        fake.embed.return_value = [0.02] * 768
        fake.compute_fisher_params.return_value = ([0.0] * 768, [1.0] * 768)
        engine._embedder = fake

        fact_id = engine.store_fast("Procurement confirmed the tariff schedule.")[0]
        spy = MagicMock(wraps=engine._embedder)
        spy._config = None
        spy._available = getattr(engine._embedder, "_available", None)
        engine._embedder = spy
        assert engine.enrich_new_facts_now([fact_id]) == 1, (
            "a fact that is already searchable by meaning must count towards the "
            "caller's receipt, otherwise the user is told it cannot be found"
        )
        spy.embed.assert_not_called()
        assert _state(engine, fact_id) == ("blob", 1), (
            "the count said this fact is findable by meaning, but the projection "
            "a meaning-based search reads is not there; the number is a claim "
            "about reachability and has to be checked against it"
        )

    def test_nothing_to_do_is_not_an_error(self, engine: MemoryEngine) -> None:
        assert engine.enrich_new_facts_now([]) == 0
        assert engine.enrich_new_facts_now(["does-not-exist"]) == 0


class TestItCannotHarmTheWrite:
    """The durable write is already committed; this runs on top of it."""

    def test_a_cold_embedder_defers_instead_of_failing(self, engine: MemoryEngine) -> None:
        fact_id = _fact_without_a_vector(engine, "A note recorded while the embedder is cold.")
        cold = _local_embedder_mock()
        cold._available = False
        engine._embedder = cold
        assert engine.enrich_new_facts_now([fact_id]) == 0
        assert _state(engine, fact_id) == ("null", 0)
        cold.embed.assert_not_called()

    def test_an_embedder_that_raises_does_not_propagate(self, engine: MemoryEngine) -> None:
        fact_id = _fact_without_a_vector(engine, "A note recorded while the embedder is broken.")
        broken = _local_embedder_mock()
        broken._available = True
        broken.embed.side_effect = RuntimeError("model not loaded")
        engine._embedder = broken
        assert engine.enrich_new_facts_now([fact_id]) == 0

    def test_the_deadline_is_honoured(self, engine: MemoryEngine) -> None:
        """Past the deadline the caller proceeds and the background pass finishes.

        This daemon serves many sessions, so a slow embedder must cost the caller
        a bounded wait and nothing more.
        """
        import threading
        import time as _time

        # Real-time bound on the caller's wait. The embed holds out for 30 s
        # unless released, so a caller that ignored its 0.2 s deadline would wait
        # ~30 s; a third of that is slack no loaded host can use up honestly.
        slow_embed_s, deadline_s = 30.0, 0.2
        caller_bound_s = slow_embed_s / 3
        fact_id = _fact_without_a_vector(engine, "A note recorded while the embedder is slow.")
        release = threading.Event()
        slow = _local_embedder_mock()
        slow._available = True
        slow.embed.side_effect = lambda _t: (release.wait(slow_embed_s), [0.1] * 768)[1]
        engine._embedder = slow
        started = _time.monotonic()
        try:
            assert engine.enrich_new_facts_now([fact_id], timeout_s=deadline_s) == 0
            assert _time.monotonic() - started < caller_bound_s, (
                "the deadline was not enforced")
        finally:
            release.set()  # let the abandoned embed finish so close() is prompt
