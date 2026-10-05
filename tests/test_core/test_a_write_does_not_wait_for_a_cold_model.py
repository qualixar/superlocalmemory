# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A write embeds inline only when that is actually cheap.

Storing a memory computes its vector on the spot so the memory can be found by
asking a question, rather than only by quoting the words it was written with.
That is worth up to a second of the write's budget -- but only if a vector comes
back. Loading the model takes 9.9-11.0 s on this machine against 42 ms once it is
loaded, so a write that starts the load can never win: it waits out the whole
deadline and stores nothing anyway.

The guard was keyed to ``_available``, which an ``EmbeddingService`` sets to
``True`` in its constructor, before the worker subprocess exists. Measured on a
copy of a real store, the first twelve writes after a cold start took a median of
1,055 ms and eleven of them stored no vector. The same twelve against a warm
model: 75 ms, and every one of them kept its vector.

So there are two halves here, and both are tested below: the guard asks whether
the model is *warm*, not whether it is *configured*; and the daemon warms it at
startup, off the request path, the way it already warms the cross-encoder.
"""

from __future__ import annotations

import threading
import time

import pytest

from superlocalmemory.core.engine import _embedder_is_warm


class TestReadinessIsNotAvailability:
    """``_available`` is set before the worker exists. ``is_warm`` is not."""

    def test_a_service_that_has_not_served_a_request_is_not_warm(self) -> None:
        class NotYetStarted:
            _available = True          # set in __init__, before any worker
            is_warm = False            # no worker has answered anything

        assert _embedder_is_warm(NotYetStarted()) is False, (
            "an embedder that has never served a request is not warm, however "
            "available it reports itself to be"
        )

    def test_a_service_that_has_served_one_is_warm(self) -> None:
        class Serving:
            _available = True
            is_warm = True

        assert _embedder_is_warm(Serving()) is True

    def test_an_embedder_with_no_opinion_falls_back_to_availability(self) -> None:
        """OllamaEmbedder has no worker of its own to start -- it talks to a
        server that is already running -- so it offers no ``is_warm`` and
        availability is the best answer there is. Removing that fallback would
        silently switch inline embedding off for every Ollama user."""
        class Ollama:
            _available = True

        class OllamaUnchecked:
            _available = None          # lazy-checked, not yet asked

        assert _embedder_is_warm(Ollama()) is True
        assert _embedder_is_warm(OllamaUnchecked()) is False


class TestTheWriteDeclinesInsteadOfWaiting:
    """The behaviour, not the predicate: what a write actually costs."""

    def _engine_with(self, embedder):
        """A bare engine object carrying just what the guard reads.

        Building a real engine here would load a real model, which is the thing
        under test.
        """
        from superlocalmemory.core.engine import MemoryEngine

        engine = MemoryEngine.__new__(MemoryEngine)
        engine._embedder = embedder
        engine._store_fast_embed_pool = None
        engine._store_fast_embed_pool_lock = threading.Lock()
        engine._store_fast_embed_pool_closed = False
        return engine

    def test_a_cold_model_costs_the_write_nothing(self) -> None:
        """Reverting the fix submits the embed to the pool and waits out its full
        deadline before giving up. That wait can only happen on the pool, so the
        proof is that the pool was never created and the model never asked --
        not a stopwatch, which a loaded host can push past any bound."""
        asked = threading.Event()

        class ColdService:
            _available = True          # what the constructor sets
            is_warm = False            # what is actually true
            _config = type("C", (), {"is_cloud": False, "is_openai_compatible": False})()

            def embed(self, text):
                asked.set()
                time.sleep(10.0)       # a real cold start is 9.9-11.0 s
                return [0.1] * 768

        engine = self._engine_with(ColdService())

        emb, fmean, fvar = engine._warm_guard_embed("a memory worth keeping")

        assert emb is None, "a cold model cannot produce a vector in time"
        assert engine._store_fast_embed_pool is None and not asked.is_set(), (
            "the write handed a cold model its embed and waited on the deadline "
            "to be told what it could have known immediately; paying the "
            "1,000 ms deadline buys nothing"
        )

    def test_a_warm_model_is_still_used(self) -> None:
        """The control. If declining were the whole answer, no memory would ever
        get its vector on the write, and every one would be findable only by
        quoting itself until the materializer caught up."""
        class WarmService:
            _available = True
            is_warm = True
            _config = type("C", (), {"is_cloud": False, "is_openai_compatible": False})()

            def embed(self, text):
                return [0.1] * 768

            def compute_fisher_params(self, emb):
                return 0.5, 0.25

        engine = self._engine_with(WarmService())
        emb, fmean, fvar = engine._warm_guard_embed("a memory worth keeping")

        assert emb is not None and len(emb) == 768
        assert (fmean, fvar) == (0.5, 0.25)


class TestTheDaemonWarmsTheModelItself:
    """Otherwise the cold start is simply moved onto whoever recalls first."""

    def test_startup_warms_a_local_model(self) -> None:
        from superlocalmemory.server.unified_daemon import _start_embedder_warmup

        embedded: list[str] = []

        class LocalService:
            _config = type("C", (), {"is_cloud": False, "is_openai_compatible": False})()

            def embed(self, text):
                embedded.append(text)
                return [0.1] * 768

        engine = type("E", (), {"_embedder": LocalService()})()
        thread = _start_embedder_warmup(engine)

        assert thread is not None
        thread.join(timeout=10)
        assert embedded, "startup did not load the model"

    def test_it_does_not_spend_a_request_on_a_hosted_model(self) -> None:
        """A hosted embedder has no model to load. Warming it would spend a
        request, and money, on a sentence nobody asked about."""
        from superlocalmemory.server.unified_daemon import _start_embedder_warmup

        called: list[str] = []

        class HostedService:
            _config = type("C", (), {"is_cloud": True, "is_openai_compatible": False})()

            def embed(self, text):
                called.append(text)
                return [0.1] * 768

        engine = type("E", (), {"_embedder": HostedService()})()

        assert _start_embedder_warmup(engine) is None
        assert called == []

    def test_a_model_that_will_not_load_does_not_stop_the_daemon(self) -> None:
        """A daemon that cannot embed still stores memories and still answers
        keyword recall. Failing to warm is not failing to start."""
        from superlocalmemory.server.unified_daemon import _start_embedder_warmup

        class BrokenService:
            _config = type("C", (), {"is_cloud": False, "is_openai_compatible": False})()

            def embed(self, text):
                raise RuntimeError("no model on this machine")

        engine = type("E", (), {"_embedder": BrokenService()})()
        thread = _start_embedder_warmup(engine)

        assert thread is not None
        thread.join(timeout=10)
        assert not thread.is_alive()

    def test_an_engine_with_no_embedder_is_not_an_error(self) -> None:
        from superlocalmemory.server.unified_daemon import _start_embedder_warmup

        assert _start_embedder_warmup(type("E", (), {"_embedder": None})()) is None
