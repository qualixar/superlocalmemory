# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Embed a recall query without letting a loading model hold the whole recall.

WHY THIS EXISTS
---------------
On a fresh daemon the local embedding model takes tens of seconds to load (a
measured 32.7 s on a fresh install). The query embedding used to be a plain
blocking call made BEFORE any channel was dispatched, so every recall in that
window sat behind the model load — keyword search included, though it needs no
embedding at all. The daemon's last-resort budget then fired and answered with
a degraded fallback, and a memory saved seconds earlier with a "queryable"
receipt came back as "No confident match".

ONLY while the embedder has never been ready (its ``is_warm`` is ``False`` and
it has not answered a single request since it was created), the embedding runs
on its own worker and the recall waits for it no longer than the per-channel hang guard
(``CHANNEL_HANG_GUARD_SECONDS``). If the vector is not back by then, the
channels that need it are reported as ``warming`` and the recall is marked
incomplete, while the channels that need no vector run and can find the memory.

ONCE THE EMBEDDER HAS BEEN READY — or when it does not say (no ``is_warm``) —
the recall embeds inline and waits exactly as 4.1.19 did, with no bound. That
includes the reload after a 30-minute idle unload or a memory-pressure kill
(``has_loaded_once``): a slow embed on a model that loaded before is waited
for; this is not a speed cap.

WHAT IT COSTS IN QUALITY
------------------------
Nothing on a warm daemon: that path is unchanged. In the cold window the
alternative was not "a recall with semantic search", it was a recall with NO
channels at all (the daemon budget fallback), so this strictly adds results.
The abandoned embed is not cancelled: it finishes on its worker and lands in
the query cache, so the same question asked again gets every channel as soon
as the model is up.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import logging
import threading
from typing import Any, Callable

from superlocalmemory.retrieval import channel_status as chstat

logger = logging.getLogger(__name__)

__all__ = ["QueryEmbedder", "embed_as_query"]


def embed_as_query(embedder: Any, text: str) -> list[float] | None:
    """Embed a question: with the embedder's question prompt when it has one, else as ``embed`` does.

    Models that take different prompts for questions and memories (the managed text
    model) define ``embed_query``; every other embedder is called exactly as before.
    Looked up statically so a duck-typed stand-in does not pretend to have it.
    """
    if inspect.getattr_static(embedder, "embed_query", None) is not None:
        return embedder.embed_query(text)
    return embedder.embed(text)


class QueryEmbedder:
    """Bounded, single-flight, cached query embedding for one retrieval engine.

    ``embed`` returns ``(vector, status)``. ``status`` is ``None`` when the
    embedder answered (the vector may still be ``None`` if the embedder itself
    returned nothing — that stays the caller's ``no_embedding`` case), and
    ``WARMING`` when a not-yet-ready model made the recall stop waiting.
    """

    def __init__(
        self, embedder: Callable[[], Any], *, cache_max_size: int = 512,
    ) -> None:
        # A provider, not the object: the owning engine's embedder can be
        # swapped after construction and every call must see the current one.
        self._provider = embedder
        self._cache: dict[str, list[float]] = {}
        self._cache_max_size = cache_max_size
        self._lock = threading.Lock()
        self._inflight: dict[str, concurrent.futures.Future] = {}
        # Created on first use: an engine with no embedder (Mode A without
        # vectors) or one only ever used from background work owns no threads.
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._closed = False

    @property
    def cache(self) -> dict[str, list[float]]:
        return self._cache

    def _remember(self, query: str, vector: list[float] | None) -> None:
        if vector is None:
            return
        with self._lock:
            if query not in self._cache and len(self._cache) >= self._cache_max_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[query] = vector

    def _compute(self, query: str) -> list[float] | None:
        vector = embed_as_query(self._provider(), query)
        self._remember(query, vector)
        return vector

    def _future_for(self, query: str) -> concurrent.futures.Future:
        with self._lock:
            fut = self._inflight.get(query)
            if fut is not None:
                return fut
            if self._executor is None:
                if self._closed:
                    raise RuntimeError("query embedder is closed")
                # Two workers: one may be parked behind a cold model load
                # while a second question still gets its turn once it is up.
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="slm-query-embed",
                )
            fut = self._executor.submit(self._compute, query)
            self._inflight[query] = fut

        def _done(_f, q=query) -> None:
            with self._lock:
                if self._inflight.get(q) is _f:
                    del self._inflight[q]

        # Outside the lock: a future that is already done runs the callback
        # inline, and the callback takes the same (non-reentrant) lock.
        fut.add_done_callback(_done)
        return fut

    def _not_ready(self) -> bool:
        """True only for a model that has NEVER loaded and says it is not ready.

        Unknown is ready. An embedder that has answered before
        (``has_loaded_once``) is ready too, even while ``is_warm`` is False
        because the idle timer or memory pressure unloaded its worker: a
        model that loaded once is reloading, and that recall waits for it
        exactly as 4.1.19 did instead of losing its semantic channels.
        """
        embedder = self._provider()
        if getattr(embedder, "has_loaded_once", None) is True:
            return False
        return getattr(embedder, "is_warm", None) is False

    def embed(self, query: str, wait_seconds: float) -> tuple[list[float] | None, str | None]:
        """See ``_embed``. Afterwards the recall on this thread will not ask the
        embedder again (the vector is cached for the question), so background
        embeds may use it while the recall runs on (``recall_gate``)."""
        try:
            return self._embed(query, wait_seconds)
        finally:
            from superlocalmemory.core.recall_gate import mark_query_embedded
            mark_query_embedded()

    def _embed(self, query: str, wait_seconds: float) -> tuple[list[float] | None, str | None]:
        """Embed ``query``; ``wait_seconds`` bounds the wait only while the
        embedder is not ready yet. A ready embedder is waited for unbounded.

        Raises whatever the embedder raised, exactly as the direct call did.
        """
        if self._provider() is None:
            return None, None
        cached = self._cache.get(query)
        if cached is not None:
            return cached, None
        if getattr(self._provider(), "needs_service", False) is True:
            return None, chstat.NEEDS_SERVICE  # no point waiting: nothing is loading
        from superlocalmemory.core.recall_gate import is_background_work

        if is_background_work() or not self._not_ready():
            # 4.1.19 behaviour, inline on the caller's thread (background
            # callers keep their thread-local priority marker). An embed for
            # this question started while the model was loading is joined,
            # not repeated.
            with self._lock:
                pending = self._inflight.get(query)
            if pending is not None:
                return pending.result(), None
            return self._compute(query), None
        fut = self._future_for(query)
        try:
            return fut.result(timeout=max(0.0, wait_seconds)), None
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Embedding model not ready within %.1fs; this recall runs "
                "without the channels that need it and is marked incomplete",
                wait_seconds,
            )
            return None, chstat.WARMING

    def worker_threads(self) -> list[threading.Thread]:
        """This embedder's pool threads (none until a cold embed created it)."""
        from superlocalmemory.core.thread_join import executor_threads

        with self._lock:
            return executor_threads(self._executor)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
