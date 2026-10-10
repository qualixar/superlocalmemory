# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recall by what a picture or a page shows.

One more channel next to the text ones. It runs only when images and documents
are on, ``media.db`` exists and the profile has at least one vector. Otherwise
it is not submitted at all and a recall is exactly what it was without it.

The question's vector comes from the already-warm media worker, asked once
before dispatch and for at most a third of a second. A cold worker is started in
the background and this recall goes on without pictures, saying so as
``warming``. Nothing here loads a model or spawns a process inside the call.

When the picture space is paired with the text model (``runtimes.space_plan``)
the question's vector is the text vector the recall already computed and the
media worker is not asked at all. If the stored pictures were built for another
space, the channel stays off.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Sequence

from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.runtimes.space_plan import SpacePlan, compatible

logger = logging.getLogger(__name__)

ACTIVE_TTL_S = 30.0
QUERY_CACHE_MAX = 256
MEDIA_SOURCE_TYPES = frozenset({"media", "document"})
_IN_CHUNK = 500
MISMATCH_LOG_S = 60.0


class MediaChannel:
    """``store_factory`` gives the media store or None; ``embedder_factory`` the
    warm query embedder or None (media off, environment not ready); ``db`` is
    the memory database that turns a memory into its facts."""

    def __init__(self, store_factory: Callable[[], Any], embedder_factory: Callable[[], Any],
                 db: Any = None, *, clock: Callable[[], float] = time.monotonic,
                 ttl_s: float = ACTIVE_TTL_S,
                 text_query_vector: Callable[[str], list[float] | None] | None = None,
                 plan_factory: Callable[[], SpacePlan | None] | None = None,
                 enabled_factory: Callable[[], bool] | None = None) -> None:
        self._store_factory = store_factory
        self._embedder_factory = embedder_factory
        self._db = db
        self._clock = clock
        self._ttl = ttl_s
        self._text_query_vector = text_query_vector
        self._plan_factory = plan_factory
        self._enabled_factory = enabled_factory
        self._mismatch_logged: dict[str, float] = {}
        self._active: dict[str, tuple[float, bool]] = {}
        self._queries: OrderedDict[tuple[str, str, str], list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def is_active(self, profile_id: str) -> bool:
        """Media on, a store, and one vector for this profile. Cached for 30 seconds."""
        now = self._clock()
        with self._lock:
            hit = self._active.get(profile_id)
            if hit is not None and now - hit[0] < self._ttl:
                return hit[1]
        active = self._probe(profile_id)
        with self._lock:
            self._active[profile_id] = (now, active)
        return active

    def _plan(self) -> SpacePlan | None:
        return self._plan_factory() if self._plan_factory is not None else None

    def min_score(self, default: float) -> float:
        """The picture evidence floor: the paired plan's own, else ``default``."""
        try:
            plan = self._plan()
        except Exception:  # noqa: BLE001 - a broken plan lookup keeps the configured floor
            return default
        floor = getattr(plan, "min_score", None)
        return default if floor is None else float(floor)

    def _on(self, plan: SpacePlan | None) -> bool:
        """Media on and, outside paired mode, the worker's environment ready."""
        if plan is not None and plan.query_from_text:
            return self._enabled_factory is None or bool(self._enabled_factory())
        return self._embedder_factory() is not None

    def _probe(self, profile_id: str) -> bool:
        try:
            plan = self._plan()
            if self._db is None or not self._on(plan):
                return False
            store = self._store_factory()
            if store is None or store.vector_count(profile_id) <= 0:
                return False
            if plan is not None and not compatible(plan, store.active_signature()):
                self._log_mismatch(profile_id)
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - a broken picture index never breaks recall
            logger.debug("picture channel is off for this recall (%s)", type(exc).__name__)
            return False

    def _log_mismatch(self, profile_id: str) -> None:
        now = self._clock()
        last = self._mismatch_logged.get(profile_id)
        if last is None or now - last >= MISMATCH_LOG_S:
            self._mismatch_logged[profile_id] = now
            logger.debug("picture channel is off: space_mismatch")

    def query_vector(self, query: str) -> list[float] | None:
        """The question as a vector, or None while the worker (or text embedder) is cold or busy."""
        plan = self._plan()
        if plan is not None and plan.query_from_text:
            return self._text_vector(plan, query)
        client = self._embedder_factory()
        if client is None:
            return None
        mode = plan.mode if plan is not None else ""
        key = (mode, str(getattr(client, "model_id", "")), query)
        with self._lock:
            cached = self._queries.get(key)
            if cached is not None:
                self._queries.move_to_end(key)
                return cached
        vector = client.embed_query(query, wait_s=0.3)
        if vector is None:
            return None
        with self._lock:
            self._queries[key] = vector
            while len(self._queries) > QUERY_CACHE_MAX:
                self._queries.popitem(last=False)
        return vector

    def _text_vector(self, plan: SpacePlan, query: str) -> list[float] | None:
        if self._text_query_vector is None:
            return None
        key = (plan.mode, plan.text_model, query)
        with self._lock:
            cached = self._queries.get(key)
            if cached is not None:
                self._queries.move_to_end(key)
                return cached
        vector = self._text_query_vector(query)
        if vector is None:
            return None
        with self._lock:
            self._queries[key] = list(vector)
            while len(self._queries) > QUERY_CACHE_MAX:
                self._queries.popitem(last=False)
        return list(vector)

    def prepare(self, query: str, profile_id: str) -> tuple[list[float] | None, str | None]:
        """``(vector, None)`` to run, ``(None, 'warming')`` when only the worker is cold,
        ``(None, None)`` when the channel does not apply."""
        if not self.is_active(profile_id):
            return None, None
        try:
            vector = self.query_vector(query)
        except Exception as exc:  # noqa: BLE001
            logger.debug("picture query vector failed (%s)", type(exc).__name__)
            vector = None
        return (vector, None) if vector is not None else (None, chstat.WARMING)

    def search(self, vector: Sequence[float], profile_id: str, top_k: int) -> list[tuple[str, float]]:
        """``[(fact_id, score)]`` best first; score is ``1 - cosine distance`` held to [0, 1]."""
        store = self._store_factory()
        if store is None or self._db is None:
            return []
        scored = {mid: min(1.0, max(0.0, 1.0 - dist))
                  for mid, dist in store.knn(vector, profile_id, top_k)}
        by_memory: dict[str, float] = {}
        for media_id, memory_ids in store.memory_ids_of(list(scored)).items():
            for memory_id in memory_ids:
                by_memory[memory_id] = max(by_memory.get(memory_id, 0.0), scored[media_id])
        best: dict[str, float] = {}
        for fact_id, memory_id in self._facts_of(list(by_memory), profile_id):
            best[fact_id] = max(best.get(fact_id, 0.0), by_memory[memory_id])
        return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))

    def _facts_of(self, memory_ids: list[str], profile_id: str) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for i in range(0, len(memory_ids), _IN_CHUNK):
            part = memory_ids[i:i + _IN_CHUNK]
            found = self._db.execute(
                "SELECT fact_id, memory_id FROM atomic_facts WHERE profile_id = ? AND memory_id IN ("
                + ",".join("?" * len(part)) + ")", (profile_id, *part))
            rows += [(r["fact_id"], r["memory_id"]) for r in found]
        return rows


# -- wiring ------------------------------------------------------------------------

def _data_root(db: Any) -> Path | None:
    path = getattr(db, "db_path", None)
    return Path(path).parent if isinstance(path, (str, Path)) else None


def for_engine(db: Any, *, text_query_vector: Callable[[str], list[float] | None] | None = None) -> MediaChannel:
    """The channel for a retrieval engine's memory database. Builds nothing heavy.

    ``text_query_vector`` is the engine's own text query embedding, used when pictures
    share the text space.
    """
    root = _data_root(db)
    holder: dict[str, Any] = {}
    lock = threading.Lock()

    def store() -> Any:
        with lock:
            if holder.get("store") is None and root is not None:
                from superlocalmemory.media import open_media_store

                holder["store"] = open_media_store(data_root=root)
            return holder.get("store")

    def embedder() -> Any:
        if root is None:
            return None
        from superlocalmemory.runtimes.worker_client import media_embedder

        return media_embedder(data_root=root)

    plan_cache: list[Any] = [0.0, None]

    def plan() -> SpacePlan | None:
        if root is None:
            return None
        now = time.monotonic()
        with lock:
            if plan_cache[1] is None or now - plan_cache[0] >= ACTIVE_TTL_S:
                from superlocalmemory.runtimes.space_plan import current_space_plan

                plan_cache[:] = [now, current_space_plan(root)]
            return plan_cache[1]

    def enabled() -> bool:
        from superlocalmemory.runtimes.features import media_enabled

        return root is not None and media_enabled(root)

    return MediaChannel(store, embedder, db if root is not None else None,
                        text_query_vector=text_query_vector, plan_factory=plan, enabled_factory=enabled)


_PAGE_STORES: dict[Path, Any] = {}
_PAGE_STORES_LOCK = threading.Lock()


def _page_store(root: Path) -> Any:
    with _PAGE_STORES_LOCK:
        if _PAGE_STORES.get(root) is None:
            from superlocalmemory.media import open_media_store

            _PAGE_STORES[root] = open_media_store(data_root=root)
        return _PAGE_STORES[root]


def _with_page_media_ids(root: Path, found: dict[str, dict]) -> dict[str, dict]:
    """Give each page source that names a document and a page (no media id yet) its page's
    media id, from one query on media.db. A page with no row keeps no media id."""
    todo = {mid: s for mid, s in found.items()
            if s.get("type") == "document" and not s.get("media_id")
            and s.get("document_id") and isinstance(s.get("page"), int)}
    if not todo:
        return found
    try:
        store = _page_store(root)
        ids = store.page_media_ids([(s["document_id"], s["page"]) for s in todo.values()]) if store else {}
    except Exception as exc:  # noqa: BLE001 - presentation only
        logger.debug("page thumbnails unavailable (%s)", type(exc).__name__)
        return found
    for mid, s in todo.items():
        media_id = ids.get((s["document_id"], s["page"]))
        if media_id:
            found[mid] = {**s, "media_id": media_id}
    return found


def memory_sources(db: Any, memory_ids: Sequence[str]) -> dict[str, dict]:
    """``{memory_id: _slm_source}`` for the images and pages among these memories.

    One batched query. Returns ``{}`` without any query when images and
    documents are off or there is no media.db: ordinary recalls cost nothing.
    """
    root = _data_root(db)
    ids = list(dict.fromkeys(i for i in memory_ids if i))
    if root is None or not ids:
        return {}
    from superlocalmemory.media import media_db_exists
    from superlocalmemory.runtimes.features import media_enabled

    if not media_db_exists(root) or not media_enabled(root):
        return {}
    try:
        rows = db.execute(
            "SELECT memory_id, metadata_json FROM memories WHERE memory_id IN ("
            + ",".join("?" * len(ids)) + ")", tuple(ids))
        out: dict[str, dict] = {}
        for row in rows:
            source = (json.loads(row["metadata_json"] or "{}") or {}).get("_slm_source")
            if isinstance(source, dict) and source.get("type") in MEDIA_SOURCE_TYPES:
                out[row["memory_id"]] = source
        return _with_page_media_ids(root, out)
    except Exception as exc:  # noqa: BLE001 - presentation only
        logger.debug("media sources unavailable (%s)", type(exc).__name__)
        return {}


__all__ = ["MEDIA_SOURCE_TYPES", "MediaChannel", "for_engine", "memory_sources"]
