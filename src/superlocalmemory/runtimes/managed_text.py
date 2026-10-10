# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Text vectors from the managed model environment's worker (the daemon's embedder).

Looks like ``core.embeddings.EmbeddingService`` to the engine, but the model runs in
the one shared worker of ``worker_client``: the text-only loadout while pictures are
off, the full loadout (the same process the picture channel uses) while they are on,
so a second copy of the model is never loaded. Only the daemon builds this class;
other processes use ``core.daemon_text_embedder``.

A worker that is not loaded yet reports ``is_warm`` False, so recall takes its
warming path and a save is left to the background materializer.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable

from superlocalmemory.runtimes import media_models, worker_client
from superlocalmemory.runtimes.worker_client import MediaWorkerError

logger = logging.getLogger(__name__)

_STATUS_TTL_S = 5.0
_OPEN: "weakref.WeakSet[ManagedTextEmbedder]" = weakref.WeakSet()
_OPEN_LOCK = threading.Lock()


class ManagedTextEmbedder:
    def __init__(self, config: Any, *, data_root: str | Path | None = None, env: Any = None,
                 client_supplier: Callable[[], Any] | None = None) -> None:
        from superlocalmemory.infra.data_root import canonical_data_root

        self._config = config
        self._root = Path(data_root) if data_root is not None else canonical_data_root()
        if env is None and client_supplier is None:
            from superlocalmemory.runtimes.media_env import media_env

            env = media_env(root=self._root / "runtimes" / "media")
        self._env, self._supplier = env, client_supplier
        profile = media_models.profile_for(config.model_name)
        self._revision = profile.revision if profile is not None else ""
        self._last: Any = None
        self._loaded_once = False
        self._closed = False
        self._status_at, self._status_ok = 0.0, False
        with _OPEN_LOCK:
            _OPEN.add(self)

    # -- what the engine reads --------------------------------------------------
    @property
    def dimension(self) -> int:
        return int(self._config.dimension)

    @property
    def model_name(self) -> str:
        return str(self._config.model_name)

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def is_available(self) -> bool:
        """The managed environment is installed and ready (checked at most every few seconds)."""
        if self._closed:
            return False
        if self._supplier is not None:
            return True
        now = time.monotonic()
        if now - self._status_at > _STATUS_TTL_S:
            try:
                self._status_ok = self._env.status().state == "ready"
            except Exception:  # noqa: BLE001 - an unreadable state file means not ready
                self._status_ok = False
            self._status_at = now
        return self._status_ok

    @property
    def _available(self) -> bool:
        return self.is_available

    @property
    def is_warm(self) -> bool:
        """The worker is loaded now (this embedder's, or the picture channel's: the same process)."""
        if self._closed:
            return False
        if self._supplier is not None:
            return bool(self._supplier().is_warm())
        return self._warm_client() is not None

    @property
    def has_loaded_once(self) -> bool:
        """True once a worker has answered; survives an idle stop, as for the other embedders."""
        if not self._loaded_once and self.is_warm:
            self._loaded_once = True
        return self._loaded_once

    def _warm_client(self) -> Any | None:
        root = str(self._env.root)
        for client in worker_client.live_clients():
            if (client.model_id == self.model_name and client.revision == self._revision
                    and client.role in ("", "text") and str(client.root) == root and client.is_warm()):
                return client
        return None

    # -- the worker ---------------------------------------------------------------
    def _client(self) -> Any | None:
        if self._supplier is not None:
            return self._supplier()
        client = worker_client.text_embedder(env=self._env, data_root=self._root,
                                             model_id=self.model_name, revision=self._revision)
        if client is None:
            return None
        previous, self._last = self._last, client
        if previous is not None and previous is not client:
            previous.stop()  # pictures were switched: never keep the old loadout beside the new one
        return client

    def _embed(self, texts: list[str], prompt: str) -> list[list[float] | None]:
        client = None if self._closed else self._client()
        if client is None:
            return [None] * len(texts)
        try:
            vectors = client.embed_texts(texts, prompt=prompt)
        except MediaWorkerError as exc:
            logger.info("managed text embedder: %s", exc)
            return [None] * len(texts)
        self._loaded_once = True
        self._check_width(vectors)
        return list(vectors)

    def _check_width(self, vectors: list[list[float]]) -> None:
        from superlocalmemory.core.embeddings import DimensionMismatchError

        for vec in vectors:
            if len(vec) != self.dimension:
                raise DimensionMismatchError(f"Embedding dimension {len(vec)} != expected {self.dimension}")

    # -- the embedder interface -----------------------------------------------------
    def embed(self, text: str) -> list[float] | None:
        """A memory's vector (the model's Document prompt)."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        from superlocalmemory.core.recall_gate import wait_for_embedder_idle

        wait_for_embedder_idle()
        return self._embed([text], "Document")[0]

    def embed_query(self, text: str) -> list[float] | None:
        """A question's vector (the model's SearchQuery prompt)."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        from superlocalmemory.core.recall_gate import wait_for_embedder_idle

        wait_for_embedder_idle()
        return self._embed([text], "SearchQuery")[0]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            raise ValueError("Cannot embed empty batch")
        from superlocalmemory.core.recall_gate import is_background_work

        if is_background_work():  # one text at a time so a recall that arrives meanwhile goes first
            return [self.embed(text) for text in texts]
        return self._embed(list(texts), "Document")

    def compute_fisher_params(self, embedding: list[float]) -> tuple[list[float], list[float]]:
        from superlocalmemory.core.embeddings import EmbeddingService

        return EmbeddingService.compute_fisher_params(self, embedding)  # type: ignore[arg-type]

    # -- stopping -----------------------------------------------------------------
    def _stop_worker_if_alone(self) -> bool:
        """Stop the worker unless pictures use it or another open embedder shares it."""
        if self._supplier is not None:
            client = self._supplier()
            client.stop()
            return True
        if worker_client.text_loadout_role(self._root) == "":
            return False
        with _OPEN_LOCK:
            shared = any(o is not self and not o._closed and o._shares(self) for o in list(_OPEN))
        client = self._last
        if shared or client is None:
            return False
        client.stop()
        return True

    def _shares(self, other: "ManagedTextEmbedder") -> bool:
        return (self.model_name == other.model_name and self._supplier is None
                and str(self._env.root) == str(other._env.root))

    def unload(self, timeout: float = 1.0) -> bool:
        return self._stop_worker_if_alone()

    def shutdown(self, timeout: float = 1.0) -> None:
        self._closed = True
        self._stop_worker_if_alone()


__all__ = ["ManagedTextEmbedder"]
