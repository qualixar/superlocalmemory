# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Text vectors from the daemon, for every process that is not the daemon.

The managed model runs once per computer, in the daemon. A command, the MCP server
or the recall worker that needs text vectors asks the daemon of its own data folder
(``/api/v3/embed``) and never starts a worker. If that daemon is not running, or its
embedder is not this space's model, the embedder is unavailable and the engine runs
keyword-only: vectors from another model would be in the wrong space.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from superlocalmemory.core.mcp_embedder_proxy import _owned_daemon_request

logger = logging.getLogger(__name__)

NEEDS_SERVICE = ("Text vectors need the SLM service running; until it is, recall uses keywords only. "
                 "Start it with: slm serve start")
_PING_PATH = "/api/v3/embed/ping"
_EMBED_PATH = "/api/v3/embed"


def describe_embedder(embedder: Any) -> dict[str, Any]:
    """What the daemon tells other processes about its embedder (the ping's ``embedder`` field)."""
    if embedder is None:
        return {"available": False, "warm": False, "model": "", "dimension": 0}
    available = getattr(embedder, "is_available", False)
    available = available() if callable(available) else available
    config = getattr(embedder, "_config", None)
    model = getattr(embedder, "model_name", None) or getattr(config, "model_name", "")
    dimension = getattr(embedder, "dimension", None) or getattr(config, "dimension", 0)
    warm = getattr(embedder, "is_warm", None)
    warm = warm() if callable(warm) else warm
    return {"available": bool(available), "warm": bool(warm) if warm is not None else bool(available),
            "model": str(model), "dimension": int(dimension or 0)}


def embed_with_prompt(embedder: Any, texts: list[str], prompt: str) -> list[Any]:
    """Vectors for ``texts`` as documents or as questions; embedders without a question prompt embed alike."""
    if prompt != "query":
        return embedder.embed_batch(texts)
    from superlocalmemory.retrieval.query_embedding import embed_as_query

    return [embed_as_query(embedder, text) for text in texts]


class DaemonTextEmbedder:
    def __init__(self, config: Any, *, timeout_s: float = 30.0, ping_ttl_s: float = 5.0) -> None:
        self._config = config
        self._timeout_s, self._ping_ttl_s = timeout_s, ping_ttl_s
        self._lock = threading.Lock()
        self._ping_at = float("-inf")
        self._ok, self._warm, self._loaded_once, self._closed = False, False, False, False

    @property
    def dimension(self) -> int:
        return int(self._config.dimension)

    @property
    def model_name(self) -> str:
        return str(self._config.model_name)

    @property
    def is_closed(self) -> bool:
        return self._closed

    # -- the daemon's answer, cached ----------------------------------------------
    def _refresh(self, *, force: bool = False) -> None:
        with self._lock:
            if not force and time.monotonic() - self._ping_at < self._ping_ttl_s:
                return
            self._ping_at = time.monotonic()
        data = _owned_daemon_request("GET", _PING_PATH, None, 2.0)
        info = data.get("embedder") if isinstance(data, dict) and data.get("ok") is True else None
        ok = (isinstance(info, dict) and info.get("available") is True
              and info.get("model") == self.model_name and info.get("dimension") == self.dimension)
        self._ok = bool(ok)
        self._warm = bool(ok and info.get("warm"))
        if self._warm:
            self._loaded_once = True

    @property
    def is_available(self) -> bool:
        """The owning daemon is up and its embedder is this space's model (checked every few seconds)."""
        if self._closed:
            return False
        self._refresh()
        return self._ok

    @property
    def _available(self) -> bool:
        """The last answer, without asking: the write path reads this on every save."""
        return self._ok and not self._closed

    @property
    def is_warm(self) -> bool:
        if self._closed:
            return False
        self._refresh()
        return self._warm

    @property
    def has_loaded_once(self) -> bool:
        return self._loaded_once

    # -- the embedder interface ---------------------------------------------------
    def _ask(self, texts: list[str], prompt: str) -> list[list[float] | None]:
        if self._closed:
            return [None] * len(texts)
        data = _owned_daemon_request("POST", _EMBED_PATH, {"texts": texts, "prompt": prompt}, self._timeout_s)
        rows = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            self._ok = self._warm = False
            return [None] * len(texts)
        rows = list(rows[:len(texts)]) + [None] * max(0, len(texts) - len(rows))
        from superlocalmemory.core.embeddings import DimensionMismatchError

        for vec in rows:
            if vec is not None and len(vec) != self.dimension:
                raise DimensionMismatchError(f"Embedding dimension {len(vec)} != expected {self.dimension}")
        self._ok = self._warm = self._loaded_once = True
        return rows

    def embed(self, text: str) -> list[float] | None:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        return self._ask([text], "document")[0]

    def embed_query(self, text: str) -> list[float] | None:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        return self._ask([text], "query")[0]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            raise ValueError("Cannot embed empty batch")
        return self._ask(list(texts), "document")

    def compute_fisher_params(self, embedding: list[float]) -> tuple[list[float], list[float]]:
        from superlocalmemory.core.embeddings import EmbeddingService

        return EmbeddingService.compute_fisher_params(self, embedding)  # type: ignore[arg-type]

    def unload(self, timeout: float = 1.0) -> bool:
        return True

    def shutdown(self, timeout: float = 1.0) -> None:
        self._closed = True


__all__ = ["DaemonTextEmbedder", "NEEDS_SERVICE", "describe_embedder", "embed_with_prompt"]
