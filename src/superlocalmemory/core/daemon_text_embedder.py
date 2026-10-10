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

import inspect
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


def embedder_identity(embedder: Any) -> tuple[str, int]:
    """``(model name, vector size)`` of an embedder."""
    config = getattr(embedder, "_config", None)
    model = getattr(embedder, "model_name", None) or getattr(config, "model_name", "")
    dimension = getattr(embedder, "dimension", None) or getattr(config, "dimension", 0)
    return str(model), int(dimension or 0)


def describe_embedder(embedder: Any) -> dict[str, Any]:
    """What the daemon tells other processes about its embedder (the ping's ``embedder`` field)."""
    if embedder is None:
        return {"available": False, "warm": False, "model": "", "dimension": 0}
    # The cached answer only: some embedders probe a server when asked ``is_available``,
    # and this runs on the daemon's event loop.
    available = getattr(embedder, "_available", None) is True
    model, dimension = embedder_identity(embedder)
    warm = getattr(embedder, "is_warm", None)
    warm = warm() if callable(warm) else warm
    return {"available": bool(available), "warm": bool(warm) if warm is not None else bool(available),
            "model": model, "dimension": dimension}


def embed_with_prompt(embedder: Any, texts: list[str], prompt: str) -> list[Any]:
    """Vectors for ``texts`` as documents or as questions; embedders without a question prompt embed alike."""
    if prompt != "query" or inspect.getattr_static(embedder, "embed_query", None) is None:
        return embedder.embed_batch(texts)
    return [embedder.embed_query(text) for text in texts]


class DaemonTextEmbedder:
    def __init__(self, config: Any, *, timeout_s: float = 30.0, ping_ttl_s: float = 5.0) -> None:
        self._config = config
        self._timeout_s, self._ping_ttl_s = timeout_s, ping_ttl_s
        self._lock = threading.Lock()
        self._ping_at = float("-inf")
        self._ok, self._warm, self._loaded_once, self._closed = False, False, False, False
        self._unreachable = False  # the last ping got no answer at all (daemon down, not loading)

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
        self._unreachable = not isinstance(data, dict)
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

    @property
    def needs_service(self) -> bool:
        """The daemon did not answer the last look (it is not running), as opposed to loading."""
        if self._closed:
            return False
        self._refresh()
        return self._unreachable

    # -- the embedder interface ---------------------------------------------------
    def _ask(self, texts: list[str], prompt: str) -> list[list[float] | None]:
        if not self.is_available:  # the daemon is down, or serves another model
            return [None] * len(texts)
        body = {"texts": texts, "prompt": prompt, "model": self.model_name, "dimension": self.dimension}
        data = _owned_daemon_request("POST", _EMBED_PATH, body, self._timeout_s)
        rows = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(rows, list):  # no answer, or "model_mismatch": look again next time
            self._ok = self._warm = False
            self._ping_at = float("-inf")
            return [None] * len(texts)
        rows = list(rows[:len(texts)]) + [None] * max(0, len(texts) - len(rows))
        from superlocalmemory.core.embeddings import DimensionMismatchError

        for vec in rows:
            if vec is not None and len(vec) != self.dimension:
                raise DimensionMismatchError(f"Embedding dimension {len(vec)} != expected {self.dimension}")
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


__all__ = ["DaemonTextEmbedder", "NEEDS_SERVICE", "describe_embedder", "embed_with_prompt", "embedder_identity"]
