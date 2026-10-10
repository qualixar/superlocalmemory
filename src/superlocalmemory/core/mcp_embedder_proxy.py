# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""V3.5.9 — McpEmbedderProxy: lightweight embedder for the MCP (LIGHT) process.

Problem: MemoryEngine(Capabilities.LIGHT) skips _init_heavy_layer(), leaving
_embedder=None permanently. Any memory stored via MCP tools has NULL embeddings
→ semantic search silently broken; health() lies, reporting unavailable.

Solution: a thin proxy that delegates embed_batch() to the running daemon via
its POST /api/embed endpoint over localhost HTTP. The daemon runs a FULL engine
with one real ONNX/Ollama worker. No second ONNX process is spawned.

Usage (engine.py LIGHT branch):
    proxy = McpEmbedderProxy()
    if proxy.is_available():
        engine._embedder = proxy

4.1.20 (audit C-8): the proxy talks only to the daemon that OWNS this
process's data root -- found through that root's daemon descriptor, its
identity checked against ``/health``, and every request carrying the
descriptor's capability and instance headers (``cli.daemon.daemon_request``).
It used to post to whatever answered on the configured port (8765), so an MCP
process for one data root embedded its memories with another installation's
daemon -- possibly a different model and dimension -- and stored the result.
No owned daemon means no proxy: keyword recall still works, and the facts are
embedded later by their own daemon.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 5.0  # seconds — fast enough for inline store() calls


def _owned_daemon_request(method: str, path: str, body: dict | None, timeout: float):
    """One request to this data root's own daemon, or None if there is none.

    Raises nothing: a refusal, a foreign occupant and an outage all mean the
    same thing to an embedder -- no vector from here.
    """
    try:
        from superlocalmemory.cli.daemon import daemon_request

        return daemon_request(method, path, body, timeout_seconds=timeout)
    except Exception as exc:  # noqa: BLE001 - DaemonRefused and transport errors alike
        logger.debug("McpEmbedderProxy request %s %s failed: %s",
                     method, path, type(exc).__name__)
        return None


class McpEmbedderProxy:
    """Proxy embedder: MCP process → its own daemon's /api/v3/embed."""

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout
        self._available: bool | None = None  # cached after first is_available() call

    def is_available(self) -> bool:
        """Is this data root's own daemon up and serving embeds? Cached on success."""
        if self._available is True:
            return True
        payload = _owned_daemon_request(
            "GET", "/api/v3/embed/ping", None, min(2.0, self._timeout),
        )
        self._available = isinstance(payload, dict) and payload.get("ok") is True
        return bool(self._available)

    # -- Embedder interface (matches embeddings.py / ollama_embedder.py) ------

    def embed(self, text: str) -> list[float] | None:
        """Embed a single text via daemon. Returns None on any error."""
        results = self.embed_batch([text])
        return results[0] if results else None

    def embed_query(self, text: str) -> list[float] | None:
        """Embed a question (the model's query prompt) via the owned daemon."""
        results = self.embed_batch([text], prompt="query")
        return results[0] if results else None

    def embed_batch(
        self, texts: list[str], prompt: str | None = None,
    ) -> list[list[float] | None]:
        """Embed a batch of texts via the owned daemon's /api/v3/embed.

        ``prompt`` is sent only when given ("query"); without it the request is
        the one every daemon has always understood.
        """
        if not texts:
            return []
        body: dict = {"texts": list(texts)}
        if prompt is not None:
            body["prompt"] = prompt
        data = _owned_daemon_request("POST", "/api/v3/embed", body, self._timeout)
        embeddings = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(embeddings, list):
            return [None] * len(texts)
        embeddings = list(embeddings[: len(texts)])
        # Pad with None if daemon returned fewer results than requested
        while len(embeddings) < len(texts):
            embeddings.append(None)
        return embeddings

    def compute_fisher_params(
        self, embedding: list[float]
    ) -> tuple[list[float] | None, list[float] | None]:
        """Fisher-Rao params stay in daemon's FULL engine; proxy returns None.

        The MCP LIGHT process stores facts with embedding=<vector> but
        fisher_mean=None, fisher_variance=None. The daemon's consolidation
        pass fills these in asynchronously — same behaviour as the
        write-through remember path.
        """
        return None, None
