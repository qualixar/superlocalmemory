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
    proxy = McpEmbedderProxy(port=8765)
    if proxy.is_available():
        engine._embedder = proxy
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 5.0  # seconds — fast enough for inline store() calls


class McpEmbedderProxy:
    """Proxy embedder: MCP process → daemon's /api/embed over localhost HTTP."""

    def __init__(self, port: int = 8765, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._base_url = f"http://127.0.0.1:{port}"
        self._timeout = timeout
        self._available: bool | None = None  # cached after first is_available() call

    def is_available(self) -> bool:
        """Ping daemon's embed endpoint. Cached after first success."""
        if self._available is True:
            return True
        try:
            import httpx
            resp = httpx.get(f"{self._base_url}/api/v3/embed/ping", timeout=2.0)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return bool(self._available)

    # -- Embedder interface (matches embeddings.py / ollama_embedder.py) ------

    def embed(self, text: str) -> list[float] | None:
        """Embed a single text via daemon. Returns None on any error."""
        results = self.embed_batch([text])
        return results[0] if results else None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a batch of texts via daemon's /api/embed endpoint."""
        if not texts:
            return []
        try:
            import httpx
            resp = httpx.post(
                f"{self._base_url}/api/v3/embed",
                json={"texts": texts},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            # Pad with None if daemon returned fewer results than requested
            while len(embeddings) < len(texts):
                embeddings.append(None)
            return embeddings
        except Exception as exc:
            logger.debug("McpEmbedderProxy.embed_batch failed: %s", exc)
            return [None] * len(texts)

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
