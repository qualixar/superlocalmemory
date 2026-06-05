# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""McpEmbedderProxy — lightweight embedder for LIGHT engine in MCP server.

Delegates embed_batch() to the daemon via HTTP instead of spawning a second
ONNX worker. Keeps the MCP process at ~60 MB while giving the LIGHT engine
a fully-functional semantic channel.

Used by MemoryEngine._try_init_proxy() when Capabilities.LIGHT is requested.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class McpEmbedderProxy:
    """Drop-in EmbeddingService replacement that proxies via daemon HTTP.

    The daemon already holds a warm ONNX worker; this class asks it for
    vectors over a local HTTP call instead of creating a second worker.
    The MCP process memory footprint stays at ~60 MB.

    The ``_is_proxy = True`` class attribute allows ``health()`` in
    ``tools_v3.py`` to report ``"proxy_via_daemon"`` instead of ``"ok"``,
    so operators can distinguish a proxied embedder from a local one.
    """

    _is_proxy = True

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        dimension: int = 768,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._dimension = dimension

    @property
    def is_available(self) -> bool:
        """Return True if the daemon is reachable on /api/stats."""
        try:
            import httpx
            r = httpx.get(self._base_url + "/api/stats", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float] | None:
        result = self.embed_batch([text])
        return result[0] if result else None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a batch of texts via POST /api/embed on the daemon."""
        try:
            import httpx
            r = httpx.post(
                self._base_url + "/api/embed",
                json={"texts": texts},
                timeout=30.0,
            )
            r.raise_for_status()
            return r.json()["vectors"]
        except Exception as exc:
            logger.warning("McpEmbedderProxy.embed_batch failed: %s", exc)
            return [None] * len(texts)

    def compute_fisher_params(
        self, embedding: list[float],
    ) -> tuple[list[float], list[float]]:
        """Fisher-Rao params are computed by the daemon; return empty here."""
        return [], []

    def unload(self) -> None:
        pass
