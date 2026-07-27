# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Ollama Embedding Provider — lightweight HTTP-based embeddings.

Uses Ollama's /api/embed endpoint for fast local embeddings without
loading PyTorch or sentence-transformers into the process.

Typical latency: <1 second (vs 30s cold start for sentence-transformers).
Memory: ~0 MB in the SLM process (Ollama manages its own memory).

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Fisher variance constants (must match EmbeddingService)
_FISHER_VAR_MIN = 0.05
_FISHER_VAR_MAX = 2.0
_FISHER_VAR_RANGE = _FISHER_VAR_MAX - _FISHER_VAR_MIN

# Ollama connect/response timeouts
_CONNECT_TIMEOUT = 5.0
_RESPONSE_TIMEOUT = 30.0


class OllamaEmbedder:
    """Embedding service backed by a local Ollama instance.

    Drop-in replacement for EmbeddingService. Implements the same
    public interface (embed, embed_batch, compute_fisher_params,
    is_available, dimension) so the engine can swap transparently.

    V3.3.27: Session-scoped LRU cache eliminates redundant HTTP calls.
    The store pipeline calls embed() 200+ times for the same texts
    across different components (type_router, scene_builder, consolidator,
    entropy_gate, sheaf_checker). Caching avoids ~215 Ollama roundtrips
    per remember call, reducing latency from 30s to ~3s on Mode B.
    """

    _CACHE_MAX_SIZE = 2048  # entries — covers a full store + recall cycle

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: int = 768,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dimension = dimension
        self._available: bool | None = None  # lazy-checked
        # V3.3.27: Session-scoped embedding cache (text -> normalized vector)
        self._embed_cache: dict[str, list[float]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # ------------------------------------------------------------------
    # Public interface (matches EmbeddingService)
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is pulled."""
        if self._available is not None:
            return self._available
        self._available = self._check_availability()
        return self._available

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float] | None:
        """Embed a single text. Returns normalized vector or None on failure.

        V3.3.27: Returns cached result if the same text was embedded
        earlier in this session, avoiding redundant Ollama HTTP calls.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        from superlocalmemory.core.recall_gate import wait_for_foreground_idle
        wait_for_foreground_idle()

        # V3.3.27: Check cache first
        cache_key = text.strip()
        if cache_key in self._embed_cache:
            self._cache_hits += 1
            return self._embed_cache[cache_key]

        try:
            result = self._call_ollama_embed(text)
            # Cache the result (evict oldest if over limit)
            if result is not None:
                if len(self._embed_cache) >= self._CACHE_MAX_SIZE:
                    # Evict first entry (oldest insertion)
                    first_key = next(iter(self._embed_cache))
                    del self._embed_cache[first_key]
                self._embed_cache[cache_key] = result
            self._cache_misses += 1
            return result
        except Exception as exc:
            logger.warning("Ollama embed failed: %s", exc)
            return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a batch of texts. Uses the batch API when available.

        V3.3.27: Skips already-cached texts, only sends uncached to Ollama.
        """
        if not texts:
            raise ValueError("Cannot embed empty batch")
        from superlocalmemory.core.recall_gate import is_background_work
        if is_background_work():
            return [self.embed(text) for text in texts]

        # V3.3.27: Split into cached and uncached
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = text.strip()
            if key in self._embed_cache:
                results[i] = self._embed_cache[key]
                self._cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return results  # All cached — zero HTTP calls

        try:
            batch_results = self._call_ollama_embed_batch(uncached_texts)
            for idx, emb in zip(uncached_indices, batch_results):
                results[idx] = emb
                if emb is not None:
                    key = texts[idx].strip()
                    if len(self._embed_cache) >= self._CACHE_MAX_SIZE:
                        first_key = next(iter(self._embed_cache))
                        del self._embed_cache[first_key]
                    self._embed_cache[key] = emb
                self._cache_misses += 1
            return results
        except Exception as exc:
            logger.warning("Ollama batch embed failed: %s", exc)
            return results  # Return whatever was cached + None for rest

    def compute_fisher_params(
        self, embedding: list[float],
    ) -> tuple[list[float], list[float]]:
        """Compute Fisher-Rao parameters from a raw embedding."""
        arr = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-10:
            mean = np.zeros(len(arr), dtype=np.float64)
            variance = np.full(len(arr), _FISHER_VAR_MAX, dtype=np.float64)
            return mean.tolist(), variance.tolist()
        mean = arr / norm
        abs_mean = np.abs(mean)
        max_val = float(np.max(abs_mean)) + 1e-10
        signal_strength = abs_mean / max_val
        variance = _FISHER_VAR_MAX - _FISHER_VAR_RANGE * signal_strength
        variance = np.clip(variance, _FISHER_VAR_MIN, _FISHER_VAR_MAX)
        return mean.tolist(), variance.tolist()

    def unload(self) -> None:
        """No-op for Ollama (Ollama manages its own model lifecycle)."""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_availability(self) -> bool:
        """Verify Ollama is running and has the embedding model."""
        import httpx

        try:
            resp = httpx.get(
                f"{self._base_url}/api/tags",
                timeout=_CONNECT_TIMEOUT,
            )
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            model_base = self._model.split(":")[0]
            if model_base not in model_names:
                logger.info(
                    "Ollama running but model '%s' not found (have: %s)",
                    self._model, ", ".join(model_names),
                )
                return False
            return True
        except Exception as exc:
            logger.debug("Ollama not reachable: %s", exc)
            return False

    def _call_ollama_embed(self, text: str) -> list[float]:
        """Call Ollama embed endpoint for a single text.

        v3.4.52: ``keep_alive: -1`` pins the embedding model in VRAM
        forever so subsequent calls have no cold-start latency. Industry
        pattern (Hindsight, Zep, Supermemory) — without this, Ollama
        unloads after 5min idle and the next call takes 20-30s.
        """
        import httpx

        resp = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": [text], "keep_alive": -1},
            timeout=httpx.Timeout(_RESPONSE_TIMEOUT, connect=_CONNECT_TIMEOUT),
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama /api/embed returns {"embeddings": [[...]]}
        vec = data["embeddings"][0]
        return self._normalize(vec)

    def _call_ollama_embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Call Ollama embed endpoint with batch input.

        v3.4.52: ``keep_alive: -1`` pins the embedding model — see
        ``_call_ollama_embed`` docstring for rationale.
        """
        import httpx

        resp = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts, "keep_alive": -1},
            timeout=httpx.Timeout(_RESPONSE_TIMEOUT, connect=_CONNECT_TIMEOUT),
        )
        resp.raise_for_status()
        data = resp.json()
        vectors = data.get("embeddings", [])
        return [self._normalize(v) for v in vectors]

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        """L2-normalize embedding vector."""
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-10:
            arr = arr / norm
        return arr.tolist()
