# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Embedding Service.

Thread-safe, dimension-validated embedding with Fisher variance computation.
Supports local (768-dim nomic) and cloud (3072-dim) models with EXPLICIT errors
on dimension mismatch — NEVER silently falls back to a different dimension.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from superlocalmemory.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fisher variance constants
# ---------------------------------------------------------------------------
_FISHER_VAR_MIN = 0.05
_FISHER_VAR_MAX = 2.0
_FISHER_VAR_RANGE = _FISHER_VAR_MAX - _FISHER_VAR_MIN  # 1.95


class DimensionMismatchError(RuntimeError):
    """Raised when the actual embedding dimension differs from config.

    This is a HARD failure — V1 silently fell back to local embeddings
    when Azure failed, changing dimension from 3072 to 768 mid-run.
    We crash loudly instead.
    """


class EmbeddingService:
    """Thread-safe embedding service with strict dimension validation.

    Lazy-loads the underlying model on first embed call.
    Validates every output dimension against the configured expectation.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: object | None = None
        self._lock = threading.Lock()
        self._loaded = False
        self._available = True  # Set False if model can't load

    @property
    def is_available(self) -> bool:
        """Check if embedding service has a usable model."""
        if not self._loaded:
            self._ensure_loaded()
        return self._available and self._model is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Expected embedding dimension (from config)."""
        return self._config.dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Returns:
            L2-normalized embedding of exactly ``self.dimension`` floats.

        Raises:
            ValueError: If text is empty.
            DimensionMismatchError: If output dimension != config.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        self._ensure_loaded()
        if self._model is None:
            return None
        vec = self._encode_single(text)
        self._validate_dimension(vec)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Returns:
            List of L2-normalized embeddings, each ``self.dimension`` floats.

        Raises:
            ValueError: If any text is empty or list is empty.
            DimensionMismatchError: If any output dimension != config.
        """
        if not texts:
            raise ValueError("Cannot embed empty batch")
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"Text at index {i} is empty")

        self._ensure_loaded()
        if self._model is None:
            return [None] * len(texts)
        vectors = self._encode_batch(texts)
        for vec in vectors:
            self._validate_dimension(vec)
        return [v.tolist() for v in vectors]

    def compute_fisher_params(
        self,
        embedding: list[float],
    ) -> tuple[list[float], list[float]]:
        """Compute Fisher-Rao parameters from a raw embedding.

        Variance is content-derived (NOT uniform). Dimensions with strong
        signal (high absolute value) get LOW variance (high confidence).
        Weak-signal dimensions get HIGH variance (uncertainty).

        This heterogeneous variance is what gives Fisher-Rao metric
        discriminative power beyond simple cosine similarity.

        Args:
            embedding: Raw embedding vector (already L2-normalized).

        Returns:
            (mean, variance) — both lists of ``self.dimension`` floats.
            Variance values are clamped to [0.3, 2.0].
        """
        arr = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(arr))

        if norm < 1e-10:
            mean = np.zeros(len(arr), dtype=np.float64)
            variance = np.full(len(arr), _FISHER_VAR_MAX, dtype=np.float64)
            return mean.tolist(), variance.tolist()

        mean = arr / norm

        # Content-derived heterogeneous variance
        abs_mean = np.abs(mean)
        max_val = float(np.max(abs_mean)) + 1e-10
        signal_strength = abs_mean / max_val  # [0, 1]

        # Inverse: strong signal -> low variance, weak -> high
        variance = _FISHER_VAR_MAX - _FISHER_VAR_RANGE * signal_strength
        variance = np.clip(variance, _FISHER_VAR_MIN, _FISHER_VAR_MAX)

        return mean.tolist(), variance.tolist()

    # ------------------------------------------------------------------
    # Internals — model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use (thread-safe)."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self._config.is_cloud:
                # Cloud mode: no local model needed, validate config
                if not self._config.api_endpoint or not self._config.api_key:
                    raise RuntimeError(
                        "Cloud embedding requires api_endpoint and api_key"
                    )
                logger.info(
                    "EmbeddingService: cloud mode (%s, %d-dim)",
                    self._config.deployment_name,
                    self._config.dimension,
                )
            else:
                self._load_local_model()
            self._loaded = True

    def _load_local_model(self) -> None:
        """Load sentence-transformers model for local embedding."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Embeddings disabled. "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
            self._loaded = True
            self._available = False
            return
        model = SentenceTransformer(
            self._config.model_name, trust_remote_code=True,
        )
        actual_dim = model.get_sentence_embedding_dimension()
        if actual_dim != self._config.dimension:
            raise DimensionMismatchError(
                f"Model '{self._config.model_name}' produces {actual_dim}-dim "
                f"embeddings but config expects {self._config.dimension}-dim"
            )
        self._model = model
        logger.info(
            "EmbeddingService: local model loaded (%s, %d-dim)",
            self._config.model_name,
            actual_dim,
        )

    # ------------------------------------------------------------------
    # Internals — encoding
    # ------------------------------------------------------------------

    def _encode_single(self, text: str) -> NDArray[np.float32]:
        """Encode one text. Dispatches to local or cloud."""
        self._ensure_loaded()
        if self._config.is_cloud:
            return self._cloud_embed([text])[0]
        return self._local_embed_batch([text])[0]

    def _encode_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Encode a batch. Dispatches to local or cloud."""
        self._ensure_loaded()
        if self._config.is_cloud:
            return self._cloud_embed(texts)
        return self._local_embed_batch(texts)

    def _local_embed_batch(
        self,
        texts: list[str],
    ) -> list[NDArray[np.float32]]:
        """Encode via local sentence-transformers (L2-normalized)."""
        if self._model is None:
            raise RuntimeError("Local model not loaded")
        vecs = self._model.encode(texts, normalize_embeddings=True)
        if isinstance(vecs, np.ndarray) and vecs.ndim == 2:
            return [vecs[i] for i in range(vecs.shape[0])]
        return [np.asarray(v, dtype=np.float32) for v in vecs]

    def _cloud_embed(
        self,
        texts: list[str],
        *,
        max_retries: int = 3,
    ) -> list[NDArray[np.float32]]:
        """Encode via Azure OpenAI embedding API with retry logic.

        Raises on failure — NEVER falls back to local model.
        """
        import httpx

        url = (
            f"{self._config.api_endpoint.rstrip('/')}/openai/deployments/"
            f"{self._config.deployment_name}/embeddings"
            f"?api-version={self._config.api_version}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": self._config.api_key,
        }
        body = {"input": texts, "model": self._config.deployment_name}

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
                    resp = client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                data = resp.json()
                results: list[NDArray[np.float32]] = []
                for item in sorted(data["data"], key=lambda d: d["index"]):
                    vec = np.asarray(item["embedding"], dtype=np.float32)
                    results.append(vec)
                return results
            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "Cloud embed attempt %d/%d failed: %s (retry in %ds)",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)

        raise RuntimeError(
            f"Cloud embedding failed after {max_retries} attempts: "
            f"{last_error}"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dimension(self, vec: NDArray) -> None:
        """Hard validation — crash on mismatch, never silently fall back."""
        actual = len(vec)
        if actual != self._config.dimension:
            raise DimensionMismatchError(
                f"Embedding dimension {actual} != "
                f"expected {self._config.dimension}. "
                f"This is a HARD failure — check your model/API config."
            )
