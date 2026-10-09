"""Embedders. torch / sentence-transformers are imported lazily inside load()."""
from __future__ import annotations

from typing import Protocol

import numpy as np

NOMIC_ID = "nomic-ai/nomic-embed-text-v1.5"
EG2_ID = "google/embeddinggemma-2"


class Embedder(Protocol):
    name: str

    def load(self) -> None: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    def embed_docs(self, texts: list[str]) -> np.ndarray: ...
    def embed_images(self, paths: list[str]) -> np.ndarray: ...


def _unit(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class Nomic:
    """nomic-embed-text-v1.5 with the task prefixes SLM uses; text only."""

    name = "nomic"

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(NOMIC_ID, trust_remote_code=True, device="cpu")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode([f"search_query: {t}" for t in texts]))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode([f"search_document: {t}" for t in texts]))

    def embed_images(self, paths: list[str]) -> np.ndarray:
        raise NotImplementedError("nomic is text only; images go through OCR text")


class EG2Full:
    """EmbeddingGemma 2, full model (text and images, one 768-d space).

    CPU settings from the measured run: sdpa attention and float32 (default
    kwargs take about 46 s per image on CPU).
    """

    name = "eg2_full"
    config_kwargs: dict | None = None

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        kwargs = {"model_kwargs": {"attn_implementation": "sdpa", "dtype": torch.float32}}
        if self.config_kwargs:
            kwargs["config_kwargs"] = self.config_kwargs
        self._model = SentenceTransformer(EG2_ID, device="cpu", **kwargs)

    def _prompt(self, name: str) -> str | None:
        return name if name in (getattr(self._model, "prompts", None) or {}) else None

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode(texts, prompt_name=self._prompt("SearchQuery")))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode(texts, prompt_name=self._prompt("Document")))

    def embed_images(self, paths: list[str]) -> np.ndarray:
        return _unit([self._model.encode({"image": p}) for p in paths])


class EG2Text(EG2Full):
    """EmbeddingGemma 2 text-only loadout (vision and audio towers not loaded)."""

    name = "eg2_text"
    config_kwargs = {"vision_config": None, "audio_config": None}

    def embed_images(self, paths: list[str]) -> np.ndarray:
        raise NotImplementedError("text-only loadout cannot embed images")


REGISTRY = {"nomic": Nomic, "eg2_text": EG2Text, "eg2_full": EG2Full}
