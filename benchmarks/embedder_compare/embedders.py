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
    """nomic-embed-text-v1.5 exactly as SLM ships it: raw text, no task prefixes."""

    name = "nomic"
    query_prefix = ""
    doc_prefix = ""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(NOMIC_ID, trust_remote_code=True, device="cpu")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode([self.query_prefix + t for t in texts]))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode([self.doc_prefix + t for t in texts]))

    def embed_images(self, paths: list[str]) -> np.ndarray:
        raise NotImplementedError("nomic is text only; images go through OCR text")


class NomicPrefixed(Nomic):
    """Reference only: nomic with its recommended task prefixes (not what SLM ships)."""

    name = "nomic_prefixed"
    query_prefix = "search_query: "
    doc_prefix = "search_document: "


class EG2Full:
    """EmbeddingGemma 2, full model (text and images, one 768-d space).

    CPU settings: sdpa attention and float32. An earlier measurement on this
    class of machine saw about 46 s per image idle with default kwargs and
    1,094 ms with these settings; this harness does not re-measure that.
    The audio tower is never loaded.
    """

    name = "eg2_full"
    config_kwargs: dict | None = {"audio_config": None}

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        kwargs = {"model_kwargs": {"attn_implementation": "sdpa", "dtype": torch.float32}}
        if self.config_kwargs:
            kwargs["config_kwargs"] = self.config_kwargs
        self._model = SentenceTransformer(EG2_ID, device="cpu", **kwargs)

    def _prompt(self, name: str) -> str:
        prompts = getattr(self._model, "prompts", None) or {}
        if name not in prompts:
            raise KeyError(f"model has no prompt named {name!r} (has {sorted(prompts)})")
        return name

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


REGISTRY = {"nomic": Nomic, "nomic_prefixed": NomicPrefixed, "eg2_text": EG2Text, "eg2_full": EG2Full}
