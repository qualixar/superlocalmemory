# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The two embedding ports: text recall keeps its own, images and pages use the other.

Nothing here routes text recall through the media model: the text port is what
``core.embeddings.EmbeddingService`` already provides, seen through a small adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


@runtime_checkable
class TextEmbedderPort(Protocol):
    model_id: str
    dim: int

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]: ...


@runtime_checkable
class MediaEmbedderPort(Protocol):
    model_id: str
    dim: int

    def embed_images(self, paths: list[Path]) -> list[list[float]]: ...

    def embed_texts(self, texts: list[str], *,
                    prompt: Literal["SearchQuery", "Document"]) -> list[list[float]]: ...


class EmbeddingServiceText:
    """``EmbeddingService`` as a :class:`TextEmbedderPort` (it is not edited to fit)."""

    def __init__(self, service: Any, model_id: str | None = None) -> None:
        self._service = service
        self.model_id = model_id or str(getattr(getattr(service, "_config", None), "model_name", ""))

    @property
    def dim(self) -> int:
        return int(self._service.dimension)

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return self._service.embed_batch(texts)


__all__ = ["EmbeddingServiceText", "MediaEmbedderPort", "TextEmbedderPort"]
