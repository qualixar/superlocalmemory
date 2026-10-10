# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The text evidence floor of the live text model.

Similarity scores are not comparable between models, so a model that has its own
floor in ``runtimes.media_models`` uses it; every other embedder keeps the
configured ``min_semantic_evidence``.
"""

from __future__ import annotations

from typing import Any


def text_semantic_floor(embedder: Any, default: float) -> float:
    name = getattr(embedder, "model_name", None)
    if not isinstance(name, str):
        return default
    from superlocalmemory.runtimes.media_models import text_min_semantic_for

    own = text_min_semantic_for(name)
    return default if own is None else own


__all__ = ["text_semantic_floor"]
