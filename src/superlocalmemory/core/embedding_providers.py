# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The embedding provider names a switch may ask for."""

from __future__ import annotations

#: "" is automatic. "slm-media" is the managed model environment's worker.
EMBEDDING_PROVIDERS: tuple[str, ...] = ("", "sentence-transformers", "ollama", "openai", "cloud", "slm-media")


def validate_embedding_provider(name: str) -> str:
    """``name`` if it is a known provider, else a ValueError that lists the known ones."""
    if name in EMBEDDING_PROVIDERS:
        return name
    known = ", ".join(p for p in EMBEDDING_PROVIDERS if p)
    raise ValueError(f"unknown embedding provider '{name}'. Known providers: {known} "
                     "(leave it empty for automatic)")


__all__ = ["EMBEDDING_PROVIDERS", "validate_embedding_provider"]
