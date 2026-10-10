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


def resolve_embedding_provider(explicit: str | None, model: str, inherited: str = "") -> str:
    """The provider a switch to ``model`` runs under, or a ValueError when the pair is wrong.

    ``explicit`` is what the request names (empty: none), ``inherited`` the live one.
    The managed model runs only under ``slm-media`` (chosen for it when none is named),
    and ``slm-media`` serves only the managed models: anything else would be embedded
    by another provider's model under a name that says otherwise.
    """
    from superlocalmemory.core.model_catalog import MANAGED_EMBEDDERS

    managed = [entry.id for entry in MANAGED_EMBEDDERS]
    named = validate_embedding_provider(explicit or "")
    if model in managed:
        if named not in ("", "slm-media"):
            raise ValueError(f"{model} runs only with the slm-media provider, not {named}")
        return "slm-media"
    if not named and inherited == "slm-media":
        return ""  # leaving the managed model: back to the automatic provider, not a refusal
    provider = named or inherited
    if provider == "slm-media":
        raise ValueError(f"slm-media serves only: {', '.join(managed)}")
    return provider


__all__ = ["EMBEDDING_PROVIDERS", "resolve_embedding_provider", "validate_embedding_provider"]
