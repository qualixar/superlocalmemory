# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Where memory text goes, worked out from the configuration (#303).

Modes A and B accept an OpenAI-compatible embedding endpoint, and Mode B a
hosted language model, so "data stays local" cannot be a fact about the mode
name. ``ModeCapabilities.data_stays_local`` stays as the mode's default; this
module answers the question for the configuration actually in force, and every
surface that reports locality asks it.

An endpoint is local when its literal host is loopback (``localhost``,
``*.localhost``, 127.0.0.0/8, ::1) or a numeric private-LAN address
(10/8, 172.16/12, 192.168/16, fc00::/7, link-local). Anything else, including
a public hostname that cannot be resolved, is remote. There is no DNS lookup:
a name is not proof of where it points, the same posture the remote reranker
and the LLM endpoint checks already take, so this reuses their host helpers.

Pure functions, no I/O.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from superlocalmemory.core.modes import get_capabilities
from superlocalmemory.retrieval.remote_reranker_config import (
    _is_loopback_host,
    _is_private_lan_host,
)
from superlocalmemory.storage.models import Mode

# What Mode B's template writes into api_base for any LLM provider when none was
# given. It is Ollama's address, so it says nothing about where a hosted
# provider's requests go.
_OLLAMA_TEMPLATE_BASE = "http://localhost:11434"

EMBEDDING_PART = "embedding endpoint"
LLM_PART = "language model"


def _host_of(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = f"//{text}"  # "localhost:8045/v1" parses as a scheme otherwise
    try:
        return urlparse(text).hostname or ""
    except ValueError:
        return ""


def is_local_endpoint(url: str) -> bool:
    """True when ``url``'s literal host is loopback or a private-LAN address."""
    host = _host_of(url)
    return bool(host) and (_is_loopback_host(host) or _is_private_lan_host(host))


def _embedding_is_remote(embedding: Any) -> bool:
    provider = getattr(embedding, "provider", "") or ""
    if provider == "openai":
        endpoint = getattr(embedding, "api_endpoint", "") or ""
        return bool(endpoint) and not is_local_endpoint(endpoint)
    if provider == "ollama":
        base = getattr(embedding, "ollama_base_url", "") or ""
        return bool(base) and not is_local_endpoint(base)
    if provider == "cloud":
        return True
    return False


def _llm_is_remote(llm: Any) -> bool:
    provider = (getattr(llm, "provider", "") or "").strip().lower()
    if provider in ("", "none"):
        return False
    base = (getattr(llm, "api_base", "") or "").strip()
    if provider == "ollama":
        return bool(base) and not is_local_endpoint(base)
    if base.rstrip("/") == _OLLAMA_TEMPLATE_BASE:
        return True
    return not (base and is_local_endpoint(base))


@dataclass(frozen=True)
class DataLocality:
    """Whether memory text stays on this computer, and what carries it away."""

    stays_local: bool
    remote_parts: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return "local-only" if self.stays_local else "provider-assisted"

    @property
    def embedding_remote(self) -> bool:
        return EMBEDDING_PART in self.remote_parts

    @property
    def llm_remote(self) -> bool:
        return LLM_PART in self.remote_parts


def effective_data_locality(
    mode: Mode, embedding: Any = None, llm: Any = None,
) -> DataLocality:
    """The mode's default, made true to the configuration.

    A mode whose default is not local stays not local. A mode whose default is
    local stays local only while the embedding endpoint is local and, for a
    mode that calls a language model, that model is too. With no configuration
    given the answer is the mode's default.
    """
    caps = get_capabilities(mode)
    parts: list[str] = []
    if embedding is not None and _embedding_is_remote(embedding):
        parts.append(EMBEDDING_PART)
    if llm is not None and caps.llm_fact_extraction and _llm_is_remote(llm):
        parts.append(LLM_PART)
    return DataLocality(
        stays_local=caps.data_stays_local and not parts,
        remote_parts=tuple(parts),
    )


def remote_parts_sentence(locality: DataLocality) -> str:
    """"" while data stays local; else one plain sentence saying what carries it away."""
    if locality.stays_local or not locality.remote_parts:
        return ""
    what = " and ".join(locality.remote_parts)
    return f"The {what} you configured is not on this computer, so memory text leaves this computer."


__all__ = [
    "DataLocality",
    "EMBEDDING_PART",
    "LLM_PART",
    "effective_data_locality",
    "is_local_endpoint",
    "remote_parts_sentence",
]
