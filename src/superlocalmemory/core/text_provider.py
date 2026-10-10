# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which embedder serves the managed ``slm-media`` text provider in this process.

The daemon runs the model (``runtimes.managed_text``); every other process asks the
daemon (``core.daemon_text_embedder``). Either embedder may be unavailable right now
(the media environment is not installed, the daemon is not running): it is returned
anyway, says so, and recovers by itself, so a long-lived process is not stuck
keyword-only after the daemon finishes starting. It never falls back to another
model, whose vectors would be in the wrong space.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from superlocalmemory.core.process_role import is_daemon_process

logger = logging.getLogger(__name__)

PROVIDER = "slm-media"
_NOT_READY = ("The managed text model is not installed or not ready, so text search uses keywords "
              "only until it is. Finish the install with: slm media enable")


def init_slm_media_embedder(config: Any) -> Any:
    emb_cfg = config.embedding
    if is_daemon_process():
        from superlocalmemory.runtimes.managed_text import ManagedTextEmbedder

        emb = ManagedTextEmbedder(emb_cfg, data_root=getattr(config, "base_dir", None))
        message = _NOT_READY
    else:
        from superlocalmemory.core.daemon_text_embedder import NEEDS_SERVICE, DaemonTextEmbedder

        emb = DaemonTextEmbedder(emb_cfg)
        message = NEEDS_SERVICE
    if not emb.is_available:
        logger.warning(message)
    return emb


def route_embedder(config: Any, *, try_ollama: Callable[[Any], Any | None],
                   try_service: Callable[[Any], Any | None]) -> Any | None:
    """The embedder for ``config.embedding.provider``; None means keyword-only.

    ``try_ollama`` / ``try_service`` build the Ollama and the sentence-transformers /
    remote-service embedders (None when they are not usable).
    """
    emb_cfg = config.embedding
    provider = emb_cfg.provider

    # The managed model (daemon: the worker; other processes: the daemon).
    if provider == PROVIDER:
        return init_slm_media_embedder(config)

    # When the provider is ollama, Ollama's own vectors are primary: the stored vectors
    # were made by Ollama's nomic-embed-text, and sentence-transformers' nomic-embed-text-v1.5
    # makes different ones, so mixing them degrades recall. The subprocess is the fallback.
    if provider == "ollama":
        result = try_ollama(emb_cfg)
        if result is not None:
            logger.info("Using Ollama embeddings (nomic-embed-text, local)")
            return result
        st_emb = try_service(emb_cfg)
        if st_emb is not None:
            logger.warning("Ollama unavailable; falling back to sentence-transformers subprocess")
            return st_emb
        return None

    if provider == "openai" and emb_cfg.is_openai_compatible:
        logger.info(
            "Using OpenAI-compatible embedding endpoint: %s (model=%s, dim=%d)",
            emb_cfg.api_endpoint, emb_cfg.model_name, emb_cfg.dimension,
        )
        return try_service(emb_cfg)

    # Explicit cloud / sentence-transformers (subprocess-isolated)
    if provider in ("cloud", "sentence-transformers") or emb_cfg.is_cloud:
        return try_service(emb_cfg)

    # Auto-detect: Ollama first (lightweight, <1s), then the subprocess, which never
    # imports torch in this process.
    ollama_emb = try_ollama(emb_cfg)
    if ollama_emb is not None:
        logger.info("Auto-detected Ollama embeddings (fast path)")
        return ollama_emb
    return try_service(emb_cfg)


def init_light_mode_embedder(config: Any) -> Any | None:
    """The managed provider's embedder for a light (MCP) engine, else None (the caller's proxy).

    The daemon-side embedder checks the daemon's model on every call and recovers when the
    daemon starts, which the plain proxy does not.
    """
    if config.embedding.provider != PROVIDER:
        return None
    from superlocalmemory.core.daemon_text_embedder import DaemonTextEmbedder

    return DaemonTextEmbedder(config.embedding)


__all__ = ["PROVIDER", "init_light_mode_embedder", "init_slm_media_embedder", "route_embedder"]
