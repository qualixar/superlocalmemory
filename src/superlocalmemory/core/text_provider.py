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


def init_light_mode_embedder(config: Any) -> Any | None:
    """The managed provider's embedder for a light (MCP) engine, else None (the caller's proxy).

    The daemon-side embedder checks the daemon's model on every call and recovers when the
    daemon starts, which the plain proxy does not.
    """
    if config.embedding.provider != PROVIDER:
        return None
    from superlocalmemory.core.daemon_text_embedder import DaemonTextEmbedder

    return DaemonTextEmbedder(config.embedding)


__all__ = ["PROVIDER", "init_light_mode_embedder", "init_slm_media_embedder"]
