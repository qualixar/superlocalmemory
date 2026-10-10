# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Is remote access set up on this machine? One yes/no for features that must not run then."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def remote_access_configured() -> bool:
    """True when a remote listener is set up, a remote key is active, or LAN mode is on.

    Fails closed: if any part cannot be read, the answer is True. Remote web-app
    connections issue remote keys, so active keys cover them and the enrollment
    journal is not read.
    """
    try:
        from superlocalmemory.core.remote_mode import is_remote_mode
        from superlocalmemory.server.remote_keys import default_store
        from superlocalmemory.server.remote_listener import try_load_config

        config, error = try_load_config()
        if config is not None or error is not None:
            return True
        if any(key.active for key in default_store().list()):
            return True
        return bool(is_remote_mode())
    except Exception as exc:  # noqa: BLE001 - an unreadable state is not "off"
        logger.warning("could not read remote access state (%s); treating it as on",
                       type(exc).__name__)
        return True


__all__ = ["remote_access_configured"]
