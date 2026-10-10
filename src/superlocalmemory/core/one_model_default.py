# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""First run only: new users get one model for text and pictures, behind one constant.

``media_models.ONE_MODEL_DEFAULT_ENABLED`` is False until the Mac memory check passes;
flipping it is a release decision. Even then it applies only to a folder with no
configuration and no memory database, on a machine with enough memory. Existing users
never switch automatically: they upgrade on request (``core/engine_upgrade``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from superlocalmemory.runtimes import media_models

logger = logging.getLogger(__name__)

_GIB = 1024 ** 3


def _total_ram_bytes() -> int:
    from superlocalmemory.runtimes.managed_env import _ram_bytes

    return _ram_bytes()


def _is_new_user(base_dir: Path) -> bool:
    return not (base_dir / "memory.db").exists() and not (base_dir / "config.json").exists()


def wanted(base_dir: Path, total_ram_bytes: int | None = None) -> bool:
    """True when this first run should start on the managed model."""
    if not media_models.ONE_MODEL_DEFAULT_ENABLED or not _is_new_user(Path(base_dir)):
        return False
    ram = _total_ram_bytes() if total_ram_bytes is None else total_ram_bytes
    return ram >= media_models.ONE_MODEL_MIN_RAM_GB * _GIB


def apply_first_run_default(config: Any, base_dir: Path, *, total_ram_bytes: int | None = None) -> bool:
    """Point a new user's config at the managed model and record the request to set it up.

    Returns True when the config was changed. Text recall is keyword-only until the
    environment finishes installing at the first daemon start (the status line says so).
    """
    try:
        if not wanted(Path(base_dir), total_ram_bytes):
            return False
        from superlocalmemory.core.engine_upgrade import upgrade_target
        from superlocalmemory.runtimes import features

        if not features.record_media_request(source="api", data_root=base_dir):
            return False
        config.embedding = upgrade_target(config.embedding)
        return True
    except Exception:  # noqa: BLE001 - the first run must never fail on an optional default
        logger.warning("could not apply the one-model default", exc_info=True)
        return False


__all__ = ["apply_first_run_default", "wanted"]
