# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The media environment: what to install for images and documents."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path

from superlocalmemory.runtimes import managed_env as _env
from superlocalmemory.runtimes import media_models
from superlocalmemory.runtimes.managed_env import EnvSpec, HuggingFaceSource, ManagedEnv
from superlocalmemory.runtimes.media_canary import media_canary

logger = logging.getLogger(__name__)

#: Top-level packages. The installed set is the hashed lock for the platform
#: (runtimes/locks/), generated from these by scripts/lock_media_env.py.
MEDIA_REQUIREMENTS: tuple[str, ...] = (
    "sentence-transformers[image]==6.1.0",
    "transformers==5.19.0",
    "torch==2.14.1",
    "torchvision==0.29.1",
    "pillow==12.3.0",
    "ImageHash==4.3.2",
    "pypdfium2==5.14.0",
    "rapidocr==3.10.0",
    "onnxruntime==1.30.0",
    "pyobjc-framework-Vision==12.2.2; sys_platform == 'darwin'",
)

#: The picture model and its pinned hub commit (checked against the hub on 2026-10-10).
#: The per-model numbers live in media_models.MODEL_PROFILES.
MEDIA_MODEL_REPO = media_models.EG2_REPO
MEDIA_MODEL_REVISION = media_models.EG2_REVISION  # 914f7f89142e33e77833254d9c9b90c3cef7303b

GIB = 1024 ** 3
MEDIA_DOWNLOAD_BYTES = int(1.5 * GIB)
#: Below this much physical memory, images and documents are refused. 15 GiB, not 16:
#: machines sold as 16 GB report a little less (a Linux VM shows about 15.6 GiB).
MEDIA_MIN_RAM_BYTES = int(15 * GIB)
LOW_RAM_OVERRIDE_ENV = "SLM_MEDIA_ALLOW_LOW_RAM"
_OVERRIDE_LOGGED = False


def media_ram_refusal(ram_bytes: int, env: Mapping[str, str] | None = None) -> str:
    """Why images and documents cannot be turned on here, or "" when they can.

    The one place that decides, for every entry point (daemon, dashboard, terminal,
    installer). Unknown memory (0) is allowed. ``SLM_MEDIA_ALLOW_LOW_RAM=1`` is a
    developer override: it allows a small machine and warns once per process. The
    installer script mirrors the threshold and the override.
    """
    global _OVERRIDE_LOGGED
    if not 0 < ram_bytes < MEDIA_MIN_RAM_BYTES:
        return ""
    if (os.environ if env is None else env).get(LOW_RAM_OVERRIDE_ENV) == "1":
        if not _OVERRIDE_LOGGED:
            _OVERRIDE_LOGGED = True
            logger.warning("%s=1: images and documents allowed on %.1f GB of memory (16 GB needed)",
                           LOW_RAM_OVERRIDE_ENV, ram_bytes / GIB)
        return ""
    return (f"Images and documents need a computer with at least 16 GB of memory; this one has "
            f"{ram_bytes / GIB:.1f} GB. Your text memories keep working.")


def media_ram_message() -> str:
    """``media_ram_refusal`` for this computer right now."""
    return media_ram_refusal(_env._ram_bytes())


def _min_free_disk() -> int:
    # Estimates; re-measure on real installs.
    return 4 * GIB if _env._platform_tag() == "darwin-arm64" else 8 * GIB


MEDIA_ENV = EnvSpec(
    name="media",
    requirements=MEDIA_REQUIREMENTS,
    model_source=HuggingFaceSource(MEDIA_MODEL_REPO, MEDIA_MODEL_REVISION, MEDIA_DOWNLOAD_BYTES),
    min_free_disk_bytes=_min_free_disk(),
    canary=media_canary,
    expected_download_bytes=MEDIA_DOWNLOAD_BYTES,
)


_SHARED: dict[Path, ManagedEnv] = {}
_SHARED_LOCK = threading.Lock()


def media_env(root: Path | None = None) -> ManagedEnv:
    """The one media environment per folder (``<data_root>/runtimes/media`` unless ``root`` is given).

    Shared so every caller in the process uses the same state-file mutex.
    """
    fresh = ManagedEnv(MEDIA_ENV, root=root)
    with _SHARED_LOCK:
        return _SHARED.setdefault(fresh.root, fresh)
