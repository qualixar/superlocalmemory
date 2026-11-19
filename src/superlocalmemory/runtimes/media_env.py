# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The media environment: what to install for images and documents."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
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
#: From this much physical memory up (15 GiB, not 16: machines sold as 16 GB report a little less,
#: a Linux VM shows about 15.6 GiB) the models download by themselves; below it they are one click.
MEDIA_MIN_RAM_BYTES = int(15 * GIB)
LOW_RAM_OVERRIDE_ENV = "SLM_MEDIA_ALLOW_LOW_RAM"
#: Set to 1 and the running daemon never starts a download on its own (the test suite sets it).
NO_AUTO_DOWNLOAD_ENV = "SLM_NO_AUTO_DOWNLOAD"
AUTO, ONE_CLICK = "auto", "one_click"
_OVERRIDE_LOGGED = False


@dataclass(frozen=True)
class MediaPolicy:
    """How images and documents get their models on this computer (Varun, 19 Nov 2026).

    ``mode`` is ``auto`` (16 GB or more: the daemon downloads everything itself) or ``one_click``
    (less, or unknown: a person taps one button). Neither refuses; ``warning`` is a plain sentence for a
    small computer, or "".
    """

    mode: str
    ram_bytes: int
    warning: str = ""


def media_policy(ram_bytes: int, env: Mapping[str, str] | None = None) -> MediaPolicy:
    """The one decision, for every entry point (daemon, dashboard, terminal, MCP, installer).

    ``ram_bytes`` of 15 GiB or more -> ``auto``. Less -> ``one_click`` with a warning. Unknown (0)
    -> ``one_click`` without one (we cannot say how much memory there is). ``SLM_MEDIA_ALLOW_LOW_RAM=1``
    is the developer override: a small computer takes the automatic path (warns once per process).
    """
    global _OVERRIDE_LOGGED
    if ram_bytes >= MEDIA_MIN_RAM_BYTES:
        return MediaPolicy(AUTO, ram_bytes)
    if ram_bytes > 0 and (os.environ if env is None else env).get(LOW_RAM_OVERRIDE_ENV) == "1":
        if not _OVERRIDE_LOGGED:
            _OVERRIDE_LOGGED = True
            logger.warning("%s=1: images and documents are automatic on %.1f GB of memory",
                           LOW_RAM_OVERRIDE_ENV, ram_bytes / GIB)
        return MediaPolicy(AUTO, ram_bytes)
    if ram_bytes <= 0:
        return MediaPolicy(ONE_CLICK, 0)
    return MediaPolicy(ONE_CLICK, ram_bytes, (
        f"This computer has {ram_bytes / GIB:.1f} GB of memory; pictures and documents may be slow "
        f"and the models are released when memory runs low."))


def current_media_policy() -> MediaPolicy:
    """``media_policy`` for this computer right now."""
    return media_policy(_env._ram_bytes())


def media_ram_refusal(ram_bytes: int, env: Mapping[str, str] | None = None) -> str:
    """Kept for old callers: images and documents are never refused for memory any more, so ``""``.

    Below 16 GB the answer is ``media_policy(...).warning`` and a one-click download.
    """
    return ""


def media_ram_message() -> str:
    """``media_ram_refusal`` for this computer right now (always ``""``; see ``media_policy``)."""
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
