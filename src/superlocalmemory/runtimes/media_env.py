# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The media environment: what to install for images and documents."""

from __future__ import annotations

import threading
from pathlib import Path

from superlocalmemory.runtimes import managed_env as _env
from superlocalmemory.runtimes import media_models
from superlocalmemory.runtimes.managed_env import EnvSpec, HuggingFaceSource, ManagedEnv
from superlocalmemory.runtimes.media_canary import media_canary

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
MEDIA_RAM_WARN_BYTES = int(7.5 * GIB)


def ram_warning_text(ram_bytes: int) -> str:
    """One plain sentence for a small machine, or "" when memory is enough or unknown.

    Shown wherever images and documents can be turned on. It informs; it never blocks.
    """
    if not 0 < ram_bytes < MEDIA_RAM_WARN_BYTES:
        return ""
    return (f"This computer has {ram_bytes / GIB:.1f} GB of memory. Images and documents work best "
            "with 8 GB or more and may slow other apps while they work. You can still turn them on.")


def _min_free_disk() -> int:
    # Estimates; re-measure on real installs.
    return 4 * GIB if _env._platform_tag() == "darwin-arm64" else 8 * GIB


MEDIA_ENV = EnvSpec(
    name="media",
    requirements=MEDIA_REQUIREMENTS,
    model_source=HuggingFaceSource(MEDIA_MODEL_REPO, MEDIA_MODEL_REVISION, MEDIA_DOWNLOAD_BYTES),
    min_free_disk_bytes=_min_free_disk(),
    min_ram_bytes_warn=MEDIA_RAM_WARN_BYTES,
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
