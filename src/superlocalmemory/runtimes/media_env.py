# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The media environment: what to install for images and documents."""

from __future__ import annotations

import threading
from pathlib import Path

from superlocalmemory.runtimes import managed_env as _env
from superlocalmemory.runtimes.managed_env import EnvSpec, HuggingFaceSource, ManagedEnv

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

#: The model repository id still has to be confirmed against the real hub.
MEDIA_MODEL_REPO = "google/embeddinggemma-2"
#: Empty until the revision is pinned; installing from the hub refuses while empty.
MEDIA_MODEL_REVISION = ""

GIB = 1024 ** 3
MEDIA_DOWNLOAD_BYTES = int(1.5 * GIB)
MEDIA_RAM_WARN_BYTES = int(7.5 * GIB)


def _min_free_disk() -> int:
    # Estimates; re-measure on real installs.
    return 4 * GIB if _env._platform_tag() == "darwin-arm64" else 8 * GIB


MEDIA_ENV = EnvSpec(
    name="media",
    requirements=MEDIA_REQUIREMENTS,
    model_source=HuggingFaceSource(MEDIA_MODEL_REPO, MEDIA_MODEL_REVISION, MEDIA_DOWNLOAD_BYTES),
    min_free_disk_bytes=_min_free_disk(),
    min_ram_bytes_warn=MEDIA_RAM_WARN_BYTES,
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
