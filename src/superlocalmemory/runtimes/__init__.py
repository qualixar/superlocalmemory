# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Managed model environments and the switches that turn optional features on."""

from __future__ import annotations

from superlocalmemory.runtimes.features import (
    disable_media, enable_media, media_enabled, media_feature_status, read_features,
    register_media_stop_hook,
)
from superlocalmemory.runtimes.managed_env import (
    EnvSpec, EnvStatus, HuggingFaceSource, LocalDirSource, ManagedEnv, ModelSource,
)
from superlocalmemory.runtimes.media_env import MEDIA_ENV, media_env

__all__ = [
    "EnvSpec", "EnvStatus", "HuggingFaceSource", "LocalDirSource", "MEDIA_ENV", "ManagedEnv",
    "ModelSource", "disable_media", "enable_media", "media_enabled", "media_env",
    "media_feature_status", "read_features", "register_media_stop_hook",
]
