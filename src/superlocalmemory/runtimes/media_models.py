# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One table of what each picture model needs: revision, width, memory cap, evidence floor.

Pure data and lookups: nothing heavy is imported, and the worker process (which has no
superlocalmemory installed) never imports this file; the client passes it what it needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

#: Cap for models without a profile (the fake models used in tests).
DEFAULT_RSS_LIMIT_MB = 1600
#: Free memory wanted before loading a model that has no profile (text models).
DEFAULT_LOAD_MB = 1500

#: Commit of google/embeddinggemma-2 on the hub, verified on 2026-10-10.
EG2_REPO = "google/embeddinggemma-2"
EG2_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


@dataclass(frozen=True)
class ModelProfile:
    repo: str
    revision: str
    dim: int
    rss_limit_mb: int        # the worker is stopped above this resident size
    media_min_score: float   # picture similarity that counts as evidence
    image_max_pixels: int    # larger pictures are shrunk before embedding; 0 = no cap
    load_mb: int = 1500      # free memory wanted before this model loads (peak while loading)


#: The EG2 numbers are provisional until the Mac memory check; the floor comes from
#: n=7 unanswerable test queries.
MODEL_PROFILES: Mapping[str, ModelProfile] = MappingProxyType({
    EG2_REPO: ModelProfile(EG2_REPO, EG2_REVISION, 768, 4500, 0.69, 0, 3000),  # the model's own processor bounds image tokens; recall was measured without a pre-shrink
    "nomic-ai/nomic-embed-vision-v1.5": ModelProfile(
        "nomic-ai/nomic-embed-vision-v1.5", "", 768, DEFAULT_RSS_LIMIT_MB, 0.084, 0, 1500),
})


def profile_for(model: str) -> ModelProfile | None:
    """The profile for a model repo id, or None (fake models, unknown models)."""
    return MODEL_PROFILES.get(model)


def rss_limit_mb_for(model: str) -> int:
    profile = profile_for(model)
    return profile.rss_limit_mb if profile is not None else DEFAULT_RSS_LIMIT_MB


def effective_rss_limit_mb(model: str) -> int:
    """The cap the worker applies to ``model``: the environment override, else the table."""
    try:
        return int(float(os.environ["SLM_MEDIA_WORKER_RSS_LIMIT_MB"]))
    except (KeyError, ValueError):
        return rss_limit_mb_for(model)


def load_mb_for(model: str) -> int:
    """Free memory to ask for before loading ``model`` (the default covers text and fake models)."""
    profile = profile_for(model)
    return profile.load_mb if profile is not None else DEFAULT_LOAD_MB


def min_score_for(model: str) -> float | None:
    """The model's own picture evidence floor, or None to keep the configured one."""
    profile = profile_for(model)
    return profile.media_min_score if profile is not None else None


def watchdog_limit_mb(default: int) -> int:
    """Memory limit the daemon's watchdog applies to the picture worker; 0 means no limit.

    The worker enforces its own cap per model after each request. The watchdog must not
    kill it earlier than that, so it uses the same override, else the largest model cap.
    """
    try:
        wanted = int(float(os.environ["SLM_MEDIA_WORKER_RSS_LIMIT_MB"]))
    except (KeyError, ValueError):
        return max([default, *(p.rss_limit_mb for p in MODEL_PROFILES.values())])
    return max(wanted, 0)


__all__ = ["DEFAULT_LOAD_MB", "DEFAULT_RSS_LIMIT_MB", "EG2_REPO", "EG2_REVISION", "effective_rss_limit_mb", "MODEL_PROFILES", "ModelProfile",
           "load_mb_for", "min_score_for", "profile_for", "rss_limit_mb_for", "watchdog_limit_mb"]
