# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""When the derivation cache must be emptied: redaction changes and erasure.

The cache package owns the file; the callers that know about redaction settings
and erasures live elsewhere, so the hooks are here. Nothing in this module raises.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

_TRUE = ("1", "on", "true", "yes")


def redaction_enabled(config: Any) -> bool:
    """The effective setting: the config value or the SLM_PII_REDACTION variable."""
    if getattr(config, "pii_redaction", False):
        return True
    return os.environ.get("SLM_PII_REDACTION", "").strip().lower() in _TRUE


def sync_cache_with_redaction(config: Any) -> bool:
    """At startup, drop a cache filled under a different redaction setting."""
    try:
        from superlocalmemory.cache import reconcile_redaction_policy

        return reconcile_redaction_policy(redaction_enabled(config))
    except Exception as exc:
        logger.warning("derivation cache redaction check skipped: %s", exc)
        return False


def clear_cache_after_erasure() -> bool:
    """After an erasure, drop the cache so no derived copy of erased text remains."""
    try:
        from superlocalmemory.cache import clear_derived_cache

        return clear_derived_cache("erasure")
    except Exception as exc:
        logger.warning("derivation cache not cleared after erasure: %s", exc)
        return False


def clears_derived_cache(erase: Callable[..., dict]) -> Callable[..., dict]:
    """Wrap an erasure method: once it returns without aborting, clear the cache."""

    @functools.wraps(erase)
    def run(*args: Any, **kwargs: Any) -> dict:
        result = erase(*args, **kwargs)
        if not (result or {}).get("erasure_aborted"):
            clear_cache_after_erasure()
        return result

    return run
