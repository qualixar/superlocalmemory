# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-02 §4.4

"""Active-model cache + integrity verification.

LLD reference: ``.backup/active-brain/lld/LLD-02-signal-pipeline-and-lightgbm.md``
Section 4.4 — every model load goes through here.

Hard rules enforced:
    M1 — ``pickle.loads`` is FORBIDDEN on ``state_bytes``.
    M2 — SHA-256 verified before ``Booster(model_str=...)``.
    M3 — Feature-name drift is logged, not silently ignored.

Cache: LRU size=4 keyed by ``profile_id``. Thread-safe.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from superlocalmemory.core.security_primitives import (
    IntegrityError,
    verify_sha256,
)
from superlocalmemory.learning.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


# ``tuple[str, ...]`` for immutability — matches LLD-02 §4.4.
_CURRENT_FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_NAMES)

_CACHE_MAX = 4


@dataclass(frozen=True)
class ActiveModel:
    """Verified, in-memory booster plus provenance."""

    profile_id: str
    booster: Any  # lightgbm.Booster — Any keeps this import-light
    feature_names: tuple[str, ...]
    trained_at: str
    sha256: str


class _LRU:
    """Tiny thread-safe LRU for ``ActiveModel | None`` entries.

    Values of ``None`` (no active model) are cached too so we don't SELECT
    on every recall when there's nothing to load.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: "OrderedDict[str, ActiveModel | None]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[bool, ActiveModel | None]:
        """Return (hit, value) — hit=False means cache miss."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return True, self._data[key]
        return False, None

    def set(self, key: str, value: ActiveModel | None) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_MODEL_CACHE = _LRU(_CACHE_MAX)

# Serialise concurrent cache-miss loads per profile so N threads only read
# from disk once (RP2 in LLD-02 §4.3).
_load_locks_lock = threading.Lock()
_load_locks: dict[str, threading.Lock] = {}


def _get_load_lock(profile_id: str) -> threading.Lock:
    with _load_locks_lock:
        lock = _load_locks.get(profile_id)
        if lock is None:
            lock = threading.Lock()
            _load_locks[profile_id] = lock
        return lock


def invalidate(profile_id: str | None = None) -> None:
    """Drop a profile (or all) from the cache — used on shutdown + tests."""
    if profile_id is None:
        _MODEL_CACHE.clear()
        return
    _MODEL_CACHE.invalidate(profile_id)


def load_active(
    db: Any, profile_id: str,
    *,
    use_cache: bool = True,
) -> ActiveModel | None:
    """Load the active model for ``profile_id`` with integrity verification.

    Args:
        db: A ``LearningDatabase`` instance that exposes
            ``load_active_model(profile_id)`` returning a dict with keys
            ``state_bytes``, ``bytes_sha256``, ``feature_names`` (JSON str),
            and ``trained_at``. None if no active row.
        profile_id: Profile key.
        use_cache: If False, bypasses the LRU — used in tests.

    Returns:
        An ``ActiveModel`` on success, or ``None`` when no active row,
        integrity check fails, or lightgbm is unavailable.
    """
    if use_cache:
        hit, value = _MODEL_CACHE.get(profile_id)
        if hit:
            return value

    lock = _get_load_lock(profile_id)
    with lock:
        # Double-checked: another thread may have populated the cache.
        if use_cache:
            hit, value = _MODEL_CACHE.get(profile_id)
            if hit:
                return value

        try:
            row = db.load_active_model(profile_id)
        except Exception as exc:
            logger.warning("model_cache: load_active_model raised: %s", exc)
            if use_cache:  # pragma: no cover — covered by two paths in tests
                _MODEL_CACHE.set(profile_id, None)
            return None

        if row is None:
            if use_cache:
                _MODEL_CACHE.set(profile_id, None)
            return None

        model = _parse_row(profile_id, row)
        # Tombstone (write-back None) on integrity/parse failure so we don't
        # retry hot. Dashboard phase computation relies on this.
        if use_cache:
            _MODEL_CACHE.set(profile_id, model)
        return model


def _parse_row(profile_id: str, row: dict) -> ActiveModel | None:
    """Verify + deserialise a single model row. Never raises."""
    state_bytes = row.get("state_bytes")
    sha_hex = row.get("bytes_sha256") or ""
    feature_names_json = row.get("feature_names") or "[]"
    trained_at = row.get("trained_at") or ""

    if not state_bytes:
        logger.warning("model_cache: empty state_bytes for %s", profile_id)
        return None
    if not isinstance(state_bytes, (bytes, bytearray)):
        # Some drivers return buffer-like; coerce once.
        try:
            state_bytes = bytes(state_bytes)
        except Exception as exc:  # pragma: no cover — defensive
            logger.error("model_cache: cannot coerce state_bytes: %s", exc)
            return None

    # M2: SHA-256 verify BEFORE touching LightGBM.
    try:
        verify_sha256(bytes(state_bytes), sha_hex)
    except IntegrityError as exc:
        logger.critical(
            "model_cache: SHA-256 mismatch for %s → tombstone: %s",
            profile_id, exc,
        )
        return None

    # Parse feature_names for drift reporting (M3).
    try:
        names = tuple(json.loads(feature_names_json))
    except (ValueError, TypeError) as exc:
        logger.warning(
            "model_cache: bad feature_names JSON for %s: %s",
            profile_id, exc,
        )
        names = ()

    if names and names != _CURRENT_FEATURE_NAMES:
        logger.info(
            "feature-drift: active model for %s has %d names; current has %d",
            profile_id, len(names), len(_CURRENT_FEATURE_NAMES),
        )

    # Native LightGBM — NOT pickle (M1).
    try:
        import lightgbm as lgb  # noqa: PLC0415 — optional dep
    except ImportError:  # pragma: no cover — optional dep
        logger.info(
            "model_cache: lightgbm unavailable; phase 3 disabled for %s",
            profile_id,
        )
        return None

    try:
        booster = lgb.Booster(model_str=bytes(state_bytes).decode("utf-8"))
    except Exception as exc:  # pragma: no cover — corrupt decode
        logger.critical(
            "model_cache: Booster parse failed for %s → tombstone: %s",
            profile_id, exc,
        )
        return None

    return ActiveModel(
        profile_id=profile_id,
        booster=booster,
        feature_names=names or _CURRENT_FEATURE_NAMES,
        trained_at=str(trained_at),
        sha256=sha_hex.lower(),
    )


# ---------------------------------------------------------------------------
# Feature-drift policy (M3 — LLD-02 §4.5)
# ---------------------------------------------------------------------------


def drift_mode(model: ActiveModel) -> str:
    """Classify feature-name drift for a loaded model.

    Returns one of:
        ``"aligned"`` — names match FEATURE_NAMES exactly.
        ``"subset"``  — active names ⊆ current → pad zeros at inference.
        ``"unknown"`` — active names have entries not in current → refuse.
    """
    active = tuple(model.feature_names)
    current = _CURRENT_FEATURE_NAMES
    if active == current:
        return "aligned"
    current_set = set(current)
    if all(n in current_set for n in active):
        return "subset"
    return "unknown"


__all__ = (
    "ActiveModel",
    "load_active",
    "invalidate",
    "drift_mode",
)
