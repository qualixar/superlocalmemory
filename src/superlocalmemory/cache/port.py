# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""The small public port every cache backend implements."""

from __future__ import annotations

from typing import Protocol

from superlocalmemory.cache.keys import CacheKey

PAYLOAD_KINDS = ("text", "vector_f32", "json", "png")


def check_payload(payload: object, kind: str) -> None:
    """Refuse anything but raw bytes of a known kind."""
    if kind not in PAYLOAD_KINDS:
        raise ValueError(f"unknown payload kind {kind!r}; expected one of {PAYLOAD_KINDS}")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("a cache payload must be bytes")


class CachePort(Protocol):
    def get(self, key: CacheKey) -> bytes | None: ...

    def put(self, key: CacheKey, payload: bytes, *, kind: str) -> None: ...

    def invalidate(self, *, deriver_id: str | None = None,
                   model_id: str | None = None) -> int: ...

    def clear(self) -> None: ...

    def stats(self) -> dict: ...
