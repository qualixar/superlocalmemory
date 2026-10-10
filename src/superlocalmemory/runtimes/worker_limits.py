# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What may be sent to the picture worker, and the hard memory limit it starts with.

The worker enforces the same input limits itself (``multimodal_worker.py`` cannot import this
file: it runs in the managed environment without superlocalmemory), but only after the data
has been sent and, for a picture, read. The daemon checks first so an oversized request never
reaches the process. A test pins every number here to the worker's.

The hard limit is the platform's own ceiling on the worker, set by the worker on itself the
moment it starts: ``RLIMIT_DATA`` on Linux, which caps private writable memory (the model's
tensors and a decoded picture) and not the address space that file-mapped weights and thread
stacks inflate, so a large allocation fails inside the worker instead of taking the machine's
memory. macOS does not enforce either limit; there the client's watchdog is the guard.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

#: Same values as multimodal_worker.MAX_TEXT_CHARS / MAX_FILE_BYTES and media_image_ops.MAX_PIXELS.
MAX_TEXT_CHARS = 8_000
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 50_000_000

DATA_LIMIT_ENV = "SLM_MEDIA_WORKER_DATA_LIMIT_MB"
#: The hard ceiling sits above the cap the client watches, so a request that is merely at
#: the cap is stopped by the watchdog with a clear message and only a runaway one hits the ceiling.
HARD_LIMIT_FACTOR = 1.5
HARD_LIMIT_HEADROOM_MB = 1024

_HEADER_BYTES = 64
_JPEG_SCAN_BYTES = 1024 * 1024
_JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


class InputTooLarge(ValueError):
    """A request the worker would refuse; the text is plain language."""


def data_limit_env(cap_mb: int, *, platform: str | None = None) -> dict[str, str]:
    """The environment entry that gives the worker its hard limit; empty where none is enforced."""
    if cap_mb <= 0 or not (sys.platform if platform is None else platform).startswith("linux"):
        return {}
    return {DATA_LIMIT_ENV: str(int(cap_mb * HARD_LIMIT_FACTOR) + HARD_LIMIT_HEADROOM_MB)}


def check_texts(texts: list[str]) -> None:
    for text in texts:
        if len(text) > MAX_TEXT_CHARS:
            raise InputTooLarge(f"That text is too long to process (over {MAX_TEXT_CHARS:,} characters).")


def check_image_file(path: str | Path) -> None:
    """Refuse a picture the worker would refuse. A file that cannot be read is left to the worker."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    if size > MAX_FILE_BYTES:
        raise InputTooLarge(f"That file is too large to process (over {MAX_FILE_BYTES // (1024 * 1024)} MB).")
    pixels = image_pixels(path)
    if pixels is not None and pixels > MAX_PIXELS:
        raise InputTooLarge(f"That picture is too large to process (over {MAX_PIXELS // 1_000_000} million pixels).")


def _png(head: bytes) -> tuple[int, int] | None:
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        return struct.unpack(">II", head[16:24])
    return None


def _gif(head: bytes) -> tuple[int, int] | None:
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return struct.unpack("<HH", head[6:10])
    return None


def _webp(head: bytes) -> tuple[int, int] | None:
    if head[:4] != b"RIFF" or head[8:12] != b"WEBP":
        return None
    kind = head[12:16]
    if kind == b"VP8X":
        return (int.from_bytes(head[24:27], "little") + 1, int.from_bytes(head[27:30], "little") + 1)
    if kind == b"VP8L" and head[20:21] == b"\x2f":
        bits = struct.unpack("<I", head[21:25])[0]
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if kind == b"VP8 " and head[23:26] == b"\x9d\x01\x2a":
        width, height = struct.unpack("<HH", head[26:30])
        return (width & 0x3FFF, height & 0x3FFF)
    return None


def _jpeg(fh) -> tuple[int, int] | None:
    if fh.read(2) != b"\xff\xd8":
        return None
    consumed = 2
    while consumed < _JPEG_SCAN_BYTES:
        marker = fh.read(2)
        if len(marker) < 2 or marker[0] != 0xFF:
            return None
        kind = marker[1]
        if kind in (0xD8, 0x01) or 0xD0 <= kind <= 0xD7:
            consumed += 2
            continue
        raw = fh.read(2)
        if len(raw) < 2:
            return None
        length = struct.unpack(">H", raw)[0]
        if kind in _JPEG_SOF:
            body = fh.read(5)
            if len(body) < 5:
                return None
            height, width = struct.unpack(">HH", body[1:5])
            return (width, height)
        if kind == 0xDA or length < 2:
            return None
        fh.seek(length - 2, os.SEEK_CUR)
        consumed += 2 + length
    return None


def image_pixels(path: str | Path) -> int | None:
    """Width times height read from the file header alone; None when the format or header is not recognised."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(_HEADER_BYTES)
            size = _png(head) or _gif(head) or _webp(head)
            if size is None and head[:2] == b"\xff\xd8":
                fh.seek(0)
                size = _jpeg(fh)
    except (OSError, struct.error):
        return None
    return size[0] * size[1] if size is not None else None


__all__ = ["DATA_LIMIT_ENV", "InputTooLarge", "MAX_FILE_BYTES", "MAX_PIXELS", "MAX_TEXT_CHARS", "check_image_file",
           "check_texts", "data_limit_env", "image_pixels"]
