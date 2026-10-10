# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The self-check run at the end of setup: one real text and one real image through the worker."""

from __future__ import annotations

import logging
import math
import struct
import tempfile
import zlib
from pathlib import Path

from superlocalmemory.runtimes.worker_client import MediaWorkerClient

logger = logging.getLogger(__name__)

EXPECTED_DIM = 768
_NORM_TOLERANCE = 1e-3


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def tiny_png() -> bytes:
    """An 8x8 RGB gradient, written with the standard library only (no imaging package)."""
    rows = b"".join(b"\x00" + bytes(v for x in range(8) for v in (x * 32, y * 32, 128)) for y in range(8))
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(rows)) + _chunk(b"IEND", b""))


class _PythonOnly:
    """Just enough of a managed environment to start the worker on a given interpreter."""

    def __init__(self, python: Path) -> None:
        self._python = Path(python)
        parents = self._python.parents
        self.root = parents[2] if len(parents) > 2 else self._python.parent

    def python(self) -> Path:
        return self._python

    def weights_dir(self) -> Path:
        return self.root / "weights"


def _unit(vec: list[float]) -> bool:
    return (len(vec) == EXPECTED_DIM and all(math.isfinite(x) for x in vec)
            and abs(math.sqrt(sum(x * x for x in vec)) - 1.0) <= _NORM_TOLERANCE)


def media_canary(python: Path, *, model_id: str | None = None, revision: str | None = None) -> bool:
    """True when the worker on ``python`` embeds a text and an image sensibly. Never raises."""
    if model_id is None:
        from superlocalmemory.runtimes.media_env import MEDIA_MODEL_REPO, MEDIA_MODEL_REVISION

        model_id, revision = MEDIA_MODEL_REPO, MEDIA_MODEL_REVISION
    client = MediaWorkerClient(_PythonOnly(python), model_id=model_id, revision=revision or "",
                               idle_s=60.0, rss_limit_mb=0)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "canary.png"
            image.write_bytes(tiny_png())
            text_vec = client.embed_texts(["a small gradient"], prompt="Document")[0]
            image_vec = client.embed_images([image])[0]
        return _unit(text_vec) and _unit(image_vec) and text_vec != image_vec
    except Exception as exc:  # noqa: BLE001 - any failure is a failed check
        logger.warning("media self-check failed: %s", exc)
        return False
    finally:
        client.stop()


__all__ = ["media_canary", "tiny_png"]
