# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A short description of a picture, written by a small local model (WS-8, flag ``media.captions``).

Fail-open: ``describe`` returns ``""`` when the part is off, its model is not on this computer, the worker
is busy, starting or too large for this moment, or anything else goes wrong, and says why in the log (the
kind of problem only, never the picture or the words). A save never waits on it past one request and never
fails because of it: the picture is saved without a description.

The description is the model's words, so it goes through the same redaction as text read from a picture and
is saved under a label that says so (``labels.CAPTION``). Everything saved from a picture already has the
origin ``model_generated`` (``trust/origin_class``, row ``media.model``, D-19), so the description joins
the memory under the same origin; a reader is never shown it as the person's own words.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Longest description kept (routing-only: it is "a few words").
MAX_CAPTION_CHARS = 400
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def tidy(text: str) -> str:
    """The description as plain words: no control characters, one line, cut at the limit."""
    return re.sub(r"\s+", " ", _CONTROL.sub("", str(text or ""))).strip()[:MAX_CAPTION_CHARS]


def describe(root: Path | str, image: Path | str, *, redact: Callable[[str], str] | None = None,
             wait_cold: bool = True, client: Any = None) -> str:
    """The description of ``image``, or ``""`` for any reason at all. Never raises.

    ``client`` is for tests and callers that already hold one; otherwise the pack's own worker client is used
    (``None`` unless the part is on and its model is installed).
    """
    try:
        if client is None:
            from superlocalmemory.runtimes import media_packs

            if not media_packs.is_active("captions", root):
                return ""
            client = media_packs.pack_client("captions", root)
        if client is None:
            return ""
        text = tidy(client.caption_image(image, wait_cold=wait_cold))
        if text and redact is not None:
            text = tidy(redact(text))
        return text
    except Exception as exc:  # noqa: BLE001 - a description never blocks a save
        logger.warning("picture description skipped (%s)", type(exc).__name__)
        return ""


__all__ = ["MAX_CAPTION_CHARS", "describe", "tidy"]
