# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The JSON-lines channel between a model worker and the process that started it.

The embedding, recall and reranker workers answer on their stdout. A model
library that prints (a native warning in the console's legacy code page, a
progress line) writes to file descriptor 1, which is that reply pipe, and the
parent then read bytes that are not UTF-8 between two replies. Its strict
decode failed the request, and through it the save being enriched.

:func:`claim_protocol_channel` keeps the pipe for replies only: replies go to a
private copy of descriptor 1, while descriptor 1 itself and ``sys.stdout`` are
pointed at stderr, which the parent discards. Both directions are UTF-8. A
worker calls it first, before any model library is imported.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO


def claim_protocol_channel() -> TextIO:
    """The stream to write protocol replies to (UTF-8, line-buffered)."""
    try:
        reply_fd = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        return sys.stdout  # no real descriptors (embedded use): nothing to move
    channel = open(reply_fd, "w", encoding="utf-8", newline="\n", buffering=1)  # noqa: SIM115
    sys.stdout = sys.stderr
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    return channel


__all__ = ["claim_protocol_channel"]
