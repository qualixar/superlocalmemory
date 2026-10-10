# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The shared RAM reservation a document job takes around each parse step."""

from __future__ import annotations

from typing import ContextManager

from superlocalmemory.core.ram_lock import ram_reservation

#: Memory a page render and read wants on top of what is already in use (an estimate,
#: well under the parse child's own cap).
PARSE_STEP_MB = 600
#: Long enough for a picture-model load or a re-index model load to finish first.
PARSE_WAIT_S = 120.0


def parse_reservation() -> ContextManager[None]:
    """Only one heavy job at a time: a parse step waits for a model load, and the other way round."""
    return ram_reservation("media-document-parse", required_mb=PARSE_STEP_MB, timeout_s=PARSE_WAIT_S)
