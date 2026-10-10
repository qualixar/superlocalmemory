# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Keep the scratch files of unfinished upload links from piling up.

The upload-link store cleans up when a link is made and when an upload starts, but a
computer where nobody makes links for days would keep an abandoned half-file until then.
This loop runs for the daemon's whole life and cleans at least hourly. A failure is
logged and never stops the loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

INTERVAL_S = 3_600.0


async def run(links: Callable[[], Any] | None = None, *, interval_s: float = INTERVAL_S) -> None:
    if links is None:
        from superlocalmemory.media.upload_links import default_links as links
    while True:
        try:
            await asyncio.to_thread(links().cleanup)
        except Exception as exc:  # noqa: BLE001 - housekeeping must never stop
            logger.warning("upload scratch cleanup failed (%s)", type(exc).__name__)
        await asyncio.sleep(interval_s)
