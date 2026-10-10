# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Link pictures to their memories without anyone running ``slm media gc``.

A picture saved while its memory was still queued is kept with no anchor memory; the memory,
once it commits, names the picture. This loop runs for the daemon's whole life, looks for such
pictures and links them (``media.gc.fill_anchors``, which never deletes anything). A failure is
logged and never stops the loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

INTERVAL_S = 600.0
FIRST_DELAY_S = 60.0


async def run(data_root: str | Path | None = None, *, fill: Callable[..., int] | None = None,
              interval_s: float = INTERVAL_S, first_delay_s: float = FIRST_DELAY_S) -> None:
    if fill is None:
        from superlocalmemory.media.gc import fill_anchors as fill
    await asyncio.sleep(first_delay_s)
    while True:
        try:
            linked = await asyncio.to_thread(lambda: fill(data_root=data_root))
            if linked:
                logger.info("media: %d pictures linked to their memories", linked)
        except Exception as exc:  # noqa: BLE001 - housekeeping must never stop
            logger.warning("picture anchor pass failed (%s)", type(exc).__name__)
        await asyncio.sleep(interval_s)
