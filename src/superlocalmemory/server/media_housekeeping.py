# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Link pictures to their memories without anyone running ``slm media gc``.

A picture saved while its memory was still queued is kept with no anchor memory; the memory,
once it commits, names the picture. This loop runs for the daemon's whole life, looks for such
pictures and links them (``media.gc.fill_anchors``, which never deletes anything). A failure is
logged and never stops the loop.

After an upgrade it also takes a second look, in small batches, at the pictures the 4.1.25 upgrade
held back from web apps (``media.revet``). Progress is recorded on each picture, so a restart
resumes it; when nothing is left to look at a pass is a single cheap query.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INTERVAL_S = 600.0
FIRST_DELAY_S = 60.0
REVET_BATCH = 50
#: Rest between batches of the second look, so it never competes with a person using SLM.
REVET_PAUSE_S = 2.0


def _default_revet(**kwargs: Any) -> Any:
    """Imported on first use, in the worker thread, so start-up never waits for it."""
    from superlocalmemory.media.revet import revet_remote_flags

    return revet_remote_flags(**kwargs)


async def _revet(data_root: str | Path | None, revet: Callable[..., Any], batch: int,
                 pause_s: float) -> None:
    """Look again at held-back pictures, a batch at a time, until none is left to look at."""
    while True:
        report = await asyncio.to_thread(lambda: revet(data_root=data_root, max_items=batch))
        if report.cleared:
            logger.info("media: %d held-back pictures are now offered to web apps", report.cleared)
        if not report.more:
            return
        await asyncio.sleep(pause_s)


async def run(data_root: str | Path | None = None, *, fill: Callable[..., int] | None = None,
              interval_s: float = INTERVAL_S, first_delay_s: float = FIRST_DELAY_S,
              revet: Callable[..., Any] | None = None, revet_batch: int = REVET_BATCH,
              revet_pause_s: float = REVET_PAUSE_S) -> None:
    if fill is None:
        from superlocalmemory.media.gc import fill_anchors as fill
    revet = revet or _default_revet
    await asyncio.sleep(first_delay_s)
    while True:
        try:
            linked = await asyncio.to_thread(lambda: fill(data_root=data_root))
            if linked:
                logger.info("media: %d pictures linked to their memories", linked)
        except Exception as exc:  # noqa: BLE001 - housekeeping must never stop
            logger.warning("picture anchor pass failed (%s)", type(exc).__name__)
        try:
            await _revet(data_root, revet, revet_batch, min(revet_pause_s, interval_s))
        except Exception as exc:  # noqa: BLE001 - housekeeping must never stop
            logger.warning("held-back picture pass failed (%s)", type(exc).__name__)
        await asyncio.sleep(interval_s)
