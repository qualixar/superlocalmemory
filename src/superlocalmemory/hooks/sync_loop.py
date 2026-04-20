# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-05 §9

"""Background sync loop — keeps every cross-platform adapter fresh.

LLD-05 §9. Runs as an asyncio task in the unified daemon's lifespan.
Default interval 900 s; first run at t=5 s so users see files quickly.
Adapter errors are logged but NEVER abort the loop (A8).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Iterable

from superlocalmemory.hooks.adapter_base import Adapter

logger = logging.getLogger(__name__)


DEFAULT_INTERVAL_SECONDS = 900
FIRST_RUN_DELAY_SECONDS = 5.0


def _interval_from_env(default: int = DEFAULT_INTERVAL_SECONDS) -> int:
    raw = os.environ.get("SLM_CROSS_PLATFORM_SYNC_INTERVAL")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(30, value)


async def cross_platform_sync_loop(
    adapters: Iterable[Adapter],
    *,
    interval: float | None = None,
    first_run_delay: float = FIRST_RUN_DELAY_SECONDS,
    iterations: int | None = None,
) -> int:
    """Top-level coroutine. Returns number of iterations run.

    ``iterations`` cap is used by tests to bound the loop. Production
    callers pass ``None`` and rely on task cancellation for shutdown.
    """
    adapters = list(adapters)
    step = float(interval) if interval is not None else float(_interval_from_env())
    await asyncio.sleep(first_run_delay)

    count = 0
    while True:
        await run_once(adapters)
        count += 1
        if iterations is not None and count >= iterations:
            return count
        try:
            await asyncio.sleep(step)
        except asyncio.CancelledError:  # pragma: no cover — shutdown
            raise


async def run_once(adapters: Iterable[Adapter]) -> dict[str, str]:
    """Run a single sync cycle over all adapters. Never raises.

    E.3 (v3.4.22 perf): ``adapter.sync()`` is synchronous file I/O
    (opens/reads/writes JSON files in ~/.cursor, ~/.antigravity, etc.)
    and used to run directly on the event loop — a slow disk or a
    large workspace could block the daemon for tens of milliseconds,
    stalling every concurrent request. We now off-load each sync to
    the default thread pool via ``asyncio.to_thread``.
    """
    results: dict[str, str] = {}

    def _one(adapter: Adapter) -> tuple[str, str, float]:
        """Sync a single adapter; returns (name, outcome, elapsed_ms)."""
        name = getattr(adapter, "name", "?")
        try:
            if not adapter.is_active():
                return name, "inactive", 0.0
            start = time.monotonic()
            wrote = adapter.sync()
            elapsed_ms = (time.monotonic() - start) * 1000
            return name, ("wrote" if wrote else "skipped"), elapsed_ms
        except Exception as exc:
            logger.warning("adapter %s sync failed: %s", name, exc)
            return name, f"error:{type(exc).__name__}", 0.0

    for adapter in adapters:
        name, outcome, elapsed_ms = await asyncio.to_thread(_one, adapter)
        results[name] = outcome
        if outcome in ("wrote", "skipped"):
            logger.debug("adapter %s sync %s (%.1f ms)",
                         name, outcome, elapsed_ms)
    return results


def schedule(adapters: Iterable[Adapter]) -> asyncio.Task:
    """Fire-and-forget scheduling for the daemon lifespan."""
    return asyncio.create_task(cross_platform_sync_loop(adapters))


__all__ = (
    "DEFAULT_INTERVAL_SECONDS",
    "FIRST_RUN_DELAY_SECONDS",
    "cross_platform_sync_loop",
    "run_once",
    "schedule",
)
