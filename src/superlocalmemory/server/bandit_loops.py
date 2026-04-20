# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §3.5 + §3.6

"""Background schedulers for the v3.4.22 contextual bandit.

Two asyncio tasks, both registered in the daemon lifespan:

  1. Reward-proxy settler — every 60 s (``SLM_BANDIT_REWARD_WINDOW_SEC``),
     calls ``reward_proxy.settle_stale_plays`` for the configured profile(s).
  2. Retention sweep — every 24 h
     (``SLM_BANDIT_PLAYS_RETENTION_INTERVAL_SEC``), calls
     ``bandit.retention_sweep`` with the configured horizon
     (``SLM_BANDIT_PLAYS_RETENTION_DAYS``, default 7).

Both honour ``SLM_BANDIT_DISABLED=1`` (caller checks before scheduling).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REWARD_INTERVAL = float(
    os.environ.get("SLM_BANDIT_REWARD_WINDOW_SEC", "60"),
)
_RETENTION_INTERVAL = float(
    os.environ.get("SLM_BANDIT_PLAYS_RETENTION_INTERVAL_SEC", "86400"),
)
_RETENTION_DAYS = int(
    os.environ.get("SLM_BANDIT_PLAYS_RETENTION_DAYS", "7"),
)


def _learning_db(config: Any) -> Path:
    if config is not None:
        cand = getattr(config, "learning_db_path", None)
        if cand is not None:
            return Path(cand)
    return Path.home() / ".superlocalmemory" / "learning.db"


def _memory_db(config: Any) -> Path:
    if config is not None:
        cand = getattr(config, "db_path", None)
        if cand is not None:
            return Path(cand)
    return Path.home() / ".superlocalmemory" / "memory.db"


def _profile_id(config: Any) -> str:
    if config is not None:
        pid = getattr(config, "default_profile", None)
        if isinstance(pid, str) and pid:
            return pid
    return "default"


async def _reward_proxy_loop(
    learning_db: Path, memory_db: Path, profile_id: str,
    interval_sec: float,
) -> None:
    """Run the proxy settler on a steady interval. Never raises."""
    from superlocalmemory.learning.reward_proxy import settle_stale_plays

    while True:
        try:
            await asyncio.sleep(interval_sec)
            # The settler is synchronous + fast; run in a thread to avoid
            # blocking the event loop on unusual DB lock stalls.
            n = await asyncio.to_thread(
                settle_stale_plays,
                profile_id, learning_db, memory_db,
            )
            if n:
                logger.debug("bandit.reward_proxy settled=%d", n)
        except asyncio.CancelledError:  # pragma: no cover — lifecycle
            raise
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("bandit.reward_proxy loop: %s", exc)


async def _retention_loop(
    learning_db: Path, interval_sec: float, retention_days: int,
) -> None:
    """Run retention_sweep on a 24h cadence. Never raises."""
    from superlocalmemory.learning.bandit import retention_sweep

    while True:
        try:
            await asyncio.sleep(interval_sec)
            deleted = await asyncio.to_thread(
                retention_sweep, learning_db, retention_days,
            )
            logger.info(
                "bandit_plays_retention_sweep tick: deleted=%d", deleted,
            )
        except asyncio.CancelledError:  # pragma: no cover — lifecycle
            raise
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("bandit.retention loop: %s", exc)


def schedule_bandit_loops(application: Any, config: Any) -> None:
    """Register both background tasks with the FastAPI app state.

    Tasks are stored on ``application.state.bandit_tasks`` so the daemon's
    shutdown path can cancel them cleanly (if added).
    """
    learning = _learning_db(config)
    memory = _memory_db(config)
    profile = _profile_id(config)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:  # pragma: no cover — defensive
        return

    tasks = []
    tasks.append(loop.create_task(
        _reward_proxy_loop(learning, memory, profile, _REWARD_INTERVAL),
    ))
    tasks.append(loop.create_task(
        _retention_loop(learning, _RETENTION_INTERVAL, _RETENTION_DAYS),
    ))
    if hasattr(application, "state"):
        application.state.bandit_tasks = tasks
    logger.info(
        "bandit loops scheduled: reward=%.0fs, retention=%.0fs, "
        "retention_days=%d",
        _REWARD_INTERVAL, _RETENTION_INTERVAL, _RETENTION_DAYS,
    )


__all__ = ("schedule_bandit_loops",)
