# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What the picture model is using, for the dashboard. Reading starts and creates nothing."""

from __future__ import annotations

from typing import Any

from superlocalmemory.runtimes import media_models

_MB = 1024 * 1024


def _rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    try:
        import psutil

        return round(psutil.Process(pid).memory_info().rss / _MB, 1)
    except Exception:  # noqa: BLE001 - a process that is gone is "not running"
        return None


def _system_total_mb() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total / _MB)
    except Exception:  # noqa: BLE001 - an unreadable total is 0 (unknown)
        return 0


def _pick(clients: list[Any]) -> Any | None:
    """The running worker if there is one, else any known client."""
    running = [c for c in clients if getattr(c, "pid", None) is not None]
    return (running or clients or [None])[0]


def ram_view() -> dict[str, Any]:
    """``{system_total_mb, worker_rss_mb (None when not running), worker_cap_mb, model}``."""
    from superlocalmemory.runtimes.worker_client import live_clients

    client = _pick(live_clients())
    model = str(getattr(client, "model_id", "") or "")
    cap = int(getattr(client, "rss_limit_mb", 0) or 0) if client is not None else 0
    if not cap and model:
        cap = media_models.effective_rss_limit_mb(model)
    return {"system_total_mb": _system_total_mb(),
            "worker_rss_mb": _rss_mb(getattr(client, "pid", None)),
            "worker_cap_mb": cap, "model": model}
