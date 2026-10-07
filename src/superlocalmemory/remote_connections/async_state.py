"""Complete local state mutations before releasing a cancelled lifecycle lock."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def finish_on_cancel(operation: Awaitable[T]) -> T:
    task = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Threads cannot be cancelled. A second cancellation must not release
        # the owning lifecycle lock while that thread can still mutate state.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # Observe a failure without replacing cancellation.
        raise


async def mutate(operation: Callable[..., T], *args, **kwargs) -> T:
    return await finish_on_cancel(asyncio.to_thread(operation, *args, **kwargs))
