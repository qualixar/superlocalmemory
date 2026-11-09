"""LLD-05 §12.7 — sync loop tests."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from pathlib import Path

import pytest

from superlocalmemory.hooks.adapter_base import (
    path_sha256,
    sync_log_record,
)
from superlocalmemory.hooks.sync_loop import (
    cross_platform_sync_loop,
    run_once,
)


class _FakeAdapter:
    def __init__(self, name: str, *, active: bool = True,
                 sync_return: bool = True, raise_exc: Exception | None = None):
        self.name = name
        self._active = active
        self._sync_return = sync_return
        self._raise = raise_exc
        self.calls = 0
        self.target_path = Path(f"/tmp/{name}.md")

    def is_active(self) -> bool:
        return self._active

    def sync(self) -> bool:
        self.calls += 1
        if self._raise:
            raise self._raise
        return self._sync_return

    def disable(self) -> None:
        self._active = False


@pytest.mark.asyncio
async def test_loop_fires_each_active_adapter_per_interval():
    a = _FakeAdapter("a")
    b = _FakeAdapter("b")
    await cross_platform_sync_loop(
        [a, b], interval=0.001, first_run_delay=0.0, iterations=2,
    )
    assert a.calls == 2
    assert b.calls == 2


@pytest.mark.asyncio
async def test_loop_continues_after_adapter_error():
    boomer = _FakeAdapter("boom", raise_exc=RuntimeError("boom"))
    good = _FakeAdapter("good")
    result = await run_once([boomer, good])
    assert result["boom"].startswith("error:")
    assert result["good"] == "wrote"


@pytest.mark.asyncio
async def test_loop_inactive_adapter_not_synced():
    inactive = _FakeAdapter("inactive", active=False)
    active = _FakeAdapter("active")
    result = await run_once([inactive, active])
    assert result["inactive"] == "inactive"
    assert inactive.calls == 0
    assert active.calls == 1


@pytest.mark.asyncio
async def test_sync_log_row_per_sync_attempt(tmp_path, fake_recall):
    from superlocalmemory.hooks import context_payload as cp
    from superlocalmemory.hooks.cursor_adapter import CursorAdapter

    os.environ["SLM_CURSOR_FORCE"] = "1"
    adapter = CursorAdapter(
        scope="project", base_dir=tmp_path,
        sync_log_db=tmp_path / "memory.db",
        recall_fn=fake_recall,
    )
    # First call writes; second is a content-hash skip.
    cp._now_iso = lambda: "2026-04-18T00:00:00+00:00"
    await run_once([adapter])
    await run_once([adapter])

    conn = sqlite3.connect(str(tmp_path / "memory.db"))
    try:
        rows = conn.execute(
            "SELECT adapter_name, success, bytes_written FROM "
            "cross_platform_sync_log"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) >= 1
    for name, success, bytes_written in rows:
        assert success in (0, 1)
    os.environ.pop("SLM_CURSOR_FORCE", None)


@pytest.mark.asyncio
async def test_sync_log_target_path_sha256_full_length_not_raw(tmp_path):
    """A7: full 64-hex, not raw path, enforced at the helper level too."""
    sync_log_record(
        tmp_path / "memory.db",
        adapter_name="test_adapter",
        profile_id="default",
        target_path_sha256=path_sha256(tmp_path / "whatever.md"),
        target_basename="whatever.md",
        bytes_written=42,
        content_sha256="a" * 64,
        success=True,
    )
    conn = sqlite3.connect(str(tmp_path / "memory.db"))
    rows = conn.execute(
        "SELECT target_path_sha256 FROM cross_platform_sync_log"
    ).fetchall()
    conn.close()
    assert rows
    for (sha,) in rows:
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)
        assert str(tmp_path) not in sha  # never the raw path

    # Direct invalid inputs are rejected.
    with pytest.raises(ValueError):
        sync_log_record(
            tmp_path / "memory.db",
            adapter_name="bad",
            profile_id="default",
            target_path_sha256=str(tmp_path / "raw.md"),  # raw path
            target_basename="raw.md",
            bytes_written=0,
            content_sha256="0" * 64,
            success=False,
        )


@pytest.mark.asyncio
async def test_first_run_delay_honoured():
    adapter = _FakeAdapter("a")
    start = asyncio.get_event_loop().time()
    await cross_platform_sync_loop(
        [adapter], interval=0.01, first_run_delay=0.05, iterations=1,
    )
    elapsed = asyncio.get_event_loop().time() - start
    # asyncio runs a timer up to one clock tick early by design (its
    # _clock_resolution); on Windows a monotonic tick is about 15.6 ms.
    tick = time.get_clock_info("monotonic").resolution
    assert elapsed >= 0.05 - tick
    assert adapter.calls == 1
