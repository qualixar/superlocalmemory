"""Scratch files of unfinished uploads are cleaned by a loop that runs for the daemon's whole life."""

from __future__ import annotations

import asyncio

import pytest

from superlocalmemory.server import upload_housekeeping as hk


class Links:
    def __init__(self, fail_first=False):
        self.calls, self.fail_first = 0, fail_first

    def cleanup(self):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise OSError("disk busy")
        return 0


@pytest.mark.asyncio
async def test_the_loop_cleans_at_every_interval_and_survives_a_failure():
    links = Links(fail_first=True)
    task = asyncio.create_task(hk.run(lambda: links, interval_s=0.01))
    await asyncio.sleep(0.15)
    assert links.calls >= 3 and not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_the_interval_is_at_most_an_hour_and_the_daemon_starts_and_stops_the_loop():
    assert 0 < hk.INTERVAL_S <= 3600
    from pathlib import Path

    source = (Path(hk.__file__).parent / "unified_daemon.py").read_text()
    assert "upload_housekeeping" in source and "_upload_housekeeping_task" in source
    assert source.count("_upload_housekeeping_task") >= 3  # created, cancelled at shutdown, guarded
