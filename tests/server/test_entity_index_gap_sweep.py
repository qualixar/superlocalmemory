# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Q7 (2026-10-06): the daemon's background gap sweep, end to end.

``run_entity_index_backfill`` still returns once the one-time backfill
completes -- an existing, tested contract
(``tests/test_retrieval/test_bridge_determinism.py::
TestTheBackfill::test_the_daemon_loop_publishes_progress_and_finishes``).
``run_entity_index_gap_sweep`` is the separate, never-terminal task chained
after it (see ``server/unified_daemon.py``'s
``_fact_entity_association_repair_loop``) that catches a coverage gap
introduced after the backfill already reported complete.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_retrieval.test_bridge_determinism import build_store

from superlocalmemory.server.entity_index_repair import (
    MAX_TICK_SECONDS,
    gap_sweep_delay,
    run_entity_index_backfill,
    run_entity_index_gap_sweep,
)
from superlocalmemory.storage import entity_index
from superlocalmemory.storage.models import AtomicFact


def _has_pair(store, fact_id: str, entity_id: str) -> bool:
    rows = store.execute(
        "SELECT 1 FROM fact_entity_associations WHERE fact_id=? AND entity_id=?",
        (fact_id, entity_id),
    )
    return bool(rows)


class TestGapSweepEndToEnd:
    def test_the_sweep_covers_a_fact_the_backfill_already_reported_complete_without(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "old.db"
        store = build_store(path)
        app = SimpleNamespace(state=SimpleNamespace())

        # The one-time backfill runs to completion against what exists now.
        asyncio.run(run_entity_index_backfill(app, path, batch_size=50, tick_seconds=0.0))
        assert app.state.entity_index_status["state"] == "complete"

        # A fact written by a path that bypasses the real-time indexing
        # hook, after the backfill already completed -- the Q7 gap.
        store.store_fact(AtomicFact(
            fact_id="gap-fact", memory_id="m0", content="gap",
            canonical_entities=["E00"],
        ))
        store.execute(
            "DELETE FROM fact_entity_associations WHERE fact_id='gap-fact'",
        )
        assert not _has_pair(store, "gap-fact", "E00")
        # The one-time backfill genuinely cannot see it: its own target is
        # frozen at the snapshot it took before this fact existed.
        again = entity_index.backfill(path, batch_size=50, max_batches=5)
        assert again["complete"] is True and again["inserted"] == 0
        assert not _has_pair(store, "gap-fact", "E00")

        async def _run_until_covered_or_timeout() -> None:
            task = asyncio.create_task(
                run_entity_index_gap_sweep(app, path, batch_size=50, tick_seconds=0.0),
            )
            try:
                for _ in range(500):
                    # covered AND reported: the status is published after the
                    # write, so waiting only for the pair races the report.
                    if (_has_pair(store, "gap-fact", "E00")
                            and getattr(app.state, "entity_index_gap_status", None)):
                        return
                    await asyncio.sleep(0.01)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.run(_run_until_covered_or_timeout())
        assert _has_pair(store, "gap-fact", "E00"), (
            "the gap sweep never covered a fact written after the backfill "
            "already reported complete"
        )
        assert app.state.entity_index_gap_status["source"] == "background_gap_sweep"


class TestGapSweepDelayPacing:
    """Pure backoff policy -- no database or event loop needed."""

    def test_normal_tick_pace_while_making_progress(self) -> None:
        assert gap_sweep_delay(0, tick_seconds=0.1) == 0.1

    def test_tick_pace_is_capped_at_the_max_tick(self) -> None:
        from superlocalmemory.server.entity_index_repair import MAX_TICK_SECONDS
        assert gap_sweep_delay(0, tick_seconds=99.0) == MAX_TICK_SECONDS

    def test_backs_off_after_an_empty_lap(self) -> None:
        first = gap_sweep_delay(1, tick_seconds=0.1)
        second = gap_sweep_delay(2, tick_seconds=0.1)
        third = gap_sweep_delay(3, tick_seconds=0.1)
        assert first < second < third

    def test_backoff_is_capped(self) -> None:
        assert gap_sweep_delay(1_000_000, tick_seconds=0.1) == 3600.0

    def test_finding_something_resets_the_backoff(self) -> None:
        """The caller resets consecutive_empty_laps to 0 whenever a lap
        inserts something -- this just pins that 0 means "fast again"."""
        assert gap_sweep_delay(0, tick_seconds=0.1) < gap_sweep_delay(5, tick_seconds=0.1)


class TestBackoffAppliesOnlyBetweenLaps:
    """CRIT catch: the backoff must pace how soon the NEXT lap starts, not
    every batch inside an active lap -- otherwise a lap that still has real
    work left would crawl at the backed-off pace instead of finishing
    promptly, making the worst-case time to notice and repair a freshly
    reintroduced gap balloon to hours instead of seconds."""

    def test_mid_lap_batches_use_the_fast_pace_even_after_backing_off(
        self, tmp_path: Path,
    ) -> None:
        app = SimpleNamespace(state=SimpleNamespace())
        sleeps: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            if len(sleeps) >= 4:
                raise asyncio.CancelledError

        # Batch 1: an empty lap completes (backs off). Batches 2-3: a NEW
        # lap is actively in progress (not yet lap_complete) and -- the
        # exact case that exposed the bug -- finds nothing to insert in
        # THIS batch either (an ordinary mid-lap batch scanning facts that
        # are already fully covered). The stale backoff state from batch 1
        # must not leak into these: lap_complete is false, so they must use
        # the fast tick pace regardless of insert count.
        results = [
            {"scanned": 0, "inserted": 0, "laps_completed": 1},
            {"scanned": 50, "inserted": 0, "laps_completed": 0},
            {"scanned": 50, "inserted": 0, "laps_completed": 0},
        ]

        async def _fake_to_thread(fn, *args, **kwargs):
            if fn.__name__ == "repair_coverage_gap":
                return results.pop(0) if results else {
                    "scanned": 0, "inserted": 0, "laps_completed": 1,
                }
            return {"state": "running", "scanned": 0, "inserted": 0,
                    "last_error": "", "updated_at": ""}

        with patch("asyncio.sleep", side_effect=_fake_sleep), \
             patch("asyncio.to_thread", side_effect=_fake_to_thread):
            try:
                asyncio.run(run_entity_index_gap_sweep(
                    app, tmp_path / "unused.db", batch_size=50, tick_seconds=0.05,
                ))
            except asyncio.CancelledError:
                pass

        assert sleeps[0] > MAX_TICK_SECONDS, (
            "the first (empty) lap must back off before the next one starts"
        )
        assert sleeps[1] == MAX_TICK_SECONDS or sleeps[1] == 0.05, (
            f"a batch still inside an active lap must use the fast tick "
            f"pace, not the backoff from the previous lap: {sleeps}"
        )
