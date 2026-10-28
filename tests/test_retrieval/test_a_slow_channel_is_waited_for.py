# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""A busy machine must not change the answer.

The channel phase runs five producers in parallel and waits once for all of
them. Whatever has not finished by then is cancelled and contributes NOTHING to
fusion -- its candidates are absent, not merely late. So the wait is the one
place in recall where the answer depends on what else the machine happened to
be doing.

Six runs of identical code against the same store logged 0, 0, 25, 2, 0 and 0
abandoned channels: in the third, `hopfield` was cut off on 13 of 140 queries
and `temporal` on 9, while the first lost nothing on those same queries. No
test covered this, which is how the suite stayed green at 9,740 passed while
recall was returning different answers to the same question.

The wait therefore has to be a guard against a wedged channel, not a latency
trim, and when it does bind the caller has to be told the answer is
incomplete. These tests pin both, and pin the ordering that justifies the
value: HARD-RULES RULE 6 puts Complete and Repeatable above Fast.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time

from superlocalmemory.retrieval import engine as engine_mod
from superlocalmemory.retrieval.engine import CHANNEL_HANG_GUARD_SECONDS


class TestTheGuardIsSizedAboveMeasuredChannelCost:
    """The value is a measurement, not a preference.

    Measured on the 0.95 GB archive, 140 queries, with the limit raised out of
    the way so nothing was truncated: the slowest channel (temporal) had a p95
    of 580 ms and a worst single run of 1,983 ms.
    """

    SLOWEST_CHANNEL_WORST_OBSERVED_MS = 1983

    def test_it_clears_the_worst_observed_channel_several_times_over(self) -> None:
        """A channel that is merely slow must never be dropped.

        At 1.4 s the worst observed single channel run (1,983 ms) already
        exceeded the limit, so the old value dropped channels on an idle
        machine, never mind a loaded one.
        """
        headroom = CHANNEL_HANG_GUARD_SECONDS * 1000.0 / self.SLOWEST_CHANNEL_WORST_OBSERVED_MS
        assert headroom >= 3.0, (
            f"the guard is only {headroom:.1f}x the worst channel run ever "
            f"measured ({self.SLOWEST_CHANNEL_WORST_OBSERVED_MS} ms); a busy "
            "machine will cross it and silently drop answers"
        )

    def test_public_deadline_can_finish_before_the_sdk_hang_guard(self) -> None:
        """The outer budget is what catches a true hang -- and it says so.

        `_recall_budget_s` is documented as a last-resort net for a genuine
        hang, not a speed cutoff, and it tells the caller when it fires
        (`retrieval_mode=degraded_lexical`). An inner limit above it would make
        the outer one unreachable and take that disclosure away.
        """
        from superlocalmemory.server.unified_daemon import _recall_budget_s

        outer = _recall_budget_s()
        # WP-6: the HTTP caller gets an explicit unjudged timeout at 3 s;
        # SDK/background work retains the measured channel correctness guard.
        assert outer <= 3.0
        assert CHANNEL_HANG_GUARD_SECONDS > outer


    def test_it_is_not_quietly_retightened_to_a_latency_figure(self) -> None:
        """Guards against the change this replaced.

        1.4 s was picked to keep a latency percentile inside a ceiling. Buying
        a percentile with missing answers inverts RULE 6's ordering, so a value
        back down in that range needs to fail loudly rather than slip through.
        """
        assert CHANNEL_HANG_GUARD_SECONDS >= 4.0, (
            "the guard is back in the range where it trims latency by "
            "discarding channels; RULE 6 orders Complete and Repeatable above "
            "Fast, so measure the quality cost and record it before lowering it"
        )


class _Recorder:
    """Collects the wait's outcome without running a real engine."""

    def __init__(self) -> None:
        self.dropped: set[str] = set()
        self.out: dict[str, list[tuple[str, float]]] = {}


def _run_wait(futures: dict[str, concurrent.futures.Future], timeout: float,
              rec: _Recorder) -> None:
    """The production wait-and-collect, isolated from engine construction.

    Building a RetrievalEngine needs a database, five channels and an embedder;
    none of that changes what is under test here, which is what the wait does
    with a future that has not finished. Kept deliberately identical in shape
    to `_run_channels`, so a divergence in behaviour shows up as a divergence
    in this helper.
    """
    _done, pending = concurrent.futures.wait(futures.values(), timeout=timeout)
    for name, fut in futures.items():
        if fut in pending:
            rec.dropped.add(name)
            fut.cancel()
            continue
        ch_name, result = fut.result()
        if result:
            rec.out[ch_name] = result


class TestASlowChannelStillContributes:
    def test_a_channel_slower_than_the_old_cutoff_is_waited_for(self) -> None:
        """The regression that mattered, expressed as behaviour.

        1.6 s is inside the guard and outside the 1.4 s cutoff it replaced, so
        this test fails on the old value and passes on the new one -- and it
        fails for the right reason: the channel's candidates are missing from
        the fused result, not merely reported late.
        """
        rec = _Recorder()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            def slow():
                time.sleep(1.6)
                return ("temporal", [("fact-a", 0.9)])

            def quick():
                return ("bm25", [("fact-b", 0.5)])

            futures = {"temporal": ex.submit(slow), "bm25": ex.submit(quick)}
            _run_wait(futures, CHANNEL_HANG_GUARD_SECONDS, rec)

        assert rec.dropped == set(), f"a 1.6 s channel was abandoned: {rec.dropped}"
        assert "temporal" in rec.out, (
            "the slow channel finished but its candidates never reached fusion"
        )
        assert rec.out["temporal"] == [("fact-a", 0.9)]

    def test_a_wedged_channel_is_abandoned_and_named(self) -> None:
        """The other half: the guard must still work, and must not be silent.

        Without this, raising the limit could be satisfied by never abandoning
        anything, which turns a wedged channel into a hung recall.
        """
        rec = _Recorder()
        release = threading.Event()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                def wedged():
                    release.wait(30)
                    return ("hopfield", [("fact-c", 0.1)])

                futures = {"hopfield": ex.submit(wedged)}
                _run_wait(futures, 0.2, rec)

                assert rec.dropped == {"hopfield"}
                assert "hopfield" not in rec.out, (
                    "a channel that never finished still contributed candidates"
                )
                release.set()
        finally:
            release.set()


class TestTheCallerIsToldWhenTheAnswerIsIncomplete:
    """A dropped channel that nobody reports is indistinguishable from no data.

    Two runs of one query can then differ with nothing in the response to
    explain it, which is what made this defect survive so long.
    """

    def test_the_response_carries_the_field(self) -> None:
        from superlocalmemory.storage.models import RecallResponse

        assert RecallResponse().incomplete_channels == (), (
            "the default must be empty, so a complete answer is not reported "
            "as degraded"
        )

    def test_it_is_ordered_so_the_field_is_itself_repeatable(self) -> None:
        """A set's iteration order would make the receipt unstable.

        Reporting ('temporal','hopfield') on one run and the reverse on the
        next would put non-determinism into the very field added to expose it.
        """
        from superlocalmemory.storage.models import RecallResponse

        dropped = {"temporal", "hopfield", "bm25"}
        first = RecallResponse(incomplete_channels=tuple(sorted(dropped)))
        second = RecallResponse(incomplete_channels=tuple(sorted(dropped)))
        assert first.incomplete_channels == second.incomplete_channels
        assert first.incomplete_channels == ("bm25", "hopfield", "temporal")

    def test_recall_populates_it_from_the_wait(self) -> None:
        """Proves the field is wired to the wait, not merely declared.

        A field that always stays empty would satisfy both tests above while
        leaving every dropped channel just as invisible as before.
        """
        import inspect

        src = inspect.getsource(engine_mod.RetrievalEngine.recall)
        assert "dropped_channels" in src, (
            "recall() does not pass a sink to _run_channels, so a dropped "
            "channel cannot reach the response"
        )
        assert "incomplete_channels=tuple(sorted(dropped_channels))" in src, (
            "the collected set is never attached to the response"
        )

    def test_the_sink_is_not_shared_between_concurrent_recalls(self) -> None:
        """It must be per-call, or two recalls report each other's losses.

        This is the v3.4.64 race the channel-disable set already had: shared
        mutable state on the engine, mutated by whichever recall is running.
        """
        import inspect

        src = inspect.getsource(engine_mod.RetrievalEngine.recall)
        assert "self._dropped_channels" not in src, (
            "the sink lives on the engine, so concurrent recalls will corrupt "
            "each other's incompleteness report"
        )
        sig = inspect.signature(engine_mod.RetrievalEngine._run_channels)
        assert sig.parameters["dropped_channels"].default is None, (
            "a mutable default would accumulate across every recall in the "
            "process and report channels that this query never lost"
        )


class TestTheWaitEndsBeforeTheCallerStopsListening:
    """#304: the guard never outlasts the caller.

    On a 37k-fact store three channels ran past the explicit recall's deadline,
    so the server gave up and the engine kept waiting for an answer nobody would
    read. With the caller's deadline known, the wait ends in time to fuse what
    finished, and the dropped channels are named as for any other timeout.
    """

    def test_no_caller_deadline_keeps_the_full_guard(self) -> None:
        assert engine_mod._channel_wait_s() == CHANNEL_HANG_GUARD_SECONDS

    def test_a_distant_deadline_keeps_the_full_guard(self) -> None:
        from superlocalmemory.retrieval.answer_check_status import caller_deadline

        with caller_deadline(time.monotonic() + 60.0):
            assert engine_mod._channel_wait_s() == CHANNEL_HANG_GUARD_SECONDS

    def test_a_near_deadline_leaves_room_to_answer(self) -> None:
        from superlocalmemory.retrieval.answer_check_status import caller_deadline

        with caller_deadline(time.monotonic() + 5.0):
            wait = engine_mod._channel_wait_s()
        assert wait <= 5.0 - engine_mod.CHANNEL_RESERVE_S + 0.05
        assert wait >= engine_mod.CHANNEL_WAIT_FLOOR_S

    def test_a_passed_deadline_still_gives_the_floor(self) -> None:
        from superlocalmemory.retrieval.answer_check_status import caller_deadline

        with caller_deadline(time.monotonic() - 1.0):
            assert engine_mod._channel_wait_s() == engine_mod.CHANNEL_WAIT_FLOOR_S
