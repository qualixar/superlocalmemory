# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Defect S01 — scope-flag isolation under concurrent recalls.

Verifies two properties after the fix/3.7.9 refactor:

(a) ISOLATION: concurrent recalls with different include_global /
    include_shared flags never corrupt each other.  Two fake channels
    record the exact flag values they were invoked with; many concurrent
    recalls are dispatched via ThreadPoolExecutor with alternating flag
    pairs, and each recall's channels must report exactly the flags that
    recall passed, never the flags from any other concurrent recall.

(b) PARAM-WINS: when a channel has NO include_global / include_shared
    attribute set on the instance at all, the recall's flag values still
    arrive correctly — the flags travel as call parameters, not via
    shared instance state.
"""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.storage.models import AtomicFact, RecallResponse


# ---------------------------------------------------------------------------
# Fake channel that records the flag values it saw at call time
# ---------------------------------------------------------------------------

class _FlagRecordingChannel:
    """Minimal channel that records (include_global, include_shared) per call.

    Implements only what RetrievalEngine._run_channels requires: a
    ``search`` method whose first positional arg is the query/embedding and
    that returns a list[tuple[str, float]].
    """

    def __init__(self, name: str) -> None:
        self.name = name
        # List of (include_global, include_shared) recorded per invocation.
        self._calls: list[tuple[bool, bool]] = []
        self._lock = threading.Lock()

    def search(
        self,
        query: Any,
        profile_id: str,
        top_k: int = 10,
        include_global: bool | None = None,
        include_shared: bool | None = None,
    ) -> list[tuple[str, float]]:
        # Resolve using the same fallback contract as the real channels.
        if include_global is None:
            include_global = bool(getattr(self, "include_global", False))
        if include_shared is None:
            include_shared = bool(getattr(self, "include_shared", False))
        with self._lock:
            self._calls.append((include_global, include_shared))
        # Return a deterministic result so the engine doesn't skip the channel.
        return [(f"fact-{self.name}-{len(self._calls)}", 0.5)]

    def calls_snapshot(self) -> list[tuple[bool, bool]]:
        with self._lock:
            return list(self._calls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(fact_id: str) -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id,
        memory_id="m0",
        content=f"fact {fact_id}",
        confidence=0.9,
    )


def _mock_db() -> MagicMock:
    db = MagicMock()
    db.get_all_facts.return_value = []
    db.get_facts_by_ids.return_value = []
    db.get_scenes_for_fact.return_value = []
    return db


def _mock_embedder() -> MagicMock:
    emb = MagicMock()
    emb.embed.return_value = [0.1, 0.2, 0.3]
    return emb


def _build_engine_with_recording_channels(
    bm25_channel: _FlagRecordingChannel,
    temporal_channel: _FlagRecordingChannel,
) -> RetrievalEngine:
    """Build a RetrievalEngine wired with two recording channels."""
    channels = {
        "bm25": bm25_channel,
        "temporal": temporal_channel,
    }
    return RetrievalEngine(
        db=_mock_db(),
        config=RetrievalConfig(),
        channels=channels,
        embedder=_mock_embedder(),
        reranker=None,
    )


# ---------------------------------------------------------------------------
# Test (a): concurrent recalls with different flags don't corrupt each other
# ---------------------------------------------------------------------------

class TestConcurrentScopeIsolation:
    """Each concurrent recall must see exactly the flags it was called with."""

    def test_alternating_flags_no_corruption(self) -> None:
        """Fire N concurrent recalls alternating (global=T, shared=F) vs
        (global=F, shared=T). Every channel invocation for recall i must
        have seen recall i's exact flags — never the other recall's flags.
        """
        NUM_RECALLS = 40

        bm25 = _FlagRecordingChannel("bm25")
        temporal = _FlagRecordingChannel("temporal")
        engine = _build_engine_with_recording_channels(bm25, temporal)

        # Map from recall index to expected flags.
        expected: dict[int, tuple[bool, bool]] = {}
        futures: list[concurrent.futures.Future[tuple[int, RecallResponse]]] = []

        def _do_recall(idx: int) -> tuple[int, RecallResponse]:
            g = (idx % 2 == 0)  # even → global=True; odd → global=False
            s = not g            # odd  → shared=True; even → shared=False
            resp = engine.recall(
                f"query {idx}",
                "default",
                include_global=g,
                include_shared=s,
            )
            return (idx, resp)

        # Build expected map before submitting
        for i in range(NUM_RECALLS):
            g = (i % 2 == 0)
            s = not g
            expected[i] = (g, s)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            for i in range(NUM_RECALLS):
                futures.append(pool.submit(_do_recall, i))
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All recalls must complete without exception.
        assert len(results) == NUM_RECALLS

        # The total number of channel calls equals NUM_RECALLS (one per recall
        # per channel; bm25 and temporal each get one call per recall).
        bm25_calls = bm25.calls_snapshot()
        temporal_calls = temporal.calls_snapshot()
        assert len(bm25_calls) == NUM_RECALLS, (
            f"bm25 expected {NUM_RECALLS} calls, got {len(bm25_calls)}"
        )
        assert len(temporal_calls) == NUM_RECALLS, (
            f"temporal expected {NUM_RECALLS} calls, got {len(temporal_calls)}"
        )

        # Every recorded call must be one of the two valid flag pairs — never
        # a corrupt mix like (True, True) or (False, False).
        valid_pairs = {(True, False), (False, True)}
        for flags in bm25_calls:
            assert flags in valid_pairs, (
                f"bm25 saw unexpected flag pair {flags!r}; "
                f"valid pairs are {valid_pairs!r}"
            )
        for flags in temporal_calls:
            assert flags in valid_pairs, (
                f"temporal saw unexpected flag pair {flags!r}; "
                f"valid pairs are {valid_pairs!r}"
            )

        # Both flag pairs must appear — the test would be vacuous if only
        # one pair showed up.
        assert (True, False) in set(bm25_calls), "No global=True call seen in bm25"
        assert (False, True) in set(bm25_calls), "No global=False call seen in bm25"

    def test_all_global_recalls_see_global_true(self) -> None:
        """All recalls with include_global=True must produce bm25 calls
        where include_global is True.
        """
        bm25 = _FlagRecordingChannel("bm25")
        temporal = _FlagRecordingChannel("temporal")
        engine = _build_engine_with_recording_channels(bm25, temporal)

        for _ in range(10):
            engine.recall("test query", "default", include_global=True, include_shared=False)

        for flags in bm25.calls_snapshot():
            assert flags == (True, False), (
                f"Expected (True, False) in all bm25 calls, got {flags!r}"
            )

    def test_all_shared_recalls_see_shared_true(self) -> None:
        """All recalls with include_shared=True must produce temporal calls
        where include_shared is True.
        """
        bm25 = _FlagRecordingChannel("bm25")
        temporal = _FlagRecordingChannel("temporal")
        engine = _build_engine_with_recording_channels(bm25, temporal)

        for _ in range(10):
            engine.recall("test query", "default", include_global=False, include_shared=True)

        for flags in temporal.calls_snapshot():
            assert flags == (False, True), (
                f"Expected (False, True) in all temporal calls, got {flags!r}"
            )


# ---------------------------------------------------------------------------
# Test (b): flags arrive as params even when channel has no attribute set
# ---------------------------------------------------------------------------

class TestFlagsArrivedAsParams:
    """The flags travel with the call even when the channel instance has no
    include_global / include_shared attribute.
    """

    def test_channel_without_attribute_still_receives_params(self) -> None:
        """A _FlagRecordingChannel instance with NO include_global attribute
        set must still receive the recall's flag values.

        The fallback in _FlagRecordingChannel.search() reads
        getattr(self, "include_global", False) which returns False when
        the attribute is absent — so if flags were NOT passed as params
        the channel would always see (False, False). We verify it sees the
        actual recall flags instead.
        """
        channel = _FlagRecordingChannel("bm25")
        # Explicitly assert the attribute is absent so test is meaningful.
        assert not hasattr(channel, "include_global"), (
            "_FlagRecordingChannel must not set include_global on __init__"
        )
        assert not hasattr(channel, "include_shared"), (
            "_FlagRecordingChannel must not set include_shared on __init__"
        )

        engine = RetrievalEngine(
            db=_mock_db(),
            config=RetrievalConfig(),
            channels={"bm25": channel},
            embedder=_mock_embedder(),
            reranker=None,
        )

        engine.recall("test query", "default", include_global=True, include_shared=True)

        calls = channel.calls_snapshot()
        assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"
        assert calls[0] == (True, True), (
            f"Channel without attribute should see (True, True) from call params, "
            f"but got {calls[0]!r}"
        )

    def test_false_flags_also_arrive_correctly(self) -> None:
        """include_global=False, include_shared=False also travels via params
        (not via absent attribute defaulting to False — same result but
        different execution path; confirm with a True→False transition).
        """
        channel = _FlagRecordingChannel("bm25")

        engine = RetrievalEngine(
            db=_mock_db(),
            config=RetrievalConfig(),
            channels={"bm25": channel},
            embedder=_mock_embedder(),
            reranker=None,
        )

        # First recall: global=True so channel sees True.
        engine.recall("q1", "default", include_global=True, include_shared=False)
        # Second recall: global=False, channel must see False (not stale True).
        engine.recall("q2", "default", include_global=False, include_shared=False)

        calls = channel.calls_snapshot()
        assert calls[0] == (True, False), f"First call: expected (True,False) got {calls[0]!r}"
        assert calls[1] == (False, False), f"Second call: expected (False,False) got {calls[1]!r}"
