# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Remember, then recall at once, while the embedding model is still loading.

The reported fresh-install defect (4.1.20 WP10): ``slm remember`` said
"Queryable", and ``slm recall`` straight after said "No confident match" for
about a minute. The contract this pins, on a fresh store through the real
engine: the memory is found, OR the answer says the search is incomplete —
never a confident "no match" from a search that did not run.
"""

from __future__ import annotations

import threading
import time

from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.retrieval import engine as retrieval_engine_mod


#: How long the fake model stays "loading" unless released. A liveness bound,
#: NOT a performance claim: every test releases it the moment its recall
#: returns, so only a recall that really waits on the model ever sits it out.
_MODEL_LOAD_HOLD_S = 30.0


class _LoadingModel:
    """An embedder whose model is still loading (``is_warm`` is False)."""

    def __init__(self, hold: float) -> None:
        self.release = threading.Event()
        # Set when any embed call has come back, i.e. the model "finished
        # loading" for that caller. Observed directly instead of guessing
        # from a stopwatch whether the recall waited for it.
        self.answered = threading.Event()
        self._hold = hold

    is_warm = False
    is_available = True

    def embed(self, text):
        self.release.wait(self._hold)
        self.answered.set()
        return [0.0] * 768


def _recall_with_loading_model(engine, query: str):
    """(response, elapsed, model_answered_before_the_recall_returned).

    The hang guard keeps its real value. This used to shrink it to 0.5 s back
    when it also bounded the cold embed wait; that wait is
    ``COLD_QUERY_EMBED_WAIT_SECONDS`` now, so the shrunken guard only gave the
    keyword channels 0.5 s to answer -- which a loaded host can miss, dropping
    the very memory this test looks for.
    """
    loading = _LoadingModel(hold=_MODEL_LOAD_HOLD_S)
    engine._retrieval_engine._embedder = loading
    try:
        t0 = time.monotonic()
        response = engine.recall(query, limit=5)
        return response, time.monotonic() - t0, loading.answered.is_set()
    finally:
        loading.release.set()


def test_a_just_saved_memory_is_found_while_the_model_loads(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    fact_ids = engine.store(
        "The zebra-quokka migration window for Project Halcyon is 14 March")
    assert fact_ids, "store returned no queryable fact"

    response, elapsed, model_answered = _recall_with_loading_model(
        engine, "When is the Halcyon migration window?")

    found = [r.fact.content for r in response.results]
    assert any("Halcyon" in c for c in found), (
        f"just-saved memory not found while the model loads: {found}, "
        f"status={response.channel_status}")
    # The recall came back while the model was still loading: it did not wait
    # for the model. Observed, not inferred from how long it took.
    assert not model_answered, (
        f"recall returned only after the loading model answered ({elapsed:.1f}s)")
    # And it did not sit out the hang guard either (the 8.06 s defect): a
    # recall that waited the guard takes at least the guard, by construction.
    # How FAST the answer is belongs to the ceiling test, not this one.
    assert elapsed < retrieval_engine_mod.CHANNEL_HANG_GUARD_SECONDS, (
        f"recall waited {elapsed:.1f}s on a loading model")
    assert response.channel_status.get("semantic") == chstat.WARMING
    assert "semantic" in response.incomplete_channels


def test_nothing_found_while_loading_is_reported_incomplete(
    engine_with_mock_deps,
) -> None:
    """A paraphrase only the vector channels could match is not found yet —
    and the answer must say why rather than claim the store has nothing."""
    engine = engine_with_mock_deps
    engine.store("The zebra-quokka migration window for Project Halcyon is 14 March")

    response, _, _ = _recall_with_loading_model(engine, "Xylophone pterodactyl")

    # Whatever came back (temporal may surface the recent memory), the answer
    # must never present itself as a complete search.
    assert "semantic" in response.incomplete_channels, (
        "an answer from a partial search was reported as complete")
    assert all(chstat.is_fault(response.channel_status[c])
               for c in response.incomplete_channels)
