# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for HopfieldConfig.prefilter_candidates default and routing behaviour.

Establishes two properties:
  1. The default pool size is 150 (not 1000), so Hopfield stays within the
     1-second latency budget on typical stores.
  2. Shrinking the pool to 150 does not change the returned fact set for a
     query whose answer ranks within the top-150 by cosine similarity —
     the no-quality-trade-off contract.

Test IDs starting at 100 to avoid collision with existing test_hopfield_channel.py
numbering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DIM = 768


def _normed_vec(seed: int, d: int = DIM) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    v /= max(float(np.linalg.norm(v)), 1e-8)
    return v.tolist()


@dataclass
class _Fact:
    fact_id: str
    profile_id: str = "p"
    embedding: list[float] | None = None


def _facts(n: int, d: int = DIM, profile: str = "p") -> list[_Fact]:
    return [
        _Fact(fact_id=f"f{i}", profile_id=profile, embedding=_normed_vec(100 + i, d))
        for i in range(n)
    ]


class _DB:
    """Complete FakeDB stub that satisfies all methods called by HopfieldChannel."""

    def __init__(self, facts: list[_Fact], reported_count: int | None = None) -> None:
        self._facts = facts
        self._reported_count = reported_count if reported_count is not None else len(facts)

    def get_fact_count(
        self, profile_id: str, include_global: bool = False, include_shared: bool = False
    ) -> int:
        return self._reported_count

    def get_all_facts(
        self,
        profile_id: str,
        limit: int | None = None,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[_Fact]:
        out = [f for f in self._facts if f.profile_id == profile_id]
        return out[:limit] if limit is not None else out

    def get_facts_by_ids(
        self,
        fact_ids: list[str],
        profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[_Fact]:
        wanted = set(fact_ids)
        return [f for f in self._facts if f.fact_id in wanted and f.profile_id == profile_id]

    def get_external_visible_facts(
        self, profile_id: str, include_global: bool = False, include_shared: bool = False
    ) -> list[_Fact]:
        # No cross-profile facts in these unit tests.
        return []

    def get_external_visible_embeddings(
        self, profile_id: str, include_global: bool = False, include_shared: bool = False
    ) -> list[tuple[str, np.ndarray]]:
        return []

    def get_fact_embeddings_by_ids(
        self,
        fact_ids: list[str],
        profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[tuple[str, np.ndarray]]:
        return [
            (f.fact_id, np.array(f.embedding, dtype=np.float32))
            for f in self.get_facts_by_ids(fact_ids, profile_id)
            if f.embedding is not None
        ]


class _VS:
    """Fake VectorStore with a configurable result list."""

    def __init__(
        self,
        available: bool = True,
        count_val: int = 0,
        results: list[tuple[str, float]] | None = None,
    ) -> None:
        self.available = available
        self._count_val = count_val
        self._results = results or []

    def count(self, profile_id: str | None = None) -> int:
        return self._count_val

    def search(
        self,
        query: list[float],
        top_k: int = 50,
        profile_id: str | None = None,
    ) -> list[tuple[str, float]]:
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# Test 100 — pin the default prefilter size
# ---------------------------------------------------------------------------

class TestPrefilterDefaultIsLargeEnoughToBeComplete:
    """The default candidate pool must be 500, and the two copies must agree.

    This stage decides final membership, so a fact the index ranks past this
    number never reaches it and can never be returned. The pool size is therefore
    a direct limit on which memories are reachable at all, not a tuning knob.

    150 was too small: it made everything the index ranked past 150th
    unreachable. 1000 was too large: it pushed retrieval channels past their
    deadline, and a channel that misses its deadline contributes nothing, so the
    same question stopped returning the same answer. 500 was measured to give
    3.3x the reachability of 150 for 142 ms, with zero channel timeouts and about
    860 ms of headroom under the 2 s recall ceiling.

    The value is declared in two places that must not drift apart, so both are
    checked here — a test that pinned only one would let them diverge silently.
    """

    def test_default_prefilter_candidates_is_500(self) -> None:
        from superlocalmemory.math.hopfield import HopfieldConfig

        cfg = HopfieldConfig()
        assert cfg.prefilter_candidates == 500, (
            f"Expected prefilter_candidates=500, got {cfg.prefilter_candidates}. "
            "Smaller silently makes facts beyond it unreachable; larger pushes "
            "channels past their deadline, which costs whole channels."
        )

    def test_the_two_declarations_agree(self) -> None:
        from superlocalmemory.core.config import HopfieldConfig as CoreHopfieldConfig
        from superlocalmemory.math.hopfield import HopfieldConfig as MathHopfieldConfig

        assert (
            MathHopfieldConfig().prefilter_candidates
            == CoreHopfieldConfig().prefilter_candidates
        ), (
            "the two prefilter_candidates defaults have drifted apart; whichever "
            "one the running code reads decides which memories are reachable"
        )


# ---------------------------------------------------------------------------
# Test 101 — the candidate pool decides whether the pre-filter runs at all
# ---------------------------------------------------------------------------

class TestPrefilterRoutesOnTheCandidatePool:
    """Whether the pre-filter runs is decided by the pool size, both ways.

    A store smaller than the pool needs no pre-filtering at all — every fact goes
    through, which is the whole point of a pool large enough to be complete. A
    store larger than the pool must pre-filter, or the matrix is unbounded.

    Both halves are checked. A test that only asserted the pre-filter runs would
    pass on a pool of 1 and read as if it proved something about routing.
    """

    def _search_calls(self, corpus_size: int) -> list[int]:
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        corpus = _facts(corpus_size)
        knn = [(f"f{i}", 0.95 - i * 0.01) for i in range(5)]
        db = _DB(corpus, reported_count=corpus_size)
        vs = _VS(available=True, count_val=corpus_size, results=knn)
        channel = HopfieldChannel(db=db, vector_store=vs)

        real_search = vs.search
        calls: list[int] = []

        def tracking_search(
            query: list[float], top_k: int = 50, profile_id: str | None = None,
        ) -> list[tuple[str, float]]:
            calls.append(top_k)
            return real_search(query, top_k, profile_id)

        vs.search = tracking_search  # type: ignore[method-assign]
        channel.search(_normed_vec(999), "p", top_k=5)
        return calls

    def test_a_store_smaller_than_the_pool_is_used_whole(self) -> None:
        """200 facts and a pool of 1000: nothing is filtered out.

        This is the behaviour the larger pool buys. Under a pool of 150 the same
        store was pre-filtered and everything the index ranked past 150th became
        unreachable.
        """
        assert self._search_calls(200) == [], (
            "the pre-filter ran on a store smaller than the candidate pool, so "
            "facts were discarded that the pool had room for"
        )

    def test_a_store_larger_than_the_pool_is_prefiltered(self) -> None:
        """The other half: the matrix must stay bounded on a large store."""
        calls = self._search_calls(1500)
        assert calls, (
            "the pre-filter did not run on a store larger than the candidate "
            "pool, so the matrix is unbounded"
        )
        assert calls[0] == 500, (
            f"the pre-filter asked for {calls[0]} candidates; it must ask for the "
            "full pool, or the pool size is not what decides reachability"
        )


# ---------------------------------------------------------------------------
# Test 102 — result set is stable when both pool sizes use the same candidates
# ---------------------------------------------------------------------------

class TestPoolSizeActuallyBoundsVSRequest:
    """The prefilter_candidates value controls how many candidates the channel
    asks the VS for, which in turn determines which facts ever reach Hopfield.

    A fact at VS-rank 151 cannot enter the Hopfield sub-matrix when
    prefilter=150, because the VS is only asked for 150 candidates.
    With prefilter=200 the same fact is inside the window and is therefore
    presented to the DB retrieval stage.

    We verify by intercepting db.get_fact_embeddings_by_ids() — the gate between the
    VS pre-filter and the Hopfield computation.  If a fact ID never appears
    as an argument to get_fact_embeddings_by_ids() it cannot affect any result,
    regardless of its embedding.

    This approach is independent of Hopfield's numerical output, which with
    many candidates and beta = 1/sqrt(d) produces a nearly-uniform softmax
    that does not reliably surface any single pattern in top-10.
    """

    def test_smaller_pool_misses_fact_ranked_beyond_its_boundary(self) -> None:
        """f151 reaches the DB only when prefilter allows it.

        Setup
        -----
        * 300 facts in the corpus, all for profile "p".
        * VS results list is ordered f0 (highest score) … f299 (lowest).
          f151 sits at index 151, making it the 152nd candidate.
        * reported_count=5000 > 200 > 150, so the prefilter path is taken
          for both runs.
        * DIM_SMALL=8 keeps computation fast; the test does not depend on
          Hopfield's output, only on which IDs were ever queried from DB.

        Assertions
        ----------
        With prefilter=150: VS returns f0–f149.  f151 is NOT in the candidate
          set → must NOT appear in the queried IDs set.

        With prefilter=200: VS returns f0–f199.  f151 IS in the candidate set
          → MUST appear in the queried IDs set.

        Backwards-compatibility check: if the VS stub reverts to returning a
        fixed list that ignores top_k, both runs would receive the same
        candidates and at least one assertion below must fail — the test would
        correctly report that the prefilter bound is not being honoured.
        """
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        DIM_SMALL = 8  # small dimension keeps Hopfield matrix ops fast

        def _sv(seed: int) -> list[float]:
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(DIM_SMALL).astype(np.float32)
            v /= max(float(np.linalg.norm(v)), 1e-8)
            return v.tolist()

        query = np.array(_sv(999), dtype=np.float32)
        n_facts = 300

        corpus = [
            _Fact(
                fact_id=f"f{i}",
                profile_id="p",
                embedding=query.tolist() if i == 151 else _sv(200 + i),
            )
            for i in range(n_facts)
        ]

        # VS result list: f0 best, f151 at index 151, f299 worst.
        vs_results = [
            (f"f{i}", max(0.0, 0.999 - i * 0.001)) for i in range(n_facts)
        ]

        def _run(prefilter: int) -> set[str]:
            """Return every fact ID ever passed to db.get_fact_embeddings_by_ids()."""
            queried_ids: set[str] = set()
            orig_db = _DB(corpus, reported_count=5000)

            class _TrackingDB:
                def get_fact_count(self, *a: Any, **kw: Any) -> int:
                    return orig_db.get_fact_count(*a, **kw)

                def get_all_facts(self, *a: Any, **kw: Any) -> list[_Fact]:
                    return orig_db.get_all_facts(*a, **kw)

                def get_fact_embeddings_by_ids(
                    self,
                    fact_ids: list[str],
                    profile_id: str,
                    include_global: bool = False,
                    include_shared: bool = False,
                ) -> list[tuple[str, np.ndarray]]:
                    queried_ids.update(fact_ids)
                    return orig_db.get_fact_embeddings_by_ids(
                        fact_ids, profile_id,
                        include_global=include_global,
                        include_shared=include_shared,
                    )

                def get_external_visible_embeddings(
                    self, *a: Any, **kw: Any
                ) -> list[tuple[str, np.ndarray]]:
                    return []

            cfg = HopfieldConfig(prefilter_candidates=prefilter, dimension=DIM_SMALL)
            vs = _VS(available=True, count_val=5000, results=vs_results)
            ch = HopfieldChannel(db=_TrackingDB(), vector_store=vs, config=cfg)
            ch.search(query.tolist(), "p", top_k=10)
            return queried_ids

        queried_at_150 = _run(150)
        queried_at_200 = _run(200)

        # With prefilter=200: f151 (at VS-rank 151) must reach the DB stage.
        assert "f151" in queried_at_200, (
            "f151 was not queried from DB with prefilter=200 even though it is "
            "at VS-rank 151, which is inside the 200-candidate window. "
            f"Queried IDs (first 20): {sorted(queried_at_200)[:20]}"
        )

        # With prefilter=150: f151 must NOT reach the DB stage.
        assert "f151" not in queried_at_150, (
            "f151 was queried from DB with prefilter=150 even though it is at "
            "VS-rank 151, outside the 150-candidate window.  Either the VS stub "
            "is not respecting top_k or the channel is ignoring prefilter_candidates. "
            f"Queried IDs (first 20): {sorted(queried_at_150)[:20]}"
        )
