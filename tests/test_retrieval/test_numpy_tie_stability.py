# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Ordering must be total: equal scores break on fact_id, not on array position.

Two independent findings in the numpy retrieval paths:

  ann_index.py  — argpartition + argsort are score-only.  When N facts share
                  a cosine score, which ones occupy the top-k depends on the
                  memory layout of the underlying array, which is determined
                  by insertion order.  The same query over the same store
                  returns a different candidate set after a restart that
                  reloads facts in a different order.

  hopfield_channel.py — argsort(-similarities) is also score-only.  Identical
                  Hopfield similarity scores (e.g. duplicate fact vectors) are
                  ordered by their position in the memory matrix, which tracks
                  the order facts were loaded from the DB.  That order varies
                  across snapshots, sessions, and DB cursor implementations.

Both files require lexsort((fact_ids, -scores)) so that ties are broken by
the fact_id string — a stable key that is independent of storage layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from superlocalmemory.retrieval.ann_index import ANNIndex

# ---------------------------------------------------------------------------
# ANNIndex — argpartition / argsort tie-breaking
# ---------------------------------------------------------------------------


class TestANNTieStability:
    """argpartition + argsort choose candidates and rank them by score only.

    When facts share a cosine score the selected set and its order depend on
    array index, which tracks insertion order and changes between restarts.
    The fix uses lexsort((fact_ids, -scores)) so both the selection and the
    ranking carry a deterministic secondary key.
    """

    def test_same_candidates_regardless_of_insertion_order(self) -> None:
        """The top-k set must be the same however the facts were inserted.

        Two indexes with the same facts in opposite orders must return the same
        fact_ids from an identical query when scores are tied.
        """
        vec = [1.0, 0.0, 0.0, 0.0]

        idx_ab = ANNIndex(dimension=4)
        idx_ab.add("fact_a", vec)
        idx_ab.add("fact_b", vec)

        idx_ba = ANNIndex(dimension=4)
        idx_ba.add("fact_b", vec)
        idx_ba.add("fact_a", vec)

        query = vec
        ids_ab = [fid for fid, _ in idx_ab.search(query, top_k=2)]
        ids_ba = [fid for fid, _ in idx_ba.search(query, top_k=2)]

        assert ids_ab == ids_ba, (
            "tied-score results differ by insertion order: "
            f"forward={ids_ab}, reverse={ids_ba}. "
            "The tie must be broken on fact_id, not on array position."
        )

    def test_tied_scores_ordered_by_fact_id_ascending(self) -> None:
        """Among equal-score facts the order must be fact_id ascending.

        fact_b is inserted first; with score-only sorting it wins the tie.
        With the fix, fact_a comes first (lexicographically smaller).
        """
        vec = [1.0, 0.0, 0.0, 0.0]

        idx = ANNIndex(dimension=4)
        idx.add("fact_b", vec)   # inserted first — scores first without tie-break
        idx.add("fact_a", vec)

        query = vec
        results = idx.search(query, top_k=2)
        ids = [fid for fid, _ in results]

        assert ids[0] == "fact_a", (
            f"fact_b (later-inserted, lexicographically larger) ranked first: {ids}. "
            "Tie must break on fact_id ascending, not on insertion order."
        )
        assert ids[1] == "fact_b"

    def test_tie_break_at_top_k_boundary(self) -> None:
        """When more facts tie than fit in top_k, the selected k must be stable.

        Three facts share identical cosine scores; top_k=2. The two selected
        must be the same regardless of insertion order, chosen by the lowest
        fact_ids lexicographically.
        """
        vec = [1.0, 0.0, 0.0, 0.0]

        idx_abc = ANNIndex(dimension=4)
        for fid in ("fact_a", "fact_b", "fact_c"):
            idx_abc.add(fid, vec)

        idx_cba = ANNIndex(dimension=4)
        for fid in ("fact_c", "fact_b", "fact_a"):
            idx_cba.add(fid, vec)

        query = vec
        ids_abc = [fid for fid, _ in idx_abc.search(query, top_k=2)]
        ids_cba = [fid for fid, _ in idx_cba.search(query, top_k=2)]

        assert ids_abc == ids_cba, (
            f"top-2 set differs by insertion order: abc={ids_abc}, cba={ids_cba}"
        )
        # The two lexicographically smallest fact_ids must win the boundary.
        assert set(ids_abc) == {"fact_a", "fact_b"}, (
            f"wrong facts selected at boundary: {ids_abc}"
        )


# ---------------------------------------------------------------------------
# HopfieldChannel._search_full_matrix — argsort tie-breaking
# ---------------------------------------------------------------------------


class TestHopfieldTieStability:
    """np.argsort(-similarities) is score-only.

    When two facts have equal Hopfield similarity (e.g. identical stored
    vectors), their relative order follows array index, which tracks the
    order rows were built into the memory matrix.  That order tracks the
    order facts were loaded from the DB cursor, which is not guaranteed
    to be stable across sessions.

    The fix: np.lexsort((np.array(fact_ids, dtype=object), -similarities))
    so ties break on fact_id.
    """

    @pytest.fixture()
    def hopfield(self):
        """Minimal HopfieldChannel with mock DB and vector_store."""
        from unittest.mock import MagicMock

        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        db = MagicMock()
        db.get_all_facts.return_value = []
        db.get_facts_by_ids.return_value = []
        db.get_external_visible_embeddings.return_value = []
        db.get_fact_embeddings_by_ids.return_value = []
        vs = MagicMock()
        vs.available = False
        vs.count.return_value = 0
        return HopfieldChannel(db=db, vector_store=vs)

    def test_same_ranking_regardless_of_matrix_row_order(self, hopfield) -> None:
        """Identical similarity scores must rank by fact_id, not row index."""
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        query = vec.copy()

        # Two facts with identical vectors → identical similarities
        memory_ab = np.array([vec, vec], dtype=np.float32)
        fact_ids_ab = ["fact_a", "fact_b"]

        memory_ba = np.array([vec, vec], dtype=np.float32)
        fact_ids_ba = ["fact_b", "fact_a"]

        results_ab = hopfield._search_full_matrix(query, memory_ab, fact_ids_ab, top_k=2)
        results_ba = hopfield._search_full_matrix(query, memory_ba, fact_ids_ba, top_k=2)

        ids_ab = [fid for fid, _ in results_ab]
        ids_ba = [fid for fid, _ in results_ba]

        assert ids_ab == ids_ba, (
            "Hopfield ties resolved differently by matrix row order: "
            f"ab={ids_ab}, ba={ids_ba}. "
            "Tie must break on fact_id, not on row index."
        )

    def test_tied_hopfield_results_ordered_by_fact_id_ascending(self, hopfield) -> None:
        """fact_b in row 0 scores identically to fact_a in row 1.

        Without a tie-break, fact_b wins because it is at index 0 in the
        argsort output.  With the fix, fact_a comes first.
        """
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        query = vec.copy()

        # fact_b first in matrix so it would win score-only argsort
        memory = np.array([vec, vec], dtype=np.float32)
        fact_ids = ["fact_b", "fact_a"]

        results = hopfield._search_full_matrix(query, memory, fact_ids, top_k=2)
        ids = [fid for fid, _ in results]

        assert ids[0] == "fact_a", (
            f"fact_b (row 0, lexicographically larger) ranked first: {ids}. "
            "Tie must break on fact_id ascending."
        )
        assert ids[1] == "fact_b"
