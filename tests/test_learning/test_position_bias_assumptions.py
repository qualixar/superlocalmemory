# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Fable audit A-20: the position curve states its assumptions and does not select on outcomes.

(1) A swap at pair k has probability SWAP_SHARE / pairs(n), so swapped cells over-represent
short lists. Deliveries are weighted by pairs(n) so both arms stand for the same mix.
(2) A pair with an empty cell was dropped: a selection on the outcome. A pair is now
measured when every cell was shown MIN_CELL_SHOWN times (a design count), and a zero
use count gives a finite rate and a wide interval.
(3) The module no longer says "without bias".
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

import pytest

from superlocalmemory.learning import position_bias as pb
from superlocalmemory.outcomes import swaps

TRUE = [1.0, 0.7, 0.5, 0.35, 0.25]
Z = NormalDist().inv_cdf(1 - pb.MODEL_ALPHA / 8)


def _rows(rng, n_deliveries, *, rel, attention, lengths=(2, 3, 4, 5, 6, 7, 8), share=0.25):
    """(delivery, swap_at, shown_position, used, list_length) like the ledger query returns."""
    out = []
    for i in range(n_deliveries):
        n = rng.choice(lengths)
        m = swaps.pairs(n)
        at = 1 + int(rng.random() * m) if rng.random() < share and m else None
        order = list(range(1, n + 1))
        if at:
            order[at - 1], order[at] = order[at], order[at - 1]
        for pos in order:
            if pos > swaps.SWAP_DEPTH:
                continue
            p = rel(n) * attention(n, pos)
            out.append((i, at, pos, rng.random() < p, n))
    return out


def _ratios(rows, *, weighted=True, monkeypatch=None):
    if not weighted:
        monkeypatch.setattr(pb, "pairs", lambda n: 1)      # every swap weighs 1, as before
    cells = pb._cells(rows)
    return [pb._ratio(k, cells[k], Z) for k in sorted(cells)]


# -- (1) list length ---------------------------------------------------------------------------------

def test_list_length_that_changes_the_curve_no_longer_biases_the_ratio(monkeypatch) -> None:
    """Short lists (n <= 4) have a flatter curve (p ** 0.3) than long ones, and a different
    relevance. The truth is the population ratio over the list-length mix. Pair 1 weighted
    lands within 0.01 of it; the unweighted ratio is off by about 0.02 (seeded, 150k deliveries)."""
    lengths = (2, 3, 4, 5, 6, 7, 8)
    rel = lambda n: 0.9 if n <= 3 else 0.5                                  # noqa: E731
    att = lambda n, pos: TRUE[pos - 1] ** (1.0 if n >= 5 else 0.3)          # noqa: E731
    rows = _rows(random.Random(9), 150_000, rel=rel, attention=att)

    def population(k):
        num = den = 0.0
        for n in lengths:
            if n >= k + 1:
                num += rel(n) * att(n, k + 1)
                den += rel(n) * att(n, k)
        return num / den

    weighted = _ratios(rows)[0].estimate
    unweighted = _ratios(rows, weighted=False, monkeypatch=monkeypatch)[0].estimate
    assert abs(weighted - population(1)) < 0.01
    assert abs(unweighted - population(1)) > abs(weighted - population(1))


def test_a_swapped_delivery_stands_for_the_lists_it_could_have_been_swapped_in() -> None:
    rows = [("a", 1, 2, True, 2), ("b", 1, 2, True, 8), ("c", None, 1, True, 8)]
    cells = pb._cells(rows)[1]
    assert (cells[0].w, cells[0].shown) == (1.0 + 4.0, 2)   # pairs(2) = 1, pairs(8) = 4
    assert (cells[1].w, cells[1].shown) == (1.0, 1)         # a control weighs 1


# -- (2) empty cells ---------------------------------------------------------------------------------

def test_a_zero_use_cell_gives_a_finite_wide_interval_not_a_dropped_pair() -> None:
    sparse = [pb.Cell.of(0, 400), pb.Cell.of(4, 400), pb.Cell.of(0, 400), pb.Cell.of(4, 400)]
    dense = [pb.Cell.of(0 + 20, 400), pb.Cell.of(40, 400), pb.Cell.of(20, 400),
             pb.Cell.of(40, 400)]
    r, d = pb._ratio(1, sparse, Z), pb._ratio(1, dense, Z)
    assert r.measured and 0 < r.estimate <= 1 and r.low < r.estimate
    assert math.log(r.estimate / r.low) > math.log(d.estimate / d.low)      # wider: less data


def test_a_pair_is_measured_by_how_often_items_were_shown_not_by_how_often_they_were_used() -> None:
    thin = [pb.Cell.of(5, pb.MIN_CELL_SHOWN - 1)] + [pb.Cell.of(5, 200)] * 3
    assert not pb._ratio(1, thin, Z).measured
    enough = [pb.Cell.of(0, pb.MIN_CELL_SHOWN)] + [pb.Cell.of(0, pb.MIN_CELL_SHOWN)] * 3
    assert pb._ratio(1, enough, Z).measured                  # no use anywhere: still measured


def test_with_sparse_use_the_joint_interval_still_covers_the_truth() -> None:
    """Relevance 0.02 over 4,000 deliveries: a cell has about 4 uses, and 64% of the
    60 seeded runs have a zero cell somewhere. The old rule dropped those pairs and
    kept only the lucky ones (a selection on the outcome). Now every pair is measured,
    and the four ratios' joint interval covers the truth in at least 90% of runs
    (observed 99%; nominal 97.5% for MODEL_ALPHA, the smoothing is conservative)."""
    covered = measured = zero_cells = 0
    for seed in range(60):
        rows = _rows(random.Random(seed), 4000, rel=lambda n: 0.02,
                     attention=lambda n, pos: TRUE[pos - 1], lengths=(5, 6, 7, 8))
        ratios = _ratios(rows)
        zero_cells += any(u == 0 for r in ratios for u, _ in r.cells)
        measured += all(r.measured for r in ratios)
        covered += all(r.low <= TRUE[r.pair] / TRUE[r.pair - 1] <= r.high for r in ratios)
    assert zero_cells >= 20                       # the test really exercises empty cells
    assert measured == 60
    assert covered / 60 >= 0.90


def test_the_curve_is_unmeasured_when_a_cell_was_never_shown() -> None:
    no_swaps = _rows(random.Random(2), 3000, rel=lambda n: 0.5,
                     attention=lambda n, pos: TRUE[pos - 1], share=0.0)
    assert not all(r.measured for r in _ratios(no_swaps))


# -- (3) the wording ----------------------------------------------------------------------------------

def test_the_texts_state_assumptions_and_do_not_claim_absence_of_bias() -> None:
    from superlocalmemory.outcomes import swaps as swaps_module

    doc = (pb.__doc__ or "")
    assert "Assumptions" in doc
    assert "without bias" not in doc and "without bias" not in (swaps_module.__doc__ or "")
    assert "consistent" in (swaps_module.__doc__ or "") and "Examination model" in doc
    assert "List length" in doc and "Empty cells" in doc


@pytest.mark.parametrize("name", ["Cell", "MIN_CELL_SHOWN"])
def test_the_public_names_are_exported(name) -> None:
    assert name in pb.__all__
