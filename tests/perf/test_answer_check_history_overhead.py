# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What the Answer Check history costs a recall: one record, microseconds.

Measured 16-22 us median on an M5 Pro under load and ~31 us on a GitHub runner
during a parallel suite (4.2.6-4.2.8; the code did not change). The gates sit
well above that and far below what an O(n) pass over the 2,048-entry ring or a
disk write per call would cost (milliseconds), so they catch the regression
they exist for, not machine noise.
"""

from __future__ import annotations

import statistics
import time

from superlocalmemory.core import answer_check_history as h

from ..test_core.test_answer_check_history import make_response

BATCHES = 200
PER_BATCH = 250


def test_record_overhead_micro() -> None:
    # Time batches, not single calls: a lone ~4 us call is dominated by timer
    # and scheduler noise on shared CI runners (a 29 us "median" blocked the
    # v4.2.6 and v4.2.7 release gates while the code itself measured ~4 us).
    h._reset_for_testing()
    h.enable(True)
    try:
        resp = make_response()
        for _ in range(2_000):                       # warm
            h.record_recall_verdict(resp, profile_id="default")
        per_call = []
        for _ in range(BATCHES):
            t0 = time.perf_counter_ns()
            for _ in range(PER_BATCH):
                h.record_recall_verdict(resp, profile_id="default")
            per_call.append((time.perf_counter_ns() - t0) / PER_BATCH)
        per_call.sort()
        median_us = statistics.median(per_call) / 1000
        p95_us = per_call[int(len(per_call) * 0.95)] / 1000
        print(f"\nrecord_recall_verdict: median {median_us:.2f} us, p95 {p95_us:.2f} us "
              f"per call over {BATCHES}x{PER_BATCH} (ring full: {len(h._ring) == h.RING_CAPACITY})")
        assert median_us < 100.0, median_us
        assert p95_us < 400.0, p95_us
    finally:
        h._reset_for_testing()
